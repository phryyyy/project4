import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
__all__ = ['UNet', 'NestedUNet']

SIZE = 32


class DPConv2(nn.Module):
    def __init__(self, features,out_features):
        """
            features：输入通道维度。
             M：分支数。
             G：卷积组的数量。
             r：计算 d 的比率，z 的长度。
             stride：步幅，默认1。
             L：论文中向量z的最小维度，默认32。
        """
        M = 3
        G = 32
        r = 16
        stride = 1
        L = 32
        super(DPConv2, self).__init__()
        d = max(int(features / r), L)  # d:特征向量z的维度 通过r来控制 默认最小为32
        self.M = M  # 分支数
        self.features = features
        self.convs = nn.ModuleList([])  # 存储各分支操作
        for i in range(M):  # 各分支 卷积 操作 循环生成
            self.convs.append(nn.Sequential(
                #nn.Conv2d(features, features, kernel_size=3 + (i * 2), stride=stride, padding=1 + i, groups=G,bias=False),
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, groups=G,bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # GAP
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(M):  # fc之后的支路 1x1代替Fc
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

        self.outconv = nn.Conv2d(features,out_features,3,padding=1)
        self.outbn = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)

        self.convT = nn.Conv2d(features*2,out_features,3,padding=1)
        self.bnT = nn.BatchNorm2d(out_features)
    def forward(self, x):

        batch_size = x.shape[0]  # 获取输入的batch

        feats = [conv(x) for conv in self.convs]  # 不同卷积核分支
        feats = torch.cat(feats, dim=1)  # cat获得融合特征U

        # 将特征图形状变成 （batch，分支数,各分支channal，x.H，x.W）
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])

        feats_U = torch.sum(feats, dim=1)  # channel 2->1  信道相加
        feats_S = self.gap(feats_U)  # gap HxW变1x1
        feats_Z = self.fc(feats_S)  # fc进行一个压缩

        attention_vectors = [fc(feats_Z) for fc in self.fcs]  # 1x1卷积获得各个支路的权重向量a，并将channel数增加到和特征图一致
        attention_vectors = torch.cat(attention_vectors, dim=1)  # 相加  channel*2
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)  # （batch，2，channel，1，1）
        attention_vectors = self.softmax(attention_vectors)  # softmax

        feats_V = torch.sum(feats * attention_vectors, dim=1)  # 支路特征图和自己支路的权值a相乘，之后sum成一个特征图后输出



        feats_out = self.outconv(feats_V)#卷积成需要的channel后输出
        feats_out = self.outbn(feats_out)
        feats_out = self.relu(feats_out)

        """feats_out = self.convT(feats)
        feats_out = self.bnT(feats_out)
        feats_out = self.relu(feats_out)"""
        return feats_out

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#scale_factor:放大的倍数  插值

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()



        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        #self.DPconv0_0 = DPConv2(features=input_channels, out_features=nb_filter[0])
        self.DPconv1_0 = DPConv2(features=nb_filter[0],out_features=nb_filter[1])
        self.DPconv2_0 = DPConv2(features=nb_filter[1], out_features=nb_filter[2])
        self.DPconv3_0 = DPConv2(features=nb_filter[2], out_features=nb_filter[3])
        self.DPconv4_0 = DPConv2(features=nb_filter[3], out_features=nb_filter[4])


        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.DPconv0_1 = DPConv2(features=nb_filter[0]+nb_filter[1], out_features=nb_filter[0])
        self.DPconv1_1 = DPConv2(features=nb_filter[1]+nb_filter[2], out_features=nb_filter[1])
        self.DPconv2_1 = DPConv2(features=nb_filter[2]+nb_filter[3], out_features=nb_filter[2])
        self.DPconv3_1 = DPConv2(features=nb_filter[3]+nb_filter[4], out_features=nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        self.DPconv0_2 = DPConv2(features=nb_filter[0]*2+nb_filter[1], out_features=nb_filter[0])
        self.DPconv1_2 = DPConv2(features=nb_filter[1]*2+nb_filter[2], out_features=nb_filter[1])
        self.DPconv2_2 = DPConv2(features=nb_filter[2]*2+nb_filter[3], out_features=nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        self.DPconv0_3 = DPConv2(features=nb_filter[0]*3+nb_filter[1], out_features=nb_filter[0])
        self.DPconv1_3 = DPConv2(features=nb_filter[1]*3+nb_filter[2], out_features=nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        self.DPconv0_4 = DPConv2(features=nb_filter[0]*4+nb_filter[1], out_features=nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.DPconv1_0(self.pool(x0_0))
        x0_1 = self.DPconv0_1(torch.cat([x0_0, self.up(x1_0)], 1))


        x2_0 = self.DPconv2_0(self.pool(x1_0))
        x1_1 = self.DPconv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.DPconv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))


        x3_0 = self.DPconv3_0(self.pool(x2_0))
        x2_1 = self.DPconv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.DPconv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.DPconv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.DPconv4_0(self.pool(x3_0))
        x3_1 = self.DPconv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.DPconv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.DPconv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.DPconv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))


        if self.deep_supervision:
            output1 = self.final1(x0_1)

            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

#获取权重值
class getWeights(nn.Module):
    def __init__(self,in_features):
        self.features = in_features
        super(getWeights, self).__init__()
        self.fc1 = nn.Linear(in_features,8)
        self.fc2 = nn.Linear(8,32)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        a = F.adaptive_avg_pool2d(x,(1,1))
        a = a.view(x.size(0),-1)
        a = self.softmax(self.fc2(self.relu(self.fc1(a))))
        #a = a.T
        a = a.unsqueeze(-1).unsqueeze(-1)
        return a #权重值

#pyramidal conv layer
#out_channels 需要等于class数
class PC(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(PC, self).__init__()
        self.inconv =nn.Conv2d(in_channels,out_channels,kernel_size=1,bias=False)
        self.yconv = nn.Conv2d(out_channels*5, out_channels, kernel_size=1,bias=False)
        self.mulconv = nn.Conv2d(out_channels,out_channels,kernel_size=1,bias=False)
        self.conv3x3 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,bias=False)
        self.conv5x5 = nn.Conv2d(out_channels,out_channels,kernel_size=5,padding=2,bias=False)
        self.conv7x7 = nn.Conv2d(out_channels,out_channels, kernel_size=7,padding=3,bias=False)
        self.conv9x9 = nn.Conv2d(out_channels,out_channels, kernel_size=9,padding=4,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.getW = getWeights(32)

    def forward(self,x):
        #不同卷积核获取特征图
        x1 = self.inconv(x)
        x1 = self.bn(x1)
        x1 = self.relu(x1)

        x2 = self.conv3x3(x1)
        x2 = self.bn(x2)
        x2 = self.relu(x2)

        x3 = self.conv5x5(x1)
        x3 = self.bn(x3)
        x3 = self.relu(x3)

        x4 = self.conv7x7(x1)
        x4 = self.bn(x4)
        x4 = self.relu(x4)

        x5 = self.conv9x9(x1)
        x5 = self.bn(x5)
        x5 = self.relu(x5)

        #权值信道相乘
        w1 = self.getW(x1)
        y1 = torch.mul(x1,w1)
        #y1 = self.mulconv(y1)

        w2 = self.getW(x2)
        y2 = torch.mul(x2, w2)
        #y2 = self.mulconv(y2)
        w3 = self.getW(x3)
        y3 = torch.mul(x3, w3)
        #y3 = self.mulconv(y3)
        w4 = self.getW(x4)
        y4 = torch.mul(x4, w4)
        #y4 = self.mulconv(y4)
        w5 = self.getW(x5)
        y5 = torch.mul(x5, w5)
        #y5 = self.mulconv(y5)

        #相加后通过1x1卷积来降低channel
        y = torch.cat((y1,y2,y3,y4,y5),dim=1)

        y = self.yconv(y)
        y = self.bn(y)
        y = self.relu(y)

        #输出
        out = x1+y

        return out

class DPNet(nn.Module):
    def __init__(self,out_channels):
        super(DPNet, self).__init__()
        self.pc1 = PC(10,10)
        self.pc2 = PC(256,10)
        self.pc3 = PC(512, 10)
        self.pc4 = PC(1024, 10)
        self.pc5 = PC(2048, 10)
        self.res = ResNet()

        self.inconv = nn.Conv2d(3,out_channels,kernel_size=7,stride=2,padding=3,bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv = nn.Conv2d(out_channels*2,out_channels,kernel_size=3,stride=1 ,padding=1,bias=False)
        self.convD = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1 ,padding=1,bias=False)
        self.convD2 = nn.Conv2d(out_channels, out_channels,kernel_size=3,padding=1,dilation=1,bias=False)
        self.convD3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.outConv = nn.Conv2d(out_channels*5,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(SIZE*SIZE*10,16)
        self.fc2 = nn.Linear(16,10)
        self.softmax = nn.Softmax(dim=1)

        self.conv5 = nn.Conv2d(2048, 10, kernel_size=1, stride=1,bias=False)
        self.conv4 = nn.Conv2d(1024, 10, kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(512, 10, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(256, 10, kernel_size=1, stride=1, bias=False)

    def forward(self,x):
        #获取X1~5
        X1 = self.relu(self.bn(self.inconv(x)))
        X1 = self.pc1(X1)#64
        X2, X3, X4, X5 =self.res(x)

        """X5 = self.conv5(X5)
        X4 = self.conv4(X4)
        X3 = self.conv3(X3)
        X2 = self.conv2(X2)"""
        #32
        X2 = self.pc2(X2)
        #16
        X3 = self.pc3(X3)
        #8
        X4 = self.pc4(X4)
        #4
        X5 = self.pc5(X5)

        #FPN
        X5 = self.relu(self.bn(self.convD(X5)))#4
        Xt = _upsample_like(X5, X4)#8

        X4 = self.relu(self.bn(self.conv(torch.cat((X4,Xt),1))))  # 8
        Xt = _upsample_like(X4, X3)  # 16

        X3 = self.relu(self.bn(self.conv(torch.cat((X3, Xt),1))))  # 16
        Xt = _upsample_like(X3, X2)  # 16

        X2 = self.relu(self.bn(self.conv(torch.cat((X2, Xt),1))))  # 32
        Xt = _upsample_like(X2, X1)  # 32

        X1 = self.relu(self.bn(self.conv(torch.cat((X1, Xt),1))))  # 64


        #下采样
        """xd = X1
        Xd1 = self.relu(self.bn(self.convD(xd)))# 64
        xd = self.pool(Xd1)

        Xd2 = self.relu(self.bn(self.convD(xd)))# 32
        xd = self.pool(Xd2)

        Xd3 = self.relu(self.bn(self.convD(xd)))  # 16
        xd = self.pool(Xd3)

        Xd4 = self.relu(self.bn(self.convD(xd)))  # 8
        xd = self.pool(Xd4)

        Xd5 = self.relu(self.bn(self.convD2(xd))) # 4

        # 上采样
        xu = self.relu(self.bn(self.convD3(xd))) # 4

        Xu5 = self.relu(self.bn(self.conv(torch.cat((xu, Xd5),1)))) # 4
        xu = _upsample_like(Xu5,Xd4)

        Xu4 = self.relu(self.bn(self.conv(torch.cat((xu, Xd4), 1))))  # 8
        xu = _upsample_like(Xu4, Xd3)

        Xu3 = self.relu(self.bn(self.conv(torch.cat((xu, Xd3), 1))))  # 16
        xu = _upsample_like(Xu3, Xd2)

        Xu2 = self.relu(self.bn(self.conv(torch.cat((xu, Xd2), 1))))  # 32
        xu = _upsample_like(Xu2, Xd1)

        Xu1 = self.relu(self.bn(self.conv(torch.cat((xu, Xd1), 1))))  # 64

        X1 = X1+Xu1+Xd1
        X1 = self.pool(X1)"""

        """X2 = X2+Xu2+Xd2
        X3 = X3+Xu3+Xd3
        X3 = _upsample_like(X3,X2)
        X4 = X4+Xu4+Xd4
        X4 = _upsample_like(X4, X2)
        X5 = X5+Xu5+Xd5"""

        X5 = _upsample_like(X5, X2)
        X1 = self.pool(X1)
        X3 = _upsample_like(X3, X2)
        X4 = _upsample_like(X4, X2)
        X5 = _upsample_like(X5, X2)

        X1 = self.relu(self.bn(self.outConv(torch.cat((X1,X2,X3,X4,X5),dim=1))))
        X1 = X1.view(-1,SIZE*SIZE*10)
        X1 = self.fc2(self.relu(self.fc1(X1)))

        """
        X5 = X5.view(-1,31360)
        X5 = self.fc2(self.relu(self.fc1(X5)))

        X4 = X4.view(-1,31360)
        X4 = self.fc2(self.relu(self.fc1(X4)))

        X3 = X3.view(-1,31360)
        X3 = self.fc2(self.relu(self.fc1(X3)))

        X2 = X2.view(-1,31360)
        X2 = self.fc2(self.relu(self.fc1(X2)))"""

        return X1#,X2,X3,X4,X5

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        X1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        X1 = F.max_pool2d(X1, kernel_size=3, stride=2, padding=1)
        X2 = self.layer1(X1)
        X3 = self.layer2(X2)
        X4 = self.layer3(X3)
        X5 = self.layer4(X4)
        return X2, X3, X4, X5

def _upsample_like(x1,x2):

    x1 = F.interpolate(x1,size=x2.shape[2:],mode='bilinear')

    return x1