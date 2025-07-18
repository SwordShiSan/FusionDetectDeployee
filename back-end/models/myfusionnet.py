import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FusionNet(nn.Module):


    def __init__(self):
        super(FusionNet, self).__init__()
        # 1.Feature extraction-----------------------------------------
        self.vis_conv1 = ConvLeakyRelu2d(1, 16)
        self.vis_rgbd1 = RGBD(16, 32)

        self.inf_conv1 = ConvLeakyRelu2d(1, 16)
        self.inf_rgbd1 = RGBD(16, 32)

        # 2.Attention enhancement--------------------------------------
        self.attention_fusion = ChannelSpatialAttention2(in_channels=32)

        # 3.Reconstruction--------------------------------------------
        self.sfc = SFC(32, 1, 4)

    def forward(self, vi, ir):
        # encode
        vi = self.vis_conv1(vi)
        vi = self.vis_rgbd1(vi)

        ir = self.inf_conv1(ir)
        ir = self.inf_rgbd1(ir)

        x = self.attention_fusion(vi, ir)  # (32,32)->32
        x = self.sfc(x) # 32->1
        return x
    
class ChannelSpatialAttention2(nn.Module):
    """
    ChannelSpatialAttention: Channel and Spatial Attention Parallel
    Construction: parallel
    Reference: refer from PSFusion SDFM;some change
    input: f_vi,f_ir;in_channels=f_vi/f_ir channels;(c,c)
    output: f;out_channels=in_channels;(c)
    版本: 在步骤1的时候权重的处理,需要满足互补的结构,
    当前版本: 选择version_1作为最终
        version_1:
            w包含了两个模态的信息
            f_cat = torch.cat([f_vi, f_ir], dim=1)
            w = self.channel_avg_attention(f_cat)   # c
            w = torch.cat([w, w], dim=1)     # (n,c,1,1)->(n,2c,1,1)
            f_cat = torch.cat([f_ir, f_vi], dim=1) * w + f_cat   # 互补,f_vi=f_vi+f_ir*w/f_ir同
            f_vi, f_ir = torch.split(f_cat, c, dim=1)   # (n,c,h,w) 增强后的f_vi,f_ir
        version_2:
            w只包含一个模态的信息,另一个模态是1-w;或者输出2w其中一个w给vi,另一个给ir
            f_cat = torch.cat([f_vi, f_ir], dim=1)  # (n,2c=vi:ir,h,w)
            w = self.channel_avg_attention(f_cat)  # (n,c,1,1) / (n,2c,1,1)
            f_vi_, f_ir_ = torch.split(f_cat * w, c, dim=1)
            f_vi = f_vi + f_ir_
            f_ir = f_ir + f_vi_
            f_cat = torch.cat([f_vi, f_ir], dim=1)
    """

    def __init__(self, in_channels=32):
        super(ChannelSpatialAttention2, self).__init__()
        self.in_channels = in_channels

        # cat:96->24->48
        self.channel_avg_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=2 * self.in_channels, out_channels=self.in_channels // 2, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(self.in_channels // 2),
            nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=self.in_channels // 2, out_channels=2 * self.in_channels, kernel_size=1, stride=1,
            #          padding=0),
            # nn.BatchNorm2d(2 * self.in_channels),
            nn.Conv2d(in_channels=self.in_channels // 2, out_channels=self.in_channels, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(self.in_channels),
            nn.Sigmoid(),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2 * self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.channel_attention = ChannelAttention(self.in_channels)
        self.spatia_attention = SpatialAttention()
        # self.sobelconv = Sobelxy(self.in_channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, f_vi, f_ir):
        # 该模块对特征进一步增强/校正,加上互补信息
        _, c, _, _ = f_vi.shape
        # 1.通道注意力(平均,平均大则强度大)
        # 这里avg通道注意力提取cat后的特征,与原特征相乘;但原cat是vi:ir,而w是包含了2个模态信息,单纯*会导致vi乘前部分c,ir乘后部分c
        # 故这里将f_att的通道输出将原来的2c变为c,复制后再乘或者分开乘;
        # 我们希望这里能够进行特征互补，而只输出c通道的w包含两个模态的共有信息,所以应该按照原代码或者使用1-w来互补
        f_cat = torch.cat([f_vi, f_ir], dim=1)  # (n,2c=vi:ir,h,w)
        w = self.channel_avg_attention(f_cat)  # (n,c,1,1) / (n,2c,1,1)

        # f_vi_, f_ir_ = torch.split(f_cat * w, c, dim=1)
        # f_vi = f_vi + f_ir_
        # f_ir = f_ir + f_vi_
        # f_cat = torch.cat([f_vi, f_ir], dim=1)

        w = torch.cat([w, w], dim=1)  # (n,c,1,1)->(n,2c,1,1)
        f_cat = torch.cat([f_ir, f_vi], dim=1) * w + f_cat  # 互补,f_vi=f_vi+f_ir*w/f_ir同
        f_vi, f_ir = torch.split(f_cat, c, dim=1)  # (n,c,h,w) 增强后的f_vi,f_ir

        # 2.通道空间并行注意力(送入注意力之前先卷积.通道和空间注意力得到的权重相乘得到注意力权重,最后再与原特征相乘),继续互补
        f_cat = self.conv1(f_cat)  # (n,2c,h,w)->(n,c,h,w)
        w = self.sigmoid(self.spatia_attention(f_cat) * self.channel_attention(f_cat))  # (n,c,h,w)*(n,c,1,1)
        # 这里空间注意力设置为互补图分开相乘，以满足互补特性(原PSFusion:空间注意力是并行的通道和空间相乘,使用1-w互补,1-w是否好？)
        f_cat = w * f_vi + (1 - w) * f_ir  # (n,c,h,w) 特征聚合
        # 是否加x或者gradient(x),使用注意力增强最好不加残差,卷据加残差更好
        # return f_cat
        # max_gradient = torch.max(self.sobelconv(f_vi), self.sobelconv(f_ir))
        # return F.leaky_relu(f_cat + max_gradient, negative_slope=0.1)
        return f_cat
    
class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)
        # return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.2)

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x


class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)

    def forward(self, x):
        return self.conv(x)


class DenseBlock(nn.Module):
    def __init__(self, channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2 * channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)

    def forward(self, x):
        x = torch.cat((x, self.conv1(x)), dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x


class RGBD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RGBD, self).__init__()
        self.dense = DenseBlock(in_channels)
        self.convdown = Conv1(3 * in_channels, out_channels)
        self.sobelconv = Sobelxy(in_channels)
        self.convup = Conv1(in_channels, out_channels)

    def forward(self, x):
        x1 = self.dense(x)
        x1 = self.convdown(x1)
        x2 = self.sobelconv(x)
        x2 = self.convup(x2)
        return F.leaky_relu(x1 + x2, negative_slope=0.1)
    
class SFC(nn.Module):
    """
    Self Fusion Conv Network: from SFC;some change between SFC and CMTFusion SFC,this is later
    in paper:4 layer
        self-expansion: group,in channel -> expansion * in channel
        hd-fusion: group,expansion * in channel -> expansion * in channel
        compression: group,expansion * in channel -> in channel
        point-wise convolution: not group,in channel -> in channel or output channel
    """

    def __init__(self, in_channel=32, out_channel=1, expansion=4):
        super(SFC, self).__init__()
        expansion_channel = int(in_channel * expansion)

        self.fused = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)

        self.se_conv = nn.Sequential(
            nn.Conv2d(in_channel, expansion_channel, 3, stride=1, padding=1, groups=in_channel),
            nn.BatchNorm2d(expansion_channel),
            nn.LeakyReLU()
        )
        self.hd_conv = nn.Sequential(
            nn.Conv2d(expansion_channel, expansion_channel, 3, stride=1, padding=1, groups=in_channel),
            nn.BatchNorm2d(expansion_channel),
            nn.LeakyReLU()
        )
        self.cp_conv = nn.Sequential(
            nn.Conv2d(expansion_channel, in_channel, 1, stride=1, padding=0, groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU()
        )
        self.pw_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fused(x)
        x = self.se_conv(x)
        x = self.hd_conv(x)
        x = self.cp_conv(x)
        x = self.pw_conv(x) / 2 + 0.5

        return x
    
class ChannelAttention(nn.Module):
    """
    ChannelAttention: Channel Attention,use mean and max pool
    return: weight,no multiply
    """

    def __init__(self, channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 这个被ONNX转换为MaxPool,要转换为GlobalMaxPool再转om的时候才不会转为MaxPool
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        # print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    """
    SpatialAttention: Spatial Attention
    return: weight,no multiply
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(kernel_size, kernel_size), padding=padding,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.sigmoid(self.conv(x))
        return x

