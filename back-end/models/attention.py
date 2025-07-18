import numpy as np
import torch
from torch import nn
from torch.nn import init
from einops import rearrange
from einops.layers.torch import Rearrange

# ================================================================== #
#                                 GAM                                #
# ================================================================== #

#GAM
class GAM(nn.Module):
    def __init__(self, c1, c2, group=True, rate=4):
        super(GAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c1, c1 // rate, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(c1, int(c1 / rate),
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // rate, c2, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(int(c1 / rate), c2,
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(c2)
        )
        self.downsample1 = nn.Conv2d(c1, c1 // 2, 1, 1)
        self.downsample2 = nn.Conv2d(c1, c1 // 2, 1, 1)

    def forward(self, x):
        x1, x2 = x[0], x[1] # rgb ir
        x = torch.cat((x1, x2), dim=1)
        residual = x
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att
        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att = channel_shuffle(x_spatial_att, 4)  # last shuffle
        x = x * x_spatial_att
        residual = self.downsample1(residual)
        out = self.downsample2(x)
        return torch.add(out, residual)

def channel_shuffle(x, groups=2):  ##shuffle channel
    B, C, H, W = x.size()
    out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
    out = out.view(B, C, H, W)
    return out

# ================================================================== #
#                                CBAM                                #
# ================================================================== #
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.channel = channel * 2 # 48 * 2 = 96
        self.ca = ChannelAttention(channel=self.channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        self.conv1 = nn.Conv2d(self.channel, self.channel // 2, 1, 1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x1, x2 = x[0], x[1]
        x = torch.cat((x1, x2), dim=1)
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        x = self.conv1(x)
        out = self.conv1(out)
        return torch.add(x, out)


class CBAMBlock2(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.channel = channel * 2
        self.ca = ChannelAttention(channel=self.channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        self.conv1 = nn.Conv2d(self.channel, self.channel // 2, 1, 1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        x = self.conv1(x)
        out = self.conv1(out)
        return torch.add(x, out)
    
if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    input2 = torch.randn(50, 512, 7, 7)
    input = [input, input2]

    kernel_size = input[0].shape[2]
    cbam = CBAMBlock(channel=512, reduction=16, kernel_size=kernel_size)
    output = cbam(input)
    print(output.shape)
