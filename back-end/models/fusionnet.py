import torch
import numbers
import numpy as np
import torch.nn as nn
from torch.nn import init
from einops import rearrange
import torch.nn.functional as F
from .attention import GAM

class CRSFGAMAdd(nn.Module):
    '''
    没有了 GAM模块融合
    相比于 CrossFusionNetS2 更改了融合模块, 从CBAM变为了GAM CRSFGAM
    从 ConvNoBN 变为了 nn.Conv2d 这里主要为了测试加 BN 的 Conv 和不加 BN 的Conv的区别
    '''
    def __init__(self):
        super().__init__()
        self.down_rgb = nn.Conv2d(3, 3, 1, 2)
        self.down_ir = nn.Conv2d(3, 3, 1, 2)
        # rgb 分支
        self.conv1_rgb = ConvNoBN(3, 16, 3, 1, 1) # in_chs out_chs kernel_size stride padding
        self.grdb1_rgb = RGBR(16, 32)
        self.self_attn_rgb = TransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.cross_attn_rgb = CrossTransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        # ir 分支
        self.conv1_ir = ConvNoBN(3, 16, 3, 1, 1) # in_chs out_chs kernel_size stride padding
        self.grdb1_ir = RGBR(16, 32)
        self.self_attn_ir = TransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.cross_attn_ir = CrossTransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        # fused
        self.gam = GAM(64, 64) # 这里 48 是单独分支的channel数量
        self.conv1_fused = Conv(32, 16, 3, 1)
        self.conv2_fused = Conv(16, 8, 3, 1)
        self.conv3_fused = nn.Sequential(
            nn.Conv2d(8, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )
        self.up = nn.ConvTranspose2d(3, 3, 2, 2)

    def forward(self, x):

        img_rgb = x[:, :3, :, :]
        img_ir  = x[:, 3:, :, :]

        img_rgb = self.down_rgb(img_rgb)
        img_ir = self.down_ir(img_ir)

        x1 = self.conv1_rgb(img_rgb) # rgb分支
        x1 = self.grdb1_rgb(x1)
        x1_out = self.self_attn_rgb(x1)

        x2 = self.conv1_ir(img_ir) # ir 分支
        x2 = self.grdb1_ir(x2)
        x2_out = self.self_attn_ir(x2)

        # cross attention 两个分支都已经计算出了中间变量才能进行 
        x1 = self.cross_attn_rgb(x1_out, x2_out) # x2 当做 k, v, 主分支还在rgb上
        x2 = self.cross_attn_ir(x2_out, x1_out) # x1 当做 k, v, 主分支还在 ir 上面

        x3 = torch.add(x1, x2) # add 

        x3 = self.conv1_fused(x3)
        x3 = self.conv2_fused(x3)
        x3 = self.conv3_fused(x3)
        
        x3 = self.up(x3)
        return x3


class CRSFGAM(nn.Module):
    '''
    相比于 CrossFusionNetS2 更改了融合模块, 从CBAM变为了GAM CRSFGAM
    从 ConvNoBN 变为了 nn.Conv2d 这里主要为了测试加 BN 的 Conv 和不加 BN 的Conv的区别
    '''
    def __init__(self):
        super().__init__()
        self.down_rgb = nn.Conv2d(3, 3, 1, 2)
        self.down_ir = nn.Conv2d(3, 3, 1, 2)
        # rgb 分支
        self.conv1_rgb = ConvNoBN(3, 16, 3, 1, 1) # in_chs out_chs kernel_size stride padding
        self.grdb1_rgb = RGBR(16, 32)
        self.self_attn_rgb = TransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.cross_attn_rgb = CrossTransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        # ir 分支
        self.conv1_ir = ConvNoBN(3, 16, 3, 1, 1) # in_chs out_chs kernel_size stride padding
        self.grdb1_ir = RGBR(16, 32)
        self.self_attn_ir = TransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.cross_attn_ir = CrossTransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        # fused
        self.gam = GAM(64, 64) # 这里 48 是单独分支的channel数量
        self.conv1_fused = Conv(32, 16, 3, 1)
        self.conv2_fused = Conv(16, 8, 3, 1)
        self.conv3_fused = nn.Sequential(
            nn.Conv2d(8, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )
        self.up = nn.ConvTranspose2d(3, 3, 2, 2)

    def forward(self, x):

        img_rgb = x[:, :3, :, :]
        img_ir  = x[:, 3:, :, :]

        img_rgb = self.down_rgb(img_rgb)
        img_ir = self.down_ir(img_ir)

        x1 = self.conv1_rgb(img_rgb) # rgb分支
        x1 = self.grdb1_rgb(x1)
        x1_out = self.self_attn_rgb(x1)

        x2 = self.conv1_ir(img_ir) # ir 分支
        x2 = self.grdb1_ir(x2)
        x2_out = self.self_attn_ir(x2)

        # cross attention 两个分支都已经计算出了中间变量才能进行 
        x1 = self.cross_attn_rgb(x1_out, x2_out) # x2 当做 k, v, 主分支还在rgb上
        x2 = self.cross_attn_ir(x2_out, x1_out) # x1 当做 k, v, 主分支还在 ir 上面

        x3 = [x1, x2] # 融合分支
        x3 = self.gam(x3)

        x3 = self.conv1_fused(x3)
        x3 = self.conv2_fused(x3)
        x3 = self.conv3_fused(x3)
        
        x3 = self.up(x3)
        return x3

class CrossFusionNetS2(nn.Module):
    '''
    相比于 CrossFusionNetS 更改了 self.down_rgb  和 self.down_ir  
    从 ConvNoBN 变为了 nn.Conv2d 这里主要为了测试加 BN 的 Conv 和不加 BN 的Conv的区别
    '''
    def __init__(self):
        super().__init__()
        self.down_rgb = nn.Conv2d(3, 3, 1, 2)
        self.down_ir = nn.Conv2d(3, 3, 1, 2)
        # rgb 分支
        self.conv1_rgb = ConvNoBN(3, 16, 3, 1, 1) # in_chs out_chs kernel_size stride padding
        self.grdb1_rgb = RGBR(16, 32)
        self.self_attn_rgb = TransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.cross_attn_rgb = CrossTransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        # ir 分支
        self.conv1_ir = ConvNoBN(3, 16, 3, 1, 1) # in_chs out_chs kernel_size stride padding
        self.grdb1_ir = RGBR(16, 32)
        self.self_attn_ir = TransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.cross_attn_ir = CrossTransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        # fused
        self.cbam = CBAMBlock(32, 8, 9) # 这里 48 是单独分支的channel数量
        self.conv1_fused = Conv(32, 16, 3, 1)
        self.conv2_fused = Conv(16, 8, 3, 1)
        self.conv3_fused = nn.Sequential(
            nn.Conv2d(8, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )
        self.up = nn.ConvTranspose2d(3, 3, 2, 2)

    def forward(self, x):

        img_rgb = x[:, :3, :, :]
        img_ir  = x[:, 3:, :, :]

        img_rgb = self.down_rgb(img_rgb)
        img_ir = self.down_ir(img_ir)

        x1 = self.conv1_rgb(img_rgb) # rgb分支
        x1 = self.grdb1_rgb(x1)
        x1_out = self.self_attn_rgb(x1)

        x2 = self.conv1_ir(img_ir) # ir 分支
        x2 = self.grdb1_ir(x2)
        x2_out = self.self_attn_ir(x2)

        # cross attention 两个分支都已经计算出了中间变量才能进行 
        x1 = self.cross_attn_rgb(x1_out, x2_out) # x2 当做 k, v, 主分支还在rgb上
        x2 = self.cross_attn_ir(x2_out, x1_out) # x1 当做 k, v, 主分支还在 ir 上面

        x3 = [x1, x2] # 融合分支
        x3 = self.cbam(x3)

        x3 = self.conv1_fused(x3)
        x3 = self.conv2_fused(x3)
        x3 = self.conv3_fused(x3)
        
        x3 = self.up(x3)
        return x3

class CrossFusionNetS(nn.Module):
    '''
       相比于  CrossFusionNet 多了 self.down_rgb, self.down_ir, self.up
       增加了一层卷积，先卷积一层的作用是降低图像的分辨率，下采样为原来的1/2.不然占得空间太大了
       增加了self.up 的作用是将图像的尺寸恢复到原来的大小
    '''
    def __init__(self):
        super().__init__()
        self.down_rgb = ConvNoBN(3, 3, 1, 2)
        self.down_ir = ConvNoBN(3, 3, 1, 2)
        # rgb 分支
        self.conv1_rgb = ConvNoBN(3, 16, 3, 1, 1) # in_chs out_chs kernel_size stride padding
        self.grdb1_rgb = RGBR(16, 32)
        self.self_attn_rgb = TransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.cross_attn_rgb = CrossTransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        # ir 分支
        self.conv1_ir = ConvNoBN(3, 16, 3, 1, 1) # in_chs out_chs kernel_size stride padding
        self.grdb1_ir = RGBR(16, 32)
        self.self_attn_ir = TransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.cross_attn_ir = CrossTransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        # fused
        self.cbam = CBAMBlock(32, 8, 9) # 这里 48 是单独分支的channel数量
        self.conv1_fused = Conv(32, 16, 3, 1)
        self.conv2_fused = Conv(16, 8, 3, 1)
        self.conv3_fused = nn.Sequential(
            nn.Conv2d(8, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )
        self.up = nn.ConvTranspose2d(3, 3, 2, 2)

    def forward(self, x):

        img_rgb = x[:, :3, :, :]
        img_ir  = x[:, 3:, :, :]

        img_rgb = self.down_rgb(img_rgb)
        img_ir = self.down_ir(img_ir)

        x1 = self.conv1_rgb(img_rgb) # rgb分支
        x1 = self.grdb1_rgb(x1)
        x1_out = self.self_attn_rgb(x1)

        x2 = self.conv1_ir(img_ir) # ir 分支
        x2 = self.grdb1_ir(x2)
        x2_out = self.self_attn_ir(x2)

        # cross attention 两个分支都已经计算出了中间变量才能进行 
        x1 = self.cross_attn_rgb(x1_out, x2_out) # x2 当做 k, v, 主分支还在rgb上
        x2 = self.cross_attn_ir(x2_out, x1_out) # x1 当做 k, v, 主分支还在 ir 上面

        x3 = [x1, x2] # 融合分支
        x3 = self.cbam(x3)

        x3 = self.conv1_fused(x3)
        x3 = self.conv2_fused(x3)
        x3 = self.conv3_fused(x3)
        
        x3 = self.up(x3)
        return x3

class CrossFusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        # rgb 分支
        self.conv1_rgb = ConvNoBN(3, 16, 3, 1, 1) # in_chs out_chs kernel_size stride padding
        self.grdb1_rgb = RGBR(16, 32)
        self.self_attn_rgb = TransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.cross_attn_rgb = CrossTransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        # ir 分支
        self.conv1_ir = ConvNoBN(3, 16, 3, 1, 1) # in_chs out_chs kernel_size stride padding
        self.grdb1_ir = RGBR(16, 32)
        self.self_attn_ir = TransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.cross_attn_ir = CrossTransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        # fused
        self.cbam = CBAMBlock(32, 8, 9) # 这里 48 是单独分支的channel数量
        self.conv1_fused = Conv(32, 16, 3, 1)
        self.conv2_fused = Conv(16, 8, 3, 1)
        self.conv3_fused = nn.Sequential(
            nn.Conv2d(8, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def forward(self, x):

        img_rgb = x[:, :3, :, :]
        img_ir  = x[:, 3:, :, :]

        x1 = self.conv1_rgb(img_rgb) # rgb分支
        x1 = self.grdb1_rgb(x1)
        x1_out = self.self_attn_rgb(x1)

        x2 = self.conv1_ir(img_ir) # ir 分支
        x2 = self.grdb1_ir(x2)
        x2_out = self.self_attn_ir(x2)

        # cross attention 两个分支都已经计算出了中间变量才能进行 
        x1 = self.cross_attn_rgb(x1_out, x2_out) # x2 当做 k, v, 主分支还在rgb上
        x2 = self.cross_attn_ir(x2_out, x1_out) # x1 当做 k, v, 主分支还在 ir 上面

        x3 = [x1, x2] # 融合分支
        x3 = self.cbam(x3)

        x3 = self.conv1_fused(x3)
        x3 = self.conv2_fused(x3)
        x3 = self.conv3_fused(x3)
        return x3

# ================================================================== #
#        Multi-DConv Head Transposed Cross-Attention (MDTCA)         #
# ================================================================== #
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        '''
            x: use to calculate query 
            y: use to calculate key and value 
        '''
        b, c, h, w = x.shape
        # calculate q, k, v seperately
        q = self.q(x)
        kv = self.kv(y)
        qkv = self.qkv_dwconv(torch.cat([q, kv], 1))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class CrossTransformerBlock(nn.Module):
    # TransformerBlock 
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(CrossTransformerBlock, self).__init__()

        self.norm1_1 = LayerNorm(dim, LayerNorm_type)
        self.norm1_2 = LayerNorm(dim, LayerNorm_type)
        self.attn = CrossAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, y):
        '''
            x : for query
            y : for key and value (from other branch)
        '''        
        shortcut_x = x
        x = self.norm1_1(x)
        y = self.norm1_2(y)
        out = self.attn(x, y)
        out = shortcut_x + out

        shortcut_out = out
        out = self.norm2(out)
        out = self.ffn(out)

        out = shortcut_out + out

        return out

# ================================================================== #
#         Multi-DConv Head Transposed Self-Attention (MDTA)          #
# ================================================================== #
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    # TransformerBlock 
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class FeedForward(nn.Module):
    # FeedForward
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

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
        x1, x2 = x[0], x[1]
        x = torch.cat((x1, x2), dim=1)
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        x = self.conv1(x)
        out = self.conv1(out)
        return torch.add(x, out)


# ================================================================== #
#                               utils                                #
# ================================================================== #
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # CBS Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class RGBR(nn.Module):
    # ResBlock with gradient block
    def __init__(self, c1, c2):
        super(RGBR, self).__init__()
        self.conv = Conv(c1, c1)
        self.convdown = nn.Conv2d(2 * c1, c2, 1, 1)
        self.sobelconv = Sobelxy(c1)
        self.convup = nn.Conv2d(c1, c2, 1, 1)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x1 = torch.cat((x, self.conv(x)), dim=1)
        x1 = self.convdown(x1)
        x2 = self.sobelconv(x)
        x2 = self.convup(x2)
        x = self.relu(x1 + x2)
        return x


class ConvNoBN(nn.Module):
    #  Without Batch Normalization
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.conv(x))


class Sobelxy(nn.Module):
    # Calculate the gradient of feature map
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x
