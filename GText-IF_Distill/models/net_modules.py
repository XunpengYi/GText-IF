import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
import math
import numbers
from einops import rearrange
import torch.nn.functional as F
from torch import einsum


class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim * 4, self.dim * 4 // reduction),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(self.dim * 4 // reduction, self.dim * 2), 
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1)
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4) # 2 B C 1 1
        return channel_weights

class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.conv_1 = nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1)
        self.conv_3 = nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=5, padding=2)

        self.act = nn.LeakyReLU(inplace=True)
        self.proj = nn.Conv2d(self.dim // reduction, 2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        x1 = self.conv_1(x)
        x2 = self.conv_3(x)
        x3 = self.conv_5(x)
        x = self.act(x1 + x2 + x3)
        x = self.sigmoid(self.proj(x))
        spatial_weights = x.reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4) # 2 B 1 H W
        return spatial_weights

class Segformer_Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_1_l = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv_2_l = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio

        if sr_ratio > 1:
            self.sr_1 = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_2 = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_1 = nn.LayerNorm(dim)
            self.norm_2 = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        N = H * W
        x1 = x1.flatten(2).transpose(1, 2).contiguous()
        x2 = x2.flatten(2).transpose(1, 2).contiguous()
        x1_input = x1.clone()
        x2_input = x2.clone()

        q_1 = self.q_1(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q_2 = self.q_2(x2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            x_1 = x1.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
            x_1 = self.sr_1(x_1).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_1 = self.norm_1(x_1)
            kv_1 = self.kv_1_l(x_1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

            x_2 = x2.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
            x_2 = self.sr_2(x_2).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_2 = self.norm_2(x_2)
            kv_2 = self.kv_2_l(x_2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        else:
            kv_1 = self.kv_1_l(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
            kv_2 = self.kv_2_l(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k_1, v_1 = kv_1[0], kv_1[1]
        k_2, v_2 = kv_2[0], kv_2[1]

        attn_1_2 = (q_1 @ k_2.transpose(-2, -1)) * self.scale
        attn_1_2 = attn_1_2.softmax(dim=-1)
        attn_1_2 = self.attn_drop(attn_1_2)

        x_2 = (attn_1_2 @ v_2).transpose(1, 2).reshape(B, N, C).contiguous()
        x_2 = x_2 + x2_input

        x2 = self.proj_2(x_2)

        attn_2_1 = (q_2 @ k_1.transpose(-2, -1)) * self.scale
        attn_2_1 = attn_2_1.softmax(dim=-1)
        attn_2_1 = self.attn_drop(attn_2_1)

        x_1 = (attn_2_1 @ v_1).transpose(1, 2).reshape(B, N, C).contiguous()
        x_1 = x_1 + x1_input

        x1 = self.proj_1(x_1)
        return x1, x2

class Information_Restoration_Module(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=8, lambda_c=.5, lambda_s=.5, sr_ratio=1):
        super(Information_Restoration_Module, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1, out_x2 


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1_conv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=qkv_bias)
        self.kv2_conv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=qkv_bias)

    def forward(self, x1, x2):
        q1 = x1.flatten(2).transpose(1, 2)
        q2 = x2.flatten(2).transpose(1, 2)
        B, N, C = q1.shape
        q1 = q1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = q2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1_conv(x1).flatten(2).transpose(1, 2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2_conv(x2).flatten(2).transpose(1, 2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        c_attn_x1 = (k1.transpose(-2, -1) @ v1) * self.scale
        c_attn_x1 = c_attn_x1.softmax(dim=-2)
        c_attn_x2 = (k2.transpose(-2, -1) @ v2) * self.scale
        c_attn_x2 = c_attn_x2.softmax(dim=-2)

        x1 = (q1 @ c_attn_x2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous() 
        x2 = (q2 @ c_attn_x1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous() 
        return x1, x2


class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2

class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels//reduction, kernel_size=1, bias=True),
                        nn.Conv2d(out_channels//reduction, out_channels//reduction, kernel_size=3, stride=1, padding=1, bias=True, groups=out_channels//reduction),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(out_channels//reduction, out_channels, kernel_size=1, bias=False),
                        norm_layer(out_channels) 
                        )
        self.norm = norm_layer(out_channels)
        
    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out

class Information_Aggregation_Module(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d, sr_ratio=1):
        super().__init__()
        self.cross_attn = Segformer_Cross_Attention(dim=dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.proj = ChannelEmbed(in_channels=dim*2, out_channels=dim, reduction=reduction, norm_layer=norm_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1_c, x2_c = self.cross_attn(x1, x2)
        x1 = x1.flatten(2).transpose(1, 2).contiguous()
        x2 = x2.flatten(2).transpose(1, 2).contiguous()
        x1 = self.norm1(x1 + x1_c)
        x2 = self.norm2(x2 + x2_c)
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.proj(merge, H, W)
        return merge

    
class Text_Interaction_Module(nn.Module):
    def __init__(self, image_dim, text_dim=None, heads=8, dim_head=64, norm_layer=nn.BatchNorm2d):
        super().__init__()
        atten_dim = dim_head * heads

        # q: text [b, n, dim]   k: image [b, h*w, dim]    v: image [b, h*w, dim]
        self.q_linear = nn.Linear(image_dim, atten_dim, bias=False)
        self.k_linear = nn.Linear(text_dim, atten_dim, bias=False)
        self.v_linear = nn.Linear(text_dim, atten_dim, bias=False)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.out_linear = nn.Linear(atten_dim, image_dim)
        self.norm = norm_layer(image_dim)

    def forward(self, x, text=None):
        b, c, h, w = x.shape  # Get the original dimensions of the image

        x_flattened = x.view(b, c, -1).permute(0, 2, 1).contiguous()  # [b, h*w, c]

        head = self.heads
        q = self.q_linear(x_flattened)

        k = self.k_linear(text)
        v = self.v_linear(text)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=head), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=head)

        out = self.out_linear(out)

        out = out.permute(0, 2, 1).view(b, c, h, w).contiguous()
        
        out_resized = self.norm(out)
        return out_resized
    
class Image_Interaction_Module(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(Image_Interaction_Module, self).__init__()
        self.out_channels = out_channels
        self.channel_embed = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels//reduction, kernel_size=1, bias=True),
                        nn.Conv2d(out_channels//reduction, out_channels//reduction, kernel_size=3, stride=1, padding=1, bias=True, groups=out_channels//reduction),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(out_channels//reduction, out_channels, kernel_size=1, bias=False),
                        norm_layer(out_channels) 
                        )
        self.norm = norm_layer(out_channels)
        
    def forward(self, x):
        x = self.channel_embed(x)
        out = self.norm(x)
        return out
    
class Semantic_Representation_Module(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, sr_ratio, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.sp_atten = SR_Attention(dim=in_channels, num_heads=num_heads, sr_ratio=sr_ratio)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.act = nn.LeakyReLU(inplace=True)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.norm = norm_layer(out_channels)

        self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.sp_atten(x) + x
        x_residual = self.conv_res(x)
        x = self.conv_1(x)
        x = self.act(x)
        x = self.conv_2(x)
        x = self.norm(x + x_residual)
        return x
    
class SR_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Linear embedding
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        B, N, C = x.shape
        # B N C -> B N num_head C//num_head -> B C//num_head N num_heads
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        return x