import numpy as np
import torch.nn as nn
import torch

from torch.nn.modules import module
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class DegradationPromptBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=16, feat_dim=192):
        """
        Improved multi-task unknown degradation prompt generation block
        """
        super().__init__()
        
        self.prompt_param = nn.Parameter(torch.randn(prompt_len, prompt_dim))

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_pool = nn.AdaptiveMaxPool2d(1) 

        self.fc = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(feat_dim, prompt_len)
        )

        self.scale_layer = nn.Linear(prompt_dim, feat_dim)
        self.bias_layer = nn.Linear(prompt_dim, feat_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        global_feat = self.global_pool(x).view(B, C)
        local_feat = self.local_pool(x).view(B, C)
        feat = torch.cat([global_feat, local_feat], dim=1)

        prompt_weights = F.softmax(self.fc(feat), dim=1)

        prompt = torch.matmul(prompt_weights, self.prompt_param)

        scale = self.scale_layer(prompt).unsqueeze(-1).unsqueeze(-1)
        bias = self.bias_layer(prompt).unsqueeze(-1).unsqueeze(-1)

        return x * scale + bias + x
    
class Visual_DecoderHead_Ori(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 out_channels=3,
                 embed_dim=64,
                 align_corners=False,
                 prompt_dim=128,
                 prompt_len=16):
        super().__init__()
        self.align_corners = align_corners
        c1_in, c2_in, c3_in, c4_in = in_channels

        self.conv_c4 = nn.Conv2d(c4_in, embed_dim, 3, padding=1)
        self.conv_c3 = nn.Conv2d(c3_in, embed_dim, 3, padding=1)
        self.conv_c2 = nn.Conv2d(c2_in, embed_dim, 3, padding=1)
        self.conv_c1 = nn.Conv2d(c1_in, embed_dim, 3, padding=1)

        self.conv_ori_c4 = nn.Conv2d(6, 16, 3, padding=1)
        self.conv_ori_c3 = nn.Conv2d(6, 16, 3, padding=1)
        self.conv_ori_c2 = nn.Conv2d(6, 16, 3, padding=1)

        self.fuse_conv_c4 = nn.Conv2d(embed_dim + 16, embed_dim, 1)
        self.fuse_conv_c3 = nn.Conv2d(embed_dim + 16, embed_dim, 1)
        self.fuse_conv_c2 = nn.Conv2d(embed_dim + 16, embed_dim, 1)

        self.upconv_c3 = nn.Conv2d(embed_dim * 2, embed_dim, 1)
        self.upconv_c2 = nn.Conv2d(embed_dim * 2, embed_dim, 1)
        self.upconv_c1 = nn.Conv2d(embed_dim * 2, embed_dim, 1)

        self.conv_img = nn.Conv2d(6, 64, 3, padding=1)
        self.linear_pred = nn.Conv2d(embed_dim + 64, out_channels, 1)

        self.prompt_block_c1 = DegradationPromptBlock(prompt_dim, prompt_len, embed_dim)
        self.prompt_block_fused = DegradationPromptBlock(prompt_dim, prompt_len, embed_dim + 64)

    def forward(self, inputs, ori_img):
        """
        inputs: list of feature maps from encoder [c1, c2, c3, c4]
        ori_img: original image (B, 6, H, W)
        """
        c1, c2, c3, c4 = inputs  # 1/4, 1/8, 1/16, 1/32 resolutions

        c4 = F.leaky_relu(self.conv_c4(c4), 0.2)
        c3 = F.leaky_relu(self.conv_c3(c3), 0.2)
        c2 = F.leaky_relu(self.conv_c2(c2), 0.2)
        c1 = F.leaky_relu(self.conv_c1(c1), 0.2)

        ori_c4 = F.interpolate(ori_img, size=c4.shape[2:], mode='bilinear', align_corners=self.align_corners)
        ori_c4 = F.leaky_relu(self.conv_ori_c4(ori_c4), 0.2)
        c4 = self.fuse_conv_c4(torch.cat([c4, ori_c4], dim=1))

        up_c4 = F.interpolate(c4, size=c3.shape[2:], mode='bilinear', align_corners=self.align_corners)
        c3 = F.leaky_relu(self.upconv_c3(torch.cat([up_c4, c3], dim=1)), 0.2)

        ori_c3 = F.interpolate(ori_img, size=c3.shape[2:], mode='bilinear', align_corners=self.align_corners)
        ori_c3 = F.leaky_relu(self.conv_ori_c3(ori_c3), 0.2)
        c3 = self.fuse_conv_c3(torch.cat([c3, ori_c3], dim=1))

        up_c3 = F.interpolate(c3, size=c2.shape[2:], mode='bilinear', align_corners=self.align_corners)
        c2 = F.leaky_relu(self.upconv_c2(torch.cat([up_c3, c2], dim=1)), 0.2)

        ori_c2 = F.interpolate(ori_img, size=c2.shape[2:], mode='bilinear', align_corners=self.align_corners)
        ori_c2 = F.leaky_relu(self.conv_ori_c2(ori_c2), 0.2)
        c2 = self.fuse_conv_c2(torch.cat([c2, ori_c2], dim=1))

        up_c2 = F.interpolate(c2, size=c1.shape[2:], mode='bilinear', align_corners=self.align_corners)
        c1 = F.leaky_relu(self.upconv_c1(torch.cat([up_c2, c1], dim=1)), 0.2)
        c1 = self.prompt_block_c1(c1)

        ori_feat = F.leaky_relu(self.conv_img(ori_img), 0.2)
        up_c1 = F.interpolate(c1, size=ori_feat.shape[2:], mode='bilinear', align_corners=self.align_corners)
        fused = torch.cat([up_c1, ori_feat], dim=1)

        fused = self.prompt_block_fused(fused)
        out = self.linear_pred(fused)

        return out