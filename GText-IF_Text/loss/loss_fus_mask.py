import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

class loss_fus_mask(nn.Module):
    def __init__(self):
        super(loss_fus_mask, self).__init__()
        self.loss_func_Grad = GradientMaxLoss_map()
        self.loss_func_Max = L_Intensity_Max_RGB_map()
        self.loss_func_color = L_color_map()

    def forward(self, image_fused, image_visible, image_infrared, mask, max_ratio=3, color_ratio=1.8, text_ratio=1):
        image_visible = self.unnormalize(image_visible)
        image_infrared = self.unnormalize(image_infrared)
        image_fused = self.unnormalize(image_fused)

        image_visible_gray = self.rgb2gray(image_visible)
        image_infrared_gray = self.rgb2gray(image_infrared)
        image_fused_gray = self.rgb2gray(image_fused)

        # Intensity loss (pixel-wise)
        loss_int_map = self.loss_func_Max(image_visible, image_infrared, image_fused)
        loss_int = self.apply_mask(loss_int_map, mask) * max_ratio

        # Color loss (pixel-wise)
        loss_color_map = self.loss_func_color(image_visible, image_fused)
        loss_color = self.apply_mask(loss_color_map, mask) * color_ratio

        # Texture loss (pixel-wise)
        loss_texture_map = self.loss_func_Grad(image_visible_gray, image_infrared_gray, image_fused_gray)
        loss_texture = self.apply_mask(loss_texture_map, mask) * text_ratio
        
        total_loss = loss_int + loss_color + loss_texture
        return total_loss

    def rgb2gray(self, image):
        b, c, h, w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray
    
    def unnormalize(self, image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        mean_t = torch.tensor(mean, device=image.device).view(1, 3, 1, 1)
        std_t = torch.tensor(std, device=image.device).view(1, 3, 1, 1)
        image = image * std_t + mean_t
        return image.clamp(0, 1)
    
    def apply_mask(self, loss_map, mask):
        if loss_map.dim() == 4:
            # shape: [B, 1, H, W] or [B, C, H, W]
            masked_loss = loss_map * mask.unsqueeze(1)
        elif loss_map.dim() == 3:
            # shape: [B, H, W]
            masked_loss = loss_map * mask
        else:
            raise ValueError("Unsupported loss_map shape for masking")
        return masked_loss.sum() / (max(mask.sum(), 1e-6))

class L_color_map(nn.Module):
    def __init__(self):
        super(L_color_map, self).__init__()

    def forward(self, image_visible, image_fused):
        ycbcr_visible = self.rgb_to_ycbcr(image_visible)
        ycbcr_fused = self.rgb_to_ycbcr(image_fused)

        cb_visible = ycbcr_visible[:, 1, :, :]
        cr_visible = ycbcr_visible[:, 2, :, :]
        cb_fused = ycbcr_fused[:, 1, :, :]
        cr_fused = ycbcr_fused[:, 2, :, :]

        loss_cb = torch.abs(cb_visible - cb_fused)
        loss_cr = torch.abs(cr_visible - cr_fused)
        loss_color_map = loss_cb + loss_cr  # shape [B,1,H,W]

        return loss_color_map

    def rgb_to_ycbcr(self, image):
        r = image[:, 0, :, :]
        g = image[:, 1, :, :]
        b = image[:, 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b

        ycbcr_image = torch.stack((y, cb, cr), dim=1)
        return ycbcr_image


class L_Intensity_Max_RGB_map(nn.Module):
    def __init__(self):
        super(L_Intensity_Max_RGB_map, self).__init__()

    def forward(self, image_visible, image_infrared, image_fused):
        gray_visible = torch.mean(image_visible, dim=1, keepdim=True)
        gray_infrared = torch.mean(image_infrared, dim=1, keepdim=True)

        mask = (gray_infrared > gray_visible).float()

        fused_target = mask * image_infrared + (1 - mask) * image_visible

        loss_map = torch.abs(fused_target - image_fused)  # [B,3,H,W]
        loss_map = torch.mean(loss_map, dim=1, keepdim=True)  # to [B,1,H,W]
        return loss_map


# use the GradientMaxLoss or L_Grad
class GradientMaxLoss_map(nn.Module):
    def __init__(self):
        super(GradientMaxLoss_map, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1],
                                                       [-2, 0, 2],
                                                       [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1],
                                                       [0, 0, 0],
                                                       [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.padding = (1, 1, 1, 1)

    def forward(self, image_A, image_B, image_fuse):
        gradient_A_x, gradient_A_y = self.gradient(image_A)
        gradient_B_x, gradient_B_y = self.gradient(image_B)
        gradient_fuse_x, gradient_fuse_y = self.gradient(image_fuse)

        grad_target_x = torch.max(gradient_A_x, gradient_B_x)
        grad_target_y = torch.max(gradient_A_y, gradient_B_y)

        loss_x = torch.abs(gradient_fuse_x - grad_target_x)
        loss_y = torch.abs(gradient_fuse_y - grad_target_y)
        loss = loss_x + loss_y
        return loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x), torch.abs(gradient_y)