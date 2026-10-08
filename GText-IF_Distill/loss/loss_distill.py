import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
    
class loss_distill_text(nn.Module):
    def __init__(self, use_mask=False, eps=1e-8, alpha=0.0, beta=1.0):
        super(loss_distill_text, self).__init__()
        self.use_mask = use_mask
        self.eps = eps
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def _as_float(x, ref=None):
        if x.is_floating_point() or x.is_complex():
            return x
        if ref is not None and (ref.is_floating_point() or ref.is_complex()):
            return x.to(ref.dtype)
        return x.to(torch.float32)

    def forward(self, pred_text, text_feat, mask=None):
        pred_text = self._as_float(pred_text)
        text_feat = self._as_float(text_feat, ref=pred_text).detach()

        # ---------- Cosine Loss ----------
        s = F.normalize(pred_text, p=2, dim=-1, eps=self.eps)
        t = F.normalize(text_feat, p=2, dim=-1, eps=self.eps)

        cos_sim  = (s * t).sum(dim=-1)     # [B, n_ctx]
        cos_loss = 1.0 - cos_sim           # [B, n_ctx]

        # ---------- L1 Loss ----------
        l1_loss = (pred_text - text_feat).abs().mean(dim=-1)  # [B, n_ctx]

        if self.use_mask and mask is not None:
            mask = self._as_float(mask, ref=cos_loss)
            denom = mask.sum().clamp_min(1.0)

            cos_loss = (cos_loss * mask).sum() / denom
            l1_loss  = (l1_loss  * mask).sum() / denom
        else:
            cos_loss = cos_loss.mean()
            l1_loss  = l1_loss.mean()

        total_loss = self.alpha * cos_loss + self.beta * l1_loss
        return total_loss