import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight
from utils.load_utils import load_pretrain
from functools import partial

from engine.logger import get_logger
from .distill_text.distill_net import VisionTransformerFixedCtx
import os

logger = get_logger()

class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        self.cfg = cfg
        
        if self.cfg['model']['text_encoder'] == 'long-clip':
            from long_clip.model import longclip
            self.text_func = longclip
            self.text_encoder = None
        else:
            raise NotImplementedError

        if self.cfg['model']['backbone'] == 'segformer-b0':
            logger.info('Using backbone: Segformer-B0')
            from .encoders.dual_segformer import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
            self.channels = [32, 64, 160, 256]
        elif self.cfg['model']['backbone'] == 'segformer-b1':
            logger.info('Using backbone: Segformer-B1')
            from .encoders.dual_segformer import mit_b1 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif self.cfg['model']['backbone'] == 'segformer-b2':
            logger.info('Using backbone: Segformer-B2')
            from .encoders.dual_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif self.cfg['model']['backbone'] == 'segformer-b4':
            logger.info('Using backbone: Segformer-B4')
            from .encoders.dual_segformer import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        else:
            raise NotImplementedError
        
        self.text_distill_net = VisionTransformerFixedCtx()
        self.aux_head = None
        
        logger.info('Using MLP Decoder')
        from .decoders.MLPDecoder import DecoderHead
        self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg['model']['num_classes'], norm_layer=norm_layer, embed_dim=512)

        from .decoders.FuseDecoder import Visual_DecoderHead_Ori
        self.visual_decode_head = Visual_DecoderHead_Ori(in_channels=self.channels, out_channels=3)

    def tokenize(self, text):
        if hasattr(self.text_func, 'module'):
            return self.text_func.module.tokenize(text)
        else:
            return self.text_func.tokenize(text)
    
    def text_distill(self, rgbs, modal_xs):
        return self.text_distill_net(rgbs, modal_xs)

    def init_func(self):
        if self.cfg['model']['text_encoder'] == 'long-clip':
            self.text_encoder, _ = self.text_func.load("./long_clip/checkpoints/longclip-B.pt", device=torch.device('cpu'))
            logger.info('init the text encoder!')

        if self.cfg['model']['backbone'] == 'segformer-b0':
            self.init_weights(self.cfg, pretrained="./pretrained_model/mit_b0.pth")
        elif self.cfg['model']['backbone'] == 'segformer-b1':
            self.init_weights(self.cfg, pretrained="./pretrained_model/mit_b1.pth")
        elif self.cfg['model']['backbone'] == 'segformer-b2':
            self.init_weights(self.cfg, pretrained="./pretrained_model/mit_b2.pth")
        elif self.cfg['model']['backbone'] == 'segformer-b4':
            self.init_weights(self.cfg, pretrained="./pretrained_model/mit_b4.pth")
        else:
            raise NotImplementedError
    
    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, 0.001, 0.1,
                mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, 0.001, 0.1,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgbs, modal_xs, text):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgbs.shape
        source_imgs = torch.cat((rgbs, modal_xs), dim=1)
        
        """if self.use_text_distill is not True:
            text_emb = self.text_encoder.encode_text_full(text)
            text_emb = text_emb.squeeze(1)
        else:
            text_emb = self.text_distill(rgbs, modal_xs).squeeze(1)
            print("use the distill text encoder!")"""

        text_emb = self.text_distill(rgbs, modal_xs).squeeze(1)
        
        x_semantic, x_visual = self.backbone(rgbs, modal_xs, text_emb)
        out_semantic = self.decode_head.forward(x_semantic)
        out_semantic = F.interpolate(out_semantic, size=orisize[2:], mode='bilinear', align_corners=False)

        out_visual = self.visual_decode_head.forward(x_visual, source_imgs)
        out_visual = F.interpolate(out_visual, size=orisize[2:], mode='bilinear', align_corners=False)
        if self.aux_head:
            aux_fm = self.aux_head(x_semantic[self.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
            return out_semantic, aux_fm, out_visual
        return out_semantic, out_visual

    def forward(self, rgb, modal_x, text):
        if self.aux_head:
            out_semantic, aux_fm, out_visual = self.encode_decode(rgb, modal_x, text)
        else:
            out_semantic, out_visual = self.encode_decode(rgb, modal_x, text)
        return out_semantic, out_visual