"""
DINOv2 ViT-B + ViT-Adapter backbone for Mask2Former
Simplified design that avoids OOM at 1024px
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


class DINOv2ViTAdapter(Backbone):
    """
    DINOv2 ViT-B/14 backbone with multi-scale feature extraction.
    Extracts features from 4 groups of ViT layers and projects to FPN dims.
    Outputs res2/res3/res4/res5 at strides 4/8/16/32.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()

        self.freeze_backbone = cfg.MODEL.DINOV2.FREEZE_BACKBONE
        vit_model            = cfg.MODEL.DINOV2.MODEL_NAME
        pretrain_path        = cfg.MODEL.WEIGHTS
        embed_dim            = 768
        out_dim              = 256
        patch_size           = 14

        # ── DINOv2 ViT-B backbone ──
        self.vit = timm.create_model(
            vit_model,
            pretrained=False,
            img_size=518,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )

        # Load pretrained weights
        if pretrain_path and pretrain_path.endswith('.pth'):
            state = torch.load(pretrain_path, map_location='cpu')
            if 'model' in state:     state = state['model']
            if 'state_dict' in state: state = state['state_dict']
            missing, unexpected = self.vit.load_state_dict(state, strict=False)
            print(f"DINOv2 loaded: {len(missing)} missing, {len(unexpected)} unexpected")

        if self.freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False
            print("DINOv2 frozen ✅")

        # ── Lightweight adapter convolutions ──
        # Takes single-scale ViT output and creates multi-scale features
        self.adapter_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim),
                nn.Conv2d(embed_dim, out_dim, 1),
                nn.GroupNorm(32, out_dim),
                nn.GELU(),
            ) for _ in range(4)
        ])

        # ── Scale adapters to create multi-scale from single ViT scale ──
        self.scale_ups = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(out_dim, out_dim, 2, stride=2),
                nn.GroupNorm(32, out_dim), nn.GELU(),
                nn.ConvTranspose2d(out_dim, out_dim, 2, stride=2),
                nn.GroupNorm(32, out_dim), nn.GELU(),
            ),  # res2: upsample 4x → stride 4 (from stride 16 ViT patch)
            nn.Sequential(
                nn.ConvTranspose2d(out_dim, out_dim, 2, stride=2),
                nn.GroupNorm(32, out_dim), nn.GELU(),
            ),  # res3: upsample 2x → stride 8
            nn.Identity(),  # res4: stride 16 (native ViT patch stride for 1024px)
            nn.Sequential(
                nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1),
                nn.GroupNorm(32, out_dim), nn.GELU(),
            ),  # res5: downsample 2x → stride 32
        ])

        self._out_features         = ['res2', 'res3', 'res4', 'res5']
        self._out_feature_strides  = {'res2': 4, 'res3': 8, 'res4': 16, 'res5': 32}
        self._out_feature_channels = {k: out_dim for k in self._out_features}

    def forward(self, x):
        B, C, H, W = x.shape

        # Pad to multiple of patch size (14)
        pad_h = (14 - H % 14) % 14
        pad_w = (14 - W % 14) % 14
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h))
        else:
            x_pad = x

        # Run DINOv2 — get features from 4 checkpoints in the 12 layers
        tokens = self.vit.patch_embed(x_pad)
        tokens = self.vit._pos_embed(tokens)
        tokens = self.vit.patch_drop(tokens)
        tokens = self.vit.norm_pre(tokens)

        depth     = len(self.vit.blocks)
        step      = depth // 4
        layer_outs = []

        for i, block in enumerate(self.vit.blocks):
            tokens = block(tokens)
            if (i + 1) % step == 0:
                # Remove CLS token, reshape to spatial
                feat   = tokens[:, 1:, :]          # [B, N, D]
                Hp     = (H + pad_h) // 14
                Wp     = (W + pad_w) // 14
                feat   = feat.reshape(B, Hp, Wp, 768).permute(0, 3, 1, 2)  # [B,D,Hp,Wp]
                # Crop back to unpadded size equivalent
                feat   = feat[:, :, :H//14 + (1 if H%14 > 0 else 0),
                                      :W//14 + (1 if W%14 > 0 else 0)]
                layer_outs.append(feat)

        # Apply adapter layers
        adapted = [adapter(f) for adapter, f in zip(self.adapter_layers, layer_outs)]

        # Use last adapted feature as base, create multi-scale
        base = adapted[-1]  # [B, 256, H/14, W/14] ~ [B, 256, 74, 74] for 1024px

        # Project to target spatial sizes
        res2 = self.scale_ups[0](base)   # ~[B, 256, H/4,  W/4]
        res3 = self.scale_ups[1](base)   # ~[B, 256, H/8,  W/8]
        res4 = self.scale_ups[2](base)   # ~[B, 256, H/14, W/14]
        res5 = self.scale_ups[3](base)   # ~[B, 256, H/28, W/28]

        # Interpolate to exact target sizes
        res2 = F.interpolate(res2, size=(H//4,  W//4),  mode='bilinear', align_corners=False)
        res3 = F.interpolate(res3, size=(H//8,  W//8),  mode='bilinear', align_corners=False)
        res4 = F.interpolate(res4, size=(H//16, W//16), mode='bilinear', align_corners=False)
        res5 = F.interpolate(res5, size=(H//32, W//32), mode='bilinear', align_corners=False)

        return {'res2': res2, 'res3': res3, 'res4': res4, 'res5': res5}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_dinov2_vitadapter_backbone(cfg, input_shape):
    return DINOv2ViTAdapter(cfg, input_shape)
