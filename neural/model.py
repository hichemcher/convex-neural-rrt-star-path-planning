"""
model.py
========
UNet with ResNet50 encoder used to predict path probability masks
on 224×224 occupancy grids.

Architecture
------------
- Encoder : ResNet50 (ImageNet pretrained, last 2 blocks fine-tuned)
- Decoder : Transposed-conv upsampling with skip connections
- Output  : 2-channel logit map (background / path)

Usage
-----
    from model import UNet, load_model
    model = load_model("best_path_mode_multipoint_224.pth")
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Building blocks ──────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 padding=1, kernel_size=3, stride=1,
                 with_nonlinearity=True, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout)
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
            x = self.dropout(x)
        return x


# ─── Encoder ──────────────────────────────────────────────────────────────────

class ResNet50Encoder(nn.Module):
    """
    ResNet50 feature pyramid encoder.
    Layers 1–2 are frozen; layers 3–4 are fine-tuned.
    Returns a list of intermediate feature maps for skip connections.
    """
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        for name, param in resnet.named_parameters():
            param.requires_grad = "layer3" in name or "layer4" in name

        self.input_block = nn.Sequential(*list(resnet.children())[:3])
        self.input_pool  = list(resnet.children())[3]
        self.down_blocks = nn.ModuleList(
            [b for b in resnet.children() if isinstance(b, nn.Sequential)]
        )

    def forward(self, x):
        features = [x]
        x = self.input_block(x)
        features.append(x)
        x = self.input_pool(x)
        for block in self.down_blocks:
            x = block(x)
            features.append(x)
        return features


# ─── Decoder ──────────────────────────────────────────────────────────────────

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 up_conv_in_channels, up_conv_out_channels,
                 upsampling_method="conv_transpose"):
        super().__init__()
        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(
                up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2
            )
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(up_conv_in_channels, up_conv_out_channels, kernel_size=1),
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], dim=1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetDecoder(nn.Module):
    def __init__(self, feature_channels_list, upsampling_method="conv_transpose"):
        super().__init__()
        self._build_channel_lists(feature_channels_list)
        self.up_blocks = nn.ModuleList([
            UpBlock(ic, oc, ui, uo, upsampling_method)
            for ic, oc, ui, uo in zip(
                self.in_channels_list, self.out_channels_list,
                self.up_conv_in_channels_list, self.up_conv_out_channels_list,
            )
        ])
        self.out = nn.Conv2d(self.out_channels_list[-1], 2, kernel_size=1)

    def _build_channel_lists(self, feature_channels_list):
        rev = feature_channels_list[::-1]
        self.in_channels_list          = [2 * c for c in rev[1:]]
        self.out_channels_list         = rev[1:]
        self.up_conv_in_channels_list  = rev[:-1]
        self.up_conv_out_channels_list = rev[1:]

    def forward(self, features):
        rev = features[::-1]
        x = rev[0]
        for feat, up_block in zip(rev[1:], self.up_blocks):
            x = up_block(x, feat)
        return self.out(x)


# ─── Full UNet ─────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    UNet with ResNet50 encoder.
    Input  : (B, 3, 224, 224) RGB-encoded occupancy map
    Output : (B, 2, 224, 224) logits  [background, path]
    """
    def __init__(self, feature_channels_list=(3, 64, 256, 512, 1024, 2048)):
        super().__init__()
        self.encoder = ResNet50Encoder()
        self.decoder = UNetDecoder(feature_channels_list)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ─── Convenience loader ────────────────────────────────────────────────────────

def load_model(weights_path: str, device: str = DEVICE) -> UNet:
    """
    Load a trained UNet from a .pth checkpoint.

    Parameters
    ----------
    weights_path : str  — path to best_path_mode_multipoint_224.pth
    device       : str  — 'cuda' or 'cpu'

    Returns
    -------
    model : UNet (eval mode, on device)
    """
    model = UNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"Model loaded from '{weights_path}' on {device}")
    return model
