import torch
import torch.nn as nn
from torchvision import models

# ----------------------------
# Channel + Spatial Attention (CBAM)
# ----------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average pooling
        avg_out = self.shared_MLP(x.mean(dim=(2, 3), keepdim=True))
        # Max pooling (applied sequentially for dim 2 then 3)
        max_out = self.shared_MLP(
            torch.max(torch.max(x, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
        )
        out = avg_out + max_out
        return self.sigmoid(out)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    def __init__(self, in_planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# ----------------------------
# HybridLeafNet (EfficientNet + CBAM + Transformer)
# ----------------------------
class HybridLeafNet(nn.Module):
    def __init__(self, num_classes=16):
        super(HybridLeafNet, self).__init__()
        backbone = models.efficientnet_b0(pretrained=True)
        self.features = backbone.features
        self.cbam = CBAM(1280)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1280, nhead=8, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)  # (B, 1280, 7, 7)
        x = self.cbam(x)
        B, C, H, W = x.shape

        # Flatten spatial dimensions for transformer
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, seq_len=49, features=1280)

        # Transformer expects (batch, seq_len, features)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average over sequence

        out = self.fc(x)
        return out
