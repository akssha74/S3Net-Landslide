"""Models and controlled losses for the reviewer-remediation experiment."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_semantics import NIR, RED, torch_ndvi, validate_nchw


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.shortcut(x))


class UNet(nn.Module):
    def __init__(self, in_ch: int = 4):
        super().__init__()
        self.inc = DoubleConv(in_ch, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv3 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        d3 = self.conv1(torch.cat([self.up1(x4), x3], dim=1))
        d2 = self.conv2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.conv3(torch.cat([self.up3(d2), x1], dim=1))
        return self.out(d1).squeeze(1)


class ResUNet(nn.Module):
    def __init__(self, in_ch: int = 4):
        super().__init__()
        self.inc = ResBlock(in_ch, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResBlock(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResBlock(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResBlock(128, 256))
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv1 = ResBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = ResBlock(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv3 = ResBlock(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        d3 = self.conv1(torch.cat([self.up1(x4), x3], dim=1))
        d2 = self.conv2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.conv3(torch.cat([self.up3(d2), x1], dim=1))
        return self.out(d1).squeeze(1)


class ControlledAttention(nn.Module):
    """Capacity-matched attention that always receives two control channels."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(feat_dim + 2, feat_dim // 2, 3, padding=1),
            nn.BatchNorm2d(feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, features: torch.Tensor, controls: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        resized = [
            F.interpolate(
                control,
                size=features.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            for control in controls
        ]
        mask = self.gate(torch.cat([features, *resized], dim=1))
        return features * mask + features


class ControlledS3Net(nn.Module):
    """Residual U-Net with none, raw-edge, or NDVI-edge control channels."""

    MODES = {"none", "raw", "ndvi"}

    def __init__(self, in_ch: int = 4, gating_mode: str = "ndvi"):
        super().__init__()
        if gating_mode not in self.MODES:
            raise ValueError(f"Unknown gating mode: {gating_mode}")
        self.gating_mode = gating_mode
        self.inc = ResBlock(in_ch, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResBlock(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResBlock(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResBlock(128, 256))
        self.attn3 = ControlledAttention(128)
        self.attn2 = ControlledAttention(64)
        self.attn1 = ControlledAttention(32)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv1 = ResBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = ResBlock(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv3 = ResBlock(64, 32)
        self.out = nn.Conv2d(32, 1, 1)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def gradient(self, value: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(value, self.sobel_x, padding=1)
        gy = F.conv2d(value, self.sobel_y, padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-8)

    def controls(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gating_mode == "none":
            zero = torch.zeros_like(x[:, :1])
            return zero, zero
        validate_nchw(x)
        if self.gating_mode == "raw":
            red = x[:, RED : RED + 1]
            nir = x[:, NIR : NIR + 1]
            return self.gradient(red), self.gradient(nir)
        ndvi = torch_ndvi(x)
        return ndvi, self.gradient(ndvi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        controls = self.controls(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        d3 = self.conv1(
            torch.cat([self.up1(x4), self.attn3(x3, controls)], dim=1)
        )
        d2 = self.conv2(
            torch.cat([self.up2(d3), self.attn2(x2, controls)], dim=1)
        )
        d1 = self.conv3(
            torch.cat([self.up3(d2), self.attn1(x1, controls)], dim=1)
        )
        return self.out(d1).squeeze(1)


class ControlledPixelLoss(nn.Module):
    """Base, plain-boundary, or equal-mass NDVI-modulated boundary weighting."""

    MODES = {"base", "boundary", "biophysical"}

    def __init__(
        self,
        mode: str = "base",
        alpha: float = 0.35,
        focal_gamma: float = 2.0,
        boundary_gamma: float = 2.0,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown loss mode: {mode}")
        self.mode = mode
        self.alpha = alpha
        self.focal_gamma = focal_gamma
        self.boundary_gamma = boundary_gamma
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def base_map(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probabilities = torch.sigmoid(logits)
        pt = targets * probabilities + (1 - targets) * (1 - probabilities)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal = alpha_t * (1 - pt).pow(self.focal_gamma) * bce
        return 0.5 * bce + 0.5 * focal

    @staticmethod
    def boundary(targets: torch.Tensor) -> torch.Tensor:
        target = targets.unsqueeze(1)
        dilated = F.max_pool2d(target, 3, stride=1, padding=1)
        eroded = -F.max_pool2d(-target, 3, stride=1, padding=1)
        return (dilated - eroded).squeeze(1)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, images: torch.Tensor
    ) -> torch.Tensor:
        losses = self.base_map(logits, targets)
        if self.mode == "base":
            return losses.mean()
        boundary = self.boundary(targets)
        if self.mode == "boundary":
            modulation = boundary
        else:
            ndvi = torch_ndvi(images)
            gx = F.conv2d(ndvi, self.sobel_x, padding=1)
            gy = F.conv2d(ndvi, self.sobel_y, padding=1)
            gradient = torch.sqrt(gx.square() + gy.square() + 1e-8).squeeze(1)
            maxima = gradient.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            signal = boundary * (gradient / maxima)
            target_mass = boundary.mean(dim=(1, 2), keepdim=True)
            signal_mass = signal.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            modulation = signal * (target_mass / signal_mass)
        weights = 1.0 + self.boundary_gamma * modulation.detach()
        return (losses * weights).mean()
