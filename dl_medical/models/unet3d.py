"""
3D U-Net —— 体数据分割网络

与 2D U-Net 的区别
==================
- 所有 Conv2d → Conv3d, MaxPool2d → MaxPool3d
- 输入: (B, C, D, H, W) 五维张量
- 计算量和显存需求显著增大, 通常需要:
  ① 减小 features 通道数
  ② 使用 patch-based 训练 (裁剪小块输入)
  ③ 混合精度训练 (AMP)

适用场景
========
- CT / MRI 体数据器官分割 (肝脏/肾脏/肿瘤等)
- 3D 医学图像中的区域标注
"""

import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):
    """3D 双卷积块: (Conv3x3x3 → BN → ReLU) × 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet3D(nn.Module):
    """
    3D U-Net

    参数
    ----
    in_channels : int  – 输入通道 (通常 1)
    out_channels : int – 分割类别数
    features : list[int] – 各层通道数, 默认 [32, 64, 128, 256] (比 2D 小)

    示例
    ----
    >>> model = UNet3D(1, 3, features=[32, 64, 128, 256])
    >>> x = torch.randn(1, 1, 64, 64, 64)
    >>> y = model(x)  # (1, 3, 64, 64, 64)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        features: list[int] | None = None,
    ):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        # 编码器
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv3D(ch, f))
            self.pools.append(nn.MaxPool3d(2))
            ch = f

        # 瓶颈
        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)

        # 解码器
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = features[-1] * 2
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose3d(ch, f, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv3D(f * 2, f))
            ch = f

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for i, (up, dec) in enumerate(zip(self.upconvs, self.decoders)):
            x = up(x)
            skip = skips[-(i + 1)]
            # 用 interpolate 对齐尺寸，避免 ONNX trace 时的 shape 比较 warning
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(
                    x, size=skip.shape[2:], mode="trilinear", align_corners=False
                )
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.final_conv(x)


if __name__ == "__main__":
    model = UNet3D(1, 2, features=[32, 64, 128, 256])
    x = torch.randn(1, 1, 64, 64, 64)
    y = model(x)
    print(f"输入: {x.shape}")
    print(f"输出: {y.shape}")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
