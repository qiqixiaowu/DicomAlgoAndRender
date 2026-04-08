"""
2D U-Net —— 经典医学图像分割网络

结构概览
========

编码器 (下采样路径)          解码器 (上采样路径)
─────────────────          ─────────────────
 Input (1, 256, 256)
   │
 ┌─▼─────────────┐        ┌──────────────┐
 │ DoubleConv 64  │───────►│ DoubleConv 64│ → Output
 └──────┬────────┘  skip   └──────▲───────┘
     MaxPool                   UpConv
 ┌──────▼────────┐        ┌──────┴───────┐
 │ DoubleConv 128│───────►│ DoubleConv128│
 └──────┬────────┘  skip   └──────▲───────┘
     MaxPool                   UpConv
 ┌──────▼────────┐        ┌──────┴───────┐
 │ DoubleConv 256│───────►│ DoubleConv256│
 └──────┬────────┘  skip   └──────▲───────┘
     MaxPool                   UpConv
 ┌──────▼────────┐        ┌──────┴───────┐
 │ DoubleConv 512│───────►│ DoubleConv512│
 └──────┬────────┘  skip   └──────▲───────┘
     MaxPool                   UpConv
 ┌──────▼────────┐
 │ Bottleneck1024│
 └───────────────┘

核心思想
========
1. 编码器: 逐层提取从低级到高级的语义特征
2. 跳跃连接 (Skip Connection): 将编码器特征拼接到解码器，保留空间细节
3. 解码器: 逐层恢复空间分辨率，融合多尺度特征
4. 最终 1×1 卷积: 映射到类别概率

关键设计点
=========
- DoubleConv: 每层两个 3×3 卷积 + BatchNorm + ReLU，提升特征表达力
- MaxPool 2×2: 下采样，扩大感受野
- 转置卷积 (ConvTranspose2d): 上采样，恢复分辨率
- 输出不加 Softmax: 因为 CrossEntropyLoss 内部已包含
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """两个连续的 (Conv3x3 → BN → ReLU) 块"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            # 第一个卷积: 改变通道数
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 第二个卷积: 保持通道数, 进一步提取特征
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2D(nn.Module):
    """
    经典 2D U-Net

    参数
    ----
    in_channels : int
        输入图像通道数 (灰度=1, RGB=3)
    out_channels : int
        输出类别数 (包含背景)
    features : list[int]
        各层的特征通道数, 默认 [64, 128, 256, 512]
        瓶颈层通道数 = features[-1] * 2

    示例
    ----
    >>> model = UNet2D(in_channels=1, out_channels=2)
    >>> x = torch.randn(1, 1, 256, 256)
    >>> out = model(x)  # shape: (1, 2, 256, 256)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        features: list[int] | None = None,
    ):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        # ---------- 编码器 ----------
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv(ch, f))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            ch = f

        # ---------- 瓶颈 ----------
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # ---------- 解码器 ----------
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        reversed_features = list(reversed(features))
        ch = features[-1] * 2
        for f in reversed_features:
            # 转置卷积上采样
            self.upconvs.append(
                nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2)
            )
            # 拼接后通道数 = f (来自 skip) + f (上采样) = 2f
            self.decoders.append(DoubleConv(f * 2, f))
            ch = f

        # ---------- 输出头 ----------
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码路径: 保存跳跃连接特征
        skip_connections = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)  # 保存当前层特征
            x = pool(x)                # 下采样

        # 瓶颈
        x = self.bottleneck(x)

        # 解码路径: 反转跳跃连接顺序
        skip_connections = skip_connections[::-1]
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)              # 上采样

            # 尺寸对齐 (处理奇数尺寸的情况)
            skip = skip_connections[i]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )

            # 跳跃连接: 在通道维度拼接
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        # 1×1 卷积映射到类别数
        return self.final_conv(x)


# ─── 快速测试 ──────────────────────────────────────────────
if __name__ == "__main__":
    model = UNet2D(in_channels=1, out_channels=2, features=[64, 128, 256, 512])
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(f"输入: {x.shape}")
    print(f"输出: {y.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
