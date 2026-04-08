"""
VoxelMorph 2D —— 基于深度学习的可变形图像配准

核心思想
========
给定一对图像:
  - Fixed image  (固定图像, 参考图)
  - Moving image (运动图像, 待配准图)

网络学习一个 **形变场 (Deformation Field)** φ, 使得:
  Moving ∘ φ ≈ Fixed

即通过空间变换将 Moving 变形到与 Fixed 对齐。

网络结构
========

  Fixed ──┐
          ├─ Concat ──► Encoder ──► Decoder ──► 形变场 φ(x,y)
  Moving ─┘                                        │
                                                    ▼
                                    Moving ──► SpatialTransformer ──► Warped

  损失 = Similarity(Fixed, Warped) + λ · Smoothness(φ)

关键组件
========
1. **编码器-解码器**: 类似 U-Net, 提取多尺度特征, 预测形变场
2. **空间变换器 (Spatial Transformer)**: 可微分的图像变形操作
3. **相似性损失**: NCC (归一化互相关) 或 MSE
4. **平滑正则化**: 约束形变场平滑, 防止不合理的剧烈变形

配准 vs 分割的区别
==================
- 分割: 输出离散类别标签 → 分类问题
- 配准: 输出连续形变场 → 回归问题
- 配准不需要标注数据！ 只需要图像对 → 自监督学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """
    可微分空间变换器 (Spatial Transformer Network, STN)

    根据形变场对输入图像进行变形:
      output[b, c, y, x] = input[b, c, y + flow_y, x + flow_x]

    原理
    ----
    1. 生成规则网格坐标 grid (identity transform)
    2. 加上形变场 flow → 得到采样坐标
    3. 使用双线性插值从输入图像采样

    为什么可微分？
    因为双线性插值是对4个邻近像素的加权求和,
    权重关于坐标连续可微 → 梯度可以反向传播到形变场 → 可以端到端训练。
    """

    def __init__(self, size: tuple[int, int]):
        super().__init__()
        # 预计算规则网格 (归一化到 [-1, 1])
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)  # (2, H, W)
        grid = grid.unsqueeze(0).float()  # (1, 2, H, W)

        # 归一化到 [-1, 1] (grid_sample 要求)
        for i in range(len(size)):
            grid[:, i, ...] = 2 * grid[:, i, ...] / (size[i] - 1) - 1

        # 注册为 buffer (不参与梯度更新, 但会随模型移动设备)
        self.register_buffer("grid", grid)

    def forward(
        self, src: torch.Tensor, flow: torch.Tensor
    ) -> torch.Tensor:
        """
        参数
        ----
        src : (B, C, H, W) – 待变形图像
        flow : (B, 2, H, W) – 形变场 (像素级位移)

        返回
        ----
        warped : (B, C, H, W) – 变形后图像
        """
        # flow 是像素位移, 需要归一化到 [-1, 1]
        shape = flow.shape[2:]
        flow_normalized = flow.clone()
        for i in range(2):
            flow_normalized[:, i, ...] = flow[:, i, ...] / (shape[i] - 1) * 2

        # 采样坐标 = 规则网格 + 归一化位移
        grid = self.grid + flow_normalized

        # grid_sample 要求: (B, H, W, 2), 且 x 在前 y 在后
        grid = grid.permute(0, 2, 3, 1)          # (B, H, W, 2)
        grid = grid[..., [1, 0]]                  # y,x → x,y

        return F.grid_sample(
            src, grid, mode="bilinear", padding_mode="border", align_corners=True
        )


class ConvBlock(nn.Module):
    """卷积块: Conv → LeakyReLU"""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VoxelMorph2D(nn.Module):
    """
    VoxelMorph 2D 配准网络

    参数
    ----
    image_size : tuple[int, int]
        输入图像尺寸 (H, W)
    enc_features : list[int]
        编码器各层通道数, 默认 [16, 32, 32, 32]
    dec_features : list[int]
        解码器各层通道数, 默认 [32, 32, 32, 16]

    前向传播
    --------
    输入: fixed (B,1,H,W), moving (B,1,H,W)
    输出: warped (B,1,H,W), flow (B,2,H,W)

    示例
    ----
    >>> model = VoxelMorph2D(image_size=(128, 128))
    >>> fixed = torch.randn(2, 1, 128, 128)
    >>> moving = torch.randn(2, 1, 128, 128)
    >>> warped, flow = model(fixed, moving)
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (128, 128),
        enc_features: list[int] | None = None,
        dec_features: list[int] | None = None,
    ):
        super().__init__()
        if enc_features is None:
            enc_features = [16, 32, 32, 32]
        if dec_features is None:
            dec_features = [32, 32, 32, 16]

        # ---- 编码器 (下采样) ----
        self.encoders = nn.ModuleList()
        in_ch = 2  # fixed + moving 拼接
        for feat in enc_features:
            self.encoders.append(ConvBlock(in_ch, feat, stride=2))
            in_ch = feat

        # ---- 解码器 (上采样 + skip) ----
        self.decoders = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        enc_reversed = list(reversed(enc_features))
        for i, feat in enumerate(dec_features):
            # 上采样后拼接 skip connection
            skip_ch = enc_reversed[i + 1] if i + 1 < len(enc_reversed) else 2
            self.upsamplers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            self.decoders.append(ConvBlock(in_ch + skip_ch, feat))
            in_ch = feat

        # ---- 最终上采样到原始分辨率 ----
        self.final_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.final_conv = ConvBlock(in_ch + 2, dec_features[-1])

        # ---- 形变场预测 (2 通道: dx, dy) ----
        self.flow_conv = nn.Conv2d(dec_features[-1], 2, kernel_size=3, padding=1)
        # 用小权重初始化, 使初始形变场接近零 (恒等变换)
        nn.init.normal_(self.flow_conv.weight, mean=0, std=1e-5)
        nn.init.zeros_(self.flow_conv.bias)

        # ---- 空间变换器 ----
        self.spatial_transformer = SpatialTransformer(image_size)

    def forward(
        self, fixed: torch.Tensor, moving: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回
        ----
        warped : 变形后的 moving 图像
        flow   : 预测的形变场
        """
        # 拼接输入
        x = torch.cat([fixed, moving], dim=1)  # (B, 2, H, W)
        input_x = x

        # 编码 (保存各层特征用于 skip)
        enc_features = [input_x]
        for encoder in self.encoders:
            x = encoder(x)
            enc_features.append(x)

        # 解码
        enc_features_reversed = list(reversed(enc_features))
        for i, (upsample, decoder) in enumerate(zip(self.upsamplers, self.decoders)):
            x = upsample(x)
            skip = enc_features_reversed[i + 2]
            # 尺寸对齐
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        # 最终上采样
        x = self.final_upsample(x)
        if x.shape[2:] != input_x.shape[2:]:
            x = F.interpolate(x, size=input_x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, input_x], dim=1)
        x = self.final_conv(x)

        # 预测形变场
        flow = self.flow_conv(x)  # (B, 2, H, W)

        # 空间变换
        warped = self.spatial_transformer(moving, flow)

        return warped, flow


if __name__ == "__main__":
    model = VoxelMorph2D(image_size=(128, 128))
    fixed = torch.randn(2, 1, 128, 128)
    moving = torch.randn(2, 1, 128, 128)
    warped, flow = model(fixed, moving)
    print(f"Fixed:  {fixed.shape}")
    print(f"Moving: {moving.shape}")
    print(f"Warped: {warped.shape}")
    print(f"Flow:   {flow.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
