"""
损失函数集合 —— 分割 & 配准

分割损失
========
- DiceLoss      : Dice 系数的损失版本, 适合类别不平衡
- DiceCELoss    : Dice + CrossEntropy 组合, 兼顾全局和局部

配准损失
========
- NCCLoss       : 归一化互相关, 对亮度变化鲁棒
- MSELoss       : 均方误差 (PyTorch 内置, 这里不重复)
- GradientLoss  : 形变场梯度平滑正则化
- BendingEnergyLoss : 弯曲能量正则化 (更强的平滑约束)

为什么需要这些自定义损失？
========================
1. 标准 CrossEntropy 对类别不平衡敏感 (医学图像中前景通常很小)
   → Dice Loss 直接优化 F1 分数

2. 配准无标注 → 用图像相似性作为监督信号
   → NCC 对灰度线性变换不变, 比 MSE 更鲁棒

3. 形变场需要正则化
   → 无约束的形变场可能出现折叠 (不合理的变形)
   → 平滑正则化确保变形的物理合理性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  分割损失
# ============================================================

class DiceLoss(nn.Module):
    """
    Dice Loss = 1 - (2 * |A ∩ B|) / (|A| + |B|)

    Dice 系数衡量两个集合的重叠度:
    - 完全重叠: Dice = 1, Loss = 0
    - 无重叠:   Dice = 0, Loss = 1

    smooth 项防止分母为零 (Laplace 平滑)
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        自动适配 2D / 3D 输入:
          2D: pred (B, C, H, W),     target (B, H, W)
          3D: pred (B, C, D, H, W),  target (B, D, H, W)
        """
        num_classes = pred.shape[1]
        spatial_ndim = pred.dim() - 2  # 2D→2, 3D→3

        # Softmax → 概率
        pred_soft = F.softmax(pred, dim=1)

        # One-hot 编码 target: (..., C) → (B, C, ...)
        target_one_hot = F.one_hot(target.long(), num_classes)
        # 原始形状: (B, [D,] H, W, C)，把 C 移到 dim=1
        # permute: 0, ndim+1, 1..ndim
        perm = [0, target_one_hot.dim() - 1] + list(range(1, target_one_hot.dim() - 1))
        target_one_hot = target_one_hot.permute(*perm).float()  # (B, C, [D,] H, W)

        # 在所有空间维度上 sum: dim=(2, 3) for 2D, dim=(2, 3, 4) for 3D
        spatial_dims = tuple(range(2, 2 + spatial_ndim))
        intersection = (pred_soft * target_one_hot).sum(dim=spatial_dims)
        union = pred_soft.sum(dim=spatial_dims) + target_one_hot.sum(dim=spatial_dims)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # 所有类别取平均, 返回 loss
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """
    Dice + CrossEntropy 组合损失

    为什么组合？
    - CE 提供像素级梯度, 训练早期收敛快
    - Dice 关注整体重叠度, 对小目标更友好
    - 组合后两全其美
    """

    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.dice_weight * self.dice(pred, target)
            + self.ce_weight * self.ce(pred, target.long())
        )


# ============================================================
#  配准损失
# ============================================================

class NCCLoss(nn.Module):
    """
    局部归一化互相关 (Local Normalized Cross-Correlation)

    NCC(I, J) = Σ[(I - μ_I)(J - μ_J)] / √[Σ(I - μ_I)² · Σ(J - μ_J)²]

    特性:
    - 取值范围 [-1, 1], 越大越相似
    - 对亮度线性变换不变 (适合多模态)
    - 使用局部窗口计算 → 捕捉局部结构

    局部计算通过卷积实现:
    - 均值 μ = Conv(I, 1/n)  (均匀核卷积 = 局部平均)
    - 这比逐像素循环快几个数量级
    """

    def __init__(self, window_size: int = 9):
        super().__init__()
        self.window_size = window_size

    def forward(
        self, fixed: torch.Tensor, warped: torch.Tensor
    ) -> torch.Tensor:
        """
        fixed, warped : (B, 1, H, W)
        返回: -NCC (取负号使其成为要最小化的损失)
        """
        ndim = len(fixed.shape) - 2  # 2 for 2D
        window = [self.window_size] * ndim

        # 构造均匀卷积核
        padding = [w // 2 for w in window]
        kernel = torch.ones(1, 1, *window, device=fixed.device) / float(
            torch.tensor(window).prod()
        )

        conv_fn = F.conv2d if ndim == 2 else F.conv3d

        # 局部均值
        I = fixed
        J = warped
        I_mean = conv_fn(I, kernel, padding=padding)
        J_mean = conv_fn(J, kernel, padding=padding)

        # 去均值
        I_centered = I - I_mean
        J_centered = J - J_mean

        # 局部互相关
        cross = conv_fn(I_centered * J_centered, kernel, padding=padding)
        I_var = conv_fn(I_centered * I_centered, kernel, padding=padding)
        J_var = conv_fn(J_centered * J_centered, kernel, padding=padding)

        # NCC
        ncc = cross / (torch.sqrt(I_var * J_var + 1e-5))

        # 返回负 NCC 作为损失 (最小化 → 最大化相似度)
        return -ncc.mean()


class GradientLoss(nn.Module):
    """
    形变场梯度平滑正则化

    L_smooth = Σ ‖∇φ‖²

    直觉: 形变场在空间上应该平滑变化, 相邻像素的位移应该接近。
    梯度大 → 变形剧烈 → 可能不合理 → 惩罚。

    计算方法: 有限差分 (相邻像素之差)
    """

    def __init__(self, penalty: str = "l2"):
        super().__init__()
        self.penalty = penalty

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """
        flow : (B, 2, H, W)
        """
        # 沿 y 方向差分
        dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
        # 沿 x 方向差分
        dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]

        if self.penalty == "l2":
            loss = (dy ** 2).mean() + (dx ** 2).mean()
        elif self.penalty == "l1":
            loss = dy.abs().mean() + dx.abs().mean()
        else:
            raise ValueError(f"未知 penalty: {self.penalty}")

        return loss


class BendingEnergyLoss(nn.Module):
    """
    弯曲能量正则化 (Bending Energy)

    L_be = Σ (∂²φ/∂x²)² + (∂²φ/∂y²)² + 2·(∂²φ/∂x∂y)²

    比一阶梯度正则化更强:
    - 一阶: 惩罚位移变化 → 偏好常数位移场
    - 二阶: 惩罚曲率 → 允许线性变化, 惩罚非线性变形
    - 结果更平滑, 变形更自然
    """

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """flow: (B, 2, H, W)"""
        # 二阶差分
        d2x = flow[:, :, :, 2:] + flow[:, :, :, :-2] - 2 * flow[:, :, :, 1:-1]
        d2y = flow[:, :, 2:, :] + flow[:, :, :-2, :] - 2 * flow[:, :, 1:-1, :]

        # 交叉项
        dxy = (
            flow[:, :, 1:, 1:] + flow[:, :, :-1, :-1]
            - flow[:, :, 1:, :-1] - flow[:, :, :-1, 1:]
        )

        return (d2x ** 2).mean() + (d2y ** 2).mean() + 2 * (dxy ** 2).mean()
