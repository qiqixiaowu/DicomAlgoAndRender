"""
3D 配准评估指标

指标
====
1. jacobian_determinant_3d : Jacobian 行列式 (检测折叠)
2. folding_ratio_3d        : 折叠比例
3. dice_3d                 : 3D Dice 分数
4. target_registration_error: 目标配准误差 (TRE)
5. ssim_3d                 : 3D 结构相似性 (简化版)
6. psnr_3d                 : 峰值信噪比

工程要点
========
- Jacobian 行列式 < 0 表示拓扑折叠
- TRE 是配准精度的金标准 (需要 landmark)
- Dice 用于评估分割标签的配准效果
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
#  Jacobian 行列式 & 折叠
# ============================================================

def jacobian_determinant_3d(flow: torch.Tensor) -> torch.Tensor:
    """
    计算 3D Jacobian 行列式

    参数
    ----
    flow : (B, 3, D, H, W) 或 (3, D, H, W), 通道顺序 (dz, dy, dx)

    返回
    ----
    det : (B, 1, D-2, H-2, W-2) 或 (D-2, H-2, W-2)
    """
    squeeze = False
    if flow.dim() == 4:
        flow = flow.unsqueeze(0)
        squeeze = True
    assert flow.dim() == 5 and flow.shape[1] == 3

    # 中心差分
    dphi_dz = (flow[:, :, 2:, 1:-1, 1:-1] - flow[:, :, :-2, 1:-1, 1:-1]) / 2.0
    dphi_dy = (flow[:, :, 1:-1, 2:, 1:-1] - flow[:, :, 1:-1, :-2, 1:-1]) / 2.0
    dphi_dx = (flow[:, :, 1:-1, 1:-1, 2:] - flow[:, :, 1:-1, 1:-1, :-2]) / 2.0

    dz_dz, dy_dz, dx_dz = dphi_dz[:, 0], dphi_dz[:, 1], dphi_dz[:, 2]
    dz_dy, dy_dy, dx_dy = dphi_dy[:, 0], dphi_dy[:, 1], dphi_dy[:, 2]
    dz_dx, dy_dx, dx_dx = dphi_dx[:, 0], dphi_dx[:, 1], dphi_dx[:, 2]

    # 变形场 φ = x + flow, Jacobian 矩阵 J = I + ∇flow
    # det(J) = det(I + ∇flow), 对角线加 1
    det = (
        (1 + dz_dz) * ((1 + dy_dy) * (1 + dx_dx) - dy_dx * dx_dy)
        - dz_dy * (dy_dz * (1 + dx_dx) - dy_dx * dx_dz)
        + dz_dx * (dy_dz * dx_dy - (1 + dy_dy) * dx_dz)
    )

    if squeeze:
        det = det.squeeze(0)
    return det


def folding_ratio_3d(flow: torch.Tensor) -> float:
    """
    计算折叠比例 (Jacobian 行列式 ≤ 0 的体素比例)

    返回
    ----
    ratio : float, 0.0 ~ 1.0 (越低越好)
    """
    with torch.no_grad():
        det = jacobian_determinant_3d(flow)
        return float((det <= 0).float().mean().item())


def jacobian_stats_3d(flow: torch.Tensor) -> dict:
    """
    Jacobian 行列式统计信息

    返回
    ----
    dict: mean, std, min, max, folding_ratio
    """
    with torch.no_grad():
        det = jacobian_determinant_3d(flow)
        return {
            "mean": float(det.mean().item()),
            "std": float(det.std().item()),
            "min": float(det.min().item()),
            "max": float(det.max().item()),
            "folding_ratio": float((det <= 0).float().mean().item()),
        }


# ============================================================
#  Dice
# ============================================================

def dice_3d(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    smooth: float = 1e-5,
) -> float:
    """
    3D Dice 分数

    参数
    ----
    pred   : (D, H, W) 二值掩码
    target : (D, H, W) 二值掩码
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    pred = pred.float()
    target = target.float()

    intersection = (pred * target).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def dice_multi_class_3d(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    num_classes: int,
    smooth: float = 1e-5,
) -> dict[str, float]:
    """
    多类 3D Dice

    返回每个类的 Dice 和平均 Dice
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    results = {}
    dices = []
    for c in range(1, num_classes):  # 跳过背景
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        d = dice_3d(pred_c, target_c, smooth)
        results[f"dice_class_{c}"] = d
        dices.append(d)
    results["dice_mean"] = float(np.mean(dices)) if dices else 0.0
    return results


# ============================================================
#  目标配准误差 (TRE)
# ============================================================

def target_registration_error(
    fixed_landmarks: np.ndarray,
    moving_landmarks: np.ndarray,
    flow: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, float]:
    """
    目标配准误差 (Target Registration Error)

    参数
    ----
    fixed_landmarks  : (N, 3) 固定图像上的 landmark 坐标 (z, y, x)
    moving_landmarks : (N, 3) 移动图像上的 landmark 坐标 (z, y, x)
    flow             : (3, D, H, W) 位移场 (dz, dy, dx)
    spacing          : 体素间距 (z, y, x)

    返回
    ----
    dict: mean, std, median, max (单位: mm)
    """
    from scipy.ndimage import map_coordinates

    # 在 moving landmark 位置采样位移
    # flow 的值表示: 该位置的像素应该移动多少
    # 所以 warped_moving = moving + flow(moving)
    # TRE = ||fixed - warped_moving||

    dz = map_coordinates(flow[0], moving_landmarks.T, order=1, mode="nearest")
    dy = map_coordinates(flow[1], moving_landmarks.T, order=1, mode="nearest")
    dx = map_coordinates(flow[2], moving_landmarks.T, order=1, mode="nearest")

    warped_moving = moving_landmarks + np.stack([dz, dy, dx], axis=-1)

    # 转换为物理坐标
    fixed_phys = fixed_landmarks * np.array(spacing)
    warped_phys = warped_moving * np.array(spacing)

    errors = np.linalg.norm(fixed_phys - warped_phys, axis=1)

    return {
        "mean": float(errors.mean()),
        "std": float(errors.std()),
        "median": float(np.median(errors)),
        "max": float(errors.max()),
        "min": float(errors.min()),
    }


# ============================================================
#  SSIM & PSNR
# ============================================================

def ssim_3d(
    img1: torch.Tensor | np.ndarray,
    img2: torch.Tensor | np.ndarray,
    win_size: int = 7,
    data_range: float = 1.0,
) -> float:
    """
    3D 结构相似性 (SSIM) — 简化版

    使用均匀窗口计算局部均值、方差、协方差。
    """
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()

    if img1.dim() == 3:
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    w = win_size
    pad = w // 2
    kernel = torch.ones(1, 1, w, w, w, device=img1.device, dtype=img1.dtype) / (w ** 3)

    mu1 = F.conv3d(img1, kernel, padding=pad)
    mu2 = F.conv3d(img2, kernel, padding=pad)

    sigma1_sq = F.conv3d(img1 * img1, kernel, padding=pad) - mu1 ** 2
    sigma2_sq = F.conv3d(img2 * img2, kernel, padding=pad) - mu2 ** 2
    sigma12 = F.conv3d(img1 * img2, kernel, padding=pad) - mu1 * mu2

    ssim_map = (
        (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    ) / (
        (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return float(ssim_map.mean().item())


def psnr_3d(
    img1: torch.Tensor | np.ndarray,
    img2: torch.Tensor | np.ndarray,
    data_range: float = 1.0,
) -> float:
    """
    3D 峰值信噪比 (PSNR)
    """
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()

    mse = F.mse_loss(img1, img2)
    if mse.item() == 0:
        return float("inf")
    return float(10.0 * torch.log10(data_range ** 2 / mse).item())


# ============================================================
#  综合评估
# ============================================================

@torch.no_grad()
def evaluate_registration_3d(
    fixed: torch.Tensor,
    warped: torch.Tensor,
    flow: torch.Tensor,
    fixed_mask: torch.Tensor | None = None,
    warped_mask: torch.Tensor | None = None,
    data_range: float = 1.0,
) -> dict:
    """
    综合评估 3D 配准结果

    参数
    ----
    fixed       : (B, 1, D, H, W) 固定图像
    warped      : (B, 1, D, H, W) 配准后图像
    flow        : (B, 3, D, H, W) 位移场
    fixed_mask  : (B, 1, D, H, W) 固定图像分割掩码 (可选)
    warped_mask : (B, 1, D, H, W) 配准后分割掩码 (可选)

    返回
    ----
    dict: 各项指标
    """
    results = {}

    # 图像相似性
    results["ssim"] = ssim_3d(fixed, warped, data_range=data_range)
    results["psnr"] = psnr_3d(fixed, warped, data_range=data_range)
    results["mse"] = float(F.mse_loss(fixed, warped).item())

    # Jacobian
    jac_stats = jacobian_stats_3d(flow)
    results.update({f"jac_{k}": v for k, v in jac_stats.items()})

    # Dice (如果有掩码)
    if fixed_mask is not None and warped_mask is not None:
        dices = []
        for b in range(fixed.shape[0]):
            d = dice_3d(
                warped_mask[b, 0] > 0.5,
                fixed_mask[b, 0] > 0.5,
            )
            dices.append(d)
        results["dice"] = float(np.mean(dices))

    return results


if __name__ == "__main__":
    print("测试 3D 配准评估指标...")

    # 模拟数据
    fixed = torch.randn(1, 1, 32, 32, 32)
    warped = fixed + torch.randn_like(fixed) * 0.1
    flow = torch.randn(1, 3, 32, 32, 32) * 0.05

    print("\n--- Jacobian ---")
    det = jacobian_determinant_3d(flow)
    print(f"  det shape: {det.shape}")
    print(f"  folding_ratio: {folding_ratio_3d(flow):.4f}")
    print(f"  stats: {jacobian_stats_3d(flow)}")

    print("\n--- 相似性 ---")
    print(f"  SSIM: {ssim_3d(fixed, warped):.4f}")
    print(f"  PSNR: {psnr_3d(fixed, warped):.4f}")

    print("\n--- Dice ---")
    mask1 = (torch.randn(32, 32, 32) > 0).float()
    mask2 = mask1.clone()
    mask2[5:10, 5:10, 5:10] = 1 - mask2[5:10, 5:10, 5:10]
    print(f"  Dice: {dice_3d(mask1, mask2):.4f}")

    print("\n--- 综合评估 ---")
    results = evaluate_registration_3d(fixed, warped, flow)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
