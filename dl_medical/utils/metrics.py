"""
评估指标 —— 分割 & 配准

分割指标
========
- Dice Coefficient (Dice 系数)
- IoU / Jaccard Index (交并比)
- Hausdorff Distance 95 (95% 豪斯多夫距离)
- Sensitivity / Specificity (敏感度 / 特异度)

配准指标
========
- Dice (配准后 label 重叠度)
- Jacobian Determinant (雅可比行列式, 评估变形合理性)
- Target Registration Error (目标配准误差)

指标的意义
=========
- Dice / IoU: 衡量预测区域与真值区域的重叠程度, 1.0 = 完美
- HD95: 衡量边界距离, 对边界质量敏感
- Jacobian < 0 的比例: 形变场的折叠率, 理想情况为 0%
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def dice_coefficient(
    pred: np.ndarray, target: np.ndarray, num_classes: int = 2
) -> dict[str, float]:
    """
    计算每个类别的 Dice 系数

    参数
    ----
    pred   : (H, W) 预测标签 (int)
    target : (H, W) 真值标签 (int)

    返回
    ----
    dict: {"class_0": 0.99, "class_1": 0.85, "mean": 0.92}
    """
    result = {}
    dices = []
    for c in range(num_classes):
        p = (pred == c).astype(float)
        t = (target == c).astype(float)
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        if union == 0:
            dice = 1.0  # 两者都为空, 视为完美匹配
        else:
            dice = 2.0 * intersection / union
        result[f"class_{c}"] = dice
        dices.append(dice)
    result["mean"] = np.mean(dices)
    return result


def iou_score(
    pred: np.ndarray, target: np.ndarray, num_classes: int = 2
) -> dict[str, float]:
    """
    IoU (Intersection over Union) / Jaccard Index

    IoU = |A ∩ B| / |A ∪ B|
    与 Dice 的关系: IoU = Dice / (2 - Dice)
    """
    result = {}
    ious = []
    for c in range(num_classes):
        p = (pred == c).astype(float)
        t = (target == c).astype(float)
        intersection = (p * t).sum()
        union = p.sum() + t.sum() - intersection
        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union
        result[f"class_{c}"] = iou
        ious.append(iou)
    result["mean"] = np.mean(ious)
    return result


def hausdorff_distance_95(
    pred: np.ndarray, target: np.ndarray
) -> float:
    """
    95% Hausdorff Distance (HD95)

    衡量两个轮廓之间的距离:
    1. 对 pred 的边界点, 计算到 target 边界的最近距离
    2. 反过来同理
    3. 取所有距离的第 95 百分位数 (比最大值更鲁棒)
    """
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)

    if not pred_bool.any() or not target_bool.any():
        return 0.0 if (not pred_bool.any() and not target_bool.any()) else float("inf")

    # 距离变换: 每个像素到最近目标边界的距离
    dist_pred = distance_transform_edt(~pred_bool)
    dist_target = distance_transform_edt(~target_bool)

    # pred 到 target 的距离
    surface_pred = dist_target[pred_bool]
    # target 到 pred 的距离
    surface_target = dist_pred[target_bool]

    all_distances = np.concatenate([surface_pred, surface_target])
    return float(np.percentile(all_distances, 95))


def sensitivity_specificity(
    pred: np.ndarray, target: np.ndarray
) -> tuple[float, float]:
    """
    前景 (类别 1) 的敏感度和特异度

    敏感度 (Recall) = TP / (TP + FN)  → 能找到多少正样本
    特异度 = TN / (TN + FP)           → 能排除多少负样本
    """
    pred_fg = (pred > 0).astype(float)
    target_fg = (target > 0).astype(float)

    tp = (pred_fg * target_fg).sum()
    fn = ((1 - pred_fg) * target_fg).sum()
    fp = (pred_fg * (1 - target_fg)).sum()
    tn = ((1 - pred_fg) * (1 - target_fg)).sum()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return float(sensitivity), float(specificity)


# ============================================================
#  配准指标
# ============================================================

def jacobian_determinant(flow: np.ndarray) -> np.ndarray:
    """
    计算 2D 形变场的雅可比行列式

    参数
    ----
    flow : (2, H, W) 形变场

    返回
    ----
    jac_det : (H-1, W-1) 雅可比行列式值

    解释
    ----
    - det(J) = 1 : 无体积变化 (刚性)
    - det(J) > 1 : 局部膨胀
    - det(J) < 1 : 局部收缩
    - det(J) ≤ 0 : 折叠! 变形不合理
    """
    # 加上恒等映射 → 得到完整坐标场
    h, w = flow.shape[1], flow.shape[2]
    gy, gx = np.mgrid[0:h, 0:w]
    phi_y = flow[0] + gy
    phi_x = flow[1] + gx

    # 计算偏导数 (有限差分)
    dphi_y_dy = phi_y[1:, :-1] - phi_y[:-1, :-1]
    dphi_y_dx = phi_y[:-1, 1:] - phi_y[:-1, :-1]
    dphi_x_dy = phi_x[1:, :-1] - phi_x[:-1, :-1]
    dphi_x_dx = phi_x[:-1, 1:] - phi_x[:-1, :-1]

    # 2x2 雅可比矩阵的行列式
    jac_det = dphi_y_dy * dphi_x_dx - dphi_y_dx * dphi_x_dy

    return jac_det


def folding_ratio(flow: np.ndarray) -> float:
    """
    计算形变场折叠率 (Jacobian determinant ≤ 0 的比例)

    理想情况: 0% (没有折叠)
    """
    jac = jacobian_determinant(flow)
    return float((jac <= 0).sum() / jac.size)
