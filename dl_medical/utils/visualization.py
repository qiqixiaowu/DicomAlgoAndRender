"""
可视化工具 —— 分割结果、配准结果、训练曲线
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path


def plot_segmentation_result(
    image: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray | None = None,
    save_path: str | None = None,
    title: str = "",
) -> None:
    """
    可视化分割结果

    image  : (H, W) 原始图像
    pred   : (H, W) 预测掩码
    target : (H, W) 真值掩码 (可选)
    """
    n_cols = 3 if target is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    # 原始图像
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("原始图像")
    axes[0].axis("off")

    # 预测 overlay
    axes[1].imshow(image, cmap="gray")
    overlay = np.ma.masked_where(pred == 0, pred)
    axes[1].imshow(overlay, cmap="jet", alpha=0.4)
    axes[1].set_title("预测分割")
    axes[1].axis("off")

    # 真值 overlay
    if target is not None:
        axes[2].imshow(image, cmap="gray")
        overlay_gt = np.ma.masked_where(target == 0, target)
        axes[2].imshow(overlay_gt, cmap="jet", alpha=0.4)
        axes[2].set_title("真值标注")
        axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_registration_result(
    fixed: np.ndarray,
    moving: np.ndarray,
    warped: np.ndarray,
    flow: np.ndarray | None = None,
    save_path: str | None = None,
) -> None:
    """
    可视化配准结果

    fixed  : (H, W) 固定图像
    moving : (H, W) 运动图像
    warped : (H, W) 配准后图像
    flow   : (2, H, W) 形变场 (可选)
    """
    n_cols = 5 if flow is not None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    axes[0].imshow(fixed, cmap="gray")
    axes[0].set_title("Fixed (固定)")
    axes[0].axis("off")

    axes[1].imshow(moving, cmap="gray")
    axes[1].set_title("Moving (运动)")
    axes[1].axis("off")

    axes[2].imshow(warped, cmap="gray")
    axes[2].set_title("Warped (配准后)")
    axes[2].axis("off")

    # 差异图
    diff = np.abs(fixed - warped)
    axes[3].imshow(diff, cmap="hot")
    axes[3].set_title("差异图 |Fixed - Warped|")
    axes[3].axis("off")

    # 形变场 (网格线)
    if flow is not None:
        ax = axes[4]
        h, w = flow.shape[1], flow.shape[2]
        step = max(h // 20, 1)
        yy, xx = np.mgrid[0:h:step, 0:w:step]
        ax.imshow(fixed, cmap="gray", alpha=0.3)
        ax.quiver(
            xx, yy,
            flow[1, ::step, ::step],
            -flow[0, ::step, ::step],
            color="cyan", scale=200, width=0.003,
        )
        ax.set_title("形变场")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    train_metrics: list[float] | None = None,
    val_metrics: list[float] | None = None,
    metric_name: str = "Dice",
    save_path: str | None = None,
) -> None:
    """绘制训练/验证曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    axes[0].plot(train_losses, label="Train Loss", color="blue")
    if val_losses:
        axes[0].plot(val_losses, label="Val Loss", color="red")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("损失曲线")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Metrics
    if train_metrics:
        axes[1].plot(train_metrics, label=f"Train {metric_name}", color="blue")
    if val_metrics:
        axes[1].plot(val_metrics, label=f"Val {metric_name}", color="red")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(metric_name)
    axes[1].set_title(f"{metric_name} 曲线")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
