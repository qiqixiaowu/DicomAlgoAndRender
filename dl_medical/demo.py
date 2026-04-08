"""
快速演示脚本 —— 无需训练, 立即运行

================================
python demo.py
================================

功能:
1. 生成合成分割数据并展示
2. 创建一个小 U-Net, 做几次迭代, 展示学习过程
3. 生成合成配准对并展示
4. 创建一个小 VoxelMorph, 做几次迭代, 展示配准效果

目的: 让你快速理解整个流程, 无需等待完整训练
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("TkAgg")  # 如果无图形界面, 改为 "Agg"
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.synthetic import SyntheticSegData, SyntheticRegData
from models.unet2d import UNet2D
from models.voxelmorph import VoxelMorph2D
from models.losses import DiceCELoss, NCCLoss, GradientLoss


def demo_segmentation() -> None:
    """分割演示"""
    print("=" * 60)
    print("  演示 1: 医学图像分割 (U-Net)")
    print("=" * 60)

    # ---- 数据 ----
    print("\n[1] 生成合成分割数据...")
    gen = SyntheticSegData(image_size=128, max_objects=3)
    images, masks = gen.generate_batch(16)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(4):
        axes[0, i].imshow(images[i], cmap="gray")
        axes[0, i].set_title(f"图像 #{i+1}")
        axes[0, i].axis("off")
        axes[1, i].imshow(masks[i], cmap="jet", interpolation="nearest")
        axes[1, i].set_title(f"标签 #{i+1}")
        axes[1, i].axis("off")
    fig.suptitle("合成分割数据: 图像 (上) + 标签 (下)", fontsize=14)
    plt.tight_layout()
    plt.savefig("demo_seg_data.png", dpi=150)
    print("  已保存: demo_seg_data.png")
    plt.show()

    # ---- 快速训练 ----
    print("\n[2] 创建 U-Net 并快速训练 20 个迭代...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D(in_channels=1, out_channels=2, features=[16, 32, 64]).to(device)
    criterion = DiceCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 准备小批次数据
    x = torch.from_numpy(images[:8]).unsqueeze(1).float().to(device)
    y = torch.from_numpy(masks[:8]).long().to(device)

    losses = []
    for step in range(20):
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 5 == 0:
            print(f"  Step {step+1:3d} | Loss: {loss.item():.4f}")

    # ---- 可视化结果 ----
    print("\n[3] 可视化预测结果...")
    model.eval()
    with torch.no_grad():
        pred = model(x[:4]).argmax(dim=1).cpu().numpy()

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for i in range(4):
        axes[0, i].imshow(images[i], cmap="gray")
        axes[0, i].set_title("原始图像")
        axes[0, i].axis("off")

        axes[1, i].imshow(masks[i], cmap="jet", interpolation="nearest")
        axes[1, i].set_title("真值标签")
        axes[1, i].axis("off")

        axes[2, i].imshow(pred[i], cmap="jet", interpolation="nearest")
        axes[2, i].set_title("预测 (20步)")
        axes[2, i].axis("off")

    fig.suptitle("U-Net 分割: 原图 / 真值 / 预测 (仅训练20步)", fontsize=14)
    plt.tight_layout()
    plt.savefig("demo_seg_result.png", dpi=150)
    print("  已保存: demo_seg_result.png")
    plt.show()


def demo_registration() -> None:
    """配准演示"""
    print("\n" + "=" * 60)
    print("  演示 2: 医学图像配准 (VoxelMorph)")
    print("=" * 60)

    # ---- 数据 ----
    print("\n[1] 生成合成配准对...")
    gen = SyntheticRegData(image_size=128, deform_alpha=12.0)
    fixed_imgs, moving_imgs = gen.generate_batch(16)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(4):
        axes[0, i].imshow(fixed_imgs[i], cmap="gray")
        axes[0, i].set_title(f"Fixed #{i+1}")
        axes[0, i].axis("off")
        axes[1, i].imshow(moving_imgs[i], cmap="gray")
        axes[1, i].set_title(f"Moving #{i+1}")
        axes[1, i].axis("off")
    fig.suptitle("合成配准数据: Fixed (上) + Moving (下)", fontsize=14)
    plt.tight_layout()
    plt.savefig("demo_reg_data.png", dpi=150)
    print("  已保存: demo_reg_data.png")
    plt.show()

    # ---- 快速训练 ----
    print("\n[2] 创建 VoxelMorph 并快速训练 30 个迭代...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VoxelMorph2D(
        image_size=(128, 128),
        enc_features=[16, 32, 32],
        dec_features=[32, 32, 16],
    ).to(device)

    sim_loss_fn = NCCLoss(window_size=9)
    smooth_loss_fn = GradientLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    f = torch.from_numpy(fixed_imgs[:8]).unsqueeze(1).float().to(device)
    m = torch.from_numpy(moving_imgs[:8]).unsqueeze(1).float().to(device)

    losses = []
    for step in range(30):
        warped, flow = model(f, m)
        sim_loss = sim_loss_fn(f, warped)
        smooth_loss = smooth_loss_fn(flow)
        loss = sim_loss + 0.01 * smooth_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 10 == 0:
            print(f"  Step {step+1:3d} | Loss: {loss.item():.4f} "
                  f"(Sim: {sim_loss.item():.4f}, Smooth: {smooth_loss.item():.4f})")

    # ---- 可视化 ----
    print("\n[3] 可视化配准结果...")
    model.eval()
    with torch.no_grad():
        warped, flow = model(f[:4], m[:4])
        warped_np = warped.squeeze(1).cpu().numpy()
        flow_np = flow.cpu().numpy()

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    titles = ["Fixed", "Moving", "Warped", "|Fixed-Warped|"]
    for i in range(4):
        axes[i, 0].imshow(fixed_imgs[i], cmap="gray")
        axes[i, 0].set_title(titles[0] if i == 0 else "")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(moving_imgs[i], cmap="gray")
        axes[i, 1].set_title(titles[1] if i == 0 else "")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(warped_np[i], cmap="gray")
        axes[i, 2].set_title(titles[2] if i == 0 else "")
        axes[i, 2].axis("off")

        diff = np.abs(fixed_imgs[i] - warped_np[i])
        axes[i, 3].imshow(diff, cmap="hot")
        axes[i, 3].set_title(titles[3] if i == 0 else "")
        axes[i, 3].axis("off")

    fig.suptitle("VoxelMorph 配准: Fixed / Moving / Warped / 差异 (30步)", fontsize=14)
    plt.tight_layout()
    plt.savefig("demo_reg_result.png", dpi=150)
    print("  已保存: demo_reg_result.png")
    plt.show()


def demo_concepts() -> None:
    """概念可视化"""
    print("\n" + "=" * 60)
    print("  演示 3: 核心概念可视化")
    print("=" * 60)

    # --- Dice 系数 ---
    print("\n[1] Dice 系数示例...")
    from utils.metrics import dice_coefficient

    s = 100
    gt = np.zeros((s, s), dtype=int)
    gt[30:70, 30:70] = 1  # 方形真值

    # 完美预测
    perfect = gt.copy()
    # 偏移预测
    shifted = np.zeros_like(gt)
    shifted[40:80, 40:80] = 1
    # 过小预测
    small = np.zeros_like(gt)
    small[40:60, 40:60] = 1

    d_perfect = dice_coefficient(perfect, gt)["class_1"]
    d_shifted = dice_coefficient(shifted, gt)["class_1"]
    d_small = dice_coefficient(small, gt)["class_1"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(gt, cmap="Blues")
    axes[0].set_title("真值")
    axes[1].imshow(perfect, cmap="Reds", alpha=0.5)
    axes[1].imshow(gt, cmap="Blues", alpha=0.3)
    axes[1].set_title(f"完美  Dice={d_perfect:.3f}")
    axes[2].imshow(shifted, cmap="Reds", alpha=0.5)
    axes[2].imshow(gt, cmap="Blues", alpha=0.3)
    axes[2].set_title(f"偏移  Dice={d_shifted:.3f}")
    axes[3].imshow(small, cmap="Reds", alpha=0.5)
    axes[3].imshow(gt, cmap="Blues", alpha=0.3)
    axes[3].set_title(f"过小  Dice={d_small:.3f}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Dice 系数: 衡量预测(红) vs 真值(蓝) 的重叠度", fontsize=14)
    plt.tight_layout()
    plt.savefig("demo_dice_concept.png", dpi=150)
    print("  已保存: demo_dice_concept.png")
    plt.show()

    print("\n全部演示完成!")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════╗")
    print("║  深度学习医学影像 - 分割与配准快速演示  ║")
    print("╚══════════════════════════════════════════╝\n")

    demo_segmentation()
    demo_registration()
    demo_concepts()
