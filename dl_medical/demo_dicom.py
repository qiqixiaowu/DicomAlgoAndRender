"""
DICOM 数据探索与训练演示

使用你的真实 DICOM 数据 (E:\\Data) 来:
1. 扫描目录, 查看有哪些数据
2. 读取并可视化 DICOM 序列
3. 用真实数据训练分割模型 (U-Net + 伪标签)
4. 用真实数据训练配准模型 (VoxelMorph + 相邻切片对)

使用方法
========
  python demo_dicom.py                       # 全部演示
  python demo_dicom.py --mode scan           # 仅扫描目录
  python demo_dicom.py --mode view           # 可视化数据
  python demo_dicom.py --mode train_seg      # 分割训练
  python demo_dicom.py --mode train_reg      # 配准训练
  python demo_dicom.py --data_root E:\\Data   # 指定数据路径
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# 配置中文字体, 防止中文标题乱码 (Windows)
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dicom_loader import (
    find_dicom_series, read_dicom_series,
    normalize_volume, extract_2d_slices, apply_window,
)

# torch 和模型相关: 仅在训练模式时导入, 避免 scan/view 模式因 torch 未安装而崩溃
def _import_torch():
    try:
        import torch
        return torch
    except ImportError:
        print("错误: 训练功能需要安装 PyTorch")
        print("  pip install torch torchvision")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DICOM 数据探索与训练")
    parser.add_argument("--data_root", type=str, default=r"E:\Data",
                        help="DICOM 数据根目录")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "scan", "view", "train_seg", "train_reg"],
                        help="运行模式")
    parser.add_argument("--series", type=str, default=None,
                        help="指定某个序列目录名 (相对于 data_root)")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


# ============================================================
#  模式 1: 扫描目录
# ============================================================

def mode_scan(args: argparse.Namespace) -> list[str]:
    """扫描并列出所有 DICOM 序列"""
    print("=" * 70)
    print(f"  扫描 DICOM 数据: {args.data_root}")
    print("=" * 70)

    dirs = find_dicom_series(args.data_root, max_depth=3)
    print(f"\n共找到 {len(dirs)} 个 DICOM 序列:\n")

    for i, d in enumerate(dirs):
        rel = Path(d).relative_to(args.data_root) if d.startswith(args.data_root) else Path(d)
        # 快速计数: os.scandir 直接用 DirEntry.is_file(), 避免额外 stat 调用
        try:
            n_files = sum(1 for e in os.scandir(d) if e.is_file(follow_symlinks=False))
        except OSError:
            n_files = -1
        print(f"  [{i+1:3d}] {str(rel):60s} ({n_files} files)")

    return dirs


# ============================================================
#  模式 2: 可视化数据
# ============================================================

def mode_view(args: argparse.Namespace) -> None:
    """读取并可视化 DICOM 数据"""
    print("=" * 70)
    print("  DICOM 数据可视化")
    print("=" * 70)

    # 选择序列
    if args.series:
        series_dir = str(Path(args.data_root) / args.series)
    else:
        dirs = find_dicom_series(args.data_root, max_depth=3)
        if not dirs:
            print("未找到 DICOM 数据!")
            return
        series_dir = dirs[0]
        print(f"自动选择第一个序列: {series_dir}")

    # 读取
    print(f"\n读取 DICOM: {series_dir}")
    volume, info = read_dicom_series(series_dir, return_info=True)

    print(f"\n  模态: {info.modality}")
    print(f"  描述: {info.series_description}")
    print(f"  体积: {volume.shape} (切片×高×宽)")
    print(f"  像素间距: {info.pixel_spacing}")
    print(f"  层厚: {info.slice_thickness}")
    print(f"  值范围: [{volume.min():.1f}, {volume.max():.1f}]")
    print(f"  窗位/窗宽: {info.window_center}/{info.window_width}")

    # ---- 可视化 1: 不同窗宽窗位 ----
    mid = volume.shape[0] // 2
    mid_slice = volume[mid]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 原始值
    axes[0, 0].imshow(mid_slice, cmap="gray")
    axes[0, 0].set_title(f"原始 (切片 {mid})")

    # 默认窗
    wc, ww = info.window_center, info.window_width
    axes[0, 1].imshow(apply_window(mid_slice, wc, ww), cmap="gray")
    axes[0, 1].set_title(f"默认窗 WC={wc:.0f} WW={ww:.0f}")

    # MinMax
    axes[0, 2].imshow(normalize_volume(mid_slice[np.newaxis], "minmax")[0], cmap="gray")
    axes[0, 2].set_title("MinMax 归一化")

    # 不同预设窗
    presets = [
        ("软组织", 40, 400),
        ("骨窗", 300, 1500),
        ("肺窗", -600, 1500),
    ]
    for j, (name, wc_, ww_) in enumerate(presets):
        axes[1, j].imshow(apply_window(mid_slice, wc_, ww_), cmap="gray")
        axes[1, j].set_title(f"{name} WC={wc_} WW={ww_}")

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(f"{info.modality} - {info.series_description}", fontsize=14)
    plt.tight_layout()
    plt.savefig("dicom_windows.png", dpi=150)
    print("\n  已保存: dicom_windows.png")
    plt.show()

    # ---- 可视化 2: 三个方向切片 ----
    norm_vol = normalize_volume(volume, "ct_soft" if info.modality == "CT" else "minmax")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    d, h, w = norm_vol.shape
    axes[0].imshow(norm_vol[d // 2], cmap="gray")
    axes[0].set_title(f"Axial (横断面, z={d//2})")

    axes[1].imshow(norm_vol[:, h // 2, :], cmap="gray", aspect=info.slice_thickness / info.pixel_spacing[0])
    axes[1].set_title(f"Coronal (冠状面, y={h//2})")

    axes[2].imshow(norm_vol[:, :, w // 2], cmap="gray", aspect=info.slice_thickness / info.pixel_spacing[1])
    axes[2].set_title(f"Sagittal (矢状面, x={w//2})")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("dicom_3views.png", dpi=150)
    print("  已保存: dicom_3views.png")
    plt.show()

    # ---- 可视化 3: 切片浏览 ----
    slices = extract_2d_slices(norm_vol, axis=0, target_size=None, skip_empty=True)
    n_show = min(16, len(slices))
    indices = np.linspace(0, len(slices) - 1, n_show, dtype=int)

    rows = int(np.ceil(n_show / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    axes = axes.flat if rows > 1 else axes
    for j, idx in enumerate(indices):
        axes[j].imshow(slices[idx], cmap="gray")
        axes[j].set_title(f"Slice {idx}")
        axes[j].axis("off")
    # 隐藏多余
    for j in range(n_show, len(list(axes)) if rows > 1 else 4):
        axes[j].axis("off")

    fig.suptitle("Axial 切片浏览 (均匀采样)", fontsize=14)
    plt.tight_layout()
    plt.savefig("dicom_slices.png", dpi=150)
    print("  已保存: dicom_slices.png")
    plt.show()


# ============================================================
#  模式 3: 用真实数据训练分割
# ============================================================

def mode_train_seg(args: argparse.Namespace) -> None:
    """用 DICOM 数据 + 伪标签训练 U-Net"""
    print("=" * 70)
    print("  用真实 DICOM 数据训练分割模型 (U-Net + Otsu 伪标签)")
    print("=" * 70)

    torch = _import_torch()
    from models.unet2d import UNet2D
    from models.losses import DiceCELoss
    from utils.metrics import dice_coefficient
    from torch.utils.data import DataLoader, random_split
    from data.dicom_dataset import DicomSliceDataset

    # 选择数据
    if args.series:
        dicom_path = str(Path(args.data_root) / args.series)
    else:
        dicom_path = args.data_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ---- 加载数据 ----
    full_ds = DicomSliceDataset(
        dicom_dirs=dicom_path,
        image_size=args.image_size,
        normalize="ct_soft",
        with_pseudo_label=True,
        transform=None,
        max_slices_per_volume=80,
    )

    if len(full_ds) < 10:
        print("数据量太少, 请指定更多序列或降低 empty_threshold")
        return

    # 划分训练/验证
    n_val = max(int(len(full_ds) * 0.2), 1)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    # 给训练集加数据增强
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"训练: {n_train} | 验证: {n_val}")

    # ---- 模型 ----
    model = UNet2D(in_channels=1, out_channels=2, features=[32, 64, 128, 256]).to(device)
    criterion = DiceCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n开始训练 ({args.epochs} epochs)...\n")

    train_losses, val_dices = [], []

    for epoch in range(1, args.epochs + 1):
        # 训练
        model.train()
        epoch_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)

        # 验证
        model.eval()
        val_dice = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                pred = model(images).argmax(dim=1).cpu().numpy()
                gt = masks.cpu().numpy()
                for i in range(len(pred)):
                    d = dice_coefficient(pred[i], gt[i])
                    val_dice += d["mean"]
                n_val_batches += len(pred)
        val_dice /= max(n_val_batches, 1)

        scheduler.step()
        train_losses.append(epoch_loss)
        val_dices.append(val_dice)

        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {epoch_loss:.4f} | Val Dice: {val_dice:.4f}")

    # ---- 可视化结果 ----
    print("\n可视化分割结果...")
    model.eval()
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    with torch.no_grad():
        for i in range(4):
            idx = np.random.randint(len(val_ds))
            img, mask = val_ds[idx]
            pred = model(img.unsqueeze(0).to(device)).argmax(dim=1).cpu().numpy()[0]
            img_np = img.squeeze().numpy()
            mask_np = mask.numpy()

            axes[0, i].imshow(img_np, cmap="gray")
            axes[0, i].set_title("DICOM 切片")
            axes[0, i].axis("off")

            axes[1, i].imshow(mask_np, cmap="jet", interpolation="nearest")
            axes[1, i].set_title("Otsu 伪标签")
            axes[1, i].axis("off")

            axes[2, i].imshow(pred, cmap="jet", interpolation="nearest")
            axes[2, i].set_title("U-Net 预测")
            axes[2, i].axis("off")

    fig.suptitle("真实 DICOM 数据分割: 图像 / 伪标签 / 预测", fontsize=14)
    plt.tight_layout()
    plt.savefig("dicom_seg_result.png", dpi=150)
    print("  已保存: dicom_seg_result.png")
    plt.show()

    # 训练曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(train_losses)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("训练损失")
    ax1.grid(True, alpha=0.3)
    ax2.plot(val_dices)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice")
    ax2.set_title("验证 Dice")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("dicom_seg_curves.png", dpi=150)
    print("  已保存: dicom_seg_curves.png")
    plt.show()


# ============================================================
#  模式 4: 用真实数据训练配准
# ============================================================

def mode_train_reg(args: argparse.Namespace) -> None:
    """用 DICOM 数据的相邻切片对训练 VoxelMorph"""
    print("=" * 70)
    print("  用真实 DICOM 数据训练配准模型 (VoxelMorph)")
    print("=" * 70)

    torch = _import_torch()
    from models.voxelmorph import VoxelMorph2D
    from models.losses import NCCLoss, GradientLoss
    from torch.utils.data import DataLoader, random_split
    from data.dicom_dataset import DicomPairDataset

    if args.series:
        dicom_path = str(Path(args.data_root) / args.series)
    else:
        dicom_path = args.data_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = min(args.image_size, 128)  # 配准用较小尺寸
    print(f"设备: {device} | 图像尺寸: {img_size}")

    # ---- 数据 ----
    full_ds = DicomPairDataset(
        dicom_dirs=dicom_path,
        image_size=img_size,
        normalize="ct_soft",
        slice_gap=3,
        max_slices_per_volume=60,
    )

    if len(full_ds) < 10:
        print("配准对太少, 请指定更多序列")
        return

    n_val = max(int(len(full_ds) * 0.2), 1)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"训练: {n_train} 对 | 验证: {n_val} 对")

    # ---- 模型 ----
    model = VoxelMorph2D(
        image_size=(img_size, img_size),
        enc_features=[16, 32, 32, 32],
        dec_features=[32, 32, 32, 16],
    ).to(device)

    sim_loss_fn = NCCLoss(window_size=9)
    smooth_loss_fn = GradientLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n开始训练 ({args.epochs} epochs)...\n")

    train_losses = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for fixed, moving in train_loader:
            fixed, moving = fixed.to(device), moving.to(device)
            warped, flow = model(fixed, moving)

            sim_loss = sim_loss_fn(fixed, warped)
            smooth_loss = smooth_loss_fn(flow)
            loss = sim_loss + 0.01 * smooth_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {epoch_loss:.4f}")

    # ---- 可视化 ----
    print("\n可视化配准结果...")
    model.eval()
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    titles = ["Fixed", "Moving", "Warped", "|Fixed-Warped|"]

    with torch.no_grad():
        for i in range(4):
            idx = np.random.randint(len(val_ds))
            fixed, moving = val_ds[idx]
            warped, flow = model(
                fixed.unsqueeze(0).to(device),
                moving.unsqueeze(0).to(device),
            )
            f_np = fixed.squeeze().numpy()
            m_np = moving.squeeze().numpy()
            w_np = warped.squeeze().cpu().numpy()

            axes[i, 0].imshow(f_np, cmap="gray")
            axes[i, 0].set_title(titles[0] if i == 0 else "")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(m_np, cmap="gray")
            axes[i, 1].set_title(titles[1] if i == 0 else "")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(w_np, cmap="gray")
            axes[i, 2].set_title(titles[2] if i == 0 else "")
            axes[i, 2].axis("off")

            diff = np.abs(f_np - w_np)
            axes[i, 3].imshow(diff, cmap="hot")
            axes[i, 3].set_title(titles[3] if i == 0 else "")
            axes[i, 3].axis("off")

    fig.suptitle("真实 DICOM 配准: Fixed / Moving / Warped / 差异", fontsize=14)
    plt.tight_layout()
    plt.savefig("dicom_reg_result.png", dpi=150)
    print("  已保存: dicom_reg_result.png")
    plt.show()


# ============================================================
#  主入口
# ============================================================

def main() -> None:
    args = parse_args()

    print("╔══════════════════════════════════════════════╗")
    print("║  DICOM 数据探索 & 深度学习训练演示          ║")
    print(f"║  数据路径: {args.data_root:<35s}║")
    print("╚══════════════════════════════════════════════╝\n")

    if args.mode in ("all", "scan"):
        mode_scan(args)
    if args.mode in ("all", "view"):
        mode_view(args)
    if args.mode in ("all", "train_seg"):
        mode_train_seg(args)
    if args.mode in ("all", "train_reg"):
        mode_train_reg(args)


if __name__ == "__main__":
    main()
