"""
血管分割模型训练脚本 (3D U-Net)

使用方法
========
  # 最简单：自动加载 E:\\traindata
  python train_vessel_seg.py

  # 指定数据路径
  python train_vessel_seg.py --data_root E:\\traindata

  # 自定义参数
  python train_vessel_seg.py --data_root E:\\traindata --epochs 100 --patch_size 64 128 128

  # 轻量模式（显存不够时）
  python train_vessel_seg.py --features 16 32 64 --patch_size 32 64 64

训练数据目录结构（支持以下任一格式）
====================================
  E:\\traindata\\
  ├── images\\
  │   ├── case001\\   ← DICOM 序列
  │   └── case002\\
  └── labels\\
      ├── case001.nii.gz   ← 二值掩码 (0=背景, 1=血管)
      └── case002.nii.gz

  或:
  E:\\traindata\\
  └── case001\\
      ├── image.nii.gz
      └── label.nii.gz

训练完成后的输出
================
  checkpoints/vessel_seg/
  ├── best_model.pth          ← 最佳模型权重
  ├── last_model.pth          ← 最后一个 epoch 的权重
  └── train_log.txt           ← 训练日志
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.unet3d import UNet3D
from models.losses import DiceCELoss
from utils.metrics import dice_coefficient
from data.vessel_dataset import VesselDataset3D


# ============================================================
#  参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="血管分割 3D U-Net 训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root", type=str, default=r"E:\traindata",
                        help="训练数据根目录")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="批大小（3D 数据占显存大，通常设 1~4）")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="初始学习率")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 128, 128],
                        metavar=("D", "H", "W"),
                        help="训练 patch 尺寸 (深度 高 宽)")
    parser.add_argument("--samples_per_volume", type=int, default=50,
                        help="每个 case 每 epoch 采样的 patch 数")
    parser.add_argument("--features", type=int, nargs="+",
                        default=[16, 32, 64, 128],
                        help="3D U-Net 各层通道数（轻量: 16 32 64，标准: 32 64 128 256）")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="验证集比例 (0~1)")
    parser.add_argument("--save_dir", type=str,
                        default="checkpoints/vessel_seg",
                        help="模型保存目录")
    parser.add_argument("--device", type=str, default=None,
                        help="训练设备 (cuda/cpu), 默认自动检测")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader 工作进程数（Windows 推荐 0）")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="使用自动混合精度 (AMP) 加速训练")
    parser.add_argument("--resume", type=str, default=None,
                        help="从检查点恢复训练 (checkpoint 路径)")
    parser.add_argument("--max_volumes", type=int, default=None,
                        help="最多使用多少个 case（调试用）")
    return parser.parse_args()


# ============================================================
#  训练循环
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
    log_interval: int = 5,
) -> tuple[float, float]:
    model.train()
    total_loss = total_dice = n = 0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            dice = dice_coefficient(
                preds.cpu().numpy(),
                masks.cpu().numpy(),
                num_classes=2,
            )["mean"]

        total_loss += loss.item()
        total_dice += dice
        n += 1

        if (batch_idx + 1) % log_interval == 0:
            print(f"    batch {batch_idx+1}/{len(loader)} | "
                  f"loss={loss.item():.4f} | dice={dice:.4f}", flush=True)

    return total_loss / max(n, 1), total_dice / max(n, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    scaler,
) -> tuple[float, float]:
    model.eval()
    total_loss = total_dice = n = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks)
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)

        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)
        dice = dice_coefficient(
            preds.cpu().numpy(),
            masks.cpu().numpy(),
            num_classes=2,
        )["mean"]
        total_loss += loss.item()
        total_dice += dice
        n += 1

    return total_loss / max(n, 1), total_dice / max(n, 1)


# ============================================================
#  主程序
# ============================================================

def main():
    args = parse_args()

    # 设备
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 打印设备信息
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024**3
        print(f"训练设备: GPU — {props.name}  ({vram_gb:.1f} GB VRAM)")
    else:
        print("训练设备: CPU（未检测到 CUDA，训练较慢）")
        print("  → 如需 GPU 训练，请运行:")
        print("  .venv\\Scripts\\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

    # AMP Scaler
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("已启用自动混合精度 (AMP) — 节省显存约 40%)")

    # 保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- 数据集 ----
    print(f"\n加载训练数据: {args.data_root}")
    patch_size = tuple(args.patch_size)

    full_dataset = VesselDataset3D(
        traindata_root=args.data_root,
        patch_size=patch_size,
        samples_per_volume=args.samples_per_volume,
        pos_fraction=0.6,
        augment=True,
        max_volumes=args.max_volumes,
    )

    # 按比例划分验证集
    total = len(full_dataset)
    val_size = max(1, int(total * args.val_split))
    train_size = total - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    # 验证集不需要增强
    val_ds.dataset.augment = False

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    print(f"训练集: {train_size} patches, 验证集: {val_size} patches")

    # ---- 模型 ----
    model = UNet3D(
        in_channels=1,
        out_channels=2,       # 0=背景, 1=血管
        features=args.features,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params/1e6:.2f} M")

    # ---- 损失函数 ----
    # 血管体素少（前景少），增大 Dice loss 权重
    criterion = DiceCELoss(dice_weight=0.7, ce_weight=0.3)

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    start_epoch = 0
    best_dice = 0.0

    # ---- 恢复训练 ----
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt.get("best_dice", 0.0)
        print(f"从 epoch {start_epoch} 恢复训练，历史最佳 Dice={best_dice:.4f}")

    # ---- 保存模型元数据 ----
    meta = {
        "model": "UNet3D",
        "in_channels": 1,
        "out_channels": 2,
        "features": args.features,
        "patch_size": list(patch_size),
        "classes": ["background", "vessel"],
        "hu_min": -150.0,
        "hu_max": 550.0,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(save_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---- 日志文件 ----
    log_path = save_dir / "train_log.txt"
    log_f = open(log_path, "a", encoding="utf-8")
    log_f.write(f"\n{'='*60}\n训练开始: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.write(f"数据路径: {args.data_root}\nPatch: {patch_size}\n")
    log_f.write(f"Epochs: {args.epochs}, BS: {args.batch_size}, LR: {args.lr}\n{'='*60}\n")

    print(f"\n{'='*60}")
    print(f"开始训练: {args.epochs} 个 epoch")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        print(f"[Epoch {epoch+1}/{args.epochs}]")

        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_dice = validate(
            model, val_loader, criterion, device, scaler
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        gpu_info = ""
        if device.type == "cuda":
            used  = torch.cuda.memory_allocated(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_info = f"  |  GPU {used:.1f}/{total:.1f}GB"
        print(f"  Train: loss={train_loss:.4f}, dice={train_dice:.4f}  |  "
              f"Val: loss={val_loss:.4f}, dice={val_dice:.4f}  |  "
              f"LR={lr_now:.2e}  |  {elapsed:.1f}s{gpu_info}")

        log_line = (f"epoch={epoch+1:03d} train_loss={train_loss:.4f} "
                    f"train_dice={train_dice:.4f} val_loss={val_loss:.4f} "
                    f"val_dice={val_dice:.4f} lr={lr_now:.2e} time={elapsed:.1f}s\n")
        log_f.write(log_line)
        log_f.flush()

        # 保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_dice": best_dice,
                "meta": meta,
            }
            torch.save(ckpt, save_dir / "best_model.pth")
            print(f"  ✓ 保存最佳模型 (val_dice={val_dice:.4f})")

        # 每 10 个 epoch 保存一次
        if (epoch + 1) % 10 == 0:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_dice": best_dice,
                "meta": meta,
            }
            torch.save(ckpt, save_dir / "last_model.pth")

    log_f.write(f"训练结束: {time.strftime('%Y-%m-%d %H:%M:%S')}, 最佳 Dice={best_dice:.4f}\n")
    log_f.close()

    print(f"\n训练完成！最佳验证 Dice: {best_dice:.4f}")
    print(f"模型保存在: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
