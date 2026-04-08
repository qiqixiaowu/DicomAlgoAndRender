"""
分割模型训练脚本

使用方法
========
  python train_segmentation.py
  python train_segmentation.py --config configs/seg_config.yaml
  python train_segmentation.py --epochs 100 --batch_size 16 --lr 0.001

训练流程
========
1. 生成/加载合成数据
2. 构建 U-Net 模型
3. 前向传播 → 计算 Loss → 反向传播 → 更新参数
4. 每个 epoch 在验证集上评估
5. 保存最佳模型

训练循环详解 (每个 batch)
========================
  images, masks = next(dataloader)       # 取一批数据
  outputs = model(images)                # 前向传播
  loss = criterion(outputs, masks)       # 计算损失
  optimizer.zero_grad()                  # 清零梯度
  loss.backward()                        # 反向传播
  optimizer.step()                       # 更新参数
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# 同级包导入
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import UNet2D
from models.losses import DiceLoss, DiceCELoss
from data.dataset import SegmentationDataset
from data.transforms import SegTransform
from utils.metrics import dice_coefficient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="医学图像分割训练")
    parser.add_argument("--config", type=str, default="configs/seg_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    config_path = Path(__file__).resolve().parent / path
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_interval: int = 10,
) -> tuple[float, float]:
    """
    训练一个 epoch

    返回: (平均 loss, 平均 dice)
    """
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    for batch_idx, (images, masks) in enumerate(loader):
        # 数据移到 GPU/CPU
        images = images.to(device)
        masks = masks.to(device)

        # 前向传播
        outputs = model(images)           # (B, C, H, W)
        loss = criterion(outputs, masks)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        n_batches += 1

        # 计算 Dice (评估用)
        with torch.no_grad():
            pred_labels = outputs.argmax(dim=1).cpu().numpy()
            gt_labels = masks.cpu().numpy()
            batch_dice = 0.0
            for i in range(len(pred_labels)):
                d = dice_coefficient(pred_labels[i], gt_labels[i])
                batch_dice += d["mean"]
            total_dice += batch_dice / len(pred_labels)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f}")

    return total_loss / n_batches, total_dice / n_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    验证

    返回: (平均 loss, 平均 dice)
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        total_loss += loss.item()
        n_batches += 1

        pred_labels = outputs.argmax(dim=1).cpu().numpy()
        gt_labels = masks.cpu().numpy()
        batch_dice = 0.0
        for i in range(len(pred_labels)):
            d = dice_coefficient(pred_labels[i], gt_labels[i])
            batch_dice += d["mean"]
        total_dice += batch_dice / len(pred_labels)

    return total_loss / n_batches, total_dice / n_batches


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # 合并配置 (命令行覆盖 yaml)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    log_cfg = cfg.get("logging", {})

    epochs = args.epochs or train_cfg.get("epochs", 50)
    batch_size = args.batch_size or train_cfg.get("batch_size", 8)
    lr = args.lr or train_cfg.get("learning_rate", 0.001)
    image_size = data_cfg.get("image_size", 256)
    num_classes = data_cfg.get("num_classes", 2)
    save_dir = Path(__file__).resolve().parent / log_cfg.get("save_dir", "checkpoints/seg")
    log_interval = log_cfg.get("log_interval", 10)

    device = get_device(args.device)
    print(f"设备: {device}")

    # ---- 数据集 ----
    print("生成合成训练数据...")
    train_ds = SegmentationDataset(
        num_samples=data_cfg.get("synthetic_train_size", 800),
        image_size=image_size,
        transform=SegTransform(p=0.5),
    )
    val_ds = SegmentationDataset(
        num_samples=data_cfg.get("synthetic_val_size", 200),
        image_size=image_size,
        transform=None,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=min(data_cfg.get("num_workers", 4), 0 if device.type == "cpu" else 4),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0,
    )
    print(f"训练集: {len(train_ds)} | 验证集: {len(val_ds)}")

    # ---- 模型 ----
    features = model_cfg.get("features", [64, 128, 256, 512])
    model = UNet2D(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=num_classes,
        features=features,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # ---- 损失函数 ----
    loss_name = train_cfg.get("loss", "DiceCE")
    if loss_name == "Dice":
        criterion = DiceLoss()
    elif loss_name == "CE":
        criterion = nn.CrossEntropyLoss()
    elif loss_name == "DiceCE":
        criterion = DiceCELoss()
    else:
        criterion = DiceCELoss()
    print(f"损失函数: {loss_name}")

    # ---- 优化器 ----
    opt_name = train_cfg.get("optimizer", "Adam")
    wd = train_cfg.get("weight_decay", 0.0001)
    if opt_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    # ---- 学习率调度器 ----
    sched_name = train_cfg.get("scheduler", "CosineAnnealing")
    if sched_name == "CosineAnnealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    else:
        scheduler = None

    # ---- 训练循环 ----
    save_dir.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0
    train_losses, val_losses = [], []
    train_dices, val_dices = [], []

    print(f"\n{'='*60}")
    print(f"开始训练  epochs={epochs}  batch_size={batch_size}  lr={lr}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # 训练
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device, log_interval
        )

        # 验证
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # 学习率调度
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f} | "
            f"LR: {current_lr:.6f} | {elapsed:.1f}s"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)

        # 保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "val_loss": val_loss,
            }, save_dir / "best_model.pth")
            print(f"  ★ 保存最佳模型  Val Dice: {val_dice:.4f}")

    # 保存训练曲线
    np.savez(
        save_dir / "training_history.npz",
        train_losses=train_losses, val_losses=val_losses,
        train_dices=train_dices, val_dices=val_dices,
    )

    print(f"\n训练完成! 最佳 Val Dice: {best_dice:.4f}")
    print(f"模型保存在: {save_dir}")


if __name__ == "__main__":
    main()
