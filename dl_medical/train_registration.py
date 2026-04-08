"""
配准模型训练脚本

使用方法
========
  python train_registration.py
  python train_registration.py --config configs/reg_config.yaml
  python train_registration.py --epochs 100 --lr 0.0001

配准训练的特殊之处
==================
1. **无需标注**: 损失 = 图像相似性 + 形变场正则化
2. **输出是连续场**: 不是分类, 而是回归形变场
3. **两个损失项需要平衡**:
   - 相似性损失太大 → 变形过度, 出现折叠
   - 正则化太大 → 变形不足, 配准不准
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import VoxelMorph2D
from models.losses import NCCLoss, GradientLoss
from data.dataset import RegistrationDataset
from data.transforms import RegTransform
from utils.metrics import folding_ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="医学图像配准训练")
    parser.add_argument("--config", type=str, default="configs/reg_config.yaml")
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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    sim_loss_fn: nn.Module,
    smooth_loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    sim_weight: float,
    smooth_weight: float,
    log_interval: int = 10,
) -> tuple[float, float, float]:
    """
    训练一个 epoch

    返回: (总 loss, 相似性 loss, 平滑 loss)
    """
    model.train()
    total_loss = 0.0
    total_sim = 0.0
    total_smooth = 0.0
    n_batches = 0

    for batch_idx, (fixed, moving) in enumerate(loader):
        fixed = fixed.to(device)
        moving = moving.to(device)

        # 前向传播
        warped, flow = model(fixed, moving)

        # 相似性损失: 配准后图像 vs 固定图像
        sim_loss = sim_loss_fn(fixed, warped)
        # 平滑正则化: 形变场应该平滑
        smooth_loss = smooth_loss_fn(flow)

        # 总损失 = 加权组合
        loss = sim_weight * sim_loss + smooth_weight * smooth_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_sim += sim_loss.item()
        total_smooth += smooth_loss.item()
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} "
                  f"(Sim: {sim_loss.item():.4f}, Smooth: {smooth_loss.item():.4f})")

    return (
        total_loss / n_batches,
        total_sim / n_batches,
        total_smooth / n_batches,
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    sim_loss_fn: nn.Module,
    smooth_loss_fn: nn.Module,
    device: torch.device,
    sim_weight: float,
    smooth_weight: float,
) -> tuple[float, float, float, float]:
    """
    验证

    返回: (总 loss, sim_loss, smooth_loss, 平均折叠率)
    """
    model.eval()
    total_loss = 0.0
    total_sim = 0.0
    total_smooth = 0.0
    total_folding = 0.0
    n_batches = 0

    for fixed, moving in loader:
        fixed = fixed.to(device)
        moving = moving.to(device)

        warped, flow = model(fixed, moving)

        sim_loss = sim_loss_fn(fixed, warped)
        smooth_loss = smooth_loss_fn(flow)
        loss = sim_weight * sim_loss + smooth_weight * smooth_loss

        total_loss += loss.item()
        total_sim += sim_loss.item()
        total_smooth += smooth_loss.item()
        n_batches += 1

        # 计算折叠率
        flow_np = flow.cpu().numpy()
        for i in range(flow_np.shape[0]):
            total_folding += folding_ratio(flow_np[i])

    total_samples = n_batches * loader.batch_size if loader.batch_size else n_batches
    return (
        total_loss / n_batches,
        total_sim / n_batches,
        total_smooth / n_batches,
        total_folding / max(total_samples, 1),
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    log_cfg = cfg.get("logging", {})

    epochs = args.epochs or train_cfg.get("epochs", 80)
    batch_size = args.batch_size or train_cfg.get("batch_size", 8)
    lr = args.lr or train_cfg.get("learning_rate", 0.0001)
    image_size = data_cfg.get("image_size", 128)
    sim_weight = train_cfg.get("sim_weight", 1.0)
    smooth_weight = train_cfg.get("smooth_weight", 0.01)
    save_dir = Path(__file__).resolve().parent / log_cfg.get("save_dir", "checkpoints/reg")
    log_interval = log_cfg.get("log_interval", 10)

    device = get_device(args.device)
    print(f"设备: {device}")

    # ---- 数据 ----
    print("生成合成配准数据...")
    train_ds = RegistrationDataset(
        num_samples=data_cfg.get("synthetic_train_size", 600),
        image_size=image_size,
        transform=RegTransform(p=0.3),
    )
    val_ds = RegistrationDataset(
        num_samples=data_cfg.get("synthetic_val_size", 150),
        image_size=image_size,
        transform=None,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    print(f"训练集: {len(train_ds)} | 验证集: {len(val_ds)}")

    # ---- 模型 ----
    model = VoxelMorph2D(
        image_size=(image_size, image_size),
        enc_features=model_cfg.get("enc_features", [16, 32, 32, 32]),
        dec_features=model_cfg.get("dec_features", [32, 32, 32, 16]),
    ).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 损失 ----
    sim_name = train_cfg.get("sim_loss", "NCC")
    if sim_name == "NCC":
        sim_loss_fn = NCCLoss(window_size=9)
    else:
        sim_loss_fn = nn.MSELoss()
    smooth_loss_fn = GradientLoss(penalty="l2")
    print(f"相似性损失: {sim_name} (权重: {sim_weight})")
    print(f"平滑正则化权重: {smooth_weight}")

    # ---- 优化器 ----
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ---- 训练 ----
    save_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_folding": []}

    print(f"\n{'='*60}")
    print(f"开始配准训练  epochs={epochs}  batch={batch_size}  lr={lr}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_sim, train_smooth = train_one_epoch(
            model, train_loader, sim_loss_fn, smooth_loss_fn,
            optimizer, device, sim_weight, smooth_weight, log_interval,
        )

        val_loss, val_sim, val_smooth, val_fold = validate(
            model, val_loader, sim_loss_fn, smooth_loss_fn,
            device, sim_weight, smooth_weight,
        )

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train: {train_loss:.4f} (sim:{train_sim:.4f} smooth:{train_smooth:.4f}) | "
            f"Val: {val_loss:.4f} fold:{val_fold:.4f} | "
            f"{elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_folding"].append(val_fold)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            }, save_dir / "best_model.pth")
            print(f"  ★ 保存最佳模型  Val Loss: {val_loss:.4f}")

    np.savez(save_dir / "training_history.npz", **history)
    print(f"\n训练完成! 最佳 Val Loss: {best_loss:.4f}")
    print(f"模型保存在: {save_dir}")


if __name__ == "__main__":
    main()
