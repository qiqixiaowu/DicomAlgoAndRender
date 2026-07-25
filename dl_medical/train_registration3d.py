"""
3D 医学图像配准训练脚本

使用方法
========
  # 默认配置 (合成肺部数据)
  python train_registration3d.py

  # 指定配置
  python train_registration3d.py --config configs/reg_config3d.yaml

  # 使用肝部预设
  python train_registration3d.py --preset liver

  # 命令行覆盖
  python train_registration3d.py --epochs 300 --batch_size 4 --lr 1e-4

特性
====
1. 支持 VoxelMorph3D (位移场) 和 VoxelMorph3DDiff (微分同胚)
2. 混合精度训练 (AMP) 节省显存
3. 梯度裁剪防止爆炸
4. 多分辨率金字塔训练
5. 完整指标评估 (SSIM, PSNR, Jacobian, Dice)
6. TensorBoard 日志
7. 学习率预热 + 余弦退火
8. 断点续训
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import (
    VoxelMorph3D,
    VoxelMorph3DDiff,
    NCCLoss3D,
    MSELoss3D,
    LNCCLoss3D,
    GradientLoss3D,
    BendingEnergyLoss3D,
    JacobianLoss,
    RegistrationLoss3D,
)
from data.dataset3d import build_dataset_3d
from utils.metrics3d import (
    evaluate_registration_3d,
    jacobian_stats_3d,
    folding_ratio_3d,
    ssim_3d,
    psnr_3d,
)


# ============================================================
#  参数解析 & 配置加载
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D 医学图像配准训练")
    parser.add_argument("--config", type=str, default="configs/reg_config3d.yaml")
    parser.add_argument("--preset", type=str, default=None,
                        choices=["lung", "liver"],
                        help="使用预设配置 (覆盖 config 中对应部分)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="断点续训的 checkpoint 路径")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="覆盖配置中的数据目录 (NIfTI 模式)")
    return parser.parse_args()


def load_config(path: str) -> dict:
    config_path = Path(__file__).resolve().parent / path
    if not config_path.exists():
        print(f"⚠ 配置文件不存在: {config_path}, 使用默认配置")
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_preset(cfg: dict, preset: str) -> dict:
    """应用预设配置 (lung / liver)"""
    preset_key = f"{preset}_preset"
    if preset_key not in cfg:
        print(f"⚠ 未找到预设: {preset_key}")
        return cfg

    preset_cfg = cfg[preset_key]
    for section in ["data", "loss"]:
        if section in preset_cfg:
            cfg.setdefault(section, {}).update(preset_cfg[section])
    print(f"✓ 应用预设: {preset}")
    return cfg


def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
#  模型构建
# ============================================================

def build_model(cfg: dict, device: torch.device) -> nn.Module:
    """构建 3D 配准模型"""
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name", "VoxelMorph3DDiff")
    enc_features = model_cfg.get("enc_features", [16, 32, 64, 64])
    dec_features = model_cfg.get("dec_features", [64, 64, 32, 16])
    volume_size = tuple(cfg.get("data", {}).get("volume_size", [64, 64, 64]))

    if name == "VoxelMorph3D":
        model = VoxelMorph3D(
            image_size=volume_size,
            enc_features=enc_features,
            dec_features=dec_features,
        )
    elif name == "VoxelMorph3DDiff":
        vecint_steps = model_cfg.get("vec_int_steps", 7)
        model = VoxelMorph3DDiff(
            image_size=volume_size,
            enc_features=enc_features,
            dec_features=dec_features,
            vecint_steps=vecint_steps,
        )
    else:
        raise ValueError(f"未知模型: {name}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型: {name}  参数量: {n_params:,}")
    return model


# ============================================================
#  损失构建
# ============================================================

def build_loss(cfg: dict, device: torch.device) -> RegistrationLoss3D:
    """构建 3D 配准组合损失"""
    loss_cfg = cfg.get("loss", {})
    sim_name = loss_cfg.get("sim_loss", "LNCC")

    if sim_name == "NCC":
        sim_loss = NCCLoss3D(window_size=loss_cfg.get("ncc_window", 9))
    elif sim_name == "MSE":
        sim_loss = MSELoss3D()
    elif sim_name == "LNCC":
        sim_loss = LNCCLoss3D(window_sizes=tuple(loss_cfg.get("lncc_windows", [5, 9])))
    else:
        raise ValueError(f"未知相似性损失: {sim_name}")

    smooth_name = loss_cfg.get("smooth_loss", "bending")
    if smooth_name == "gradient":
        smooth_loss = GradientLoss3D(penalty="l2")
    elif smooth_name == "bending":
        smooth_loss = BendingEnergyLoss3D()
    else:
        raise ValueError(f"未知正则化损失: {smooth_name}")

    reg_loss = RegistrationLoss3D(
        sim_loss=sim_loss,
        smooth_loss=smooth_loss,
        sim_weight=loss_cfg.get("sim_weight", 1.0),
        smooth_weight=loss_cfg.get("smooth_weight", 0.5),
        jac_weight=loss_cfg.get("jac_weight", 0.1),
    ).to(device)

    print(f"损失: sim={sim_name}  smooth={smooth_name}  "
          f"w_sim={reg_loss.sim_weight}  w_smooth={reg_loss.smooth_weight}  "
          f"w_jac={reg_loss.jac_weight}")
    return reg_loss


# ============================================================
#  优化器 & 调度器
# ============================================================

def build_optimizer(
    model: nn.Module, cfg: dict
) -> tuple[torch.optim.Optimizer, object]:
    """构建优化器和学习率调度器"""
    train_cfg = cfg.get("training", {})
    lr = train_cfg.get("learning_rate", 1e-4)
    wd = train_cfg.get("weight_decay", 1e-5)
    opt_name = train_cfg.get("optimizer", "Adam")

    if opt_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=wd, momentum=0.9
        )
    else:
        raise ValueError(f"未知优化器: {opt_name}")

    sched_cfg = train_cfg.get("scheduler", "CosineAnnealingLR")
    sched_params = train_cfg.get("scheduler_params", {})
    epochs = train_cfg.get("epochs", 200)

    if sched_cfg == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=sched_params.get("T_max", epochs),
            eta_min=sched_params.get("eta_min", 1e-6),
        )
    elif sched_cfg == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_params.get("step_size", 50),
            gamma=sched_params.get("gamma", 0.5),
        )
    elif sched_cfg == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10,
        )
    else:
        scheduler = None

    return optimizer, scheduler


def warmup_lr(
    optimizer: torch.optim.Optimizer,
    current_step: int,
    warmup_steps: int,
    base_lr: float,
) -> None:
    """线性学习率预热"""
    if warmup_steps <= 0 or current_step >= warmup_steps:
        return
    lr = base_lr * (current_step + 1) / warmup_steps
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ============================================================
#  训练 & 验证
# ============================================================

def forward_model(
    model: nn.Module, fixed: torch.Tensor, moving: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    统一前向传播, 处理 VoxelMorph3D (2返回) 和 VoxelMorph3DDiff (3返回)

    返回: (warped, flow)
    """
    output = model(fixed, moving)
    if len(output) == 3:
        warped, flow, velocity = output
    else:
        warped, flow = output
    return warped, flow


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    reg_loss_fn: RegistrationLoss3D,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    warmup_steps: int,
    base_lr: float,
    grad_clip: float | None = None,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    log_interval: int = 10,
    global_step: int = 0,
) -> tuple[dict, int]:
    """训练一个 epoch"""
    model.train()
    loss_accum = {"sim": 0.0, "smooth": 0.0, "jac": 0.0, "total": 0.0}
    metric_accum = {"ssim": 0.0, "psnr": 0.0, "folding": 0.0}
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        fixed = batch["fixed"].to(device, non_blocking=True)
        moving = batch["moving"].to(device, non_blocking=True)

        # 学习率预热
        warmup_lr(optimizer, global_step, warmup_steps, base_lr)

        # 前向传播
        if use_amp and device.type == "cuda":
            with torch.amp.autocast('cuda'):
                warped, flow = forward_model(model, fixed, moving)
                loss, loss_dict = reg_loss_fn(fixed, warped, flow)
        else:
            warped, flow = forward_model(model, fixed, moving)
            loss, loss_dict = reg_loss_fn(fixed, warped, flow)

        # 反向传播
        optimizer.zero_grad()
        if use_amp and device.type == "cuda" and scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # 累积损失
        for k in loss_accum:
            loss_accum[k] += loss_dict[k]
        n_batches += 1
        global_step += 1

        # 评估指标 (每 log_interval 步)
        if (batch_idx + 1) % log_interval == 0:
            with torch.no_grad():
                metric_accum["ssim"] += ssim_3d(fixed, warped)
                metric_accum["psnr"] += psnr_3d(fixed, warped)
                metric_accum["folding"] += folding_ratio_3d(flow)

            avg_loss = loss_accum["total"] / n_batches
            print(
                f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                f"loss={loss_dict['total']:.4f} "
                f"(sim:{loss_dict['sim']:.4f} "
                f"smooth:{loss_dict['smooth']:.4f} "
                f"jac:{loss_dict['jac']:.4f}) "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    # 平均
    for k in loss_accum:
        loss_accum[k] /= max(n_batches, 1)
    metric_count = max(n_batches // log_interval, 1)
    for k in metric_accum:
        metric_accum[k] /= metric_count

    return loss_accum, metric_accum, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    reg_loss_fn: RegistrationLoss3D,
    device: torch.device,
    use_amp: bool = False,
) -> dict:
    """验证"""
    model.eval()
    loss_accum = {"sim": 0.0, "smooth": 0.0, "jac": 0.0, "total": 0.0}
    metric_accum = {"ssim": 0.0, "psnr": 0.0, "folding": 0.0}
    n_batches = 0

    for batch in loader:
        fixed = batch["fixed"].to(device, non_blocking=True)
        moving = batch["moving"].to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast('cuda'):
                warped, flow = forward_model(model, fixed, moving)
                loss, loss_dict = reg_loss_fn(fixed, warped, flow)
        else:
            warped, flow = forward_model(model, fixed, moving)
            loss, loss_dict = reg_loss_fn(fixed, warped, flow)

        for k in loss_accum:
            loss_accum[k] += loss_dict[k]

        metric_accum["ssim"] += ssim_3d(fixed, warped)
        metric_accum["psnr"] += psnr_3d(fixed, warped)
        metric_accum["folding"] += folding_ratio_3d(flow)
        n_batches += 1

    for k in loss_accum:
        loss_accum[k] /= max(n_batches, 1)
    for k in metric_accum:
        metric_accum[k] /= max(n_batches, 1)

    return {**loss_accum, **metric_accum}


# ============================================================
#  主函数
# ============================================================

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # 应用预设
    if args.preset:
        cfg = apply_preset(cfg, args.preset)

    # 命令行覆盖
    if args.data_dir:
        cfg.setdefault("data", {})["data_dir"] = args.data_dir
        cfg["data"]["source"] = "nifti"

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    log_cfg = cfg.get("logging", {})

    epochs = args.epochs or train_cfg.get("epochs", 200)
    batch_size = args.batch_size or train_cfg.get("batch_size", 2)
    lr = args.lr or train_cfg.get("learning_rate", 1e-4)
    num_workers = data_cfg.get("num_workers", 4)
    log_interval = log_cfg.get("log_interval", 10)
    val_interval = log_cfg.get("val_interval", 5)
    use_amp = train_cfg.get("amp", True)
    grad_clip = train_cfg.get("gradient_clip", 1.0)
    warmup_epochs = train_cfg.get("warmup_epochs", 5)

    save_dir = Path(__file__).resolve().parent / log_cfg.get("save_dir", "checkpoints/reg3d")
    save_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- 数据 ----
    print("\n--- 构建数据集 ---")
    train_cfg_for_ds = {**cfg, "data": {**data_cfg}}
    val_data_cfg = {**data_cfg}
    val_data_cfg["n_samples"] = data_cfg.get("n_val_samples", 40)
    val_data_cfg["augment"] = False
    val_cfg_for_ds = {**cfg, "data": val_data_cfg}

    train_ds = build_dataset_3d(cfg, mode="train")
    val_ds = build_dataset_3d(val_cfg_for_ds, mode="val")
    print(f"训练集: {len(train_ds)} | 验证集: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    # ---- 模型 ----
    print("\n--- 构建模型 ---")
    model = build_model(cfg, device)

    # ---- 损失 ----
    print("\n--- 构建损失 ---")
    reg_loss_fn = build_loss(cfg, device)

    # ---- 优化器 ----
    optimizer, scheduler = build_optimizer(model, cfg)
    print(f"优化器: {train_cfg.get('optimizer', 'Adam')}  lr={lr}  wd={train_cfg.get('weight_decay', 1e-5)}")

    # ---- AMP ----
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == "cuda" else None
    if scaler:
        print("混合精度训练: 开启")

    # ---- TensorBoard ----
    writer = None
    if log_cfg.get("tensorboard", True):
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(save_dir / "tensorboard"))
            print(f"TensorBoard: {save_dir / 'tensorboard'}")
        except ImportError:
            print("⚠ 未安装 tensorboard, 跳过日志记录")

    # ---- 断点续训 ----
    start_epoch = 1
    best_val_loss = float("inf")
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt.get("optimizer_state_dict", optimizer.state_dict()))
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        global_step = ckpt.get("global_step", 0)
        print(f"✓ 断点续训: epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ---- 训练循环 ----
    warmup_steps = warmup_epochs * len(train_loader)
    history = {"train_loss": [], "val_loss": [], "val_ssim": [], "val_folding": []}

    print(f"\n{'='*70}")
    print(f"  3D 配准训练")
    print(f"  epochs={epochs}  batch={batch_size}  lr={lr}")
    print(f"  amp={use_amp}  grad_clip={grad_clip}  warmup={warmup_epochs}ep")
    print(f"  save_dir={save_dir}")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        train_losses, train_metrics, global_step = train_one_epoch(
            model, train_loader, reg_loss_fn, optimizer, device,
            epoch, epochs, warmup_steps, lr,
            grad_clip=grad_clip, use_amp=use_amp, scaler=scaler,
            log_interval=log_interval, global_step=global_step,
        )

        # 学习率调度
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_losses["total"])
            else:
                scheduler.step()

        elapsed = time.time() - t0

        # 验证
        if epoch % val_interval == 0 or epoch == epochs:
            val_results = validate(
                model, val_loader, reg_loss_fn, device, use_amp=use_amp
            )

            print(
                f"\nEpoch {epoch:3d}/{epochs} ({elapsed:.1f}s)\n"
                f"  Train: loss={train_losses['total']:.4f} "
                f"(sim:{train_losses['sim']:.4f} smooth:{train_losses['smooth']:.4f} "
                f"jac:{train_losses['jac']:.4f})\n"
                f"  Train Metrics: SSIM={train_metrics['ssim']:.4f} "
                f"PSNR={train_metrics['psnr']:.2f} "
                f"Folding={train_metrics['folding']:.4f}\n"
                f"  Val:   loss={val_results['total']:.4f} "
                f"(sim:{val_results['sim']:.4f} smooth:{val_results['smooth']:.4f} "
                f"jac:{val_results['jac']:.4f})\n"
                f"  Val Metrics:   SSIM={val_results['ssim']:.4f} "
                f"PSNR={val_results['psnr']:.2f} "
                f"Folding={val_results['folding']:.4f}"
            )

            # TensorBoard
            if writer:
                writer.add_scalar("val/loss", val_results["total"], epoch)
                writer.add_scalar("val/sim_loss", val_results["sim"], epoch)
                writer.add_scalar("val/smooth_loss", val_results["smooth"], epoch)
                writer.add_scalar("val/jac_loss", val_results["jac"], epoch)
                writer.add_scalar("val/ssim", val_results["ssim"], epoch)
                writer.add_scalar("val/psnr", val_results["psnr"], epoch)
                writer.add_scalar("val/folding", val_results["folding"], epoch)

            # 保存最佳模型
            if val_results["total"] < best_val_loss:
                best_val_loss = val_results["total"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_results["total"],
                    "val_metrics": val_results,
                    "config": cfg,
                    "global_step": global_step,
                }, save_dir / "best_model.pth")
                print(f"  ★ 保存最佳模型  Val Loss: {val_results['total']:.4f}")

            history["val_loss"].append(val_results["total"])
            history["val_ssim"].append(val_results["ssim"])
            history["val_folding"].append(val_results["folding"])
        else:
            print(
                f"Epoch {epoch:3d}/{epochs} ({elapsed:.1f}s) "
                f"Train: loss={train_losses['total']:.4f} "
                f"SSIM={train_metrics['ssim']:.4f} "
                f"Folding={train_metrics['folding']:.4f}"
            )

        # TensorBoard (train)
        if writer:
            writer.add_scalar("train/loss", train_losses["total"], epoch)
            writer.add_scalar("train/sim_loss", train_losses["sim"], epoch)
            writer.add_scalar("train/smooth_loss", train_losses["smooth"], epoch)
            writer.add_scalar("train/jac_loss", train_losses["jac"], epoch)
            writer.add_scalar("train/ssim", train_metrics["ssim"], epoch)
            writer.add_scalar("train/psnr", train_metrics["psnr"], epoch)
            writer.add_scalar("train/folding", train_metrics["folding"], epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        history["train_loss"].append(train_losses["total"])

        # 保存 last checkpoint (每 10 epoch 或最后一个)
        if epoch % 10 == 0 or epoch == epochs:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": cfg,
                "global_step": global_step,
            }, save_dir / "last_model.pth")

    # 保存训练历史
    np.savez(save_dir / "training_history.npz", **history)

    if writer:
        writer.close()

    print(f"\n{'='*70}")
    print(f"训练完成!")
    print(f"  最佳 Val Loss: {best_val_loss:.4f}")
    print(f"  模型保存在: {save_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
