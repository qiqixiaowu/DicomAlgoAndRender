"""
3D 配准推理脚本

使用方法
========
  # 基本推理
  python infer_registration3d.py --checkpoint checkpoints/reg3d/best_model.pth \
      --fixed data/lung_fixed.nii.gz --moving data/lung_moving.nii.gz \
      --output outputs/

  # 使用合成数据测试
  python infer_registration3d.py --checkpoint checkpoints/reg3d/best_model.pth \
      --synthetic --output outputs/

  # 可视化结果
  python infer_registration3d.py --checkpoint checkpoints/reg3d/best_model.pth \
      --synthetic --visualize --output outputs/

输出
====
  warped.nii.gz         : 配准后图像
  flow.nii.gz           : 位移场 (3通道)
  jacobian.nii.gz       : Jacobian 行列式图
  metrics.json          : 评估指标
  visualization.png     : 可视化对比 (可选)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import (
    VoxelMorph3D,
    VoxelMorph3DDiff,
)
from data.dataset3d import load_nifti, resample_volume
from data.synthetic3d import SyntheticLungData3D, SyntheticLiverData3D
from data.transforms3d import Normalize3D
from utils.metrics3d import (
    evaluate_registration_3d,
    jacobian_stats_3d,
    ssim_3d,
    psnr_3d,
    folding_ratio_3d,
    jacobian_determinant_3d,
)


# ============================================================
#  参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D 配准推理")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型 checkpoint 路径")
    parser.add_argument("--fixed", type=str, default=None,
                        help="固定图像路径 (NIfTI)")
    parser.add_argument("--moving", type=str, default=None,
                        help="移动图像路径 (NIfTI)")
    parser.add_argument("--synthetic", action="store_true",
                        help="使用合成数据测试")
    parser.add_argument("--synthetic_type", type=str, default="lung",
                        choices=["lung", "liver"],
                        help="合成数据类型")
    parser.add_argument("--output", type=str, default="outputs/reg3d",
                        help="输出目录")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--visualize", action="store_true",
                        help="生成可视化对比图")
    parser.add_argument("--save_nifti", action="store_true", default=True,
                        help="保存结果为 NIfTI 格式")
    return parser.parse_args()


# ============================================================
#  模型加载
# ============================================================

def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    """加载模型和配置"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
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
        model = VoxelMorph3DDiff(
            image_size=volume_size,
            enc_features=enc_features,
            dec_features=dec_features,
            vecint_steps=model_cfg.get("vec_int_steps", 7),
        )
    else:
        raise ValueError(f"未知模型: {name}")

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"模型加载成功: {name}")
    print(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}")
    print(f"  Val loss: {ckpt.get('val_loss', '?'):.4f}")
    return model, cfg


# ============================================================
#  推理
# ============================================================

@torch.no_grad()
def infer(
    model: torch.nn.Module,
    fixed: np.ndarray,
    moving: np.ndarray,
    device: torch.device,
    normalize: str = "percentile",
) -> dict:
    """
    执行配准推理

    参数
    ----
    fixed  : (D, H, W) numpy array
    moving : (D, H, W) numpy array
    device : torch.device

    返回
    ----
    dict: warped, flow, jacobian, metrics
    """
    # 归一化
    normalizer = Normalize3D(mode=normalize)
    fixed_norm, moving_norm = normalizer(fixed.copy(), moving.copy())

    # 转 tensor
    fixed_t = torch.from_numpy(fixed_norm).unsqueeze(0).unsqueeze(0).to(device)
    moving_t = torch.from_numpy(moving_norm).unsqueeze(0).unsqueeze(0).to(device)

    # 前向传播
    output = model(fixed_t, moving_t)
    if len(output) == 3:
        warped_t, flow_t, velocity_t = output
    else:
        warped_t, flow_t = output

    # 转 numpy
    warped = warped_t.squeeze().cpu().numpy()
    flow = flow_t.squeeze().cpu().numpy()  # (3, D, H, W)

    # Jacobian 行列式
    jac_det = jacobian_determinant_3d(flow_t).squeeze().cpu().numpy()

    # 评估指标
    metrics = evaluate_registration_3d(
        fixed_t, warped_t, flow_t, data_range=1.0
    )

    return {
        "warped": warped,
        "flow": flow,
        "jacobian": jac_det,
        "metrics": metrics,
        "fixed_norm": fixed_norm,
        "moving_norm": moving_norm,
    }


# ============================================================
#  保存结果
# ============================================================

def save_nifti(
    data: np.ndarray,
    filepath: str,
    reference_path: str | None = None,
) -> None:
    """保存为 NIfTI 格式"""
    try:
        import SimpleITK as sitk

        if reference_path:
            ref_img = sitk.ReadImage(reference_path)
            img = sitk.GetImageFromArray(data)
            img.SetSpacing(ref_img.GetSpacing())
            img.SetOrigin(ref_img.GetOrigin())
            img.SetDirection(ref_img.GetDirection())
        else:
            img = sitk.GetImageFromArray(data)

        sitk.WriteImage(img, filepath)
        print(f"  保存: {filepath}")

    except ImportError:
        try:
            import nibabel as nib

            if data.ndim == 4:  # flow (3, D, H, W) → (X, Y, Z, 3)
                data = np.transpose(data, (2, 1, 0, 3))
            else:
                data = np.transpose(data, (2, 1, 0))

            img = nib.Nifti1Image(data, affine=np.eye(4))
            nib.save(img, filepath)
            print(f"  保存: {filepath}")

        except ImportError:
            print(f"  ⚠ 无法保存 NIfTI (需要 SimpleITK 或 nibabel), 改存 .npy")
            np.save(filepath.replace(".nii.gz", ".npy").replace(".nii", ".npy"), data)


def save_results(
    results: dict,
    output_dir: str,
    fixed_path: str | None = None,
) -> None:
    """保存所有结果"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n--- 保存结果 ---")

    # 配准后图像
    save_nifti(
        results["warped"].astype(np.float32),
        str(out / "warped.nii.gz"),
        reference_path=fixed_path,
    )

    # 位移场
    save_nifti(
        results["flow"].astype(np.float32),
        str(out / "flow.nii.gz"),
        reference_path=fixed_path,
    )

    # Jacobian 行列式
    save_nifti(
        results["jacobian"].astype(np.float32),
        str(out / "jacobian.nii.gz"),
        reference_path=fixed_path,
    )

    # 指标
    metrics_path = out / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results["metrics"], f, indent=2, ensure_ascii=False)
    print(f"  保存: {metrics_path}")


# ============================================================
#  可视化
# ============================================================

def visualize_results(
    results: dict,
    output_path: str,
    slice_idx: int | None = None,
) -> None:
    """生成可视化对比图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fixed = results["fixed_norm"]
    moving = results["moving_norm"]
    warped = results["warped"]
    flow = results["flow"]
    jac = results["jacobian"]

    D = fixed.shape[0]
    if slice_idx is None:
        slice_idx = D // 2

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 第一行: 图像对比
    axes[0, 0].imshow(fixed[slice_idx], cmap="gray")
    axes[0, 0].set_title("Fixed (Target)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(moving[slice_idx], cmap="gray")
    axes[0, 1].set_title("Moving (Source)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(warped[slice_idx], cmap="gray")
    axes[0, 2].set_title("Warped (Registered)")
    axes[0, 2].axis("off")

    # 第二行: 差异图和形变场
    diff_before = np.abs(fixed[slice_idx] - moving[slice_idx])
    diff_after = np.abs(fixed[slice_idx] - warped[slice_idx])

    axes[1, 0].imshow(diff_before, cmap="hot")
    axes[1, 0].set_title(f"|Fixed - Moving| (before)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(diff_after, cmap="hot")
    axes[1, 1].set_title(f"|Fixed - Warped| (after)")
    axes[1, 1].axis("off")

    # 形变场 (箭头图)
    # flow: (3, D, H, W), 取当前 slice 的 y, x 分量
    dy = flow[1, slice_idx]
    dx = flow[2, slice_idx]
    H, W = dy.shape
    step = max(1, H // 20)
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    axes[1, 2].quiver(xx, yy, dx[::step, ::step], dy[::step, ::step],
                      color="blue", scale=50)
    axes[1, 2].set_title(f"Deformation Field (slice {slice_idx})")
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_aspect("equal")

    plt.suptitle(
        f"3D Registration Result  |  "
        f"SSIM={results['metrics']['ssim']:.4f}  "
        f"PSNR={results['metrics']['psnr']:.2f}  "
        f"Folding={results['metrics']['jac_folding_ratio']:.4f}",
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  可视化: {output_path}")


# ============================================================
#  主函数
# ============================================================

def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"设备: {device}")

    # 加载模型
    model, cfg = load_model(args.checkpoint, device)

    # 获取数据
    if args.synthetic:
        print(f"\n使用合成 {args.synthetic_type} 数据测试...")
        vol_size = tuple(cfg.get("data", {}).get("volume_size", [64, 64, 64]))
        if args.synthetic_type == "lung":
            gen = SyntheticLungData3D(vol_size)
        else:
            gen = SyntheticLiverData3D(vol_size)
        fixed, moving = gen.generate_one()
        fixed_path = None
    else:
        if not args.fixed or not args.moving:
            raise ValueError("非合成模式需要 --fixed 和 --moving 参数")
        print(f"\n加载 NIfTI 数据...")
        vol_size = tuple(cfg.get("data", {}).get("volume_size", [64, 64, 64]))
        fixed = load_nifti(args.fixed)
        moving = load_nifti(args.moving)
        fixed = resample_volume(fixed, vol_size)
        moving = resample_volume(moving, vol_size)
        fixed_path = args.fixed

    print(f"  Fixed:  {fixed.shape}  range: [{fixed.min():.2f}, {fixed.max():.2f}]")
    print(f"  Moving: {moving.shape} range: [{moving.min():.2f}, {moving.max():.2f}]")

    # 推理
    print("\n--- 执行配准推理 ---")
    t0 = __import__("time").time()
    results = infer(model, fixed, moving, device)
    elapsed = __import__("time").time() - t0
    print(f"  推理时间: {elapsed:.3f}s")

    # 打印指标
    print("\n--- 评估指标 ---")
    for k, v in results["metrics"].items():
        print(f"  {k:20s}: {v:.4f}")

    # 保存结果
    save_results(results, args.output, fixed_path)

    # 可视化
    if args.visualize:
        vis_path = Path(args.output) / "visualization.png"
        visualize_results(results, str(vis_path))

    print(f"\n✓ 推理完成! 结果保存在: {args.output}")


if __name__ == "__main__":
    main()
