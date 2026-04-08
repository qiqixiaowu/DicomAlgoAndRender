"""
血管分割推理脚本

功能
====
1. 加载训练好的 3D U-Net 模型
2. 读取一个 DICOM 序列（CT）
3. 滑动窗口推理，得到全分辨率血管分割掩码
4. 导出供 C++ OpenGL 渲染器使用的 .raw 文件和元数据
5. 用 matplotlib 可视化分割结果

使用方法
========
  # 推理并可视化
  python infer_vessel_seg.py --dicom_dir "E:\\PatientData\\CTA_001"

  # 指定模型路径和输出
  python infer_vessel_seg.py \\
      --dicom_dir  "E:\\PatientData\\CTA_001" \\
      --model_path checkpoints/vessel_seg/best_model.pth \\
      --output_dir outputs/vessel_result

输出文件
========
  outputs/vessel_result/
  ├── vessel_mask.raw          ← uint8 二值掩码 (D×H×W, 0=背景, 255=血管)
  ├── vessel_mask_meta.json    ← 体数据维度（C++ 加载时需要）
  └── vessel_overlay.png       ← 三平面可视化预览

C++ 使用方法
============
  在 main_optimized_example.cpp 的 main() 中，加载 DICOM 之后调用:
    auto dlMask = loadDLVesselMask("outputs/vessel_result/vessel_mask.raw",
                                   "outputs/vessel_result/vessel_mask_meta.json");
    g_segMaskTexture = createOrUpdateSegMaskTexture(dlMask, g_segMaskTexture);
    renderState.enableSegmentation = true;
    renderState.segColor = glm::vec3(1.0f, 0.2f, 0.1f);  // 红色血管
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.unet3d import UNet3D
from data.vessel_dataset import (
    VesselInferenceVolume, normalize_vessel_ct,
    _load_dicom_series,
)


# ============================================================
#  参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="血管分割推理 + 导出 OpenGL 可用的 .raw 掩码",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dicom_dir", type=str, required=True,
                        help="要推理的 DICOM 序列目录")
    parser.add_argument("--model_path", type=str,
                        default="checkpoints/vessel_seg/best_model.pth",
                        help="训练好的模型检查点路径")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/vessel_result",
                        help="输出文件保存目录")
    parser.add_argument("--patch_size", type=int, nargs=3,
                        default=None, metavar=("D", "H", "W"),
                        help="推理 patch 大小（默认读取 model_meta.json）")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="滑动窗口重叠率 (0~1)，越大越准确但越慢")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="二值化阈值（血管概率 > threshold 判为血管）")
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备 (cuda/cpu)")
    parser.add_argument("--no_vis", action="store_true",
                        help="跳过可视化（仅导出 .raw 文件）")
    parser.add_argument("--no_remove_bed", action="store_true",
                        help="禁用床板自动去除（默认启用）")
    parser.add_argument("--body_thresh", type=float, default=-200.0,
                        help="身体/床板分割阈值 HU（默认 -200）")
    return parser.parse_args()


# ============================================================
#  床板去除预处理
# ============================================================

def remove_bed_artifact(
    volume_hu: np.ndarray,
    body_thresh: float = -200.0,
    air_hu: float = -1000.0,
) -> np.ndarray:
    """
    自动去除 CT 床板 (patient table) 伪影

    方法
    ----
    1. 圆形 FOV 裁剪：CT 重建域是圆形，边角置为空气，消除重建伪影
    2. 逐轴向切片连通域分析：阈值 > body_thresh 后找最大连通域（患者身体），
       填充内部孔洞（肺等低密度器官），屏蔽床板
    3. 用身体掩码将非身体区域替换为 air_hu，避免床板影响推理

    参数
    ----
    volume_hu   : (D, H, W) float32，原始 HU 值体数据
    body_thresh : 阈值，默认 -200 HU（空气约 -1000，软组织约 50）
    air_hu      : 替换非身体区域所用的 HU 值，默认 -1000

    返回
    ----
    cleaned : 与输入同形状，床板/边角区域已替换为 air_hu
    """
    from scipy.ndimage import label as _label, binary_fill_holes

    D, H, W = volume_hu.shape

    # ---- Step 1: 圆形 FOV 掩码 ----
    # CT 重建圆的半径略小于图像短边 / 2
    cy, cx = H / 2.0, W / 2.0
    radius = min(H, W) * 0.5
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    fov_circle = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2   # (H, W)

    # ---- Step 2: 逐切片提取患者身体轮廓 ----
    body_mask_3d = np.zeros((D, H, W), dtype=bool)
    for z in range(D):
        sl = volume_hu[z]          # (H, W)
        binary = (sl > body_thresh) & fov_circle
        if not binary.any():
            continue

        labeled, n_comp = _label(binary)
        if n_comp == 0:
            continue

        # 找面积最大的连通域（通常是患者身体截面）
        comp_sizes = np.bincount(labeled.ravel())
        comp_sizes[0] = 0          # 背景排除
        largest_id = int(np.argmax(comp_sizes))
        body_slice = labeled == largest_id

        # 填充内部孔洞（肺、空腔等低密度区域）
        body_mask_3d[z] = binary_fill_holes(body_slice)

    # ---- Step 3: 应用掩码，替换非身体区域 ----
    result = volume_hu.copy()
    result[~body_mask_3d] = air_hu

    removed = int((~body_mask_3d).sum())
    total   = body_mask_3d.size
    print(f"  [床板去除] 屏蔽体素: {removed:,} / {total:,} "
          f"({100.0 * removed / total:.1f}%) → 替换为 {air_hu:.0f} HU")
    return result


# ============================================================
#  滑动窗口推理
# ============================================================

def sliding_window_inference(
    model: torch.nn.Module,
    volume: np.ndarray,
    patch_size: tuple[int, int, int],
    overlap: float,
    device: torch.device,
) -> np.ndarray:
    """
    3D 滑动窗口推理

    参数
    ----
    model      : 训练好的模型
    volume     : (D, H, W) float32, 已归一化到 [0,1]
    patch_size : (pd, ph, pw)
    overlap    : 相邻窗口重叠率
    device     : 推理设备

    返回
    ----
    prob_map : (D, H, W) float32, 血管概率图
    """
    D, H, W = volume.shape
    pd, ph, pw = patch_size

    # 计算步长
    stride = (
        max(1, int(pd * (1 - overlap))),
        max(1, int(ph * (1 - overlap))),
        max(1, int(pw * (1 - overlap))),
    )

    # 对结果累加（高斯权重窗避免块状伪影）
    prob_sum = np.zeros((D, H, W), dtype=np.float32)
    weight_sum = np.zeros((D, H, W), dtype=np.float32)
    gauss_w = _gaussian_weight_map(patch_size)

    # 生成所有 patch 起点
    starts_d = _get_starts(D, pd, stride[0])
    starts_h = _get_starts(H, ph, stride[1])
    starts_w = _get_starts(W, pw, stride[2])

    total_patches = len(starts_d) * len(starts_h) * len(starts_w)
    n_done = 0

    model.eval()
    with torch.no_grad():
        for z0 in starts_d:
            for y0 in starts_h:
                for x0 in starts_w:
                    z1, y1, x1 = z0 + pd, y0 + ph, x0 + pw

                    # 提取 patch（含边界 padding）
                    patch = _extract_patch(volume, z0, y0, x0, z1, y1, x1)

                    # 推理
                    t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                    out = model(t)                          # (1, 2, pd, ph, pw)
                    prob = torch.softmax(out, dim=1)[0, 1] # (pd, ph, pw) 血管概率
                    prob_np = prob.cpu().numpy()

                    # 写回（注意 clamp 到体数据范围）
                    az0 = max(0, z0); az1 = min(D, z1)
                    ay0 = max(0, y0); ay1 = min(H, y1)
                    ax0 = max(0, x0); ax1 = min(W, x1)
                    pz0 = az0 - z0; pz1 = az1 - z0
                    py0 = ay0 - y0; py1 = ay1 - y0
                    px0 = ax0 - x0; px1 = ax1 - x0

                    prob_sum[az0:az1, ay0:ay1, ax0:ax1] += (
                        prob_np[pz0:pz1, py0:py1, px0:px1]
                        * gauss_w[pz0:pz1, py0:py1, px0:px1]
                    )
                    weight_sum[az0:az1, ay0:ay1, ax0:ax1] += (
                        gauss_w[pz0:pz1, py0:py1, px0:px1]
                    )

                    n_done += 1
                    if n_done % max(1, total_patches // 10) == 0:
                        pct = 100 * n_done / total_patches
                        print(f"  推理进度: {n_done}/{total_patches} ({pct:.0f}%)",
                              flush=True)

    # 加权平均
    mask = np.where(weight_sum > 0, prob_sum / weight_sum, 0.0)
    return mask.astype(np.float32)


def _get_starts(length: int, patch: int, stride: int) -> list[int]:
    """生成滑动窗口起点列表，确保覆盖全部区域"""
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch, stride))
    if starts[-1] + patch < length:
        starts.append(length - patch)
    return starts


def _extract_patch(
    volume: np.ndarray,
    z0: int, y0: int, x0: int,
    z1: int, y1: int, x1: int,
) -> np.ndarray:
    """提取 patch，越界部分 zero-pad"""
    D, H, W = volume.shape
    az0, ay0, ax0 = max(0, z0), max(0, y0), max(0, x0)
    az1, ay1, ax1 = min(D, z1), min(H, y1), min(W, x1)
    patch_shape = (z1 - z0, y1 - y0, x1 - x0)
    patch = np.zeros(patch_shape, dtype=np.float32)
    pz0 = az0 - z0; pz1 = az1 - z0
    py0 = ay0 - y0; py1 = ay1 - y0
    px0 = ax0 - x0; px1 = ax1 - x0
    patch[pz0:pz1, py0:py1, px0:px1] = volume[az0:az1, ay0:ay1, ax0:ax1]
    return patch


def _gaussian_weight_map(patch_size: tuple[int, int, int]) -> np.ndarray:
    """生成高斯权重图，中心权重高，边缘权重低，减少拼接伪影"""
    pd, ph, pw = patch_size
    z = np.linspace(-1, 1, pd)
    y = np.linspace(-1, 1, ph)
    x = np.linspace(-1, 1, pw)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    sigma = 0.5
    w = np.exp(-(zz**2 + yy**2 + xx**2) / (2 * sigma**2))
    return w.astype(np.float32)


# ============================================================
#  导出 .raw 文件（供 C++ 加载）
# ============================================================

def export_mask_raw(
    mask: np.ndarray,
    output_dir: Path,
    volume_shape: tuple[int, int, int],
):
    """
    导出 uint8 二值掩码为 .raw 文件，并写入元数据 JSON。

    .raw 存储格式: uint8, shape=(D, H, W), 行主序 (C order)
    值: 0=背景, 255=血管

    C++ 读取对应 RegionGrowingResult.mask 格式:
      mask[z * H * W + y * W + x] = 0 or 255
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    D, H, W = volume_shape
    raw_mask = (mask * 255).astype(np.uint8)  # 0 or 255

    raw_path = output_dir / "vessel_mask.raw"
    raw_mask.tofile(str(raw_path))

    vessel_count = int((raw_mask > 0).sum())
    total_voxels = D * H * W

    meta = {
        "width":  W,
        "height": H,
        "depth":  D,
        "voxel_count": vessel_count,
        "total_voxels": total_voxels,
        "vessel_ratio": round(vessel_count / max(1, total_voxels) * 100, 3),
        "dtype": "uint8",
        "layout": "D_H_W (depth-height-width, C order)",
        "values": "0=background, 255=vessel",
    }
    meta_path = output_dir / "vessel_mask_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n已导出:")
    print(f"  掩码文件: {raw_path}  ({raw_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  元数据:   {meta_path}")
    print(f"  血管体素: {vessel_count:,} / {total_voxels:,} ({meta['vessel_ratio']}%)")
    return raw_path, meta_path


# ============================================================
#  可视化
# ============================================================

def visualize_result(
    volume: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
):
    """三平面 (轴/冠/矢) 叠加显示原始 CT + 血管掩码"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        D, H, W = volume.shape
        vessel_cmap = LinearSegmentedColormap.from_list(
            "vessel", [(1, 0, 0, 0), (1, 0.1, 0.1, 0.85)]
        )

        # 选取包含血管最多的切片
        axial_idx   = int(np.argmax(mask.sum(axis=(1, 2))))
        coronal_idx = int(np.argmax(mask.sum(axis=(0, 2))))
        sagit_idx   = int(np.argmax(mask.sum(axis=(0, 1))))

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor("#0a0a0a")

        titles = [
            f"Axial (z={axial_idx})",
            f"Coronal (y={coronal_idx})",
            f"Sagittal (x={sagit_idx})",
        ]
        slices_vol = [
            volume[axial_idx],
            volume[:, coronal_idx, :],
            volume[:, :, sagit_idx],
        ]
        slices_msk = [
            mask[axial_idx].astype(float),
            mask[:, coronal_idx, :].astype(float),
            mask[:, :, sagit_idx].astype(float),
        ]

        for ax, title, sv, sm in zip(axes, titles, slices_vol, slices_msk):
            ax.imshow(sv, cmap="gray", vmin=0.1, vmax=0.8, origin="upper")
            ax.imshow(np.where(sm > 0, sm, np.nan),
                      cmap=vessel_cmap, vmin=0, vmax=1,
                      alpha=0.7, origin="upper")
            ax.set_title(title, color="white", fontsize=12)
            ax.axis("off")
            ax.set_facecolor("black")

        plt.suptitle("血管分割结果 (红色=血管)", color="white", fontsize=14, y=0.99)
        plt.tight_layout()

        out_img = output_dir / "vessel_overlay.png"
        plt.savefig(str(out_img), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  可视化图像: {out_img}")
    except Exception as e:
        print(f"  [警告] 可视化失败: {e}")


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
        print(f"推理设备: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("推理设备: CPU")

    output_dir = Path(args.output_dir)

    # ---- 加载模型 ----
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"错误: 找不到模型文件 {model_path}")
        print("请先运行训练脚本: python train_vessel_seg.py")
        sys.exit(1)

    print(f"\n加载模型: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # 读取模型结构元数据
    meta = ckpt.get("meta", {})
    features = meta.get("features", [16, 32, 64, 128])
    in_ch    = meta.get("in_channels", 1)
    out_ch   = meta.get("out_channels", 2)
    patch_size = tuple(
        args.patch_size if args.patch_size
        else meta.get("patch_size", [64, 128, 128])
    )
    hu_min = meta.get("hu_min", -150.0)
    hu_max = meta.get("hu_max", 550.0)

    model = UNet3D(in_channels=in_ch, out_channels=out_ch, features=features).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"  模型结构: UNet3D features={features}")
    print(f"  推理 patch: {patch_size}")
    print(f"  HU 范围: [{hu_min}, {hu_max}]")

    # ---- 加载 DICOM ----
    print(f"\n加载 DICOM 序列: {args.dicom_dir}")
    volume_hu = _load_dicom_series(args.dicom_dir)
    D, H, W = volume_hu.shape
    print(f"  体数据尺寸: D={D}, H={H}, W={W}")
    print(f"  HU 范围: [{volume_hu.min():.0f}, {volume_hu.max():.0f}]")

    # ---- 床板去除（默认启用）----
    if not args.no_remove_bed:
        print("\n预处理: 自动去除床板伪影...")
        volume_hu = remove_bed_artifact(
            volume_hu,
            body_thresh=args.body_thresh,
            air_hu=-1000.0,
        )
    else:
        print("\n[提示] 床板去除已禁用 (--no_remove_bed)")

    # 归一化（使用与训练时相同的参数）
    volume_norm = np.clip(volume_hu, hu_min, hu_max)
    volume_norm = ((volume_norm - hu_min) / (hu_max - hu_min)).astype(np.float32)

    # ---- 滑动窗口推理 ----
    print(f"\n开始滑动窗口推理 (overlap={args.overlap})...")
    import time
    t0 = time.time()

    prob_map = sliding_window_inference(
        model=model,
        volume=volume_norm,
        patch_size=patch_size,
        overlap=args.overlap,
        device=device,
    )

    elapsed = time.time() - t0
    print(f"推理完成，耗时 {elapsed:.1f}s")

    # ---- 二值化 ----
    binary_mask = (prob_map >= args.threshold).astype(np.uint8)
    vessel_count = int(binary_mask.sum())
    print(f"血管体素数: {vessel_count:,} ({100*vessel_count/binary_mask.size:.2f}%)")

    # ---- 导出 .raw ----
    print("\n导出 .raw 文件...")
    export_mask_raw(binary_mask, output_dir, (D, H, W))

    # ---- 可视化 ----
    if not args.no_vis:
        print("\n生成可视化图像...")
        visualize_result(volume_norm, binary_mask, output_dir)

    print(f"\n{'='*60}")
    print("推理完成！")
    print(f"{'='*60}")
    print("\nC++ 中加载此分割结果:")
    print("  在 main_optimized_example 的 main() 中，DICOM 加载之后添加:")
    print(f"""
    // --- 加载深度学习血管分割掩码 ---
    auto dlMask = loadDLVesselMask(
        "{(output_dir / 'vessel_mask.raw').as_posix()}",
        "{(output_dir / 'vessel_mask_meta.json').as_posix()}"
    );
    if (!dlMask.mask.empty()) {{
        g_segMaskTexture = createOrUpdateSegMaskTexture(dlMask, g_segMaskTexture);
        renderState.enableSegmentation = true;
        renderState.segColor   = glm::vec3(1.0f, 0.15f, 0.1f); // 血管红色
        renderState.segOpacity = 0.85f;
        std::cout << "血管分割掩码已加载" << std::endl;
    }}
""")


if __name__ == "__main__":
    main()
