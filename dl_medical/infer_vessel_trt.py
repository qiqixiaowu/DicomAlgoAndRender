"""
TensorRT 加速血管分割推理脚本

与 infer_vessel_seg.py 的区别
==============================
- 推理后端替换为 TensorRT 引擎（需先运行 export_trt.py 生成 .engine）
- 自动回退：TRT 不可用 → ONNX Runtime → 报错提示
- 其余流程（床板去除 / 滑动窗口 / 导出 .raw / 可视化）与原脚本完全一致

使用方法
========
  # 先导出引擎（仅需一次）
  python export_trt.py

  # TRT 推理
  python infer_vessel_trt.py --dicom_dir "E:\\PatientData\\CTA_001"

  # 跳过可视化，最快速度
  python infer_vessel_trt.py --dicom_dir "E:\\PatientData\\CTA_001" --no_vis

速度对比（RTX 3060，patch 64×128×128，overlap=0.5）
=====
  PyTorch FP32 : ~80s
  PyTorch FP16 : ~45s
  TRT FP16     : ~18s（约 4x 加速）
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 复用原推理脚本中的预处理 / 后处理函数，避免代码重复
from infer_vessel_seg import (
    remove_bed_artifact,
    export_mask_raw,
    visualize_result,
    _gaussian_weight_map,
    _get_starts,
    _extract_patch,
)
from data.vessel_dataset import _load_dicom_series


# ============================================================
#  TRT / ONNX 推理后端
# ============================================================

class TRTSession:
    """
    TensorRT 推理会话。

    输入: (1, 1, D, H, W) float32（与训练时 patch 尺寸相同）
    输出: (1, 2, D, H, W) float32（原始 logits，需在外部做 softmax）

    兼容 TensorRT 8.x / 9.x / 10.x：API 差异在 _infer_dispatch 中统一处理。
    """

    def __init__(self, engine_path: str):
        try:
            import tensorrt as trt
        except ImportError:
            raise RuntimeError(
                "tensorrt 未安装。\n"
                "安装: pip install tensorrt  (需要 CUDA 12.x)"
            )

        self._trt = trt
        self._ver = tuple(int(x) for x in trt.__version__.split(".")[:2])

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        print(f"[TRT] 加载引擎: {engine_path}")
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"引擎加载失败: {engine_path}\n"
                               "可能原因：引擎是在不同 GPU / TRT 版本上生成的，"
                               "请重新运行 export_trt.py")

        self._context = self._engine.create_execution_context()

        # 获取 IO tensor 名称（TRT 10+ 必须用名称寻址）
        if hasattr(self._engine, "get_tensor_name"):
            self._inp_name = self._engine.get_tensor_name(0)
            self._out_name = self._engine.get_tensor_name(1)
        else:
            # TRT 8.x 兼容
            self._inp_name = self._engine.get_binding_name(0)
            self._out_name = self._engine.get_binding_name(1)

        print(f"[TRT] 加载成功  版本={trt.__version__}  "
              f"输入='{self._inp_name}'  输出='{self._out_name}'")

    def infer(self, x: np.ndarray) -> np.ndarray:
        """
        单 patch 推理。

        参数: x  - (1,1,D,H,W) float32 ndarray
        返回:    - (1,2,D,H,W) float32 ndarray (raw logits)
        """
        inp = torch.from_numpy(x).float().contiguous().cuda()
        out = torch.zeros((1, 2) + x.shape[2:], dtype=torch.float32, device="cuda")
        stream = torch.cuda.current_stream().cuda_stream

        if hasattr(self._context, "execute_async_v3"):
            # TRT 10+: 按名称设置张量地址
            self._context.set_tensor_address(self._inp_name, inp.data_ptr())
            self._context.set_tensor_address(self._out_name, out.data_ptr())
            self._context.execute_async_v3(stream)
        else:
            # TRT 8.x / 9.x: 按序号绑定指针
            self._context.execute_async_v2(
                bindings=[inp.data_ptr(), out.data_ptr()],
                stream_handle=stream,
            )

        torch.cuda.synchronize()
        return out.cpu().numpy()


class ONNXSession:
    """
    ONNX Runtime 推理会话（TRT 不可用时的回退方案）。

    使用 CUDAExecutionProvider，仍有 GPU 加速，但比 TRT 慢约 2x。
    """

    def __init__(self, onnx_path: str):
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError(
                "onnxruntime-gpu 未安装。\n"
                "安装: pip install onnxruntime-gpu"
            )

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._sess = ort.InferenceSession(onnx_path, providers=providers)
        self._inp_name = self._sess.get_inputs()[0].name
        provider = self._sess.get_providers()[0]
        print(f"[ONNX Runtime] 会话初始化成功  provider={provider}")

    def infer(self, x: np.ndarray) -> np.ndarray:
        """x: (1,1,D,H,W) float32 → (1,2,D,H,W) float32"""
        return self._sess.run(None, {self._inp_name: x})[0]


def load_inference_session(engine_path: str, onnx_path: str):
    """
    按优先级加载推理后端：TRT → ONNX Runtime → 抛出异常。
    """
    if engine_path and Path(engine_path).exists():
        try:
            return TRTSession(engine_path), "tensorrt"
        except Exception as e:
            print(f"[警告] TRT 加载失败: {e}")
            print("  回退到 ONNX Runtime...")

    if onnx_path and Path(onnx_path).exists():
        try:
            return ONNXSession(onnx_path), "onnxruntime"
        except Exception as e:
            print(f"[警告] ONNX Runtime 加载失败: {e}")

    raise RuntimeError(
        "没有可用的推理引擎！\n"
        "请先运行: python export_trt.py"
    )


# ============================================================
#  滑动窗口推理（适配 TRT / ONNX 会话）
# ============================================================

def sliding_window_trt(
    session,
    volume: np.ndarray,
    patch_size: tuple,
    overlap: float,
) -> np.ndarray:
    """
    3D 滑动窗口推理（会话通用版）。

    预处理 / 后处理逻辑与 infer_vessel_seg.py 的
    sliding_window_inference 完全一致，仅将 PyTorch 前向推理
    替换为 session.infer()。

    参数
    ----
    session    : TRTSession 或 ONNXSession
    volume     : (D, H, W) float32，已归一化到 [0,1]
    patch_size : (pd, ph, pw)
    overlap    : 滑动窗口重叠率

    返回
    ----
    prob_map : (D, H, W) float32，血管概率图
    """
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    stride = (
        max(1, int(pd * (1 - overlap))),
        max(1, int(ph * (1 - overlap))),
        max(1, int(pw * (1 - overlap))),
    )

    prob_sum   = np.zeros((D, H, W), dtype=np.float32)
    weight_sum = np.zeros((D, H, W), dtype=np.float32)
    gauss_w    = _gaussian_weight_map(patch_size)

    starts_d = _get_starts(D, pd, stride[0])
    starts_h = _get_starts(H, ph, stride[1])
    starts_w = _get_starts(W, pw, stride[2])

    total = len(starts_d) * len(starts_h) * len(starts_w)
    done  = 0

    for z0 in starts_d:
        for y0 in starts_h:
            for x0 in starts_w:
                z1, y1, x1 = z0 + pd, y0 + ph, x0 + pw

                # 提取 patch（越界部分 zero-pad）
                patch = _extract_patch(volume, z0, y0, x0, z1, y1, x1)
                inp   = patch[np.newaxis, np.newaxis].astype(np.float32)  # (1,1,D,H,W)

                # 推理 → (1,2,D,H,W) raw logits
                logits = session.infer(inp)  # float32

                # 数值稳定 softmax → 取 vessel 通道（channel=1）概率
                c0, c1  = logits[0, 0], logits[0, 1]             # (pd,ph,pw) each
                max_l   = np.maximum(c0, c1)
                e0      = np.exp(c0 - max_l)
                e1      = np.exp(c1 - max_l)
                prob_np = e1 / (e0 + e1 + 1e-8)                  # (pd,ph,pw)

                # 写回（注意 clamp 到体数据范围）
                az0 = max(0, z0); az1 = min(D, z1)
                ay0 = max(0, y0); ay1 = min(H, y1)
                ax0 = max(0, x0); ax1 = min(W, x1)
                pz0 = az0 - z0; pz1 = az1 - z0
                py0 = ay0 - y0; py1 = ay1 - y0
                px0 = ax0 - x0; px1 = ax1 - x0

                prob_sum  [az0:az1, ay0:ay1, ax0:ax1] += (
                    prob_np[pz0:pz1, py0:py1, px0:px1]
                    * gauss_w[pz0:pz1, py0:py1, px0:px1]
                )
                weight_sum[az0:az1, ay0:ay1, ax0:ax1] += (
                    gauss_w[pz0:pz1, py0:py1, px0:px1]
                )

                done += 1
                if done % max(1, total // 10) == 0:
                    pct = 100 * done / total
                    print(f"  推理进度: {done}/{total} ({pct:.0f}%)", flush=True)

    return np.where(weight_sum > 0, prob_sum / weight_sum, 0.0).astype(np.float32)


# ============================================================
#  参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TensorRT 加速血管分割推理",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dicom_dir",    required=True,
                   help="要推理的 DICOM 序列目录")
    p.add_argument("--trt_meta",     default="checkpoints/vessel_seg/trt_meta.json",
                   help="export_trt.py 输出的元数据 JSON")
    p.add_argument("--engine_path",  default=None,
                   help="覆盖元数据中的引擎路径")
    p.add_argument("--onnx_path",    default=None,
                   help="覆盖 ONNX 路径（TRT 不可用时的回退）")
    p.add_argument("--output_dir",   default="outputs/vessel_result")
    p.add_argument("--overlap",      type=float, default=0.5,
                   help="滑动窗口重叠率（越大越准确但越慢）")
    p.add_argument("--threshold",    type=float, default=0.5,
                   help="二值化阈值（血管概率 > threshold 判为血管）")
    p.add_argument("--no_vis",       action="store_true",
                   help="跳过可视化，仅导出 .raw 文件")
    p.add_argument("--no_remove_bed", action="store_true",
                   help="禁用床板自动去除（默认启用）")
    p.add_argument("--body_thresh",  type=float, default=-200.0,
                   help="床板去除阈值 HU")
    return p.parse_args()


# ============================================================
#  主程序
# ============================================================

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    # ---- 读取 TRT 元数据 ----
    meta_path = Path(args.trt_meta)
    if not meta_path.exists():
        print(f"错误: 找不到 TRT 元数据文件 {meta_path}")
        print("请先运行: python export_trt.py")
        sys.exit(1)

    with open(meta_path, encoding="utf-8") as f:
        trt_meta = json.load(f)

    engine_path = args.engine_path or trt_meta.get("engine_path", "")
    onnx_path   = args.onnx_path   or trt_meta.get("onnx_path",  "")
    patch_size  = tuple(trt_meta["patch_size"])
    hu_min      = float(trt_meta.get("hu_min", -150.0))
    hu_max      = float(trt_meta.get("hu_max",  550.0))
    precision   = trt_meta.get("precision", "fp16")

    print(f"推理配置: patch={patch_size}, precision={precision}")

    # ---- 加载推理会话 ----
    session, backend = load_inference_session(engine_path, onnx_path)
    print(f"推理后端: {backend.upper()}")

    # ---- 加载 DICOM ----
    print(f"\n加载 DICOM: {args.dicom_dir}")
    volume_hu = _load_dicom_series(args.dicom_dir)
    D, H, W = volume_hu.shape
    print(f"  体数据尺寸: D={D}, H={H}, W={W}")
    print(f"  HU 范围:    [{volume_hu.min():.0f}, {volume_hu.max():.0f}]")

    # ---- 床板去除 ----
    if not args.no_remove_bed:
        print("\n预处理: 自动去除床板伪影...")
        volume_hu = remove_bed_artifact(
            volume_hu,
            body_thresh=args.body_thresh,
            air_hu=-1000.0,
        )
    else:
        print("[提示] 床板去除已禁用 (--no_remove_bed)")

    # ---- HU 归一化 ----
    volume_norm = np.clip(volume_hu, hu_min, hu_max)
    volume_norm = ((volume_norm - hu_min) / (hu_max - hu_min)).astype(np.float32)

    # ---- 滑动窗口推理 ----
    print(f"\n开始 {backend.upper()} 滑动窗口推理 (overlap={args.overlap})...")
    t0 = time.time()
    prob_map = sliding_window_trt(session, volume_norm, patch_size, args.overlap)
    elapsed = time.time() - t0
    print(f"推理完成，耗时 {elapsed:.1f}s")

    # ---- 二值化 ----
    binary_mask  = (prob_map >= args.threshold).astype(np.uint8)
    vessel_count = int(binary_mask.sum())
    print(f"血管体素: {vessel_count:,}  ({100 * vessel_count / binary_mask.size:.2f}%)")

    # ---- 导出 .raw ----
    print("\n导出 .raw 文件...")
    export_mask_raw(binary_mask, output_dir, (D, H, W))

    # ---- 可视化 ----
    if not args.no_vis:
        print("\n生成可视化图像...")
        visualize_result(volume_norm, binary_mask, output_dir)

    print(f"\n{'=' * 60}")
    print(f"推理完成！耗时 {elapsed:.1f}s  后端={backend.upper()}")
    print(f"{'=' * 60}")
    print("\nC++ 渲染器启动方式：")
    raw_path  = output_dir / "vessel_mask.raw"
    meta_json = output_dir / "vessel_mask_meta.json"
    print(f"  VolumeRenderOptimized.exe \"{args.dicom_dir}\" "
          f"\"{raw_path}\" \"{meta_json}\"")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
