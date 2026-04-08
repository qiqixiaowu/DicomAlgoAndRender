"""
TensorRT 引擎导出工具

功能
====
1. 加载训练好的 PyTorch 3D U-Net (.pth)
2. 导出为 ONNX 格式（固定 patch 尺寸，opset 17）
3. 使用 TensorRT 将 ONNX 编译为优化引擎 (.engine)
4. 输出 trt_meta.json 供 infer_vessel_trt.py 使用

使用方法
========
  # 默认 FP16（推荐，RTX 3060 速度约提升 2~3x）
  python export_trt.py

  # FP32
  python export_trt.py --no_fp16

  # 指定模型路径
  python export_trt.py --model_path checkpoints/vessel_seg/best_model.pth

输出文件
========
  checkpoints/vessel_seg/
  ├── vessel_seg.onnx              ← ONNX 模型（可用 ONNX Runtime 推理）
  ├── vessel_seg_fp16.engine       ← TensorRT FP16 引擎
  └── trt_meta.json                ← 元数据（patch_size / HU 范围 / 文件路径）

注意
====
  - 首次构建引擎约需 2~5 分钟（会 profile 算子）
  - 引擎与 GPU 型号/TRT 版本绑定，换 GPU 须重新构建
  - 支持 TensorRT 8.x / 9.x / 10.x
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.unet3d import UNet3D


# ============================================================
#  TRT 版本兼容工具
# ============================================================

def _trt_set_workspace(trt, config, gb: int):
    """兼容 TRT 8.x（max_workspace_size）和 9.x/10.x（set_memory_pool_limit）"""
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, gb << 30)
    except AttributeError:
        config.max_workspace_size = gb << 30  # TRT < 8.5


def _trt_explicit_batch_flag(trt) -> int:
    """获取 EXPLICIT_BATCH 标志位（各版本一致，但防御性处理）"""
    try:
        return 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    except Exception:
        return 1  # 历史默认值


# ============================================================
#  ONNX 导出
# ============================================================

def export_onnx(
    model: torch.nn.Module,
    patch_size: tuple,
    onnx_path: Path,
    opset: int = 17,
) -> Path:
    """
    将 UNet3D 导出为固定输入形状的 ONNX 文件。

    patch_size 必须与推理时使用的 patch 尺寸完全一致，
    因为 TRT 静态引擎的输入形状在编译时已固定。

    注意：trace 在 CPU 上进行，避免 cuDNN 算子查找失败
    （"GET was unable to find an engine" 错误）。
    ONNX 文件与设备无关，不影响后续 TRT FP16 编译。
    """
    pd, ph, pw = patch_size
    print(f"  导出 ONNX: input=(1,1,{pd},{ph},{pw}), opset={opset}")

    # 临时移到 CPU trace，避免 CUDA cuDNN dispatch 报错
    orig_device = next(model.parameters()).device
    cpu_model = model.cpu().eval()
    dummy = torch.zeros(1, 1, pd, ph, pw)  # CPU tensor

    with torch.no_grad():
        torch.onnx.export(
            cpu_model,
            dummy,
            str(onnx_path),
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )

    # 恢复原始设备
    model.to(orig_device)

    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"  ONNX 已保存: {onnx_path}  ({size_mb:.1f} MB)")
    return onnx_path


def verify_onnx(onnx_path: Path, patch_size: tuple) -> bool:
    """用 onnxruntime 做一次前向推理验证 ONNX 正确性"""
    try:
        import onnxruntime as ort
        import numpy as np

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = ort.InferenceSession(str(onnx_path), providers=providers)
        pd, ph, pw = patch_size
        dummy = numpy_dummy = __import__("numpy").zeros(
            (1, 1, pd, ph, pw), dtype="float32"
        )
        out = sess.run(None, {"input": dummy})[0]
        provider = sess.get_providers()[0]
        print(f"  ONNX Runtime 验证通过: output={out.shape}, provider={provider}")
        return True
    except ImportError:
        print("  提示: 安装 onnxruntime-gpu 可验证 ONNX 正确性")
        return True
    except Exception as e:
        print(f"  [警告] ONNX 验证失败: {e}")
        return False


# ============================================================
#  TensorRT 引擎构建
# ============================================================

def build_trt_engine(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = True,
    workspace_gb: int = 4,
) -> bool:
    """
    从 ONNX 文件构建 TensorRT 序列化引擎。

    兼容 TRT 8.x / 9.x / 10.x。
    引擎固化了 GPU 型号和精度，换卡或换 TRT 版本需重新构建。
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("\n[TRT] tensorrt 未安装，跳过引擎构建")
        print("  安装方法: pip install tensorrt  (需要 CUDA 12.x)")
        print("  安装后重新运行本脚本即可生成 .engine 文件")
        return False

    ver_str = trt.__version__
    ver = tuple(int(x) for x in ver_str.split(".")[:2])
    print(f"\n[TRT] TensorRT 版本: {ver_str}")

    logger = trt.Logger(trt.Logger.WARNING)

    # 在创建 Builder 之前先检测 CUDA 初始化，捕获驱动版本不兼容
    try:
        builder_obj = trt.Builder(logger)
    except (TypeError, Exception) as e:
        print(f"\n[TRT] CUDA 初始化失败，无法创建 Builder: {e}")
        print("\n  原因: TensorRT 10.x 捆绑的 CUDA 运行时需要更新的 GPU 驱动。")
        print(f"  当前驱动: 537.58 (支持到 CUDA 12.2)")
        print(f"  TRT {ver_str} 需要: CUDA 12.4+ (驱动 ≥ 550.xx)")
        print("\n  解决方案 A (推荐): 升级 GPU 驱动至最新版本")
        print("    https://www.nvidia.com/drivers")
        print("\n  解决方案 B: 降级 TRT 至与 CUDA 12.2 兼容的版本")
        print("    pip install tensorrt==9.3.0.post12.dev5")
        print("    或: pip install tensorrt==8.6.1")
        print("\n  当前 ONNX 文件已生成，可先使用 ONNX Runtime 进行推理:")
        print("    pip install onnxruntime-gpu")
        print("    python infer_vessel_trt.py --dicom_dir <DICOM目录>")
        return False

    with builder_obj as builder, \
         builder.create_network(_trt_explicit_batch_flag(trt)) as network, \
         trt.OnnxParser(network, logger) as parser:

        config = builder.create_builder_config()
        _trt_set_workspace(trt, config, workspace_gb)

        # FP16 精度
        if fp16:
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("[TRT] FP16 精度已启用（速度约提升 2x）")
            else:
                print("[TRT] 当前 GPU 不支持快速 FP16，将使用 FP32")

        # 解析 ONNX
        print("[TRT] 解析 ONNX 模型...")
        with open(str(onnx_path), "rb") as f:
            raw = f.read()
        if not parser.parse(raw):
            for i in range(parser.num_errors):
                err = parser.get_error(i)
                print(f"  [ONNX 解析错误 {i}] {err}")
            return False
        print(f"[TRT]   输入: {network.get_input(0).shape}")
        print(f"[TRT]   输出: {network.get_output(0).shape}")

        # 构建（可能需要数分钟）
        print("[TRT] 开始编译引擎（首次约 2~5 分钟，请耐心等待）...")
        t0 = time.time()
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            print("[TRT] 引擎构建失败！请检查：")
            print("      1. ONNX 模型可以在 Netron 中正常打开")
            print("      2. 显存是否充足（当前 workspace_gb 设置为", workspace_gb, "GB）")
            return False

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(engine_path), "wb") as f:
            f.write(serialized)

        elapsed = time.time() - t0
        size_mb = engine_path.stat().st_size / 1024 / 1024
        print(f"[TRT] 引擎构建完成！耗时 {elapsed:.1f}s，大小 {size_mb:.1f} MB")
        print(f"[TRT] 保存至: {engine_path}")
        return True


# ============================================================
#  参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="导出 3D U-Net → ONNX → TensorRT 引擎",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_path", default="checkpoints/vessel_seg/best_model.pth",
                   help="PyTorch 检查点路径")
    p.add_argument("--output_dir", default="checkpoints/vessel_seg",
                   help="ONNX / .engine / trt_meta.json 保存目录")
    p.add_argument("--fp16", action="store_true", default=True,
                   help="启用 FP16 半精度（推荐）")
    p.add_argument("--no_fp16", dest="fp16", action="store_false",
                   help="禁用 FP16，使用 FP32")
    p.add_argument("--workspace_gb", type=int, default=4,
                   help="TRT 构建阶段最大显存（GB）")
    p.add_argument("--opset", type=int, default=17,
                   help="ONNX opset 版本")
    return p.parse_args()


# ============================================================
#  主程序
# ============================================================

def main():
    args = parse_args()
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"错误: 找不到模型文件 {model_path}")
        print("请先运行训练: python train_vessel_seg.py --data_root E:\\traindata")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("错误: CUDA 不可用。TensorRT 导出需要 GPU 环境")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")

    # ---- 加载检查点（始终在 CPU，ONNX 导出不需要 GPU）----
    print(f"\n加载检查点: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    meta       = ckpt.get("meta", {})
    features   = meta.get("features",    [16, 32, 64, 128])
    in_ch      = meta.get("in_channels",  1)
    out_ch     = meta.get("out_channels", 2)
    patch_size = tuple(meta.get("patch_size", [64, 128, 128]))
    hu_min     = meta.get("hu_min", -150.0)
    hu_max     = meta.get("hu_max",  550.0)

    model = UNet3D(in_channels=in_ch, out_channels=out_ch, features=features)
    model.load_state_dict(ckpt["model"])
    model.eval()  # 保持在 CPU；export_onnx 内部也在 CPU trace

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  UNet3D features={features}, 参数量={total_params:,}")
    print(f"  推理 patch: {patch_size}")
    print(f"  HU 范围:    [{hu_min}, {hu_max}]")

    # ============================================================
    #  步骤 1/2: 导出 ONNX
    # ============================================================
    print("\n" + "─" * 50)
    print("步骤 1/2: 导出 ONNX")
    print("─" * 50)
    onnx_path = output_dir / "vessel_seg.onnx"
    export_onnx(model, patch_size, onnx_path, opset=args.opset)
    verify_onnx(onnx_path, patch_size)

    # ============================================================
    #  步骤 2/2: 构建 TensorRT 引擎
    # ============================================================
    print("\n" + "─" * 50)
    print("步骤 2/2: 构建 TensorRT 引擎")
    print("─" * 50)
    suffix = "fp16" if args.fp16 else "fp32"
    engine_path = output_dir / f"vessel_seg_{suffix}.engine"

    trt_ok = build_trt_engine(
        onnx_path, engine_path,
        fp16=args.fp16,
        workspace_gb=args.workspace_gb,
    )

    # ---- 保存 TRT 元数据（infer_vessel_trt.py 读取）----
    trt_meta = {
        "onnx_path":    str(onnx_path),
        "engine_path":  str(engine_path) if trt_ok else "",
        "precision":    "fp16" if (args.fp16 and trt_ok) else "fp32",
        "patch_size":   list(patch_size),
        "features":     features,
        "in_channels":  in_ch,
        "out_channels": out_ch,
        "hu_min":       hu_min,
        "hu_max":       hu_max,
        "gpu":          gpu_name,
    }
    meta_path = output_dir / "trt_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(trt_meta, f, indent=2, ensure_ascii=False)

    # ---- 汇总 ----
    print("\n" + "=" * 60)
    print("导出完成！")
    print("=" * 60)
    print(f"  ONNX  : {onnx_path}")
    if trt_ok:
        print(f"  Engine: {engine_path}")
    else:
        print("  Engine: 未生成（TRT 未安装，将使用 ONNX Runtime 回退）")
    print(f"  元数据: {meta_path}")
    print()
    print("下一步 — 运行 TRT 推理:")
    print("  python infer_vessel_trt.py --dicom_dir <DICOM目录>")
    print()
    if not trt_ok:
        print("安装 TensorRT 后再次运行本脚本可生成 .engine 文件:")
        print("  pip install tensorrt")
    print("=" * 60)


if __name__ == "__main__":
    main()
