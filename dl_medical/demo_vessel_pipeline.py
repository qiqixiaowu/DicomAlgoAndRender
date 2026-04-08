"""
血管分割完整流水线演示
=====================

一键运行从「数据扫描」到「模型训练」到「推理导出」的全流程，
并在最后输出供 C++ 渲染器直接使用的命令。

使用方法
--------
  # 步骤 0：先检查你的数据能否被正确识别（不需要 GPU 也不需要安装深度学习库）
  python demo_vessel_pipeline.py --mode scan --data_root E:\\traindata

  # 步骤 1：训练
  python demo_vessel_pipeline.py --mode train --data_root E:\\traindata

  # 步骤 2：对新的 CT 序列推理
  python demo_vessel_pipeline.py --mode infer --dicom_dir E:\\PatientData\\CTA_001

  # 一次性运行扫描 + 训练 + 推理
  python demo_vessel_pipeline.py --mode all \\
      --data_root E:\\traindata \\
      --dicom_dir E:\\PatientData\\CTA_001

数据目录结构（你现在的结构，直接支持）
--------------------------------------
  E:\\traindata\\
  ├── images\\
  │   ├── case001\\      ← 每个 case 的 DICOM 序列目录
  │   │   ├── 0001.dcm
  │   │   └── ...
  │   └── case002\\
  └── labels\\
      ├── case001.nii.gz  ← 二值掩码（0=背景, 1=血管）
      └── case002.nii.gz

  若 images\\ 里直接是 .nii.gz 文件也支持：
  E:\\traindata\\
  ├── images\\
  │   ├── case001.nii.gz
  │   └── case002.nii.gz
  └── labels\\
      ├── case001.nii.gz
      └── case002.nii.gz
"""

import argparse
import subprocess
import sys
from pathlib import Path

# 确保可以 import 同级包
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ============================================================
#  参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="血管分割完整流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["scan", "train", "infer", "all"], default="scan",
        help="运行模式: scan=扫描数据 | train=训练 | infer=推理 | all=全部"
    )
    parser.add_argument("--data_root", type=str, default=r"E:\traindata",
                        help="训练数据根目录（包含 images/ 和 labels/）")
    parser.add_argument("--dicom_dir", type=str, default=None,
                        help="推理时使用的 DICOM 序列目录")
    parser.add_argument("--model_path", type=str,
                        default="checkpoints/vessel_seg/best_model.pth",
                        help="推理使用的模型路径")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/vessel_result",
                        help="推理结果输出目录")
    # 训练超参数（透传给 train_vessel_seg.py）
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--features",   type=int, nargs="+",
                        default=[16, 32, 64, 128],
                        help="U-Net 通道数（显存不足可改为 16 32 64）")
    parser.add_argument("--max_volumes", type=int, default=None,
                        help="调试时只用前 N 个 case")
    return parser.parse_args()


# ============================================================
#  模式 0：扫描数据，验证自动配对
# ============================================================

def mode_scan(data_root: str):
    print("=" * 65)
    print(f"  扫描训练数据: {data_root}")
    print("=" * 65)

    from data.vessel_dataset import scan_traindata
    pairs = scan_traindata(data_root)

    if not pairs:
        print("\n[错误] 未能找到任何 image-label 对！")
        print("请检查目录结构：")
        print(f"  {data_root}/images/  ← 存放 DICOM 序列目录 或 .nii.gz 文件")
        print(f"  {data_root}/labels/  ← 存放对应的 .nii.gz 标注文件")
        print("\n标注文件命名规则：与图像目录名（或文件名）相同，后缀 .nii.gz")
        print("例如: images/case001/ → labels/case001.nii.gz")
        return []

    print(f"\n共找到 {len(pairs)} 个 case：\n")
    print(f"  {'编号':>4}  {'case 名称':<30}  {'类型':<6}  标注文件")
    print(f"  {'-'*4}  {'-'*30}  {'-'*6}  {'-'*30}")
    for i, p in enumerate(pairs, 1):
        lbl_name = Path(p["label_path"]).name
        print(f"  [{i:3d}]  {p['case_name']:<30}  {p['image_type']:<6}  {lbl_name}")

    print(f"\n数据格式：{pairs[0]['image_type'].upper()}"
          f"{'（DICOM 序列）' if pairs[0]['image_type']=='dicom' else '（NIfTI 体数据）'}")
    print("\n✓ 数据识别成功！运行训练命令：")
    print(f"  python train_vessel_seg.py --data_root \"{data_root}\"")
    return pairs


# ============================================================
#  模式 1：训练
# ============================================================

def mode_train(args: argparse.Namespace):
    print("=" * 65)
    print("  开始训练血管分割模型（3D U-Net）")
    print("=" * 65)

    # 先扫描，确认数据可用
    from data.vessel_dataset import scan_traindata
    pairs = scan_traindata(args.data_root)
    if not pairs:
        print(f"\n[错误] 在 {args.data_root} 中未找到训练数据，终止。")
        return False

    print(f"\n将使用 {len(pairs)} 个 case 训练，epochs={args.epochs}，"
          f"batch_size={args.batch_size}")
    print(f"U-Net 通道数: {args.features}")
    print(f"模型将保存到: checkpoints/vessel_seg/best_model.pth\n")

    # 直接调用训练模块（与 train_vessel_seg.py 共用同一套代码）
    from models.unet3d import UNet3D
    from models.losses import DiceCELoss
    from utils.metrics import dice_coefficient
    from data.vessel_dataset import VesselDataset3D
    import torch
    from torch.utils.data import DataLoader, random_split
    import json
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {'GPU (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else 'CPU'}")

    patch_size = (64, 128, 128)
    save_dir = Path("checkpoints/vessel_seg")
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = VesselDataset3D(
        traindata_root=args.data_root,
        patch_size=patch_size,
        samples_per_volume=40,
        pos_fraction=0.6,
        augment=True,
        max_volumes=args.max_volumes,
    )
    total = len(dataset)
    val_size = max(1, int(total * 0.15))
    train_ds, val_ds = random_split(dataset, [total - val_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=0)

    model = UNet3D(in_channels=1, out_channels=2, features=args.features).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params/1e6:.2f} M\n")

    criterion = DiceCELoss(dice_weight=0.7, ce_weight=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    meta = {
        "model": "UNet3D",
        "in_channels": 1, "out_channels": 2,
        "features": args.features,
        "patch_size": list(patch_size),
        "classes": ["background", "vessel"],
        "hu_min": -150.0, "hu_max": 550.0,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(save_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    best_dice = 0.0
    for epoch in range(args.epochs):
        # --- 训练 ---
        model.train()
        tr_loss = tr_dice = n = 0
        for imgs, masks in train_loader:
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                with torch.amp.autocast("cuda"):
                    out  = model(imgs)
                    loss = criterion(out, masks)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                out  = model(imgs)
                loss = criterion(out, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            with torch.no_grad():
                d = dice_coefficient(out.argmax(1).cpu().numpy(),
                                     masks.cpu().numpy(), num_classes=2)["mean"]
            tr_loss += loss.item(); tr_dice += d; n += 1
        scheduler.step()

        # --- 验证 ---
        model.eval()
        vl_loss = vl_dice = vn = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs  = imgs.to(device)
                masks = masks.to(device)
                if scaler:
                    with torch.amp.autocast("cuda"):
                        out  = model(imgs)
                        loss = criterion(out, masks)
                else:
                    out  = model(imgs)
                    loss = criterion(out, masks)
                d = dice_coefficient(out.argmax(1).cpu().numpy(),
                                     masks.cpu().numpy(), num_classes=2)["mean"]
                vl_loss += loss.item(); vl_dice += d; vn += 1

        avg_tr = (tr_loss/max(n,1), tr_dice/max(n,1))
        avg_vl = (vl_loss/max(vn,1), vl_dice/max(vn,1))
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[{epoch+1:3d}/{args.epochs}] "
              f"train loss={avg_tr[0]:.4f} dice={avg_tr[1]:.4f} | "
              f"val  loss={avg_vl[0]:.4f} dice={avg_vl[1]:.4f} | "
              f"lr={lr_now:.1e}", flush=True)

        if avg_vl[1] > best_dice:
            best_dice = avg_vl[1]
            ckpt = {"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_dice": best_dice, "meta": meta}
            torch.save(ckpt, save_dir / "best_model.pth")
            print(f"  → 保存最佳模型 val_dice={best_dice:.4f}")

    print(f"\n训练完成！最佳验证 Dice = {best_dice:.4f}")
    print(f"模型路径: {(save_dir / 'best_model.pth').resolve()}")
    return True


# ============================================================
#  模式 2：推理
# ============================================================

def mode_infer(dicom_dir: str, model_path: str, output_dir: str):
    print("=" * 65)
    print(f"  推理: {dicom_dir}")
    print("=" * 65)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"\n[错误] 找不到模型文件: {model_path}")
        print("请先运行训练: python demo_vessel_pipeline.py --mode train ...")
        return None, None

    import torch, json, time, numpy as np
    from models.unet3d import UNet3D
    from data.vessel_dataset import _load_dicom_series

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"推理设备: {device}")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    meta = ckpt.get("meta", {})
    features   = meta.get("features", [16, 32, 64, 128])
    patch_size = tuple(meta.get("patch_size", [64, 128, 128]))
    hu_min, hu_max = meta.get("hu_min", -150.0), meta.get("hu_max", 550.0)

    model = UNet3D(1, 2, features=features).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"模型加载完成 (features={features}, patch={patch_size})")

    print(f"\n读取 DICOM: {dicom_dir}")
    volume_hu = _load_dicom_series(dicom_dir)
    D, H, W = volume_hu.shape
    print(f"体数据尺寸: D={D} H={H} W={W}")

    volume = np.clip(volume_hu, hu_min, hu_max)
    volume = ((volume - hu_min) / (hu_max - hu_min)).astype(np.float32)

    # 滑动窗口推理
    from infer_vessel_seg import sliding_window_inference, export_mask_raw, visualize_result
    print(f"\n执行滑动窗口推理 (overlap=0.5)...")
    t0 = time.time()
    prob_map = sliding_window_inference(model, volume, patch_size, 0.5, device)
    print(f"推理耗时: {time.time()-t0:.1f}s")

    binary_mask = (prob_map >= 0.5).astype(np.uint8)
    print(f"血管体素: {binary_mask.sum():,} ({100*binary_mask.sum()/binary_mask.size:.2f}%)")

    out_dir = Path(output_dir)
    raw_path, meta_path = export_mask_raw(binary_mask, out_dir, (D, H, W))
    visualize_result(volume, binary_mask, out_dir)

    return str(raw_path), str(meta_path)


# ============================================================
#  输出 C++ 使用命令
# ============================================================

def print_cpp_usage(dicom_dir: str, raw_path: str):
    exe = Path("x64/Debug/VolumeRenderOptimized/VolumeRenderOptimized.exe").resolve()
    raw = Path(raw_path).resolve()

    print("\n" + "=" * 65)
    print("  在 C++ OpenGL 渲染器中加载分割结果")
    print("=" * 65)
    print("\n方法 1 — 命令行直接启动（推荐）：")
    print(f"""
  {exe} \\
    "{dicom_dir}" \\
    "{raw}"
""")
    print("启动后：")
    print("  • 血管自动以红色叠加在 CT 体绘制上")
    print("  • F1       — 切换血管叠加显示/隐藏")
    print("  • F5       — 切换血管边界线高亮")
    print("  • 鼠标右键 — 仍可用区域生长做对比")
    print()
    print("方法 2 — 在代码中加载（main_optimized_example.cpp 主函数里）：")
    print(f"""
    // DICOM 加载完成后，插入以下代码:
    auto dlMask = loadDLVesselMask(
        "{raw.as_posix()}",
        "{raw.with_suffix('').with_suffix('').as_posix()}_meta.json"
    );
    if (!dlMask.mask.empty()) {{
        g_segMaskTexture = createOrUpdateSegMaskTexture(dlMask, g_segMaskTexture);
        renderState.enableSegmentation = true;
        renderState.segColor   = glm::vec3(1.0f, 0.15f, 0.1f);
        renderState.segOpacity = 0.85f;
    }}
""")


# ============================================================
#  主程序
# ============================================================

def main():
    args = parse_args()

    if args.mode == "scan":
        mode_scan(args.data_root)

    elif args.mode == "train":
        mode_train(args)

    elif args.mode == "infer":
        if not args.dicom_dir:
            print("[错误] --infer 模式需要指定 --dicom_dir")
            sys.exit(1)
        raw_path, _ = mode_infer(args.dicom_dir, args.model_path, args.output_dir)
        if raw_path:
            print_cpp_usage(args.dicom_dir, raw_path)

    elif args.mode == "all":
        # 扫描
        pairs = mode_scan(args.data_root)
        if not pairs:
            sys.exit(1)

        # 训练
        ok = mode_train(args)
        if not ok:
            sys.exit(1)

        # 推理（如果提供了 dicom_dir）
        if args.dicom_dir:
            raw_path, _ = mode_infer(args.dicom_dir, args.model_path, args.output_dir)
            if raw_path:
                print_cpp_usage(args.dicom_dir, raw_path)
        else:
            print("\n提示: 训练完成后，运行推理命令：")
            print(f"  python demo_vessel_pipeline.py --mode infer "
                  f"--dicom_dir <CT序列目录>")


if __name__ == "__main__":
    main()
