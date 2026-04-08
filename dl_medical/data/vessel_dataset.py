"""
血管分割数据集

支持的 E:\\traindata 目录结构
============================

格式 A — images + labels 子目录（DICOM序列 + NIfTI标注）:
    traindata/
    ├── images/
    │   ├── case001/   ← DICOM 序列目录
    │   │   ├── 0001.dcm
    │   │   └── ...
    │   └── case002/
    └── labels/
        ├── case001.nii.gz   ← 二值掩码 (0=背景, 1=血管)
        └── case002.nii.gz

格式 B — 每个 case 一个子目录（NIfTI image + NIfTI label）:
    traindata/
    └── case001/
        ├── image.nii.gz
        └── label.nii.gz

格式 C — images + labels 子目录（NIfTI image + NIfTI label）:
    traindata/
    ├── images/
    │   ├── case001.nii.gz
    │   └── case002.nii.gz
    └── labels/
        ├── case001.nii.gz
        └── case002.nii.gz

格式 D — 每个 case 包含 DICOM 序列目录 + label 文件:
    traindata/
    └── case001/
        ├── images/  (或 image/ 或 dicom/ 直接含.dcm)
        │   ├── 0001.dcm
        │   └── ...
        └── label.nii.gz

WVesselDataset3D:
- 随机裁剪 patch_size 大小的 3D patch
- 保证正样本比例 (pos_fraction): 多次尝试裁剪到包含血管体素的区域
- 数据增强: 随机翻转、随机缩放、高斯噪声

VesselInferenceVolume:
- 返回完整的归一化体数据 (用于推理)
"""

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


# ============================================================
#  CT 预处理 (血管专用窗宽窗位)
# ============================================================

# 血管 CTA 常用窗口: 窗位 200 HU, 窗宽 700 HU => [-150, 550]
VESSEL_HU_MIN = -150.0
VESSEL_HU_MAX = 550.0


def normalize_vessel_ct(volume_hu: np.ndarray) -> np.ndarray:
    """
    将 HU 值归一化到 [0, 1] (血管窗)
    volume_hu: float32, 单位 HU
    """
    v = np.clip(volume_hu, VESSEL_HU_MIN, VESSEL_HU_MAX)
    v = (v - VESSEL_HU_MIN) / (VESSEL_HU_MAX - VESSEL_HU_MIN)
    return v.astype(np.float32)


# ============================================================
#  读取辅助函数
# ============================================================

def _load_nifti(path: str) -> np.ndarray:
    """读取 NIfTI 文件, 返回 (D, H, W) float32"""
    if not HAS_SITK:
        raise RuntimeError("需要安装 SimpleITK: pip install SimpleITK")
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (D, H, W)
    return arr


def _load_dicom_series(directory: str) -> np.ndarray:
    """读取 DICOM 序列目录, 返回 (D, H, W) float32 HU 值"""
    if not HAS_SITK:
        raise RuntimeError("需要安装 SimpleITK: pip install SimpleITK")
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(directory))
    if not dicom_names:
        raise ValueError(f"目录中没有 DICOM 文件: {directory}")
    reader.SetFileNames(dicom_names)
    img = reader.Execute()
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (D, H, W)
    return arr


def _is_dicom_dir(path: Path) -> bool:
    """判断目录是否包含 DICOM 文件"""
    exts = {".dcm", ".ima", ".dicom"}
    for f in path.iterdir():
        if f.is_file() and (f.suffix.lower() in exts or _check_dicom_magic(f)):
            return True
    return False


def _check_dicom_magic(filepath: Path) -> bool:
    try:
        with open(filepath, "rb") as fh:
            fh.seek(128)
            return fh.read(4) == b"DICM"
    except (OSError, IOError):
        return False


def _find_dicom_subdir(case_dir: Path) -> Optional[Path]:
    """在 case 目录下寻找包含 DICOM 的子目录"""
    # 直接在 case_dir 里
    if _is_dicom_dir(case_dir):
        return case_dir
    # 常见子目录名
    for name in ("images", "image", "dicom", "DICOM", "CT", "ct"):
        sub = case_dir / name
        if sub.is_dir() and _is_dicom_dir(sub):
            return sub
    # 任意子目录
    for sub in case_dir.iterdir():
        if sub.is_dir() and _is_dicom_dir(sub):
            return sub
    return None


# ============================================================
#  自动检测 traindata 目录格式
# ============================================================

def scan_traindata(root: str) -> list[dict]:
    """
    扫描 traindata 根目录, 自动匹配 image-label 对。

    返回:
        list of {"image_path": ..., "label_path": ..., "image_type": "nifti"|"dicom"}
    """
    root = Path(root)
    pairs: list[dict] = []

    images_dir = root / "images"
    labels_dir = root / "labels"

    # --- 格式 A / C: images + labels 子目录 ---
    if images_dir.is_dir() and labels_dir.is_dir():
        # 格式 A: images/caseXXX/ (DICOM 序列目录) + labels/caseXXX.nii.gz
        for case_dir in sorted(images_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            case_name = case_dir.name
            if _is_dicom_dir(case_dir):
                for label_file in _find_label_files(labels_dir, case_name):
                    pairs.append({
                        "image_path": str(case_dir),
                        "label_path": str(label_file),
                        "image_type": "dicom",
                        "case_name": case_name,
                    })

        # 格式 C: images/*.nii.gz + labels/*.nii.gz（NIfTI 文件直接放在 images/ 下）
        # 支持 nnU-Net 风格: topcow_mr_001_0000.nii.gz → labels/topcow_mr_001.nii.gz
        nifti_exts = {".nii", ".gz"}
        for img_file in sorted(images_dir.iterdir()):
            if img_file.is_file() and img_file.suffix.lower() in nifti_exts:
                raw_stem  = _stem(img_file)                  # e.g. topcow_mr_001_0000
                case_name = _strip_channel_suffix(raw_stem)  # e.g. topcow_mr_001
                for label_file in _find_label_files(labels_dir, raw_stem):
                    # 去重：同一个 case（剥离通道后缀后相同）只加一次
                    if not any(p["case_name"] == case_name for p in pairs):
                        pairs.append({
                            "image_path": str(img_file),
                            "label_path": str(label_file),
                            "image_type": "nifti",
                            "case_name": case_name,
                        })

    # --- 格式 B / D: 每个子目录一个 case ---
    if not pairs:
        for case_dir in sorted(root.iterdir()):
            if not case_dir.is_dir():
                continue
            # 寻找 label 文件
            label_file = _find_label_in_case(case_dir)
            if label_file is None:
                continue
            # 寻找 image
            dicom_sub = _find_dicom_subdir(case_dir)
            nifti_img = _find_nifti_image(case_dir)
            if dicom_sub:
                pairs.append({
                    "image_path": str(dicom_sub),
                    "label_path": str(label_file),
                    "image_type": "dicom",
                    "case_name": case_dir.name,
                })
            elif nifti_img:
                pairs.append({
                    "image_path": str(nifti_img),
                    "label_path": str(label_file),
                    "image_type": "nifti",
                    "case_name": case_dir.name,
                })

    return pairs


def _stem(p: Path) -> str:
    """获取不含扩展名的文件名, .nii.gz 去掉两层"""
    name = p.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return p.stem


def _strip_channel_suffix(stem: str) -> str:
    """
    剥离 nnU-Net 风格的通道后缀，返回 case 名。

    例如:
      topcow_mr_001_0000  →  topcow_mr_001
      case001_0000        →  case001
      case001             →  case001  (无后缀时原样返回)

    规则: 若末尾为 _XXXX（四位数字），则去掉该部分。
    """
    import re
    return re.sub(r'_\d{4}$', '', stem)


def _find_label_files(labels_dir: Path, case_name: str) -> list[Path]:
    """
    在 labels_dir 中找 case_name 对应的标注文件。

    同时尝试剥离 nnU-Net 通道后缀（_0000）后的 case 名，
    以支持 images/topcow_mr_001_0000.nii.gz → labels/topcow_mr_001.nii.gz 的配对。
    """
    # 候选 case 名：原始名 + 剥离通道后缀后的名
    stripped = _strip_channel_suffix(case_name)
    lookup_names = [case_name] if case_name == stripped else [case_name, stripped]

    found: list[Path] = []
    for name in lookup_names:
        candidates = [
            labels_dir / f"{name}.nii.gz",
            labels_dir / f"{name}.nii",
            labels_dir / f"{name}_label.nii.gz",
            labels_dir / f"{name}_mask.nii.gz",
            labels_dir / name / "label.nii.gz",
        ]
        for c in candidates:
            if c.exists() and c not in found:
                found.append(c)
    return found


def _find_label_in_case(case_dir: Path) -> Optional[Path]:
    """在 case 目录下寻找标注文件"""
    candidates = ["label.nii.gz", "mask.nii.gz", "seg.nii.gz",
                  "label.nii", "mask.nii", "annotation.nii.gz"]
    for name in candidates:
        f = case_dir / name
        if f.exists():
            return f
    # 也可能直接在目录里
    for f in case_dir.iterdir():
        if f.is_file() and ("label" in f.name.lower() or "mask" in f.name.lower()
                            or "seg" in f.name.lower()):
            if f.suffix.lower() in {".gz", ".nii"}:
                return f
    return None


def _find_nifti_image(case_dir: Path) -> Optional[Path]:
    """在 case 目录下寻找图像 NIfTI 文件"""
    candidates = ["image.nii.gz", "img.nii.gz", "ct.nii.gz",
                  "image.nii", "img.nii"]
    for name in candidates:
        f = case_dir / name
        if f.exists():
            return f
    for f in case_dir.iterdir():
        if f.is_file() and f.suffix.lower() in {".gz", ".nii"}:
            if not any(kw in f.name.lower() for kw in ("label", "mask", "seg")):
                return f
    return None


# ============================================================
#  体数据缓存: 加载 + 归一化
# ============================================================

def load_pair(pair: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    加载一对 (image, label)。
    image: 归一化 float32 (D, H, W) in [0,1]
    label: uint8 (D, H, W), 0=背景, 1=血管
    """
    # 读取图像
    if pair["image_type"] == "dicom":
        volume_hu = _load_dicom_series(pair["image_path"])
    else:
        volume_raw = _load_nifti(pair["image_path"])
        # 尝试判断是否已经是 HU（绝对值 > 10 认为是原始 HU）
        if np.abs(volume_raw).max() > 10.0:
            volume_hu = volume_raw
        else:
            # 已归一化，反推（粗略）
            volume_hu = volume_raw * (VESSEL_HU_MAX - VESSEL_HU_MIN) + VESSEL_HU_MIN

    image = normalize_vessel_ct(volume_hu)  # (D, H, W) float32 [0, 1]

    # 读取标注
    label_raw = _load_nifti(pair["label_path"])
    label = (label_raw > 0.5).astype(np.uint8)  # 二值化

    # 统一形状（确保 image 与 label 空间维度一致）
    if image.shape != label.shape:
        # 尝试简单 resize（最近邻）
        from scipy.ndimage import zoom
        scale = [s1 / s2 for s1, s2 in zip(image.shape, label.shape)]
        label = zoom(label, scale, order=0).astype(np.uint8)

    return image, label


# ============================================================
#  训练数据集: 随机 Patch
# ============================================================

class VesselDataset3D(Dataset):
    """
    3D 血管分割 Patch 数据集

    参数
    ----
    traindata_root : E:\\traindata 路径
    patch_size     : (D, H, W) 随机裁剪大小
    samples_per_volume : 每个体数据随机采样多少个 patch
    pos_fraction   : 正样本（含血管）patch 的比例 (0~1)
    augment        : 是否开启数据增强

    用法
    ----
    >>> ds = VesselDataset3D("E:/traindata", patch_size=(64, 128, 128))
    >>> img, mask = ds[0]
    >>> # img:  Tensor (1, 64, 128, 128) float
    >>> # mask: Tensor (64, 128, 128) long
    """

    def __init__(
        self,
        traindata_root: str,
        patch_size: tuple[int, int, int] = (64, 128, 128),
        samples_per_volume: int = 50,
        pos_fraction: float = 0.5,
        augment: bool = True,
        max_volumes: Optional[int] = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.pos_fraction = pos_fraction
        self.augment = augment

        # 扫描数据集
        pairs = scan_traindata(traindata_root)
        if not pairs:
            raise ValueError(
                f"在 {traindata_root} 中没有找到有效的 image-label 对。\n"
                "请检查目录结构（见 vessel_dataset.py 文件头部说明）。"
            )
        if max_volumes:
            pairs = pairs[:max_volumes]

        print(f"[VesselDataset] 找到 {len(pairs)} 个 case，加载中...")

        # 预加载所有体数据（内存允许时）
        self.volumes: list[tuple[np.ndarray, np.ndarray]] = []
        for i, pair in enumerate(pairs):
            try:
                img, lbl = load_pair(pair)
                self.volumes.append((img, lbl))
                vessel_voxels = int(lbl.sum())
                print(f"  [{i+1}/{len(pairs)}] {pair['case_name']}: "
                      f"shape={img.shape}, 血管体素={vessel_voxels:,}")
            except Exception as e:
                print(f"  [警告] 加载 {pair['case_name']} 失败: {e}")

        if not self.volumes:
            raise RuntimeError("所有数据加载失败，请检查数据格式和依赖库安装。")

        print(f"[VesselDataset] 成功加载 {len(self.volumes)} 个体数据，"
              f"总 patch 数 = {len(self)}")

    def __len__(self) -> int:
        return len(self.volumes) * self.samples_per_volume

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        vol_idx = idx // self.samples_per_volume
        image, label = self.volumes[vol_idx]

        # 随机 patch（带正样本偏好）
        patch_img, patch_lbl = self._random_patch(image, label)

        # 数据增强
        if self.augment:
            patch_img, patch_lbl = self._augment(patch_img, patch_lbl)

        img_t = torch.from_numpy(patch_img).unsqueeze(0).float()   # (1, D, H, W)
        lbl_t = torch.from_numpy(patch_lbl.astype(np.int64)).long()  # (D, H, W)
        return img_t, lbl_t

    def _random_patch(
        self, image: np.ndarray, label: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        D, H, W = image.shape
        pd, ph, pw = self.patch_size

        # 有 pos_fraction 概率尝试裁到包含血管的区域
        use_pos = random.random() < self.pos_fraction and label.any()

        if use_pos:
            vessel_z, vessel_y, vessel_x = np.where(label > 0)
            # 随机选一个血管体素作为中心
            idx = random.randrange(len(vessel_z))
            cz, cy, cx = int(vessel_z[idx]), int(vessel_y[idx]), int(vessel_x[idx])
            # 以该点为中心确定起点（并clamp到合法范围）
            z0 = int(np.clip(cz - pd // 2, 0, max(0, D - pd)))
            y0 = int(np.clip(cy - ph // 2, 0, max(0, H - ph)))
            x0 = int(np.clip(cx - pw // 2, 0, max(0, W - pw)))
        else:
            z0 = random.randint(0, max(0, D - pd))
            y0 = random.randint(0, max(0, H - ph))
            x0 = random.randint(0, max(0, W - pw))

        patch_img = image[z0:z0+pd, y0:y0+ph, x0:x0+pw]
        patch_lbl = label[z0:z0+pd, y0:y0+ph, x0:x0+pw]

        # Pad 如果体数据小于 patch_size
        if patch_img.shape != self.patch_size:
            patch_img = _pad_to(patch_img, self.patch_size)
            patch_lbl = _pad_to(patch_lbl, self.patch_size)

        return patch_img.copy(), patch_lbl.copy()

    def _augment(
        self, image: np.ndarray, label: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # 随机翻转（三个轴）
        for axis in range(3):
            if random.random() < 0.5:
                image = np.flip(image, axis=axis)
                label = np.flip(label, axis=axis)

        # 随机高斯噪声
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)

        # 随机亮度调整
        if random.random() < 0.3:
            gamma = random.uniform(0.8, 1.2)
            image = np.power(np.clip(image, 1e-6, 1.0), gamma)

        return image, label


def _pad_to(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """将数组 zero-pad 到 target_shape（每个维度右侧补零）"""
    pad_width = [(0, max(0, t - s)) for s, t in zip(arr.shape, target_shape)]
    return np.pad(arr, pad_width, mode="constant", constant_values=0)


# ============================================================
#  推理: 加载完整体数据
# ============================================================

class VesselInferenceVolume:
    """
    加载单个 DICOM 序列用于推理。

    返回归一化的 (D, H, W) float32 数组。
    """

    def __init__(self, dicom_dir: str):
        self.dicom_dir = dicom_dir
        volume_hu = _load_dicom_series(dicom_dir)
        self.volume = normalize_vessel_ct(volume_hu)
        self.shape = self.volume.shape  # (D, H, W)
        print(f"[VesselInference] 加载体数据: shape={self.shape}, "
              f"HU范围=[{VESSEL_HU_MIN}, {VESSEL_HU_MAX}]")

    def get_tensor(self) -> torch.Tensor:
        """返回 (1, 1, D, H, W) 张量（batch=1, channel=1）"""
        return torch.from_numpy(self.volume).unsqueeze(0).unsqueeze(0).float()
