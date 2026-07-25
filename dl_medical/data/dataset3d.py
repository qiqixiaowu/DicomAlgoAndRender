"""
3D 配准数据集

数据集
======
1. RegistrationDataset3D       : 合成数据数据集 (支持肺/肝/通用)
2. NIfTIDataset3D               : NIfTI 格式真实医学数据
3. PairedNIfTIDataset3D         : 配对 NIfTI 数据 (fixed + moving)

NIfTI 加载
==========
- 使用 SimpleITK 或 nibabel 加载 .nii/.nii.gz
- 自动处理方向矩阵和原点
- 支持重采样到统一 spacing

工程要点
========
- 合成数据用于快速验证和调试
- NIfTI 数据集用于真实训练
- 支持半精度加载 (节省显存)
"""

from __future__ import annotations

import os
import glob
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from .synthetic3d import (
    SyntheticLungData3D,
    SyntheticLiverData3D,
    SyntheticRegData3D,
)
from .transforms3d import (
    Compose3D,
    get_default_transforms_3d,
    get_val_transforms_3d,
)


# ============================================================
#  合成数据数据集
# ============================================================

class RegistrationDataset3D(Dataset):
    """
    3D 合成配准数据集

    参数
    ----
    n_samples     : 样本数
    volume_size   : 体数据尺寸 (D, H, W)
    data_type     : "lung" | "liver" | "generic"
    transform     : 数据增强
    deform_sigma  : 弹性变形平滑度
    deform_alpha  : 弹性变形幅度
    """

    def __init__(
        self,
        n_samples: int = 100,
        volume_size: tuple[int, int, int] = (64, 64, 64),
        data_type: str = "lung",
        transform: Compose3D | None = None,
        deform_sigma: float = 8.0,
        deform_alpha: float = 12.0,
    ):
        self.n_samples = n_samples
        self.volume_size = volume_size
        self.data_type = data_type
        self.transform = transform

        if data_type == "lung":
            self.generator = SyntheticLungData3D(volume_size, deform_sigma, deform_alpha)
        elif data_type == "liver":
            self.generator = SyntheticLiverData3D(volume_size, deform_sigma, deform_alpha)
        else:
            self.generator = SyntheticRegData3D(volume_size, deform_sigma, deform_alpha)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        fixed, moving = self.generator.generate_one()

        if self.transform is not None:
            fixed, moving = self.transform(fixed, moving)

        return {
            "fixed": torch.from_numpy(fixed).unsqueeze(0),  # (1, D, H, W)
            "moving": torch.from_numpy(moving).unsqueeze(0),
        }


# ============================================================
#  NIfTI 数据集
# ============================================================

def load_nifti(
    filepath: str,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    加载 NIfTI 文件

    优先使用 SimpleITK (更好的方向处理), 回退到 nibabel。

    返回
    ----
    volume : (D, H, W) numpy array
    """
    try:
        import SimpleITK as sitk

        sitk_img = sitk.ReadImage(filepath)
        arr = sitk.GetArrayFromImage(sitk_img)  # (D, H, W) already
        return arr.astype(dtype)

    except ImportError:
        pass

    try:
        import nibabel as nib

        img = nib.load(filepath)
        arr = np.asanyarray(img.dataobj)
        # nibabel 默认 (X, Y, Z), 转为 (Z, Y, X) = (D, H, W)
        arr = np.transpose(arr, (2, 1, 0))
        return arr.astype(dtype)

    except ImportError:
        raise ImportError(
            "需要安装 SimpleITK 或 nibabel 来加载 NIfTI 文件:\n"
            "  pip install SimpleITK   (推荐)\n"
            "  pip install nibabel"
        )


def resample_volume(
    volume: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """
    将体数据重采样到目标尺寸 (三线性插值)

    参数
    ----
    volume       : (D, H, W)
    target_shape : (D', H', W')
    """
    if volume.shape == target_shape:
        return volume

    try:
        import SimpleITK as sitk

        img = sitk.GetImageFromArray(volume)
        original_size = img.GetSize()
        original_spacing = img.GetSpacing()

        new_spacing = [
            original_spacing[i] * (original_size[i] / target_shape[2 - i])
            for i in range(3)
        ]

        img.SetSpacing(new_spacing)
        resampled = sitk.Resample(
            img,
            target_shape[::-1],  # SimpleITK 使用 (W, H, D)
            sitk.Transform(),
            sitk.sitkLinear,
            img.GetOrigin(),
            new_spacing,
            img.GetDirection(),
            0,
            img.GetPixelID(),
        )
        return sitk.GetArrayFromImage(resampled).astype(np.float32)

    except ImportError:
        # 回退到 scipy
        from scipy.ndimage import zoom

        factors = [
            target_shape[i] / volume.shape[i]
            for i in range(3)
        ]
        return zoom(volume, factors, order=1).astype(np.float32)


class NIfTIDataset3D(Dataset):
    """
    单个 NIfTI 文件数据集

    从一个目录加载所有 .nii/.nii.gz 文件,
    每次随机选取两个作为 (fixed, moving) 对。
    适用于同一患者不同时相的配准。
    """

    def __init__(
        self,
        data_dir: str,
        volume_size: tuple[int, int, int] = (128, 128, 128),
        transform: Compose3D | None = None,
        file_pattern: str = "*.nii*",
    ):
        self.data_dir = data_dir
        self.volume_size = volume_size
        self.transform = transform

        self.files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"在 {data_dir} 中未找到匹配 '{file_pattern}' 的文件"
            )

        # 预加载所有数据到内存 (如果数据量不大)
        self.volumes = []
        for f in self.files:
            vol = load_nifti(f)
            vol = resample_volume(vol, volume_size)
            self.volumes.append(vol)

    def __len__(self) -> int:
        return len(self.volumes)

    def __getitem__(self, idx: int) -> dict:
        fixed = self.volumes[idx].copy()
        # 随机选择 moving
        moving_idx = np.random.randint(0, len(self.volumes))
        moving = self.volumes[moving_idx].copy()

        if self.transform is not None:
            fixed, moving = self.transform(fixed, moving)

        return {
            "fixed": torch.from_numpy(fixed).unsqueeze(0),
            "moving": torch.from_numpy(moving).unsqueeze(0),
            "fixed_path": self.files[idx],
            "moving_path": self.files[moving_idx],
        }


class PairedNIfTIDataset3D(Dataset):
    """
    配对 NIfTI 数据集

    从两个目录 (fixed_dir, moving_dir) 加载配对数据,
    文件名需一一对应。

    适用于:
    - 同一患者不同时相 (inhale/exhale)
    - 不同模态配准 (CT/MRI)
    """

    def __init__(
        self,
        fixed_dir: str,
        moving_dir: str,
        volume_size: tuple[int, int, int] = (128, 128, 128),
        transform: Compose3D | None = None,
        file_pattern: str = "*.nii*",
    ):
        self.fixed_dir = fixed_dir
        self.moving_dir = moving_dir
        self.volume_size = volume_size
        self.transform = transform

        self.fixed_files = sorted(glob.glob(os.path.join(fixed_dir, file_pattern)))
        self.moving_files = sorted(glob.glob(os.path.join(moving_dir, file_pattern)))

        if len(self.fixed_files) == 0:
            raise FileNotFoundError(f"在 {fixed_dir} 中未找到文件")
        if len(self.moving_files) == 0:
            raise FileNotFoundError(f"在 {moving_dir} 中未找到文件")
        if len(self.fixed_files) != len(self.moving_files):
            raise ValueError(
                f"Fixed ({len(self.fixed_files)}) 和 Moving ({len(self.moving_files)}) "
                f"文件数不匹配"
            )

    def __len__(self) -> int:
        return len(self.fixed_files)

    def __getitem__(self, idx: int) -> dict:
        fixed = load_nifti(self.fixed_files[idx])
        moving = load_nifti(self.moving_files[idx])

        fixed = resample_volume(fixed, self.volume_size)
        moving = resample_volume(moving, self.volume_size)

        if self.transform is not None:
            fixed, moving = self.transform(fixed, moving)

        return {
            "fixed": torch.from_numpy(fixed).unsqueeze(0),
            "moving": torch.from_numpy(moving).unsqueeze(0),
            "fixed_path": self.fixed_files[idx],
            "moving_path": self.moving_files[idx],
        }


# ============================================================
#  数据集构建工厂
# ============================================================

def build_dataset_3d(
    config: dict,
    mode: str = "train",
) -> Dataset:
    """
    根据配置构建 3D 数据集

    配置格式 (config["data"]):
    -------------------------
    source: "synthetic" | "nifti" | "paired_nifti"
    data_type: "lung" | "liver" | "generic"  (synthetic 模式)
    data_dir: path                            (nifti 模式)
    fixed_dir / moving_dir: path              (paired_nifti 模式)
    volume_size: [D, H, W]
    n_samples: int                            (synthetic 模式)
    normalize: "percentile" | "zscore" | "minmax"
    augment: bool
    """
    data_cfg = config["data"]
    volume_size = tuple(data_cfg.get("volume_size", [64, 64, 64]))

    if mode == "train":
        transform = get_default_transforms_3d(
            normalize=data_cfg.get("normalize", "percentile")
        ) if data_cfg.get("augment", True) else get_val_transforms_3d(
            normalize=data_cfg.get("normalize", "percentile")
        )
    else:
        transform = get_val_transforms_3d(
            normalize=data_cfg.get("normalize", "percentile")
        )

    source = data_cfg.get("source", "synthetic")

    if source == "synthetic":
        return RegistrationDataset3D(
            n_samples=data_cfg.get("n_samples", 100),
            volume_size=volume_size,
            data_type=data_cfg.get("data_type", "lung"),
            transform=transform,
            deform_sigma=data_cfg.get("deform_sigma", 8.0),
            deform_alpha=data_cfg.get("deform_alpha", 12.0),
        )

    elif source == "nifti":
        return NIfTIDataset3D(
            data_dir=data_cfg["data_dir"],
            volume_size=volume_size,
            transform=transform,
        )

    elif source == "paired_nifti":
        return PairedNIfTIDataset3D(
            fixed_dir=data_cfg["fixed_dir"],
            moving_dir=data_cfg["moving_dir"],
            volume_size=volume_size,
            transform=transform,
        )

    else:
        raise ValueError(f"未知数据源: {source}")


if __name__ == "__main__":
    print("测试 3D 数据集...")

    # 合成数据
    print("\n--- 合成肺部数据 ---")
    ds = RegistrationDataset3D(
        n_samples=5,
        volume_size=(64, 64, 64),
        data_type="lung",
        transform=get_default_transforms_3d(),
    )
    sample = ds[0]
    print(f"  Fixed:  {sample['fixed'].shape}  range: [{sample['fixed'].min():.3f}, {sample['fixed'].max():.3f}]")
    print(f"  Moving: {sample['moving'].shape} range: [{sample['moving'].min():.3f}, {sample['moving'].max():.3f}]")

    print("\n--- 合成肝部数据 ---")
    ds = RegistrationDataset3D(
        n_samples=5,
        volume_size=(64, 64, 64),
        data_type="liver",
        transform=get_default_transforms_3d(),
    )
    sample = ds[0]
    print(f"  Fixed:  {sample['fixed'].shape}  range: [{sample['fixed'].min():.3f}, {sample['fixed'].max():.3f}]")

    print("\n数据集模块测试完成!")
