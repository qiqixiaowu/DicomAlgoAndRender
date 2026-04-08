"""
基于真实 DICOM 数据的 PyTorch Dataset

功能
====
1. DicomSliceDataset   — 从多个 DICOM 序列提取 2D 切片, 用于分割/预训练
2. DicomPairDataset    — 从多个序列中构造 (fixed, moving) 对, 用于配准训练
3. DicomAutoSegDataset — 自动生成伪标签 (阈值分割), 用于无标注的分割训练入门

使用方式
========
>>> from data.dicom_dataset import DicomSliceDataset
>>> ds = DicomSliceDataset(
...     dicom_dirs=["E:/Data/AC_CT_BRAIN_3_0_H31S_0009"],
...     image_size=256,
...     normalize="ct_brain",
... )
>>> img, pseudo_mask = ds[0]   # (1,256,256) float, (256,256) long
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from .dicom_loader import (
    find_dicom_series,
    read_dicom_series,
    normalize_volume,
    extract_2d_slices,
)
from .transforms import SegTransform, RegTransform


class DicomSliceDataset(Dataset):
    """
    从真实 DICOM 序列提取 2D 切片用于分割训练

    由于你暂时没有人工标注, 提供两种模式:
    1. with_pseudo_label=True  → 自动阈值生成伪标签 (前景/背景)
    2. with_pseudo_label=False → 仅返回图像 (用于自监督/预训练)

    参数
    ----
    dicom_dirs : list[str]
        DICOM 序列目录列表, 或包含多个序列的根目录
    image_size : int
        输出切片尺寸
    normalize : str
        归一化方式: "minmax", "ct_soft", "ct_bone", "ct_brain", "ct_lung"
    axis : int
        切片方向: 0=axial (横断面), 1=coronal (冠状面), 2=sagittal (矢状面)
    with_pseudo_label : bool
        是否生成伪标签 (Otsu 阈值分割)
    transform : SegTransform | None
        数据增强
    max_slices_per_volume : int | None
        每个序列最多取多少切片 (None=全部)
    """

    def __init__(
        self,
        dicom_dirs: list[str] | str,
        image_size: int = 256,
        normalize: str = "ct_soft",
        axis: int = 0,
        with_pseudo_label: bool = True,
        transform: SegTransform | None = None,
        max_slices_per_volume: int | None = None,
    ):
        super().__init__()
        self.image_size = image_size
        self.with_pseudo_label = with_pseudo_label
        self.transform = transform

        # 解析目录
        if isinstance(dicom_dirs, str):
            root = Path(dicom_dirs)
            if self._dir_has_dicom(root):
                dirs = [str(root)]
            else:
                dirs = find_dicom_series(dicom_dirs, max_depth=3)
        else:
            dirs = []
            for d in dicom_dirs:
                p = Path(d)
                if self._dir_has_dicom(p):
                    dirs.append(str(p))
                else:
                    dirs.extend(find_dicom_series(d, max_depth=3))

        if not dirs:
            raise ValueError(f"未找到 DICOM 数据, 请检查路径")

        # 读取并提取切片
        self.slices: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []

        print(f"加载 DICOM 数据 ({len(dirs)} 个序列)...")

        for i, d in enumerate(dirs):
            try:
                volume = read_dicom_series(d, return_info=False)
                volume = normalize_volume(volume, method=normalize)
                sl = extract_2d_slices(
                    volume, axis=axis, target_size=image_size,
                    skip_empty=True, empty_threshold=0.02,
                )

                if max_slices_per_volume and len(sl) > max_slices_per_volume:
                    # 均匀采样
                    indices = np.linspace(0, len(sl) - 1, max_slices_per_volume, dtype=int)
                    sl = [sl[j] for j in indices]

                for s in sl:
                    self.slices.append(s)
                    if with_pseudo_label:
                        self.labels.append(self._otsu_segment(s))

                print(f"  [{i+1}/{len(dirs)}] {Path(d).name}: "
                      f"{volume.shape} → {len(sl)} 切片")
            except Exception as e:
                print(f"  [{i+1}/{len(dirs)}] 跳过 {Path(d).name}: {e}")

        print(f"共加载 {len(self.slices)} 个切片")

    @staticmethod
    def _dir_has_dicom(d: Path) -> bool:
        """判断目录是否直接包含 DICOM 文件"""
        if not d.is_dir():
            return False
        for f in d.iterdir():
            if f.is_file():
                try:
                    with open(f, "rb") as fp:
                        fp.seek(128)
                        if fp.read(4) == b"DICM":
                            return True
                except (OSError, IOError):
                    pass
        return False

    @staticmethod
    def _otsu_segment(image: np.ndarray) -> np.ndarray:
        """
        Otsu 自动阈值分割 — 生成伪标签

        原理: 找到一个阈值 t, 使前景和背景的类间方差最大
        适合 CT 图像的简单前景/背景分离
        """
        from skimage.filters import threshold_otsu
        from scipy.ndimage import binary_fill_holes

        if image.max() - image.min() < 0.01:
            return np.zeros_like(image, dtype=np.int32)

        thresh = threshold_otsu(image)
        mask = (image > thresh).astype(np.int32)
        mask = binary_fill_holes(mask).astype(np.int32)
        return mask

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.slices[idx].copy()

        if self.with_pseudo_label:
            mask = self.labels[idx].copy()
            if self.transform is not None:
                image, mask = self.transform(image, mask)
            image_t = torch.from_numpy(image).unsqueeze(0).float()
            mask_t = torch.from_numpy(mask).long()
            return image_t, mask_t
        else:
            if self.transform is not None:
                image, _ = self.transform(image, np.zeros_like(image, dtype=np.int32))
            image_t = torch.from_numpy(image).unsqueeze(0).float()
            return image_t, image_t  # 自监督: 输入=输出


class DicomPairDataset(Dataset):
    """
    从 DICOM 构造配准训练对

    策略: 同一个 volume 的不同切片作为 (fixed, moving) 对
    - 相邻切片: 变形小, 适合入门
    - 间隔切片: 变形大, 更有挑战

    参数
    ----
    dicom_dirs : DICOM 序列目录
    image_size : 输出尺寸
    normalize : 归一化方法
    slice_gap : fixed 与 moving 之间间隔多少切片
    transform : 数据增强
    """

    def __init__(
        self,
        dicom_dirs: list[str] | str,
        image_size: int = 128,
        normalize: str = "ct_soft",
        slice_gap: int = 3,
        transform: RegTransform | None = None,
        max_slices_per_volume: int | None = None,
    ):
        super().__init__()
        self.image_size = image_size
        self.transform = transform
        self.slice_gap = slice_gap

        if isinstance(dicom_dirs, str):
            dirs = find_dicom_series(dicom_dirs, max_depth=3)
        else:
            dirs = []
            for d in dicom_dirs:
                dirs.extend(find_dicom_series(d, max_depth=3))

        # 读取 volume 并构造切片对
        self.pairs: list[tuple[np.ndarray, np.ndarray]] = []

        print(f"加载配准数据 ({len(dirs)} 个序列, gap={slice_gap})...")

        for i, d in enumerate(dirs):
            try:
                volume = read_dicom_series(d, return_info=False)
                volume = normalize_volume(volume, method=normalize)
                slices = extract_2d_slices(
                    volume, axis=0, target_size=image_size,
                    skip_empty=True,
                )

                if max_slices_per_volume and len(slices) > max_slices_per_volume:
                    indices = np.linspace(0, len(slices) - 1, max_slices_per_volume, dtype=int)
                    slices = [slices[j] for j in indices]

                # 构造对: (slice[i], slice[i + gap])
                count = 0
                for j in range(len(slices) - slice_gap):
                    self.pairs.append((slices[j], slices[j + slice_gap]))
                    count += 1

                print(f"  [{i+1}/{len(dirs)}] {Path(d).name}: {count} 对")
            except Exception as e:
                print(f"  [{i+1}/{len(dirs)}] 跳过 {Path(d).name}: {e}")

        print(f"共构造 {len(self.pairs)} 个配准对")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fixed, moving = self.pairs[idx]
        fixed = fixed.copy()
        moving = moving.copy()

        if self.transform is not None:
            fixed, moving = self.transform(fixed, moving)

        fixed_t = torch.from_numpy(fixed).unsqueeze(0).float()
        moving_t = torch.from_numpy(moving).unsqueeze(0).float()
        return fixed_t, moving_t
