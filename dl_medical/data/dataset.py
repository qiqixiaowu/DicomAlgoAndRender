"""
PyTorch Dataset 类 —— 分割与配准

Dataset 的作用
=============
- 将数据组织为 PyTorch 可用的格式
- 支持索引访问 dataset[i]
- 配合 DataLoader 实现批量加载、打乱、多线程

数据流
======
合成数据生成 → numpy → Dataset (增强) → Tensor → DataLoader → 模型
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .synthetic import SyntheticSegData, SyntheticRegData
from .transforms import SegTransform, RegTransform


class SegmentationDataset(Dataset):
    """
    分割数据集

    参数
    ----
    num_samples : int – 数据集大小
    image_size  : int – 图像尺寸
    transform   : 数据增强 (可选)

    示例
    ----
    >>> ds = SegmentationDataset(num_samples=100, image_size=256)
    >>> img, mask = ds[0]
    >>> # img:  Tensor (1, 256, 256)
    >>> # mask: Tensor (256, 256) long
    """

    def __init__(
        self,
        num_samples: int = 500,
        image_size: int = 256,
        transform: SegTransform | None = None,
    ):
        super().__init__()
        self.transform = transform

        # 预生成全部数据 (内存中, 适合合成数据)
        gen = SyntheticSegData(image_size=image_size)
        self.images, self.masks = gen.generate_batch(num_samples)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx].copy()
        mask = self.masks[idx].copy()

        # 数据增强
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # numpy → tensor
        # 图像: (H, W) → (1, H, W) 添加通道维度
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor


class RegistrationDataset(Dataset):
    """
    配准数据集

    参数
    ----
    num_samples : int – 数据集大小
    image_size  : int – 图像尺寸
    transform   : 数据增强 (可选)

    返回
    ----
    fixed  : (1, H, W) float tensor
    moving : (1, H, W) float tensor
    """

    def __init__(
        self,
        num_samples: int = 500,
        image_size: int = 128,
        transform: RegTransform | None = None,
    ):
        super().__init__()
        self.transform = transform

        gen = SyntheticRegData(image_size=image_size)
        self.fixed_images, self.moving_images = gen.generate_batch(num_samples)

    def __len__(self) -> int:
        return len(self.fixed_images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fixed = self.fixed_images[idx].copy()
        moving = self.moving_images[idx].copy()

        if self.transform is not None:
            fixed, moving = self.transform(fixed, moving)

        fixed_tensor = torch.from_numpy(fixed).unsqueeze(0).float()
        moving_tensor = torch.from_numpy(moving).unsqueeze(0).float()

        return fixed_tensor, moving_tensor
