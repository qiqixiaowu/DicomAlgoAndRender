"""
数据增强 (Data Augmentation)

为什么需要数据增强？
==================
1. 医学影像数据量小 → 容易过拟合
2. 通过几何和灰度变换扩充训练集
3. 提升模型泛化能力

分割增强
========
- 图像和标签需同步变换 (旋转/翻转)
- 灰度变换只作用于图像 (对比度/亮度)

配准增强
========
- 两幅图可独立灰度变换 (模拟多模态)
- 弹性变形 (已在合成数据中实现)
"""

import numpy as np
from scipy.ndimage import rotate, gaussian_filter


class SegTransform:
    """
    分割数据增强

    变换列表:
    1. 随机水平翻转
    2. 随机垂直翻转
    3. 随机旋转 (±15°)
    4. 随机亮度/对比度调整
    5. 高斯噪声
    """

    def __init__(self, p: float = 0.5):
        """p: 每种增强被应用的概率"""
        self.p = p
        self.rng = np.random.default_rng()

    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        image: (H, W) float32
        mask:  (H, W) int32
        """
        # 水平翻转
        if self.rng.random() < self.p:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # 垂直翻转
        if self.rng.random() < self.p:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        # 随机旋转
        if self.rng.random() < self.p:
            angle = self.rng.uniform(-15, 15)
            image = rotate(image, angle, reshape=False, order=1, mode="reflect")
            mask = rotate(mask, angle, reshape=False, order=0, mode="reflect")

        # 亮度/对比度 (只作用于图像)
        if self.rng.random() < self.p:
            alpha = self.rng.uniform(0.8, 1.2)  # 对比度
            beta = self.rng.uniform(-0.1, 0.1)  # 亮度
            image = np.clip(alpha * image + beta, 0, 1).astype(np.float32)

        # 高斯噪声
        if self.rng.random() < self.p:
            noise = self.rng.normal(0, 0.02, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0, 1)

        return image, mask


class RegTransform:
    """
    配准数据增强

    变换列表:
    1. 同步翻转 (两幅图相同变换)
    2. 独立灰度扰动 (模拟模态差异)
    3. 高斯模糊 (模拟分辨率差异)
    """

    def __init__(self, p: float = 0.5):
        self.p = p
        self.rng = np.random.default_rng()

    def __call__(
        self, fixed: np.ndarray, moving: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """fixed, moving: (H, W) float32"""

        # 同步水平翻转
        if self.rng.random() < self.p:
            fixed = np.flip(fixed, axis=1).copy()
            moving = np.flip(moving, axis=1).copy()

        # 独立灰度扰动
        if self.rng.random() < self.p:
            alpha = self.rng.uniform(0.85, 1.15)
            beta = self.rng.uniform(-0.08, 0.08)
            fixed = np.clip(alpha * fixed + beta, 0, 1).astype(np.float32)

        if self.rng.random() < self.p:
            alpha = self.rng.uniform(0.85, 1.15)
            beta = self.rng.uniform(-0.08, 0.08)
            moving = np.clip(alpha * moving + beta, 0, 1).astype(np.float32)

        # 随机高斯模糊 (其中一幅)
        if self.rng.random() < self.p * 0.5:
            sigma = self.rng.uniform(0.5, 1.5)
            if self.rng.random() < 0.5:
                fixed = gaussian_filter(fixed, sigma).astype(np.float32)
            else:
                moving = gaussian_filter(moving, sigma).astype(np.float32)

        return fixed, moving
