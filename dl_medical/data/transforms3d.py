"""
3D 数据增强变换

变换
====
1. RandomFlip3D       : 随机轴翻转
2. RandomRotate90_3D  : 随机 90° 旋转
3. RandomIntensity3D  : 随机亮度/对比度
4. RandomGaussianNoise3D: 随机高斯噪声
5. Compose3D          : 组合多个变换

注意
====
- 配准数据增强需对 (fixed, moving) 同步变换
- 翻转/旋转需同步, 否则破坏对应关系
- 强度变换可独立应用
"""

from __future__ import annotations

import numpy as np
import torch


class Compose3D:
    """组合多个 3D 变换"""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(
        self, fixed: np.ndarray, moving: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            fixed, moving = t(fixed, moving)
        return fixed, moving


class RandomFlip3D:
    """
    随机 3D 翻转

    对 fixed 和 moving 同步翻转相同轴。
    """

    def __init__(self, p: float = 0.5, axes: tuple[int, ...] = (0, 1, 2)):
        self.p = p
        self.axes = axes

    def __call__(
        self, fixed: np.ndarray, moving: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self.p:
            for ax in self.axes:
                if np.random.rand() < 0.5:
                    fixed = np.flip(fixed, axis=ax)
                    moving = np.flip(moving, axis=ax)
        return np.ascontiguousarray(fixed), np.ascontiguousarray(moving)


class RandomRotate90_3D:
    """
    随机 90° 旋转 (在轴向平面上)

    在 (D, H), (H, W), (D, W) 平面上随机旋转 90° 的倍数。
    对 fixed 和 moving 同步。
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, fixed: np.ndarray, moving: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self.p:
            # 随机选择旋转平面
            plane = [(0, 1), (1, 2), (0, 2)][np.random.randint(3)]
            k = np.random.randint(1, 4)  # 1, 2, 3 次 90°
            fixed = np.rot90(fixed, k=k, axes=plane)
            moving = np.rot90(moving, k=k, axes=plane)
        return np.ascontiguousarray(fixed), np.ascontiguousarray(moving)


class RandomIntensity3D:
    """
    随机亮度/对比度变换

    对 fixed 和 moving 独立应用 (模拟不同扫描参数)。
    """

    def __init__(
        self,
        brightness: float = 0.1,
        contrast: float = 0.1,
        p: float = 0.5,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(
        self, fixed: np.ndarray, moving: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self.p:
            fixed = self._apply(fixed)
        if np.random.rand() < self.p:
            moving = self._apply(moving)
        return fixed, moving

    def _apply(self, img: np.ndarray) -> np.ndarray:
        # 亮度
        if np.random.rand() < 0.5:
            delta = np.random.uniform(-self.brightness, self.brightness)
            img = img + delta
        # 对比度
        if np.random.rand() < 0.5:
            alpha = 1.0 + np.random.uniform(-self.contrast, self.contrast)
            img = (img - img.mean()) * alpha + img.mean()
        return img.astype(np.float32)


class RandomGaussianNoise3D:
    """
    随机高斯噪声

    对 fixed 和 moving 独立添加。
    """

    def __init__(self, sigma: float = 0.02, p: float = 0.3):
        self.sigma = sigma
        self.p = p

    def __call__(
        self, fixed: np.ndarray, moving: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self.p:
            noise = np.random.normal(0, self.sigma, fixed.shape).astype(np.float32)
            fixed = fixed + noise
        if np.random.rand() < self.p:
            noise = np.random.normal(0, self.sigma, moving.shape).astype(np.float32)
            moving = moving + noise
        return fixed.astype(np.float32), moving.astype(np.float32)


class Normalize3D:
    """
    3D 体数据归一化

    模式:
    - "minmax" : 归一化到 [0, 1]
    - "zscore" : 零均值单位方差
    - "percentile": 百分位截断后归一化 (推荐用于 CT)
    """

    def __init__(
        self,
        mode: str = "percentile",
        p_low: float = 0.5,
        p_high: float = 99.5,
    ):
        self.mode = mode
        self.p_low = p_low
        self.p_high = p_high

    def __call__(
        self, fixed: np.ndarray, moving: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._norm(fixed), self._norm(moving)

    def _norm(self, img: np.ndarray) -> np.ndarray:
        if self.mode == "minmax":
            mn, mx = img.min(), img.max()
            return ((img - mn) / (mx - mn + 1e-8)).astype(np.float32)

        elif self.mode == "zscore":
            return ((img - img.mean()) / (img.std() + 1e-8)).astype(np.float32)

        elif self.mode == "percentile":
            lo = np.percentile(img, self.p_low)
            hi = np.percentile(img, self.p_high)
            img = np.clip(img, lo, hi)
            return ((img - lo) / (hi - lo + 1e-8)).astype(np.float32)

        else:
            raise ValueError(f"未知归一化模式: {self.mode}")


def get_default_transforms_3d(
    normalize: str = "percentile",
) -> Compose3D:
    """
    获取默认 3D 配准数据增强

    包含: 翻转 + 旋转 + 强度 + 噪声 + 归一化
    """
    return Compose3D([
        RandomFlip3D(p=0.5, axes=(1, 2)),  # 只翻转 H, W (不翻转 D, 保持上下关系)
        RandomRotate90_3D(p=0.3),
        RandomIntensity3D(brightness=0.05, contrast=0.05, p=0.3),
        RandomGaussianNoise3D(sigma=0.01, p=0.3),
        Normalize3D(mode=normalize),
    ])


def get_val_transforms_3d(
    normalize: str = "percentile",
) -> Compose3D:
    """验证集变换 (仅归一化)"""
    return Compose3D([
        Normalize3D(mode=normalize),
    ])


if __name__ == "__main__":
    print("测试 3D 数据增强...")

    fixed = np.random.randn(32, 32, 32).astype(np.float32)
    moving = np.random.randn(32, 32, 32).astype(np.float32)

    transform = get_default_transforms_3d()
    f, m = transform(fixed, moving)
    print(f"  Fixed:  {f.shape}  range: [{f.min():.3f}, {f.max():.3f}]")
    print(f"  Moving: {m.shape}  range: [{m.min():.3f}, {m.max():.3f}]")

    val_transform = get_val_transforms_3d()
    f, m = val_transform(fixed, moving)
    print(f"  Val Fixed:  {f.shape}  range: [{f.min():.3f}, {f.max():.3f}]")
