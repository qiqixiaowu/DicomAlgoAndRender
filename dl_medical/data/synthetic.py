"""
合成数据生成器 —— 无需真实数据即可开始学习

为什么用合成数据？
==================
1. 医学图像数据受隐私保护, 获取困难
2. 合成数据可以快速验证网络是否正确运行
3. 控制数据复杂度, 逐步增加难度
4. 已有完美标注 → 评估指标更可靠

分割数据
========
生成带有圆形/椭圆形"器官"的灰度图像 + 对应 mask

配准数据
========
生成一对图像: fixed 和 moving (带有随机弹性变形)
"""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


class SyntheticSegData:
    """
    合成分割数据生成器

    生成包含多个"器官" (圆形/椭圆) 的灰度图像:
    - 背景: 低强度 + 噪声
    - 器官: 高强度 + 不同灰度级 + 噪声
    - 标签: 0=背景, 1=前景

    示例
    ----
    >>> gen = SyntheticSegData(image_size=256)
    >>> image, mask = gen.generate_one()
    >>> # image: (256, 256) float32, range [0, 1]
    >>> # mask:  (256, 256) int32,   values {0, 1}
    """

    def __init__(self, image_size: int = 256, max_objects: int = 5):
        self.size = image_size
        self.max_objects = max_objects
        self.rng = np.random.default_rng()

    def generate_one(self) -> tuple[np.ndarray, np.ndarray]:
        """生成一对 (图像, 分割标签)"""
        image = np.zeros((self.size, self.size), dtype=np.float32)
        mask = np.zeros((self.size, self.size), dtype=np.int32)

        # 背景纹理
        bg_noise = self.rng.normal(0.15, 0.05, (self.size, self.size)).astype(np.float32)
        bg_noise = gaussian_filter(bg_noise, sigma=3)
        image += bg_noise

        # 随机生成 1~max_objects 个器官
        n_objects = self.rng.integers(1, self.max_objects + 1)
        for _ in range(n_objects):
            self._add_organ(image, mask)

        # 全局噪声
        noise = self.rng.normal(0, 0.02, image.shape).astype(np.float32)
        image = image + noise
        image = np.clip(image, 0, 1)

        return image, mask

    def _add_organ(self, image: np.ndarray, mask: np.ndarray) -> None:
        """在图像上添加一个随机器官"""
        s = self.size

        # 随机中心 (避免边界)
        cy = self.rng.integers(s // 5, 4 * s // 5)
        cx = self.rng.integers(s // 5, 4 * s // 5)

        # 随机半径
        ry = self.rng.integers(s // 15, s // 4)
        rx = self.rng.integers(s // 15, s // 4)

        # 随机旋转
        angle = self.rng.uniform(0, 2 * np.pi)

        # 生成坐标网格
        yy, xx = np.mgrid[0:s, 0:s]
        # 平移到中心
        dy = yy - cy
        dx = xx - cx
        # 旋转
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dy_rot = cos_a * dy + sin_a * dx
        dx_rot = -sin_a * dy + cos_a * dx

        # 椭圆方程
        ellipse = (dy_rot / ry) ** 2 + (dx_rot / rx) ** 2 <= 1.0

        # 器官灰度
        intensity = self.rng.uniform(0.5, 0.95)
        organ_texture = intensity + self.rng.normal(0, 0.03, image.shape).astype(np.float32)
        organ_texture = gaussian_filter(organ_texture, sigma=2)

        image[ellipse] = organ_texture[ellipse]
        mask[ellipse] = 1

    def generate_batch(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """生成 N 对数据"""
        images, masks = [], []
        for _ in range(n):
            img, msk = self.generate_one()
            images.append(img)
            masks.append(msk)
        return np.stack(images), np.stack(masks)


class SyntheticRegData:
    """
    合成配准数据生成器

    生成配准对 (fixed, moving):
    1. 创建一个"模板"图像 (圆形/矩形结构)
    2. 对模板施加随机弹性变形 → moving
    3. 原始模板 → fixed

    弹性变形原理:
    - 生成随机位移场 (高斯平滑后)
    - 用插值将图像按位移场变形
    - 平滑的位移场 → 自然的变形效果

    示例
    ----
    >>> gen = SyntheticRegData(image_size=128)
    >>> fixed, moving = gen.generate_one()
    """

    def __init__(self, image_size: int = 128, deform_sigma: float = 10.0,
                 deform_alpha: float = 15.0):
        self.size = image_size
        self.sigma = deform_sigma    # 高斯平滑 sigma (控制变形平滑度)
        self.alpha = deform_alpha    # 位移幅度 (控制变形强度)
        self.rng = np.random.default_rng()

    def generate_one(self) -> tuple[np.ndarray, np.ndarray]:
        """生成 (fixed, moving) 配准对"""
        s = self.size

        # 创建模板图像 (3~5 个形状)
        template = np.zeros((s, s), dtype=np.float32)
        template += self.rng.normal(0.1, 0.02, (s, s))

        n_shapes = self.rng.integers(3, 6)
        for _ in range(n_shapes):
            template = self._add_shape(template)

        template = np.clip(template, 0, 1).astype(np.float32)

        # 弹性变形
        moving = self._elastic_deform(template)

        return template, moving

    def _add_shape(self, image: np.ndarray) -> np.ndarray:
        """添加随机形状"""
        s = self.size
        yy, xx = np.mgrid[0:s, 0:s]

        cy = self.rng.integers(s // 5, 4 * s // 5)
        cx = self.rng.integers(s // 5, 4 * s // 5)
        r = self.rng.integers(s // 10, s // 3)
        intensity = self.rng.uniform(0.4, 0.95)

        shape_type = self.rng.choice(["circle", "rect", "ring"])

        if shape_type == "circle":
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            region = dist < r
        elif shape_type == "rect":
            region = (np.abs(yy - cy) < r // 2) & (np.abs(xx - cx) < r)
        else:  # ring
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            region = (dist > r * 0.6) & (dist < r)

        image[region] = intensity
        return image

    def _elastic_deform(self, image: np.ndarray) -> np.ndarray:
        """
        弹性变形

        步骤:
        1. 生成随机位移场 dx, dy
        2. 高斯滤波平滑 → 变形连续
        3. 用 map_coordinates 双线性插值采样
        """
        s = self.size

        # 随机位移
        dx = self.rng.standard_normal((s, s)) * self.alpha
        dy = self.rng.standard_normal((s, s)) * self.alpha

        # 高斯平滑 → 使变形平滑
        dx = gaussian_filter(dx, self.sigma)
        dy = gaussian_filter(dy, self.sigma)

        # 生成采样坐标
        yy, xx = np.mgrid[0:s, 0:s]
        coords = np.array([yy + dy, xx + dx])

        # 双线性插值变形
        deformed = map_coordinates(image, coords, order=1, mode="reflect")
        return deformed.astype(np.float32)

    def generate_batch(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """生成 N 对"""
        fixed_list, moving_list = [], []
        for _ in range(n):
            f, m = self.generate_one()
            fixed_list.append(f)
            moving_list.append(m)
        return np.stack(fixed_list), np.stack(moving_list)
