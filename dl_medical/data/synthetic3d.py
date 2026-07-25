"""
3D 合成体数据生成器 —— 模拟肺部/肝部结构用于配准训练

生成策略
========
1. SyntheticLungData3D : 模拟肺部 CT (含气管树、血管、肺实质)
2. SyntheticLiverData3D: 模拟肝脏 (含血管、肿瘤、分叶)
3. SyntheticRegData3D  : 通用 3D 配准对 (弹性变形)

每个生成器输出 (fixed, moving) 对:
- fixed  : 原始模板
- moving : 对模板施加随机弹性变形后的图像

弹性变形
========
1. 生成随机 3D 位移场 (高斯平滑)
2. 用 map_coordinates 三线性插值采样
3. 平滑的位移场 → 自然的器官变形
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


# ============================================================
#  3D 弹性变形工具
# ============================================================

def elastic_deform_3d(
    image: np.ndarray,
    sigma: float = 8.0,
    alpha: float = 10.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    对 3D 图像施加弹性变形

    参数
    ----
    image : (D, H, W) float32
    sigma : 高斯平滑 sigma (控制变形平滑度)
    alpha : 位移幅度 (控制变形强度)
    rng   : 随机数生成器

    返回
    ----
    deformed : (D, H, W) float32
    """
    if rng is None:
        rng = np.random.default_rng()

    shape = image.shape

    # 生成随机位移场
    dz = rng.standard_normal(shape) * alpha
    dy = rng.standard_normal(shape) * alpha
    dx = rng.standard_normal(shape) * alpha

    # 高斯平滑 → 使变形连续
    dz = gaussian_filter(dz, sigma)
    dy = gaussian_filter(dy, sigma)
    dx = gaussian_filter(dx, sigma)

    # 构建采样坐标
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    coords = np.array([zz + dz, yy + dy, xx + dx])

    # 三线性插值采样
    deformed = map_coordinates(image, coords, order=1, mode="reflect")
    return deformed.astype(np.float32)


# ============================================================
#  肺部合成数据
# ============================================================

class SyntheticLungData3D:
    """
    模拟肺部 CT 体数据

    结构:
    - 肺实质: 低密度椭圆体 (左肺 + 右肺)
    - 气管树: 管状结构从顶部向下分叉
    - 血管: 细管状结构随机分布
    - 结节: 小球形高密度

    配准对:
    - fixed  = 原始模板
    - moving = 弹性变形后的模板 (模拟呼吸运动)
    """

    def __init__(
        self,
        volume_size: tuple[int, int, int] = (64, 64, 64),
        deform_sigma: float = 8.0,
        deform_alpha: float = 12.0,
    ):
        self.size = volume_size
        self.sigma = deform_sigma
        self.alpha = deform_alpha
        self.rng = np.random.default_rng()

    def generate_one(self) -> tuple[np.ndarray, np.ndarray]:
        """生成 (fixed, moving) 配准对"""
        template = self._create_lung_template()
        moving = elastic_deform_3d(
            template, self.sigma, self.alpha, self.rng
        )
        return template, moving

    def generate_batch(
        self, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """生成 N 对数据"""
        fixed_list, moving_list = [], []
        for _ in range(n):
            f, m = self.generate_one()
            fixed_list.append(f)
            moving_list.append(m)
        return np.stack(fixed_list), np.stack(moving_list)

    def _create_lung_template(self) -> np.ndarray:
        """创建肺部模板"""
        D, H, W = self.size
        img = np.full((D, H, W), -0.3, dtype=np.float32)  # 背景低密度

        # 背景纹理
        bg = self.rng.normal(0, 0.02, (D, H, W)).astype(np.float32)
        bg = gaussian_filter(bg, sigma=2)
        img += bg

        # 左肺 (较小, 偏左)
        self._add_ellipsoid(
            img, center=(D // 2, H // 2, W // 4),
            radii=(D // 3, H // 3, W // 5),
            intensity=0.3,
        )
        # 右肺 (较大, 偏右)
        self._add_ellipsoid(
            img, center=(D // 2, H // 2, 3 * W // 4),
            radii=(D // 3, H // 3, W // 4),
            intensity=0.35,
        )

        # 气管 (从顶部向下)
        self._add_tube(
            img, start=(0, H // 2, W // 2),
            end=(D // 2, H // 2, W // 2),
            radius=3, intensity=0.8,
        )
        # 左支气管
        self._add_tube(
            img, start=(D // 2, H // 2, W // 2),
            end=(2 * D // 3, H // 2, W // 4),
            radius=2, intensity=0.7,
        )
        # 右支气管
        self._add_tube(
            img, start=(D // 2, H // 2, W // 2),
            end=(2 * D // 3, H // 2, 3 * W // 4),
            radius=2, intensity=0.7,
        )

        # 血管 (随机细管)
        n_vessels = self.rng.integers(3, 7)
        for _ in range(n_vessels):
            start = self.rng.integers([D // 4, H // 4, W // 6], [3 * D // 4, 3 * H // 4, 5 * W // 6])
            end = start + self.rng.integers(-10, 11, size=3)
            self._add_tube(
                img, start=tuple(start), end=tuple(end),
                radius=self.rng.integers(1, 3),
                intensity=self.rng.uniform(0.6, 0.9),
            )

        # 结节 (小球)
        n_nodules = self.rng.integers(0, 3)
        for _ in range(n_nodules):
            center = self.rng.integers([D // 4, H // 4, W // 6], [3 * D // 4, 3 * H // 4, 5 * W // 6])
            r = self.rng.integers(2, 5)
            self._add_sphere(img, center=tuple(center), radius=r, intensity=0.9)

        # 全局噪声
        noise = self.rng.normal(0, 0.02, img.shape).astype(np.float32)
        img = np.clip(img + noise, -1, 1)
        return img

    def _add_ellipsoid(
        self, img: np.ndarray,
        center: tuple[int, int, int],
        radii: tuple[int, int, int],
        intensity: float,
    ) -> None:
        """添加椭球体"""
        D, H, W = img.shape
        zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
        d = ((zz - center[0]) / radii[0]) ** 2 \
          + ((yy - center[1]) / radii[1]) ** 2 \
          + ((xx - center[2]) / radii[2]) ** 2
        mask = d <= 1.0
        img[mask] = intensity

    def _add_sphere(
        self, img: np.ndarray,
        center: tuple[int, int, int],
        radius: int,
        intensity: float,
    ) -> None:
        """添加球体"""
        D, H, W = img.shape
        zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
        d = np.sqrt(
            (zz - center[0]) ** 2
            + (yy - center[1]) ** 2
            + (xx - center[2]) ** 2
        )
        img[d <= radius] = intensity

    def _add_tube(
        self, img: np.ndarray,
        start: tuple[int, int, int],
        end: tuple[int, int, int],
        radius: int,
        intensity: float,
    ) -> None:
        """添加管状结构 (圆柱体)"""
        D, H, W = img.shape
        zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]

        # 线段参数
        s = np.array(start, dtype=np.float32)
        e = np.array(end, dtype=np.float32)
        seg = e - s
        seg_len = np.linalg.norm(seg)
        if seg_len < 1:
            return
        seg_dir = seg / seg_len

        # 点到线段距离
        pts = np.stack([zz, yy, xx], axis=-1).astype(np.float32)  # (D, H, W, 3)
        vec = pts - s  # (D, H, W, 3)
        t = np.clip((vec @ seg_dir), 0, seg_len)
        proj = s + t[..., None] * seg_dir
        dist = np.linalg.norm(pts - proj, axis=-1)

        img[dist <= radius] = intensity


# ============================================================
#  肝部合成数据
# ============================================================

class SyntheticLiverData3D:
    """
    模拟肝脏 CT 体数据

    结构:
    - 肝实质: 不规则分叶状高密度体
    - 门静脉: 管状分支结构
    - 肿瘤: 小球形低密度/高密度
    - 边缘: 梯度过渡 (模拟肝脏边缘)

    配准对:
    - fixed  = 原始模板
    - moving = 弹性变形后的模板 (模拟呼吸/体位变化)
    """

    def __init__(
        self,
        volume_size: tuple[int, int, int] = (64, 64, 64),
        deform_sigma: float = 6.0,
        deform_alpha: float = 10.0,
    ):
        self.size = volume_size
        self.sigma = deform_sigma
        self.alpha = deform_alpha
        self.rng = np.random.default_rng()

    def generate_one(self) -> tuple[np.ndarray, np.ndarray]:
        template = self._create_liver_template()
        moving = elastic_deform_3d(
            template, self.sigma, self.alpha, self.rng
        )
        return template, moving

    def generate_batch(
        self, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        fixed_list, moving_list = [], []
        for _ in range(n):
            f, m = self.generate_one()
            fixed_list.append(f)
            moving_list.append(m)
        return np.stack(fixed_list), np.stack(moving_list)

    def _create_liver_template(self) -> np.ndarray:
        """创建肝脏模板"""
        D, H, W = self.size
        img = np.full((D, H, W), -0.4, dtype=np.float32)

        # 背景纹理
        bg = self.rng.normal(0, 0.02, (D, H, W)).astype(np.float32)
        bg = gaussian_filter(bg, sigma=2)
        img += bg

        # 肝实质 (大椭球 + 分叶)
        liver_center = (D // 2, H // 2, W // 2)
        self._add_ellipsoid(
            img, center=liver_center,
            radii=(D // 3, H // 3, W // 3),
            intensity=0.5,
        )
        # 分叶 (小椭球叠加)
        n_lobes = self.rng.integers(2, 5)
        for _ in range(n_lobes):
            offset = self.rng.integers(-8, 9, size=3)
            center = tuple(np.array(liver_center) + offset)
            self._add_ellipsoid(
                img, center=center,
                radii=tuple(self.rng.integers(8, 16, size=3)),
                intensity=0.5,
            )

        # 肝脏边缘模糊 (高斯平滑边缘)
        liver_mask = img > 0.3
        edge = gaussian_filter(liver_mask.astype(np.float32), sigma=1)
        img = np.where(edge > 0.3, img * (0.5 + edge * 0.5), img)

        # 门静脉 (从肝门向各叶分支)
        hilum = (D // 2, H // 2, W // 2)
        n_branches = self.rng.integers(3, 6)
        for _ in range(n_branches):
            end = hilum + self.rng.integers(-15, 16, size=3)
            self._add_tube(
                img, start=hilum, end=tuple(end),
                radius=self.rng.integers(1, 3),
                intensity=self.rng.uniform(0.1, 0.3),
            )

        # 肿瘤 (小球)
        n_tumors = self.rng.integers(1, 4)
        for _ in range(n_tumors):
            center = hilum + self.rng.integers(-12, 13, size=3)
            r = self.rng.integers(3, 6)
            intensity = self.rng.choice([0.2, 0.8])  # 低密度或高密度
            self._add_sphere(img, center=tuple(center), radius=r, intensity=float(intensity))

        # 噪声
        noise = self.rng.normal(0, 0.02, img.shape).astype(np.float32)
        img = np.clip(img + noise, -1, 1)
        return img

    def _add_ellipsoid(
        self, img: np.ndarray,
        center: tuple[int, int, int],
        radii: tuple[int, int, int],
        intensity: float,
    ) -> None:
        D, H, W = img.shape
        zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
        d = ((zz - center[0]) / radii[0]) ** 2 \
          + ((yy - center[1]) / radii[1]) ** 2 \
          + ((xx - center[2]) / radii[2]) ** 2
        mask = d <= 1.0
        img[mask] = intensity

    def _add_sphere(
        self, img: np.ndarray,
        center: tuple[int, int, int],
        radius: int,
        intensity: float,
    ) -> None:
        D, H, W = img.shape
        zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
        d = np.sqrt(
            (zz - center[0]) ** 2
            + (yy - center[1]) ** 2
            + (xx - center[2]) ** 2
        )
        img[d <= radius] = intensity

    def _add_tube(
        self, img: np.ndarray,
        start: tuple[int, int, int],
        end: tuple[int, int, int],
        radius: int,
        intensity: float,
    ) -> None:
        D, H, W = img.shape
        zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
        s = np.array(start, dtype=np.float32)
        e = np.array(end, dtype=np.float32)
        seg = e - s
        seg_len = np.linalg.norm(seg)
        if seg_len < 1:
            return
        seg_dir = seg / seg_len
        pts = np.stack([zz, yy, xx], axis=-1).astype(np.float32)
        vec = pts - s
        t = np.clip((vec @ seg_dir), 0, seg_len)
        proj = s + t[..., None] * seg_dir
        dist = np.linalg.norm(pts - proj, axis=-1)
        img[dist <= radius] = intensity


# ============================================================
#  通用 3D 配准数据
# ============================================================

class SyntheticRegData3D:
    """
    通用 3D 配准数据生成器

    生成包含多种形状的体数据, 施加弹性变形作为 moving。
    """

    def __init__(
        self,
        volume_size: tuple[int, int, int] = (64, 64, 64),
        deform_sigma: float = 8.0,
        deform_alpha: float = 12.0,
    ):
        self.size = volume_size
        self.sigma = deform_sigma
        self.alpha = deform_alpha
        self.rng = np.random.default_rng()

    def generate_one(self) -> tuple[np.ndarray, np.ndarray]:
        D, H, W = self.size
        template = np.full((D, H, W), 0.1, dtype=np.float32)
        template += self.rng.normal(0, 0.02, (D, H, W)).astype(np.float32)
        template = gaussian_filter(template, sigma=2)

        # 随机形状
        n_shapes = self.rng.integers(3, 7)
        for _ in range(n_shapes):
            self._add_shape(template)

        template = np.clip(template, 0, 1).astype(np.float32)
        moving = elastic_deform_3d(template, self.sigma, self.alpha, self.rng)
        return template, moving

    def generate_batch(
        self, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        fixed_list, moving_list = [], []
        for _ in range(n):
            f, m = self.generate_one()
            fixed_list.append(f)
            moving_list.append(m)
        return np.stack(fixed_list), np.stack(moving_list)

    def _add_shape(self, img: np.ndarray) -> None:
        D, H, W = img.shape
        shape_type = self.rng.choice(["sphere", "ellipsoid", "tube"])

        center = self.rng.integers([D // 5, H // 5, W // 5], [4 * D // 5, 4 * H // 5, 4 * W // 5])
        intensity = self.rng.uniform(0.4, 0.95)

        if shape_type == "sphere":
            r = self.rng.integers(3, D // 6)
            zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
            d = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
            img[d <= r] = intensity

        elif shape_type == "ellipsoid":
            radii = self.rng.integers(3, D // 5, size=3)
            zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
            d = ((zz - center[0]) / radii[0]) ** 2 \
              + ((yy - center[1]) / radii[1]) ** 2 \
              + ((xx - center[2]) / radii[2]) ** 2
            img[d <= 1] = intensity

        else:  # tube
            end = center + self.rng.integers(-15, 16, size=3)
            radius = self.rng.integers(1, 4)
            zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
            s = center.astype(np.float32)
            e = end.astype(np.float32)
            seg = e - s
            seg_len = np.linalg.norm(seg)
            if seg_len < 1:
                return
            seg_dir = seg / seg_len
            pts = np.stack([zz, yy, xx], axis=-1).astype(np.float32)
            vec = pts - s
            t = np.clip((vec @ seg_dir), 0, seg_len)
            proj = s + t[..., None] * seg_dir
            dist = np.linalg.norm(pts - proj, axis=-1)
            img[dist <= radius] = intensity


if __name__ == "__main__":
    print("测试 3D 合成数据生成器...")

    print("\n--- 肺部数据 ---")
    lung_gen = SyntheticLungData3D(volume_size=(64, 64, 64))
    fixed, moving = lung_gen.generate_one()
    print(f"  Fixed:  {fixed.shape}  range: [{fixed.min():.2f}, {fixed.max():.2f}]")
    print(f"  Moving: {moving.shape} range: [{moving.min():.2f}, {moving.max():.2f}]")

    print("\n--- 肝部数据 ---")
    liver_gen = SyntheticLiverData3D(volume_size=(64, 64, 64))
    fixed, moving = liver_gen.generate_one()
    print(f"  Fixed:  {fixed.shape}  range: [{fixed.min():.2f}, {fixed.max():.2f}]")
    print(f"  Moving: {moving.shape} range: [{moving.min():.2f}, {moving.max():.2f}]")

    print("\n--- 通用数据 ---")
    gen = SyntheticRegData3D(volume_size=(64, 64, 64))
    fixed, moving = gen.generate_one()
    print(f"  Fixed:  {fixed.shape}  range: [{fixed.min():.2f}, {fixed.max():.2f}]")
    print(f"  Moving: {moving.shape} range: [{moving.min():.2f}, {moving.max():.2f}]")
