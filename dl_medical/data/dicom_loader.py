"""
DICOM 数据加载工具

功能
====
1. 扫描目录, 自动发现所有 DICOM 序列
2. 读取 DICOM 序列为 3D numpy 数组
3. 提取关键 DICOM tag (模态/层厚/像素间距等)
4. 窗宽窗位 (Window/Level) 调整
5. 从 3D volume 截取 2D 切片用于训练

支持的数据路径格式
=================
E:\\Data\\
├── AC_CT_BRAIN\\           ← 直接包含 .dcm/.IMA 文件
│   ├── xxx.0001.IMA
│   └── xxx.0002.IMA
├── CTMR\\
│   └── patient01\\         ← 嵌套子目录
│       └── series01\\
│           ├── 001.dcm
│           └── 002.dcm
"""

import os
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False


# ============================================================
#  数据结构
# ============================================================

@dataclass
class DicomSeriesInfo:
    """一个 DICOM 序列的元信息"""
    directory: str
    num_slices: int = 0
    modality: str = ""              # CT, MR, PT, NM ...
    patient_name: str = ""
    series_description: str = ""
    pixel_spacing: tuple[float, float] = (1.0, 1.0)
    slice_thickness: float = 1.0
    rows: int = 0
    cols: int = 0
    window_center: float = 40.0
    window_width: float = 400.0


# ============================================================
#  DICOM 扫描与发现
# ============================================================

def find_dicom_series(root_dir: str, max_depth: int = 5) -> list[str]:
    """
    迭代扫描目录, 找到所有包含 DICOM 文件的文件夹 (避免递归栈溢出)

    参数
    ----
    root_dir : 根目录路径
    max_depth : 最大递归深度

    返回
    ----
    包含 DICOM 文件的目录路径列表 (已排序)
    """
    _dicom_extensions = {".dcm", ".ima", ".dicom", ""}
    dicom_dirs = []

    # 用显式栈代替递归: (目录路径, 当前深度)
    stack: list[tuple[str, int]] = [(str(root_dir), 0)]

    while stack:
        cur_dir, depth = stack.pop()
        if depth > max_depth:
            continue

        has_dicom = False
        subdirs: list[tuple[str, int]] = []

        try:
            with os.scandir(cur_dir) as it:
                for entry in it:
                    # 跳过符号链接, 避免环形遍历
                    if entry.is_symlink():
                        continue
                    if entry.is_file(follow_symlinks=False):
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext in _dicom_extensions and not has_dicom:
                            if _is_dicom_file(Path(entry.path)):
                                has_dicom = True
                    elif entry.is_dir(follow_symlinks=False):
                        subdirs.append((entry.path, depth + 1))
        except (PermissionError, OSError):
            continue

        if has_dicom:
            dicom_dirs.append(cur_dir)

        stack.extend(subdirs)

    return sorted(dicom_dirs)


def _is_dicom_file(filepath: Path) -> bool:
    """快速判断文件是否为 DICOM 格式 (读取前 132 字节检查魔数)"""
    try:
        with open(filepath, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except (OSError, IOError):
        return False


# ============================================================
#  DICOM 读取 (使用 SimpleITK, 更稳定)
# ============================================================

def read_dicom_series(
    directory: str,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, DicomSeriesInfo]:
    """
    读取一个目录中的 DICOM 序列为 3D numpy 数组

    参数
    ----
    directory : DICOM 文件所在目录
    return_info : 是否同时返回元信息

    返回
    ----
    volume : (D, H, W) float32 数组  (D=切片数)
    info   : DicomSeriesInfo (如果 return_info=True)
    """
    if HAS_SITK:
        return _read_with_sitk(directory, return_info)
    elif HAS_PYDICOM:
        return _read_with_pydicom(directory, return_info)
    else:
        raise ImportError(
            "需要安装 SimpleITK 或 pydicom:\n"
            "  pip install SimpleITK\n"
            "  pip install pydicom"
        )


def _read_with_sitk(
    directory: str, return_info: bool
) -> np.ndarray | tuple[np.ndarray, DicomSeriesInfo]:
    """使用 SimpleITK 读取"""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(directory)

    if not series_ids:
        raise ValueError(f"目录中没有找到 DICOM 序列: {directory}")

    # 取第一个序列
    file_names = reader.GetGDCMSeriesFileNames(directory, series_ids[0])
    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    image = reader.Execute()
    volume = sitk.GetArrayFromImage(image).astype(np.float32)  # (D, H, W)

    if not return_info:
        return volume

    # 提取元信息
    info = DicomSeriesInfo(directory=directory)
    info.num_slices = volume.shape[0]
    info.rows = volume.shape[1]
    info.cols = volume.shape[2]

    spacing = image.GetSpacing()  # (x, y, z)
    info.pixel_spacing = (spacing[0], spacing[1])
    info.slice_thickness = spacing[2]

    # 从第一个切片读 DICOM tag
    if reader.GetMetaDataKeys(0):
        info.modality = _get_meta(reader, 0, "0008|0060", "Unknown")
        info.patient_name = _get_meta(reader, 0, "0010|0010", "Anonymous")
        info.series_description = _get_meta(reader, 0, "0008|103e", "")
        wc = _get_meta(reader, 0, "0028|1050", "40")
        ww = _get_meta(reader, 0, "0028|1051", "400")
        try:
            info.window_center = float(wc.split("\\")[0])
            info.window_width = float(ww.split("\\")[0])
        except (ValueError, IndexError):
            pass

    return volume, info


def _get_meta(reader, slice_idx: int, tag: str, default: str) -> str:
    """安全读取 SimpleITK meta data"""
    try:
        if reader.HasMetaDataKey(slice_idx, tag):
            return reader.GetMetaData(slice_idx, tag).strip()
    except RuntimeError:
        pass
    return default


def _read_with_pydicom(
    directory: str, return_info: bool
) -> np.ndarray | tuple[np.ndarray, DicomSeriesInfo]:
    """使用 pydicom 读取"""
    dcm_dir = Path(directory)
    dcm_files = []

    for f in dcm_dir.iterdir():
        if f.is_file() and _is_dicom_file(f):
            dcm_files.append(f)

    if not dcm_files:
        raise ValueError(f"目录中没有找到 DICOM 文件: {directory}")

    # 读取所有切片
    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(str(f), force=True)
        if hasattr(ds, "pixel_array"):
            slices.append(ds)

    if not slices:
        raise ValueError(f"无法从 DICOM 文件中读取像素数据: {directory}")

    # 按 InstanceNumber 或 ImagePositionPatient 排序
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except (AttributeError, TypeError, IndexError):
        try:
            slices.sort(key=lambda s: int(s.InstanceNumber))
        except (AttributeError, TypeError):
            pass

    # 组装 3D 体数据
    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])

    # 应用 RescaleSlope / RescaleIntercept (CT HU 值)
    ds0 = slices[0]
    slope = float(getattr(ds0, "RescaleSlope", 1))
    intercept = float(getattr(ds0, "RescaleIntercept", 0))
    if slope != 1 or intercept != 0:
        volume = volume * slope + intercept

    if not return_info:
        return volume

    info = DicomSeriesInfo(directory=directory)
    info.num_slices = len(slices)
    info.rows = ds0.Rows
    info.cols = ds0.Columns
    info.modality = str(getattr(ds0, "Modality", "Unknown"))
    info.patient_name = str(getattr(ds0, "PatientName", "Anonymous"))
    info.series_description = str(getattr(ds0, "SeriesDescription", ""))

    ps = getattr(ds0, "PixelSpacing", [1.0, 1.0])
    info.pixel_spacing = (float(ps[0]), float(ps[1]))
    info.slice_thickness = float(getattr(ds0, "SliceThickness", 1.0))

    wc = getattr(ds0, "WindowCenter", 40)
    ww = getattr(ds0, "WindowWidth", 400)
    info.window_center = float(wc) if not isinstance(wc, pydicom.multival.MultiValue) else float(wc[0])
    info.window_width = float(ww) if not isinstance(ww, pydicom.multival.MultiValue) else float(ww[0])

    return volume, info


# ============================================================
#  预处理工具
# ============================================================

def apply_window(
    image: np.ndarray,
    window_center: float = 40.0,
    window_width: float = 400.0,
) -> np.ndarray:
    """
    窗宽窗位调整 (Window/Level)

    将 CT HU 值映射到 [0, 1]:
    - 骨窗: WC=300, WW=1500
    - 肺窗: WC=-600, WW=1500
    - 脑窗: WC=40, WW=80
    - 腹部: WC=40, WW=400
    """
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    result = np.clip(image, min_val, max_val)
    result = (result - min_val) / (max_val - min_val + 1e-8)
    return result.astype(np.float32)


def normalize_volume(volume: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    体数据归一化

    method:
    - "minmax" : 线性缩放到 [0, 1]
    - "zscore" : 零均值单位方差 (减均值/除标准差)
    - "ct_soft": CT 软组织窗 WC=40, WW=400
    - "ct_bone": CT 骨窗 WC=300, WW=1500
    - "ct_lung": CT 肺窗 WC=-600, WW=1500
    - "ct_brain": CT 脑窗 WC=40, WW=80
    """
    if method == "minmax":
        vmin, vmax = volume.min(), volume.max()
        if vmax - vmin < 1e-8:
            return np.zeros_like(volume)
        return ((volume - vmin) / (vmax - vmin)).astype(np.float32)
    elif method == "zscore":
        mean, std = volume.mean(), volume.std()
        if std < 1e-8:
            return np.zeros_like(volume)
        return ((volume - mean) / std).astype(np.float32)
    elif method == "ct_soft":
        return apply_window(volume, 40, 400)
    elif method == "ct_bone":
        return apply_window(volume, 300, 1500)
    elif method == "ct_lung":
        return apply_window(volume, -600, 1500)
    elif method == "ct_brain":
        return apply_window(volume, 40, 80)
    else:
        raise ValueError(f"未知归一化方法: {method}")


def extract_2d_slices(
    volume: np.ndarray,
    axis: int = 0,
    target_size: int | None = 256,
    skip_empty: bool = True,
    empty_threshold: float = 0.01,
) -> list[np.ndarray]:
    """
    从 3D volume 中提取 2D 切片

    参数
    ----
    volume : (D, H, W) 3D 体数据 (已归一化到 [0,1])
    axis : 沿哪个轴切片 (0=axial, 1=coronal, 2=sagittal)
    target_size : 目标尺寸 (resize), None=不缩放
    skip_empty : 跳过几乎全黑的切片
    empty_threshold : 非零像素比例低于此值视为空

    返回
    ----
    切片列表, 每个 shape = (target_size, target_size) 或原始尺寸
    """
    from skimage.transform import resize as sk_resize

    slices = []
    n = volume.shape[axis]

    for i in range(n):
        if axis == 0:
            s = volume[i, :, :]
        elif axis == 1:
            s = volume[:, i, :]
        elif axis == 2:
            s = volume[:, :, i]
        else:
            raise ValueError(f"axis 必须为 0, 1, 2, 得到: {axis}")

        # 跳过空切片
        if skip_empty:
            nonzero_ratio = (s > 0.01).sum() / s.size
            if nonzero_ratio < empty_threshold:
                continue

        # 缩放到目标尺寸
        if target_size is not None and (s.shape[0] != target_size or s.shape[1] != target_size):
            s = sk_resize(s, (target_size, target_size),
                          preserve_range=True, anti_aliasing=True).astype(np.float32)

        slices.append(s)

    return slices


def scan_and_summarize(root_dir: str, max_series: int = 50) -> list[DicomSeriesInfo]:
    """
    扫描目录, 汇总所有找到的 DICOM 序列信息

    返回
    ----
    DicomSeriesInfo 列表
    """
    dirs = find_dicom_series(root_dir)
    print(f"在 {root_dir} 下找到 {len(dirs)} 个 DICOM 序列目录")

    results = []
    for i, d in enumerate(dirs[:max_series]):
        try:
            _, info = read_dicom_series(d, return_info=True)
            results.append(info)
            print(f"  [{i+1:3d}] {info.modality:4s} | "
                  f"{info.num_slices:4d} slices | "
                  f"{info.rows}×{info.cols} | "
                  f"spacing=({info.pixel_spacing[0]:.2f},{info.pixel_spacing[1]:.2f},{info.slice_thickness:.2f}) | "
                  f"{info.series_description[:40]}")
        except Exception as e:
            print(f"  [{i+1:3d}] 错误: {d} → {e}")

    return results
