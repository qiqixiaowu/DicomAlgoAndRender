# NailPrint3D — 3D美甲打印系统

## 概述

NailPrint3D 是一个完整的3D美甲打印工程，支持三大核心模块：

1. **3D重建 (Reconstruction)** — 从点云/STL网格重建可打印的曲面模型
2. **RIP切片 (Slicing)** — 将3D模型切分为2D层，生成打印路径
3. **色彩管理 (Color Management)** — ICC色彩转换、LUT调色、多色映射、抖动量化

## 项目结构

```
NailPrint3D/
├── include/                     # 头文件
│   ├── nail_types.h             # 核心数据结构 (Vec3, Mesh, Polygon, Color等)
│   ├── stl_loader.h             # STL加载器 + 美甲网格生成器
│   ├── nail_reconstruction.h    # 3D重建 (Marching Cubes, Poisson, 网格修复)
│   ├── nail_slicer.h            # RIP切片引擎
│   ├── nail_color.h             # 色彩管理 (ICC, LUT, 抖动, 多色映射)
│   ├── nail_gcode.h             # G-code生成器
│   └── nail_renderer.h          # OpenGL渲染器
├── src/                         # 源文件
│   ├── stl_loader.cpp
│   ├── nail_reconstruction.cpp
│   ├── nail_slicer.cpp
│   ├── nail_color.cpp
│   ├── nail_gcode.cpp
│   ├── nail_renderer.cpp
│   └── main_nail_print.cpp      # 主程序入口
├── shaders/                     # GLSL着色器
│   ├── nail_mesh.vert/frag      # 实体渲染 (Blinn-Phong + 菲涅尔)
│   ├── nail_normal.vert/frag    # 法线可视化
│   ├── nail_slice_line.vert/frag # 切片路径预览
│   └── nail_palette.vert/frag   # 调色板颜色预览
├── NailPrint3D.vcxproj          # VS工程文件
└── NailPrint3D.vcxproj.filters  # VS过滤器文件
```

## 核心模块详解

### 1. 3D重建模块 (`nail_reconstruction.h/cpp`)

| 组件 | 功能 |
|------|------|
| `MarchingCubes` | 等值面提取 (从体素/SDF生成网格) |
| `PoissonReconstruction` | 点云到网格重建 (法线估计+泼溅) |
| `MeshRepair` | 网格修复 (补洞、法线统一、顶点焊接、拉普拉斯平滑) |
| `Voxelizer` | 网格转体素 + SDF生成 |
| `ReconstructionPipeline` | 完整重建流水线 |

### 2. RIP切片模块 (`nail_slicer.h/cpp`)

| 组件 | 功能 |
|------|------|
| `NailSlicer` | 等距/自适应切片、三角形-平面求交、轮廓连接 |
| `NailPrintSlicer` | 美甲专用切片 (底胶层/颜色层/封层分层) |
| 填充模式 | 直线、网格、蜂窝、同心圆 |
| 路径优化 | 最近邻贪心TSP、空行程插入 |
| 支撑生成 | 悬垂检测、稀疏支撑 |

### 3. 色彩管理模块 (`nail_color.h/cpp`)

| 组件 | 功能 |
|------|------|
| `ColorSpaceConverter` | sRGB↔XYZ↔Lab↔CMYK, DeltaE/DeltaE2000 |
| `ICCColorManager` | ICC配置文件加载、色彩空间转换 |
| `LUTManager` | 3D LUT加载/保存/生成 (Identity/Gamma/白平衡) |
| `DitherProcessor` | Floyd-Steinberg、Bayer抖动、调色板量化 |
| `MultiColorMapper` | 纹理到层的颜色映射、颜色过渡混合 |
| `ColorCalibrator` | 色块校准、DeltaE评估 |
| `VoxelColorManager` | 体素颜色管理、3D纹理导出 |

### 4. G-code生成 (`nail_gcode.h/cpp`)

- 多色打印支持 (M163颜色切换)
- UV固化控制 (M240/M241)
- 回抽/回退、擦嘴
- 层/路径/颜色切换/UV固化分段生成
- 打印统计输出

### 5. OpenGL渲染 (`nail_renderer.h/cpp`)

- 轨道相机 (鼠标旋转/平移/缩放)
- 4种渲染模式: 实体/线框/切片预览/颜色预览
- Blinn-Phong光照 + 菲涅尔光泽效果
- 切片路径逐层高亮预览

## 打印流程

```
生成美甲网格 → 网格修复 → RIP切片 → 色彩管理 → G-code生成 → 实时预览
     │              │           │           │           │
     ▼              ▼           ▼           ▼           ▼
  nail_patch.stl  顶点焊接    等距/自适应   ICC转换    nail_print.gcode
  (参数化曲面)    补洞平滑    轮廓+填充     LUT调色    (多色+UV固化)
                             路径优化      抖动量化
```

## 美甲专用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `nailBedWidth` | 15mm | 美甲宽度 |
| `nailBedLength` | 20mm | 美甲长度 |
| `nailCurvature` | 0.3 | 曲率 |
| `baseThickness` | 0.3mm | 底胶层厚度 |
| `colorLayerHeight` | 0.1mm | 颜色层高 |
| `topCoatThickness` | 0.2mm | 封层厚度 |
| `colorCount` | 4 | 多色数量 |

## 构建与运行

### 依赖

- Visual Studio 2022 (v143工具集)
- C++17
- OpenGL 3.3+
- GLFW, GLAD, GLM (与主解决方案共享)

### 构建

1. 在VS2022中打开 `OpenglRender.sln`
2. 选择 `NailPrint3D` 项目
3. 配置: `Debug|x64` 或 `Release|x64`
4. 生成解决方案

### 运行

程序启动后自动执行完整流程：
1. 生成美甲网格并保存 `nail_patch.stl`
2. 网格修复
3. RIP切片
4. 色彩管理 (ICC + LUT + 抖动)
5. 生成 `nail_print.gcode`
6. 打开OpenGL预览窗口

### 操作说明

| 操作 | 功能 |
|------|------|
| 鼠标左键拖动 | 旋转视角 |
| 鼠标右键拖动 | 平移视角 |
| 鼠标滚轮 | 缩放 |
| 键盘 `1` | 实体渲染 |
| 键盘 `2` | 线框渲染 |
| 键盘 `3` | 切片预览 (↑↓切换层) |
| 键盘 `4` | 颜色预览 |
| `ESC` | 退出 |

## 输出文件

| 文件 | 说明 |
|------|------|
| `nail_patch.stl` | 生成的美甲3D网格 |
| `nail_colors.bin` | 体素颜色数据 |
| `nail_print.gcode` | 打印G-code (含多色+UV固化) |

## 技术要点

- **参数化美甲曲面**: 使用余弦函数生成自然弧度的指甲表面
- **自适应切片**: 根据轮廓复杂度动态调整层高
- **CIEDE2000色差公式**: 精确的颜色匹配和量化
- **Floyd-Steinberg抖动**: 减少多色打印的颜色条带
- **UV固化G-code**: 每层打印后自动插入UV固化指令
