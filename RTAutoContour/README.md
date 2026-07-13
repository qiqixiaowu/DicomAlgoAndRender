# RTAutoContour — 放射治疗自动勾画算法库

> 提炼自联影 McsfAlgoAutoContour 工程，纯 C++17 header-only 实现，零外部依赖

## 目录结构

```
RTAutoContour/
├── CMakeLists.txt                     # 构建配置
├── README.md                          # 本文件
├── include/                           # Header-only 算法库
│   ├── rt_types.hpp                   # 公共类型（Image2D/3D, 轮廓, 定位点）
│   ├── morphology.hpp                 # 2D/3D 形态学（腐蚀/膨胀/闭运算/孔洞填充）
│   ├── connected_component.hpp        # 连通域标记 + 区域属性 + 边界追踪
│   ├── contour_mask.hpp               # 轮廓 ↔ 掩膜互转（含超采样）
│   ├── live_wire.hpp                  # Live-Wire 智能剪刀（Dijkstra）
│   ├── interpolation.hpp              # 三线性/最近邻插值重采样
│   ├── ct_reset_match.hpp             # CT 切片匹配（骨结构投影 + NCC）
│   ├── fiducial_detection.hpp         # 金属定位点自动检测
│   └── bed_board_detection.hpp        # 治疗床板检测与校准
└── demo/
    └── main_demo.cpp                  # 综合演示程序
```

## 构建与运行

```bash
cd RTAutoContour
mkdir build && cd build
cmake ..
cmake --build . --config Release
./rtac_demo              # Linux
.\Release\rtac_demo.exe  # Windows
```

## 算法概览

### 1. 形态学操作 (`morphology.hpp`)

| 函数 | 说明 |
|------|------|
| `Erosion2D` | 2D 单像素腐蚀（4邻域） |
| `Dilation2D` | 2D 单像素膨胀（4邻域） |
| `Opening2D` | 2D 开运算 = 先腐蚀后膨胀 |
| `Closing2D` | 2D 闭运算 = 先膨胀后腐蚀 |
| `FillHoles2D` | 2D 孔洞填充（floodfill 外部取反） |
| `Erosion3D` | 3D 球形结构元素腐蚀 |
| `Dilation3D` | 3D 球形结构元素膨胀 |
| `Closing3D` | 3D 闭运算 |
| `GenerateBallMask` | 生成球形结构元素 |

### 2. 连通域分析 (`connected_component.hpp`)

| 函数 | 说明 |
|------|------|
| `LabelConnectedComponents2D` | 2D BFS 连通域标记（4/8连通） |
| `LabelConnectedComponents3D` | 3D BFS 连通域标记（6/26连通） |
| `ComputeRegionProps` | 面积、质心、包围盒、主轴、偏心率 |
| `KeepLargestComponent2D/3D` | 仅保留最大连通域 |
| `ExtractBoundary2D` | Moore 边界追踪提取有序轮廓 |

### 3. 轮廓 ↔ 掩膜互转 (`contour_mask.hpp`)

| 函数 | 说明 |
|------|------|
| `ContourToMask` | 轮廓→掩膜（闭合边缘 + floodfill + 超采样） |
| `MaskToContours` | 掩膜→轮廓（连通域 + Moore 边界追踪） |
| `ContourToEdgeMask` | 轮廓光栅化为边缘掩膜 |
| `InterpolateLinear` | 轮廓线性加密 |
| `InterpolateSpline` | Catmull-Rom 样条细分 |
| `ContoursToVolumeMask` | 逐层轮廓→3D体积掩膜（XOR 处理嵌套） |

### 4. Live-Wire 智能剪刀 (`live_wire.hpp`)

基于 Dijkstra 最短路径的交互式轮廓勾画工具。

```cpp
LiveWire lw;
lw.initialize(image);    // 计算梯度代价图
lw.setAnchor(x0, y0);   // 用户点击锚点
auto path = lw.getLiveWire(x1, y1);  // 实时最优路径
lw.commitSegment(x1, y1);            // 确认路径段
auto mask = lw.closeAndFill();       // 闭合填充
```

### 5. 插值重采样 (`interpolation.hpp`)

| 函数 | 说明 |
|------|------|
| `TrilinearSample` | 单点三线性插值 |
| `NearestSample` | 单点最近邻插值 |
| `ResampleToSize` | 按目标尺寸重采样 |
| `ResampleToSpacing` | 按目标 spacing 重采样 |
| `ResampleZUniform` | Z 轴非等间距→等间距 |

### 6. CT 切片匹配 (`ct_reset_match.hpp`)

将计划 CT 单层与重建 CT 体数据匹配，找最相似的 Z 位置。

```
骨结构阈值分割(300HU) → 行/列投影 → 归一化互相关(NCC) → 最佳切片
```

### 7. 金属定位点检测 (`fiducial_detection.hpp`)

```
双阈值候选检出 → 3D连通域 → 形态筛选 → Left/Right/Top分类 → Z分组 → Center计算
```

### 8. 床板检测 (`bed_board_detection.hpp`)

```
Z-MIP → 阈值分割 → 形态学净化 → 逐层行位置追踪 → 二次多项式拟合
```

## 与原工程的对应关系

| 本库模块 | 原工程模块 | 原工程文件 |
|----------|-----------|------------|
| `morphology.hpp` | AutoContourCommon | McsfAlgoAutoContourCommon.cpp |
| `connected_component.hpp` | AutoContourCommon | McsfAlgoAutoContourCommon.cpp |
| `contour_mask.hpp` | BasicFunction | McsfAlgoAutoContourBasicFunction.cpp |
| `live_wire.hpp` | SmartContour/IntellSci | McsfAlgoRTSmartContour.cpp |
| `interpolation.hpp` | McsfAlgoInterpolation | McsfAlgoAutoContourInterpolation.cpp |
| `ct_reset_match.hpp` | BasicFunction | McsfAlgoAutoContourBasicFunction.cpp |
| `fiducial_detection.hpp` | FiducialDetection | McsfAlgoRTFiducialDetection.cpp |
| `bed_board_detection.hpp` | BedBoardCalibration | McsfAlgoBedBoardCalibration.cpp |

## 设计说明

**与原工程的差异**：
- 原工程依赖 IPP/MKL/Boost/OpenMP 等重量级库
- 本库为纯 C++17 header-only，零外部依赖
- 使用模板 `Image2D<T>` / `Image3D<T>` 替代原工程的裸指针
- 所有算法内存安全，无裸指针泄漏风险
- 算法核心逻辑保持一致，接口更现代化
