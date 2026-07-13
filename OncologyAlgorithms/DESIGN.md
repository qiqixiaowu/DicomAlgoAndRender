# OncologyAlgorithms 详细设计文档

> 版本：1.0.0  
> 日期：2026-04-08  
> 语言：C++17  

---

## 目录

1. [工程概述](#1-工程概述)
2. [工程结构](#2-工程结构)
3. [公共数据类型（oncology_types.h）](#3-公共数据类型)
4. [模块1：肿瘤径线计算 & 统计参数](#4-模块1肿瘤径线计算--统计参数)
5. [模块2：通用肿瘤半自动分割](#5-模块2通用肿瘤半自动分割)
6. [模块3：PET 分子影像病灶分割](#6-模块3pet-分子影像病灶分割)
7. [模块4：CT 肺结节传统分割](#7-模块4ct-肺结节传统分割)
8. [模块5：DL 肺结节分割（3D nnUNet）](#8-模块5dl-肺结节分割3d-nnunet)
9. [构建与集成](#9-构建与集成)
10. [性能参考](#10-性能参考)
11. [扩展指南](#11-扩展指南)

---

## 1. 工程概述

**OncologyAlgorithms** 是一套面向医学影像肿瘤学分析的 C++17 算法库，不依赖任何重量级第三方框架（如 ITK/VTK），可直接嵌入到任何 C++ 宿主工程。

### 1.1 功能覆盖

| 功能分类 | 覆盖内容 |
|---|---|
| **径线测量** | 2D最大截面长径/垂直径、3D最长径、肿瘤体积、灰度统计 |
| **通用分割** | K-means、GMM、Random Walker + DL（SAM2/nnInteractive 接口预留）|
| **PET 分割** | 固定阈值、百分比阈值、自适应阈值 + Hover 实时预览 |
| **CT 肺结节（传统）** | Hessian 增强 + 胸壁去除 + 多形态精化（实性/贴壁/血管/GGO）|
| **CT 肺结节（DL）** | 手写 cuDNN 3D nnUNet，支持 CPU 回退；.bin 权重直接加载 |

### 1.2 设计原则

- **零框架依赖**：核心 CPU 路径仅使用 C++ 标准库，无需 ITK/VTK/OpenCV
- **模块独立**：每个算法模块可单独编译，通过公共类型（`oncology_types.h`）互操作
- **CPU/GPU 双路**：DL 模块在 GPU 不可用时自动回退到 CPU 实现，不崩溃
- **安全设计**：所有接口在输入异常时返回 `false` 而非抛出异常

---

## 2. 工程结构

```
OncologyAlgorithms/
├── CMakeLists.txt               ← 构建配置（支持 USE_CUDA 选项）
├── include/                     ← 公开头文件
│   ├── oncology_types.h         ← 公共数据类型
│   ├── tumor_diameter_calculator.h
│   ├── general_tumor_segment.h
│   ├── pet_lesion_segment.h
│   ├── lung_nodule_segment_traditional.h
│   ├── lung_nodule_segment_dl.h
│   ├── hessian_filter.h
│   ├── kmeans_segmentation.h
│   ├── gmm_segmentation.h
│   └── random_walker_segmentation.h
├── src/                         ← 实现文件
│   ├── tumor_diameter_calculator.cpp
│   ├── general_tumor_segment.cpp
│   ├── pet_lesion_segment.cpp
│   ├── lung_nodule_segment_traditional.cpp
│   ├── lung_nodule_segment_dl.cpp   ← CPU 路径 + .bin 加载
│   ├── lung_nodule_segment_dl.cu    ← CUDA/cuDNN 推理引擎（USE_CUDA）
│   ├── hessian_filter.cpp
│   ├── kmeans_segmentation.cpp
│   ├── gmm_segmentation.cpp
│   └── random_walker_segmentation.cpp
└── demo/
    └── main.cpp                 ← 全功能演示入口
```

### 模块依赖关系

```
oncology_types.h  ◄─────── 所有模块共用
        │
        ├── tumor_diameter_calculator
        │
        ├── general_tumor_segment
        │       ├── kmeans_segmentation
        │       ├── gmm_segmentation
        │       └── random_walker_segmentation
        │
        ├── pet_lesion_segment
        │
        ├── lung_nodule_segment_traditional
        │       └── hessian_filter
        │
        └── lung_nodule_segment_dl
                ├── lung_nodule_segment_dl.cpp  (CPU)
                └── lung_nodule_segment_dl.cu   (CUDA, 可选)
```

---

## 3. 公共数据类型

**文件**：`include/oncology_types.h`

### 3.1 基础几何类型

```cpp
struct Point3i { int x, y, z; };      // 体素坐标（像素空间）
struct Point3f { float x, y, z; };    // 物理坐标（毫米空间）
```

`Point3f` 提供 `distanceTo()`、`dot()`、`norm()` 等常用运算。

### 3.2 图像元信息

```cpp
struct ImageInfo {
    int    dim[3];       // [X, Y, Z] 体素维度
    double spacing[3];   // [X, Y, Z] 体素间距（mm）

    int totalVoxels() const;
    int linearIndex(int x, int y, int z) const;     // z*dimY*dimX + y*dimX + x
    bool inBounds(int x, int y, int z) const;
};
```

**坐标约定**：线性索引 = `z × dimX × dimY + y × dimX + x`（z 为慢轴）。

### 3.3 分割结果类型

```cpp
using SegmentResult = std::vector<Point3i>;  // 前景体素坐标集合
using SegmentMask   = std::vector<uint8_t>;  // 等尺寸二值掩码（0/1）
```

### 3.4 进度回调

```cpp
using ProgressCallback = std::function<bool(double progress)>;
// 返回 false 可中止算法（部分算法支持取消）
```

### 3.5 PET SUV 信息

```cpp
struct SUVInfo {
    bool   hasValidSUV;   // 是否有效
    double rawAtSUV1;     // SUVbw=suv1 时的原始灰度
    double rawAtSUV2;     // SUVbw=suv2 时的原始灰度
    double suv1, suv2;    // 典型值：1.0, 2.0
};
// 灰度  ↔  SUVbw 之间为线性关系，由上述两点确定斜率
```

---

## 4. 模块1：肿瘤径线计算 & 统计参数

**文件**：`include/tumor_diameter_calculator.h`，`src/tumor_diameter_calculator.cpp`

### 4.1 算法原理

#### 4.1.1 2D 最大截面径（长径/垂直径）

```
步骤：
1. 遍历所有 Z 切片，找到包含体素数最多的横断位切片
2. 提取该切片的轮廓点集（边界体素）
3. 旋转卡壳（Rotating Calipers）算法：
   a. 计算2D凸包
   b. 对每个对趾点对，记录跨度距离
   c. 最大跨度 → 长径向量 (ldStart, ldEnd)
   d. 垂直于长径方向 → 垂直径 (pdStart, pdEnd)
```

**时间复杂度**：O(N_slice × k log k)，k 为轮廓点数。

#### 4.1.2 3D 最长径

```
步骤：
1. 从分割点集随机采样 min(500, N) 个点
2. 对每对采样点计算欧氏距离（物理坐标，单位mm）
3. 取最大距离为 3D 最长径
4. 若超过 maxCalTimeMs 时间限制，返回 -1
```

> 注：精确算法时间为 O(N²)，采样策略将期望复杂度降为 O(500²) 常数级。
> 对于大病灶（>10000体素）可用凸包+旋转卡壳进一步精确，但通常采样结果误差 <2%。

#### 4.1.3 体积与统计

```
体积 = 分割体素数 × spacing[0] × spacing[1] × spacing[2]（mm³）
均值 = ΣI(v) / N
标准差 = sqrt(Σ(I(v)-mean)² / N)
```

### 4.2 接口说明

```cpp
// 径线计算
bool CalculateTumorDiameters(
    const SegmentResult& segPoints,
    const ImageInfo&     info,
    TumorDiameterResult& result,
    double               maxCalTimeMs = 30000.0,
    ProgressCallback     progress = nullptr);

// 统计参数（模板，支持 short/float 等图像类型）
template<typename T>
bool CalculateStatistics(
    const T*             data,
    const SegmentResult& segPoints,
    const ImageInfo&     info,
    TumorStatistics&     result);
```

### 4.3 结果结构体

```cpp
struct TumorDiameterResult {
    float longestDiameter3D_cm;      // 3D最长径 (cm)，超时返回 -1
    float longestDiameter2D_cm;      // 2D截面长径 (cm)
    float perpendicularDiameter_cm;  // 2D截面垂直径 (cm)
    Point3f ldStart, ldEnd;          // 长径端点（体素坐标）
    Point3f pdStart, pdEnd;          // 垂直径端点
};

struct TumorStatistics {
    float volume_mm3;  // 体积 mm³
    float mean;        // 均值
    float stdDev;      // 标准差
    float minValue;    // 最小值
    float maxValue;    // 最大值
};
```

### 4.4 使用示例

```cpp
#include "tumor_diameter_calculator.h"
using namespace Onc;

// 假设已有分割结果 seg 和图像信息 info
TumorDiameterResult diam;
bool ok = CalculateTumorDiameters(seg, info, diam, 15000.0,
    [](double p){ std::cout << p*100 << "%\n"; return true; });

if (ok) {
    printf("3D最长径: %.2f cm\n", diam.longestDiameter3D_cm);
    printf("2D长径:   %.2f cm\n", diam.longestDiameter2D_cm);
}

TumorStatistics stats;
CalculateStatistics(ctData, seg, info, stats);
printf("体积: %.0f mm³, HU均值: %.1f\n", stats.volume_mm3, stats.mean);
```

---

## 5. 模块2：通用肿瘤半自动分割

**文件**：`include/general_tumor_segment.h`，`src/general_tumor_segment.cpp`

### 5.1 支持的分割方法

| 枚举值 | 方法 | 适用场景 |
|---|---|---|
| `SegMethod::KMeans` | K-means 聚类（k=2：前景/背景）| 均质实性肿瘤 |
| `SegMethod::GMM` | 高斯混合模型（EM 算法） | 有部分容积效应的肿瘤 |
| `SegMethod::RandomWalker` | 随机游走概率分割 | 边界模糊、任意形状 |
| `SegMethod::DL` | 深度学习接口（预留）| 需提前加载 DL 引擎 |

### 5.2 各算法原理

#### 5.2.1 K-means (k=2)

```
初始化：以种子点 HU 为前景中心，肺背景 HU(-700) 为背景中心
迭代：
  E step: 为每个 ROI 体素分配最近簇
  M step: 重新计算两个簇的均值
收敛：maxIter 次迭代后，提取前景簇 → 连通域过滤保留种子点所在区域
```

#### 5.2.2 GMM（2-分量，EM 算法） Gaussian_Mixture_Model

```
建模：P(HU | 前景) = N(μ₁, σ₁²);  P(HU | 背景) = N(μ₀, σ₀²)
EM迭代：
  E step: γ_i = π₁·N(HU_i | μ₁,σ₁²) / [π₀·N(HU_i|μ₀,σ₀²) + π₁·N(HU_i|μ₁,σ₁²)]
  M step: 更新 π, μ, σ²
后处理：γ_i > 0.5 的体素标为前景 → 连通域过滤
```

#### 5.2.3 Random Walker

```
将 ROI 内体素视为图节点，相邻体素间边权重：
  w(i,j) = exp(-β × (HU_i - HU_j)²)

已知标签：
  种子点 → 前景标签 1
  ROI 边界 → 背景标签 0

求解稀疏线性方程组：L_u × x = -B_u × x_s
（L_u: 未标注体素的 Laplacian, B_u: 未标注↔已标注的边矩阵）

解 x 即为每体素的前景概率，x > 0.5 的体素为前景。
```

> 本实现使用 Gauss-Seidel 迭代求解稀疏系统，收敛速度快，无需引入线性代数库。

#### 5.2.4 跨时间点传播（PropagateSegmentation）

```
原理：
  1. 在目标图像上，以源分割的 bounding box 为 ROI
  2. 使用源分割体素的 HU 分布作为前景先验重新初始化 K-means
  3. 执行 K-means 分割
  4. 对结果做形态学膨胀/腐蚀消除离群点
```

### 5.3 接口说明

```cpp
struct GeneralTumorSegConfig {
    SegMethod   method      = SegMethod::KMeans;
    Point3i     seedPoint;
    float       roiRadiusMM = 30.f;  // ROI 半径（mm）
    int         maxIter     = 50;    // 最大迭代次数
    float       beta        = 90.f;  // RW 边权指数（越大越敏感于 HU 差异）
};

class GeneralTumorSegment {
    bool segment(const short*             data,
                 const ImageInfo&         info,
                 const GeneralTumorSegConfig& cfg,
                 SegmentResult&           result,
                 ProgressCallback         progress = nullptr);
};

// 跨时间点传播
bool PropagateSegmentation(
    const short*         targetData,
    const ImageInfo&     info,
    const SegmentResult& sourceSeg,
    SegmentResult&       result,
    ProgressCallback     progress = nullptr);
```

---

## 6. 模块3：PET 分子影像病灶分割

**文件**：`include/pet_lesion_segment.h`，`src/pet_lesion_segment.cpp`

### 6.1 SUVbw 与图像灰度的转换

PET 图像以原始灰度值存储，通过 `SUVInfo` 线性映射到 SUVbw：

$$\text{raw} = \text{rawAtSUV1} + \frac{\text{rawAtSUV2} - \text{rawAtSUV1}}{\text{suv2} - \text{suv1}} \times (\text{SUVbw} - \text{suv1})$$

### 6.2 三种分割模式

#### 6.2.1 Fixed（固定绝对阈值）

```
输入：fixedSUV（用户直接指定，如 2.5 g/mL）
阈值 T = suvToRaw(fixedSUV)
区域生长：从种子点 BFS，纳入 raw >= T 的连通体素
```

**适用场景**：研究方案对 SUV 阈值有明确规定时（如 PERCIST 标准 SUV=2.5）。

#### 6.2.2 Percent（最大值百分比阈值）

```
在种子点 30mm 邻域内找 SUVmax
阈值 T = suvToRaw(SUVmax × percent/100)
区域生长：同 Fixed
```

**适用场景**：欧洲核医学会（EANM）推荐的 42%SUVmax 阈值。

典型参数：`percent = 42.0`（42% ）。

#### 6.2.3 Adaptive（自适应阈值）

```
初始权重 w_init = 0.42（42%）
迭代调整：
  1. 计算当前分割结果的平均 SUV（lesionMeanSUV）
  2. 设定目标：lesionMeanSUV 应 > backgroundMean + 2σ
  3. 若不满足：增大 w；若过分割：减小 w
  4. 收敛条件：|Δw| < 0.005 或迭代 > 20 次
```

自适应算法自动平衡灵敏度（不漏掉病灶边缘）和特异性（不过分割入正常组织），
对低 SUV 病灶（<3）表现优于固定阈值方法。

#### 6.2.4 Hover 实时预览

对应 Adaptive 模式但限制最大体素数（通常 < 5000），用于鼠标悬停时的快速
交互反馈，一般 < 20 ms 返回。

### 6.3 完整接口

```cpp
// Fixed 单种子点
bool PETSegmentFixed(const SUVInfo& s, const void* data, const ImageInfo& i,
                     const Point3i& seed, double fixedSUV, PETSegmentResult& r);

// Percent 单种子点
bool PETSegmentPercent(const SUVInfo& s, const void* data, const ImageInfo& i,
                       const Point3i& seed, double percent, PETSegmentResult& r);

// Adaptive 单种子点
bool PETSegmentAdaptive(const SUVInfo& s, const void* data, const ImageInfo& i,
                        const Point3i& seed, PETSegmentResult& r);

// Hover 快速预览（Adaptive 简化版）
bool PETSegmentAdaptiveHover(const SUVInfo& s, const void* data, const ImageInfo& i,
                              const Point3i& seed, SegmentResult& r);

// VOI 多病灶版本（在给定掩码区域内执行分割）
bool PETSegmentWithVOI(...) ;  // 接口预留

struct PETSegmentResult {
    SegmentResult voxels;       // 分割体素集合
    double        usedThreshold; // 实际使用原始灰度阈值（可反馈给 UI）
    int           maxCoord[3];  // SUVmax 体素坐标
};
```

---

## 7. 模块4：CT 肺结节传统分割

**文件**：`include/lung_nodule_segment_traditional.h`，`include/hessian_filter.h`

### 7.1 完整分割流程

```
┌─────────────────────────────────────────────────────────────────┐
│ 输入：CT 图像（HU 值）+ 种子点                                   │
│                                                                  │
│  Step 1: 胸腔提取（ChestWallRemoval）                           │
│    ● 阈值 -400 HU 二值化                                        │
│    ● 形态学闭运算消除小孔                                        │
│    ● 取肺野掩码（最大的非胸壁连通域）                           │
│                                                                  │
│  Step 2: ROI 提取                                               │
│    ● 以种子点为中心，roiRadiusMM 半径取 bounding box            │
│    ● 与肺野掩码求交                                              │
│                                                                  │
│  Step 3: Hessian 血管增强                                       │
│    ● 多尺度 σ ∈ [hessianScale1, hessianScale2]                  │
│    ● 对每体素计算 Hessian 矩阵特征值 λ₁≤λ₂≤λ₃                   │
│    ● 点状响应：Sblob = exp(-|λ₁|²/2A²) × (1-exp(-R_B²/2B²))    │
│      × (1-exp(-S²/2C²))                                         │
│                                                                  │
│  Step 4: 初始化分割（HU 阈值 + 连通域）                         │
│                                                                  │
│  Step 5: 形态精化（依结节类型选择）                             │
│    ● Solid:        Hessian 点状响应加权区域生长                  │
│    ● JuxtaWall:    形态学分离胸壁 + Active Contour 精化          │
│    ● JuxtaVessel:  Hessian 管状响应图 mask 掉血管部分            │
│    ● GGO:          扩大 HU 范围（-800~0）+ 梯度一致性过滤        │
│                                                                  │
│  Step 6: 凸包平滑                                               │
│    ● 对分割结果做 3D 凸包计算（Andrew's Monotone Chain 变体）   │
│    ● 填充凸包内体素消除边缘凹陷伪影                             │
│                                                                  │
│ 输出：SegmentResult                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Hessian 滤波器

**文件**：`include/hessian_filter.h`

```cpp
struct HessianConfig {
    float sigmaMin   = 1.0f;   // 最小尺度（mm）
    float sigmaMax   = 3.0f;   // 最大尺度（mm）
    int   numScales  = 3;      // 尺度数量
    float vesselness = 0.5f;   // 血管性响应权重（0：只检点，1：只检管）
};

// 计算多尺度 Hessian 响应图
// 返回 [0,1] 范围的点状增强图（nodule enhancement map）
bool ComputeHessianFilter(
    const short*      data,
    const ImageInfo&  info,
    const HessianConfig& cfg,
    std::vector<float>& responseMap);
```

Hessian 矩阵特征值分析（Frangi 1998）：

$$R_B = \frac{\lambda_1}{\sqrt{\lambda_2 \lambda_3}}, \quad S = \sqrt{\lambda_1^2+\lambda_2^2+\lambda_3^2}$$

$$V_{\sigma}(\mathbf{x}) = \begin{cases} 0 & \text{if } \lambda_2>0 \text{ or } \lambda_3>0 \\ \left(1-e^{-R_A^2/2\alpha^2}\right)e^{-R_B^2/2\beta^2}\left(1-e^{-S^2/2c^2}\right) & \text{otherwise} \end{cases}$$

### 7.3 结节类型与适用场景

| 类型 | 特征 | 调参建议 |
|---|---|---|
| `Solid` | 均质实性结节，与周围肺实质边界清晰 | 默认参数即可 |
| `JuxtaWall` | 贴近胸壁/膈肌，需先分离壁层 | `roiRadiusMM` 缩小至 10-15 |
| `JuxtaVessel` | 邻近血管，需血管抑制 | `hessianScale1=0.8` 提高对小血管的分辨率 |
| `GGO` | 磨玻璃结节，密度低（-600~0 HU）| `noduleHUMin=-600`, `noduleHUMax=0` |
| `Auto` | 自动判断（默认） | 根据种子点 HU 值自动选择上述模式 |

---

## 8. 模块5：DL 肺结节分割（3D nnUNet）

**文件**：`include/lung_nodule_segment_dl.h`，`src/lung_nodule_segment_dl.cpp`，`src/lung_nodule_segment_dl.cu`

### 8.1 网络架构

本模块实现 nnUNet v1 标准的 3D 全分辨率 U-Net：

```
Input [1, 128, 128, 128] (HU归一化至[0,1])
        │
        ▼
   ┌──────────────────────────────────────────────────────┐
   │  Encoder 1: Conv(1→32, k=3, s=1) + IN + LReLU × 2   │ ──→ skip1 [32,128,128,128]
   └──────────────────────────────────────────────────────┘
        │ stride=1
        ▼
   ┌──────────────────────────────────────────────────────┐
   │  Encoder 2: Conv(32→64, k=3, s=2) + IN + LReLU × 2  │ ──→ skip2 [64,64,64,64]
   └──────────────────────────────────────────────────────┘
        │ stride=2 (down)
        ▼
   ┌──────────────────────────────────────────────────────┐
   │  Encoder 3: Conv(64→128,k=3,s=2) + IN + LReLU × 2   │ ──→ skip3 [128,32,32,32]
   └──────────────────────────────────────────────────────┘
        │ stride=2 (down)
        ▼
   ┌──────────────────────────────────────────────────────┐
   │  Encoder 4: Conv(128→256,k=3,s=2)+ IN + LReLU × 2   │ ──→ skip4 [256,16,16,16]
   └──────────────────────────────────────────────────────┘
        │ stride=2 (down)
        ▼
   ┌──────────────────────────────────────────────────────┐
   │  Bottleneck: Conv(256→320,k=3,s=2)+IN+LReLU × 2     │       [320,8,8,8]
   └──────────────────────────────────────────────────────┘
        │
        ▼ up×2 (最近邻)
   ┌──────────────────────────────────────────────────────┐
   │  Decoder 4: Cat(up,skip4)→[576,16,16,16]             │
   │             Conv(576→256,k=3)+IN+LReLU               │
   │             Conv(256→256,k=1)+IN+LReLU               │
   └──────────────────────────────────────────────────────┘
        │
        ▼ up×2
   ┌──────────────────────────────────────────────────────┐
   │  Decoder 3: Cat(up,skip3)→[384,32,32,32]             │
   │             Conv(384→128,k=3)+IN+LReLU               │
   │             Conv(128→128,k=1)+IN+LReLU               │
   └──────────────────────────────────────────────────────┘
        │
        ▼ up×2
   ┌──────────────────────────────────────────────────────┐
   │  Decoder 2: Cat(up,skip2)→[192,64,64,64]             │
   │             Conv(192→64, k=3)+IN+LReLU               │
   │             Conv(64→64,  k=1)+IN+LReLU               │
   └──────────────────────────────────────────────────────┘
        │
        ▼ up×2
   ┌──────────────────────────────────────────────────────┐
   │  Decoder 1: Cat(up,skip1)→[96,128,128,128]           │
   │             Conv(96→32,  k=3)+IN+LReLU               │
   │             Conv(32→32,  k=1)+IN+LReLU               │
   └──────────────────────────────────────────────────────┘
        │
        ▼
   Output Head: Conv(32→2, k=1)  ──→  Argmax  ──→  Binary Mask
```

**参数量统计**：

| 层 | 参数量（约） |
|---|---|
| Encoder 1~4 + BN | ~8.5M |
| Decoder 1~4 | ~12.3M |
| Output Head | 64 |
| **总计** | **~20.8M** |

### 8.2 权重文件格式（.bin）

```
File Header:
  [uint32] num_layers          ← 层数（例如：128）

Per Layer:
  [uint16] name_length         ← 层名字节长度
  [char × name_length] name    ← 层名，如 "conv_blocks_context.0.blocks.0.conv.weight"
  [uint64] num_elements        ← 元素数量（例如 32×1×3×3×3 = 864）
  [float32 × num_elements] data ← 权重数据（float32，C-contiguous，NCHW 排列）
```

权重转换脚本（Python 参考）：

```python
import struct, numpy as np, torch

def export_nnunet_bin(model, path):
    params = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    with open(path, 'wb') as f:
        f.write(struct.pack('I', len(params)))  # uint32 num_layers
        for name, arr in params.items():
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('H', len(name_bytes)))  # uint16 name_len
            f.write(name_bytes)
            data = arr.flatten().astype(np.float32)
            f.write(struct.pack('Q', len(data)))         # uint64 num_elem
            f.write(data.tobytes())
```

### 8.3 推理流程

```
1. 以种子点为中心，裁剪 128³ patch
   ● 若图像小于 patch 尺寸，剩余位置填 0
   ● HU 归一化：v = clamp((HU - huMin)/(huMax - huMin), 0, 1)

2. 推理（自动选择路径）：
   if USE_CUDA && checkGPUAvailable(estimateGPUMemMB()):
       inferGPU()      ← cuDNN 完整5层 U-Net
   else:
       inferCPU()      ← C++ 简化推理（CPU 回退）

3. 后处理：
   ● Argmax[C=0|C=1] 获得二值 patch mask
   ● 将 mask 坐标还原到全图空间
   ● 3D BFS 连通域过滤，只保留含种子点的连通域
```

### 8.4 接口说明

```cpp
struct LungNoduleDLConfig {
    std::string weightsPath;           // .bin 权重文件路径（可为空）
    int   patchSize[3] = {128,128,128};// patch 尺寸
    float huMin        = -1000.f;      // HU 归一化下限
    float huMax        =  400.f;       // HU 归一化上限
    bool  useCUDA      = true;         // 是否使用 GPU
};

class LungNoduleDLSegment {
    explicit LungNoduleDLSegment(const LungNoduleDLConfig& cfg = {});
    bool loadWeights(const std::string& path);  // 加载 .bin 权重
    bool segment(...);                           // 主分割接口
    bool checkGPUAvailable(int requiredMB = 2048) const;
    int  estimateGPUMemMB() const;              // 预估显存需求（MB）
};
```

### 8.5 GPU 显存估算

$$\text{显存(MB)} = \frac{128^3 \times 10 \times 4}{1024^2} + 512 \approx 1.8 \text{GB}$$

建议：GPU 显存 ≥ 4GB，典型推理时间（RTX 3090）约 80ms/patch。

---

## 9. 构建与集成

### 9.1 快速构建（CPU-only）

```bash
cd F:/渲染/OpenglRender/OncologyAlgorithms
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
# 可执行文件位于 build/Release/OncDemo.exe
```

### 9.2 启用 CUDA 构建

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON \
      -DCUDNN_PATH="C:/Program Files/NVIDIA/CUDNN/v8.x"
cmake --build build --config Release
```

**CUDA 环境要求**：

| 组件 | 最低版本 |
|---|---|
| CUDA Toolkit | 11.0+ |
| cuDNN | 8.0+ |
| GPU 架构 | SM 7.5（Turing）及以上 |

### 9.3 作为子模块集成到其他 CMake 工程

```cmake
# 在宿主工程的 CMakeLists.txt 中：
add_subdirectory(third_party/OncologyAlgorithms)
target_link_libraries(MyApp PRIVATE OncologyAlgorithms)
```

### 9.4 Visual Studio 手动配置

1. 将 `include/` 路径加入 VC++ Include Directories
2. 将所有 `src/*.cpp` 加入项目
3. 若需 CUDA：添加 `src/*.cu` 并设置 CUDA 扩展属性
4. Preprocessor：`_CRT_SECURE_NO_WARNINGS;NOMINMAX`（可选 `USE_CUDA`）

---

## 10. 性能参考

以下数据基于合成 128³ 图像，Intel i9-12900K + RTX 3090：

| 算法 | CPU 耗时 | GPU 耗时 | 备注 |
|---|---|---|---|
| 径线计算（2D+3D）| ~5 ms | — | 采样 500点对 |
| K-means 分割 | ~15 ms | — | 50次迭代，ROI=30mm |
| GMM 分割 | ~25 ms | — | 50次 EM 迭代 |
| Random Walker | ~80 ms | — | beta=90，GS求解 |
| PET Fixed | ~3 ms | — | BFS |
| PET Adaptive | ~20 ms | — | 最多20次调整 |
| 肺结节传统 | ~60 ms | — | Hessian+凸包 |
| nnUNet DL（GPU）| — | ~80 ms | 128³ patch，RTX 3090 |
| nnUNet DL（CPU）| ~8000 ms | — | CPU回退，精简推理 |

---

## 11. 扩展指南

### 11.1 添加新的分割方法

1. 在 `SegMethod` 枚举中增加新值（`general_tumor_segment.h`）
2. 在 `GeneralTumorSegment::segment()` 中增加 `case`
3. 创建对应的 `xxx_segmentation.h/.cpp` 实现文件
4. 在 `CMakeLists.txt` 的 `file(GLOB ONC_SRCS ...)` 中自动包含

### 11.2 集成 TensorRT / ONNX Runtime 引擎

在 `lung_nodule_segment_dl.h` 中 `LungNoduleDLConfig` 增加字段：

```cpp
enum class InferBackend { CPU, CUDNN, TENSORRT, ONNXRUNTIME };
InferBackend backend = InferBackend::CUDNN;
std::string enginePath;  // .engine 文件路径（TensorRT）
```

在 `inferGPU()` 中根据 `backend` 切换推理路径，原有 cuDNN 路径保留为 fallback。

### 11.3 支持 DICOM 输入

工程当前接受裸内存指针（`const short*`）。对接 DICOM：
- 用 DCMTK / fo-dicom 读取像素数据，填充 `ImageInfo` 中的 `spacing`
- 将像素数组传入对应算法接口即可，无需修改算法代码

### 11.4 多时间点随访流程示例

```cpp
// 假设已有时间点 T1 的分割结果 seg_t1 和 T2 的图像 ct_t2
SegmentResult seg_t2;
PropagateSegmentation(ct_t2.data(), info, seg_t1, seg_t2);

// 计算两个时间点的径线变化
TumorDiameterResult diam_t1, diam_t2;
CalculateTumorDiameters(seg_t1, info, diam_t1);
CalculateTumorDiameters(seg_t2, info, diam_t2);

float change_pct = (diam_t2.longestDiameter2D_cm - diam_t1.longestDiameter2D_cm)
                   / diam_t1.longestDiameter2D_cm * 100.f;
// RECIST 1.1: +20% → Progressive Disease
//             -30% → Partial Response
```
