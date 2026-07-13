# 医学图像分割算法库

> 提炼自 `AlgoSegmentationConan` 工程，完整实现 7 种核心分割算法 + 2 种部署框架

## 目录结构

```
MedicalSegmentation/
├── CMakeLists.txt                  # 构建配置
├── README.md                       # 本文件
├── include/                        # Header-only 算法库
│   ├── image2d.hpp                 # 基础 2D/3D 图像容器 + PGM 读写
│   ├── chan_vese.hpp               # Chan-Vese Level Set (2D)
│   ├── drlse.hpp                   # DRLSE 距离正则化水平集 (2D)
│   ├── snake_gvf.hpp              # Snake 活动轮廓 + GVF
│   ├── graph_cut.hpp              # Graph Cut 图割分割
│   ├── region_growing.hpp         # Region Growing 区域生长 (2D/3D)
│   ├── dense_crf.hpp              # Dense CRF 全连接条件随机场
│   ├── sliding_window.hpp         # 滑动窗口推理框架
│   └── two_stage.hpp              # 两阶段 Coarse-to-Fine 分割
└── demo/
    └── main_demo.cpp              # 综合演示程序
```

## 构建与运行

```bash
cd MedicalSegmentation
mkdir build && cd build
cmake ..
cmake --build . --config Release
./medseg_demo              # Linux
.\Release\medseg_demo.exe  # Windows
```

运行后生成 PGM 格式结果图像，可用 GIMP/IrfanView/Python 查看。

---

## 算法详解

### 1. Chan-Vese Level Set ([chan_vese.hpp](include/chan_vese.hpp))

**分类**: 基于区域的活动轮廓

**核心思想**: 不依赖图像梯度，通过最小化区域内外灰度拟合能量来驱动水平集演化。

**能量泛函**:
```
E(φ) = μ·∫|∇H(φ)|dx + λ₁·∫(I-c₁)²H(φ)dx + λ₂·∫(I-c₂)²(1-H(φ))dx
```

**演化方程**:
```
∂φ/∂t = δ(φ)[μ·κ - λ₁(I-c₁)² + λ₂(I-c₂)²]
```

其中:
- `φ`: 水平集函数，`{φ=0}` 即分割轮廓
- `c₁, c₂`: 前景/背景全局灰度均值
- `κ`: 曲率 `= (φ_xx·φ_y² + φ_yy·φ_x² - 2φ_xφ_yφ_xy) / |∇φ|³`
- `μ`: 曲率正则系数

**关键实现技术**:
| 技术 | 说明 |
|------|------|
| **窄带加速** | 仅在 `\|φ\| < 1.2` 处计算，大幅减少计算量 |
| **Sussman 重初始化** | 定期求解 `\|∇ψ\|=1` 修正 SDF 性质 |
| **CFL 步长** | `Δt = 0.45 / max(\|Δφ\|)` 自适应稳定 |
| **局部均值** | 窗口内计算 c₁, c₂，适应非均匀图像 |

**适用场景**: MR 脑组织分割、CT 器官分割（边界模糊情况）

---

### 2. DRLSE ([drlse.hpp](include/drlse.hpp))

**分类**: 距离正则化水平集

**核心思想**: 在能量中加入距离正则项 `R(φ)`，使 `|∇φ|` 自动维持为 1（SDF 性质），完全消除重初始化需求。

**能量泛函**:
```
E(φ) = μ·R(φ) + λ·L_g(φ) + α·A_g(φ)

R(φ) = ∫ p(|∇φ|) dx          -- 距离正则项
L_g(φ) = ∫ g·δ(φ)·|∇φ| dx    -- 加权边缘长度
A_g(φ) = ∫ g·H(-φ) dx         -- 面积加速项
```

**双势阱函数** `p(s)`:
```
p(s) = (s-1)²/4,                s ≤ 1
p(s) = (1/2π)(1-cos(2πs)),      s > 1
```

**边缘停止函数**:
```
g(I) = 1 / (1 + |∇(G_σ * I)|²)
```

**优势** vs 传统 Level Set:
- 无需重初始化
- 初始化可以是简单二值函数
- 数值更稳定

---

### 3. Snake + GVF ([snake_gvf.hpp](include/snake_gvf.hpp))

**分类**: 参数化活动轮廓

**核心思想**: 轮廓表示为有序控制点序列，在内力（弹性+刚度）和外力（GVF 场）驱动下演化。

**能量**:
```
E = ∫[α|v'|² + β|v''|²]ds + ∫E_ext(v)ds
     ───────────────────    ─────────────
         内力(正则)            外力(图像)
```

**GVF 迭代**:
```
u ← u + μ·∇²u - |∇f|²·(u - f_x)
v ← v + μ·∇²v - |∇f|²·(v - f_y)
```

**求解**:
离散化后每步求解 `(A + γI)·v_new = γ·v_old + f_ext`，其中 A 是环形五对角矩阵。

**GVF 的作用**: 将边缘梯度信息扩散到整个图像域，解决传统 Snake 捕获范围有限的问题。

---

### 4. Graph Cut ([graph_cut.hpp](include/graph_cut.hpp))

**分类**: 基于图论的分割

**核心思想**: 将图像建模为图，通过最小割/最大流实现最优二分类。

**能量**:
```
E(L) = Σ_p D_p(L_p) + λ·Σ_{p,q∈N} V_pq(L_p, L_q)
       ─────────────   ───────────────────────────
        数据项(T-link)        平滑项(N-link)
```

**数据项**（基于直方图）:
```
D_p(fg) = -log(P_bg(I_p) + 0.01)    -- 归为前景的代价
D_p(bg) = -log(P_fg(I_p) + 0.01)    -- 归为背景的代价
```

**平滑项**:
```
V_pq = exp(-(I_p - I_q)² / 2σ²)
```

**最大流算法**: Edmonds-Karp (BFS 增广路径)，时间复杂度 O(VE²)

**工作流程**:
1. 用户标记前景/背景种子点
2. 从种子构建灰度直方图
3. 建立图（T-link + N-link）
4. BFS 增广路径求最大流
5. 最小割 → 源侧=前景，汇侧=背景

---

### 5. Region Growing ([region_growing.hpp](include/region_growing.hpp))

**分类**: 基于区域相似性的分割

**核心思想**: 从种子点 BFS 扩展，满足相似度准则的邻域像素被纳入。

**三种内置准则**:
| 准则 | 公式 | 适用场景 |
|------|------|----------|
| 阈值法 | `I(q) ∈ [low, high]` | 已知目标灰度范围 |
| 差值法 | `\|I(q) - μ_seed\| ≤ tol` | 基于种子相似度 |
| 比值法 | `\|I(q)/μ_seed - 1\| ≤ tol` | 比例不变 |

**支持**: 2D (4/8连通) + 3D (6/26连通)、自定义准则函数、多种子点、最大体素限制

---

### 6. Dense CRF ([dense_crf.hpp](include/dense_crf.hpp))

**分类**: 概率图模型后处理

**核心思想**: 对分类器输出的概率图做全局优化，使空间相邻且外观相似的像素标签一致。

**CRF 能量**:
```
E(x) = Σ_i ψ_u(x_i) + Σ_{i<j} ψ_p(x_i, x_j)
```

**Pairwise 核**:
```
外观核: k_app = exp(-|p_i-p_j|²/2θ_α² - |I_i-I_j|²/2θ_β²)
平滑核: k_sm  = exp(-|p_i-p_j|²/2θ_γ²)
```

**均值场推理迭代**:
```
1. 初始化 Q = softmax(-ψ_u)
2. 消息传递: Q̃ = Σ_j k(f_i,f_j)·Q_j    (Permutohedral Lattice 加速)
3. 兼容性: ψ̂ = Σ_l' μ(l,l')·Q̃(l')
4. 更新: Q = softmax(-ψ_u - ψ̂)
```

**Permutohedral Lattice**: 将 O(N²) 的全连接滤波降为 O(N·d)，通过 splat→blur→slice 三步完成。

---

### 7. 滑动窗口推理 ([sliding_window.hpp](include/sliding_window.hpp))

**分类**: 深度学习部署框架

**核心思想**: 大图像切分为重叠 patch，逐个推理后高斯加权融合。

**Patch 计算**:
```
stride = patch_size × (1 - overlap_ratio)
N_patches = ceil((img_size - patch_size) / stride) + 1
```

**高斯加权融合**:
```
weight(x,y) = exp(-((x-cx)² + (y-cy)²) / 2σ²)
result(x,y) = Σ_patch[output·weight] / Σ_patch[weight]
```

边缘贡献小、中心贡献大，消除拼接边界伪影。

---

### 8. 两阶段分割 ([two_stage.hpp](include/two_stage.hpp))

**分类**: Coarse-to-Fine 分割策略

**流程**:
```
Stage 1 (粗):  原图 → 降采样 → 粗分割 → BBox 定位
                              ↓
Stage 2 (精):  原图 → 按 BBox 裁剪 → 精分割 → 回映射
```

**空间映射**: `BBox_orig = BBox_coarse × (W_orig / W_coarse)`

**连通域分析**: Flood fill + 保留最大连通域，去除噪声区域

**优势**: 暴力全图推理显存和计算量大，两阶段策略先粗后精，兼顾速度与精度。

---

## 与 AlgoSegmentationConan 原工程的对应关系

| 本库模块 | 原工程模块 | 说明 |
|----------|-----------|------|
| `chan_vese.hpp` | `McsfAlgoActiveContour` | ChanVese2D 部分 |
| `drlse.hpp` | `McsfAlgoActiveContour` | ActiveContour3D 的 DRLSE |
| `snake_gvf.hpp` | `McsfAlgoActiveContour` | SnakeActiveContour 部分 |
| `graph_cut.hpp` | `McsfAlgoGraphCut` | 完整图割 + BK 最大流 |
| `region_growing.hpp` | `McsfAlgoRegionGrowing` | 模板化区域生长 |
| `dense_crf.hpp` | `McsfAlgoDenseCRF` | DenseCRF + Permutohedral |
| `sliding_window.hpp` | `McsfAlgoSlideWindow` / `McsfAlgoDLSegFramework` | 滑窗推理核心 |
| `two_stage.hpp` | `McsfAlgoSeg2Stage` | 两阶段分割 |

**简化说明**: 原工程依赖 CUDA/cuDNN/TensorRT/IPP/MKL/Boost 等重量级库，本库为纯 C++17 header-only 实现，零外部依赖，便于学习和集成。

## 参考文献

1. Chan T, Vese L. "Active contours without edges." IEEE TIP, 2001.
2. Li C, et al. "Distance regularized level set evolution." IEEE TIP, 2010.
3. Xu C, Prince J. "Snakes, shapes, and gradient vector flow." IEEE TIP, 1998.
4. Boykov Y, Kolmogorov V. "Min-cut/max-flow algorithms for energy minimization." IEEE TPAMI, 2004.
5. Adams R, Bischof L. "Seeded region growing." IEEE TPAMI, 1994.
6. Krähenbühl P, Koltun V. "Efficient inference in fully connected CRFs." NeurIPS, 2011.
7. Isensee F, et al. "nnU-Net: self-configuring deep learning segmentation." Nature Methods, 2021.
