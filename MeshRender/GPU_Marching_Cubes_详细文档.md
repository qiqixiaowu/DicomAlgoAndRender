# GPU Marching Cubes (Compute Shader) 详细文档

> **项目**: OpenglRender / MeshRender 模块  
> **技术**: OpenGL 4.3 Compute Shader + SSBO + Image Load/Store  
> **硬件**: NVIDIA GeForce RTX 3060 (OpenGL 4.3.0 NVIDIA 551.61)  
> **数据**: DICOM CT 序列 512×512×216, 间距 1/1/3.5mm  
> **日期**: 2026-08-28

---

## 目录

1. [概述与动机](#1-概述与动机)
2. [Marching Cubes 算法原理](#2-marching-cubes-算法原理)
3. [GPU 并行化设计](#3-gpu-并行化设计)
4. [文件结构与职责](#4-文件结构与职责)
5. [Compute Shader 详解 (`marching_cubes.comp`)](#5-compute-shader-详解-marching_cubescomp)
6. [C++ 封装详解 (`gpu_marching_cubes.hpp`)](#6-c-封装详解-gpu_marching_cubeshpp)
7. [集成与调用链](#7-集成与调用链)
8. [踩坑记录与解决方案](#8-踩坑记录与解决方案)
9. [性能基准](#9-性能基准)
10. [操作指南](#10-操作指南)
11. [扩展方向](#11-扩展方向)

---

## 1. 概述与动机

### 1.1 什么是 Marching Cubes

Marching Cubes 是经典的等值面提取算法，从 3D 标量场（如 CT/MRI 体数据）中提取出三角网格表面。它逐个体素"行进"，根据 8 个角点与等值面的关系查表确定三角形拓扑。

### 1.2 为什么需要 GPU 加速

| 维度 | CPU 版 | GPU 版 |
|------|--------|--------|
| **并行度** | 单线程逐体素 | 数千线程并行 |
| **512×512×216 体数据** | ~15-30 秒 | ~280ms (计算部分) |
| **顶点输出** | `push_back` 动态增长 | `atomicAdd` 预分配写入 |
| **法线计算** | 额外遍历 | 与顶点同步生成 |
| **适用场景** | 小数据 / 调试 | 实时交互 / 大数据 |

CPU 版的瓶颈在于：约 5600 万个体素需要逐个遍历，且 `std::vector::push_back` 的内存重分配开销巨大。GPU 版利用 Compute Shader 的海量并行性，每个线程独立处理一个体素，通过原子计数器无锁分配输出位置。

### 1.3 技术依赖

```
OpenGL 4.3+  →  GL_COMPUTE_SHADER
                GL_ARB_shader_storage_buffer_object (SSBO)
                GL_ARB_shader_image_load_store (imageLoad/Store)
                GL_ARB_compute_shader
```

---

## 2. Marching Cubes 算法原理

### 2.1 体素与角点编号

每个体素有 8 个角点，编号 0-7，构成一个立方体：

```
        v4 ────── v5
       /|         /|
      v7────── v6 |
      | |       | |
      | v0─────|─v1
      |/        |/
      v3────── v2

  z轴朝上 (v0→v4 方向)
```

角点偏移定义（在 `.comp` 中）：

```glsl
const ivec3 cornerOffset[8] = ivec3[8](
    ivec3(0, 0, 0),  // v0
    ivec3(1, 0, 0),  // v1
    ivec3(1, 1, 0),  // v2
    ivec3(0, 1, 0),  // v3
    ivec3(0, 0, 1),  // v4
    ivec3(1, 0, 1),  // v5
    ivec3(1, 1, 1),  // v6
    ivec3(0, 1, 1)   // v7
);
```

### 2.2 12 条边编号

体素有 12 条边，每条边连接两个角点：

```glsl
const ivec2 edgeEndpoints[12] = ivec2[12](
    ivec2(0, 1), ivec2(1, 2), ivec2(2, 3), ivec2(3, 0),  // 底面 4 边 (e0-e3)
    ivec2(4, 5), ivec2(5, 6), ivec2(6, 7), ivec2(7, 4),  // 顶面 4 边 (e4-e7)
    ivec2(0, 4), ivec2(1, 5), ivec2(2, 6), ivec2(3, 7)   // 竖直 4 边 (e8-e11)
);
```

### 2.3 配置索引 (cubeConfig)

对每个体素，采样 8 个角点的密度值，与等值面阈值比较：

```
if val[i] < isovalue → bit i = 1  (角点在等值面"内部")
```

8 个角点 → 8-bit 配置号 (0-255)：
- `config == 0`：所有角点都在外部，等值面不穿过此体素
- `config == 255`：所有角点都在内部，同上
- 其他值：等值面穿过此体素，需要生成三角形

### 2.4 两张查找表

#### edgeTable[256] — 边查找表

12-bit 掩码，表示哪些边与等值面相交：

```glsl
int edges = edgeTable[config];
// edges 的第 e 位为 1 → 边 e 与等值面相交
if ((edges & (1 << e)) != 0) { /* 在边 e 上插值顶点 */ }
```

#### triTable[256][16] — 三角形查找表

每个配置最多生成 5 个三角形（15 个顶点 + 1 个 -1 终止符 = 16 个条目）。每 3 个连续条目是一个三角形的 3 条边索引：

```glsl
for (int i = 0; i < 16; i += 3) {
    int e0 = triTableFlat[config * 16 + i];
    if (e0 == -1) break;  // 三角形列表结束
    int e1 = triTableFlat[config * 16 + i + 1];
    int e2 = triTableFlat[config * 16 + i + 2];
    // 用 edgeVertex[e0], edgeVertex[e1], edgeVertex[e2] 构成三角形
}
```

> **注意**：GLSL 不支持二维 `const` 数组，所以 `triTable[256][16]` 被展平为一维 `triTableFlat[4096]`。

### 2.5 边插值

对于每条与等值面相交的边 (a→b)，用线性插值找到交点：

$$t = \frac{\text{isovalue} - v_a}{v_b - v_a}$$

$$P = C_a + t \cdot (C_b - C_a)$$

其中 $v_a, v_b$ 是角点密度值，$C_a, C_b$ 是角点物理坐标。

### 2.6 梯度法线

用中心差分计算密度梯度，取负方向作为外法线：

$$\nabla f = \left(\frac{f_{x+1} - f_{x-1}}{2},\ \frac{f_{y+1} - f_{y-1}}{2},\ \frac{f_{z+1} - f_{z-1}}{2}\right)$$

$$\mathbf{n} = -\frac{\nabla f}{|\nabla f|}$$

---

## 3. GPU 并行化设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      C++ 主机端                              │
│                                                             │
│  1. 编译 Compute Shader (缓存)                              │
│  2. 上传体数据 → 3D 纹理 (GL_R8UI)                          │
│  3. 上传查找表 → SSBO (edgeTable + triTableFlat)            │
│  4. 预分配输出 SSBO (vertices + normals + counter)          │
│  5. 设置 Uniforms (尺寸/间距/等值面/原点)                    │
│  6. glDispatchCompute()                                     │
│  7. glMemoryBarrier() + glFinish()                          │
│  8. 读回 counter → actualVerts                              │
│  9. 读回 vertices/normals → MeshData                        │
│ 10. 清理 GPU 资源                                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    GPU Compute Shader                       │
│                                                             │
│  每个工作组: 8×8×8 = 512 线程                               │
│  每个线程: 处理一个体素                                      │
│                                                             │
│  Thread(x,y,z):                                             │
│    1. 采样 8 角点 → cubeConfig                              │
│    2. 查 edgeTable → 哪些边相交                             │
│    3. 边插值 → 交点位置 + 梯度法线                          │
│    4. 查 triTableFlat → 三角形拓扑                          │
│    5. atomicAdd(counter, 3) → 分配输出位置                  │
│    6. 写入 vertices[] / normals[]                           │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 工作组划分

```
local_size = (8, 8, 8)  →  每个工作组 512 线程
dispatch   = (ceil((W-1)/8), ceil((H-1)/8), ceil((D-1)/8))
```

对于 512×512×216 的体数据：

```
groupsX = (512 + 6) / 8 = 64
groupsY = (512 + 6) / 8 = 64
groupsZ = (216 + 6) / 8 = 27

总工作组 = 64 × 64 × 27 = 110,592
总线程   = 110,592 × 512 = 56,623,104
```

> 每个线程处理一个体素，边界线程（coord ≥ W-1 等）直接 return。

### 3.3 无锁输出策略：atomicAdd

CPU 版用 `std::vector::push_back` 逐个追加顶点，存在频繁内存重分配。GPU 版采用**预分配 + 原子计数器**策略：

```glsl
// 每个三角形需要 3 个顶点位置
uint idx = atomicAdd(vertexCount, 3);

// 溢出保护
if (idx + 3 > uMaxVertices)
    return;

// 直接写入预分配的缓冲区
vertices[idx]     = vec4(edgeVertex[e0], 0.0);
vertices[idx + 1] = vec4(edgeVertex[e1], 0.0);
vertices[idx + 2] = vec4(edgeVertex[e2], 0.0);
```

**优点**：
- 无锁并行写入，线程间无需同步
- 写入位置由原子操作保证唯一性
- 预分配避免了动态内存分配

**代价**：
- 非索引模式：每个三角形 3 个独立顶点（共享顶点未合并）
- 需要预估最大顶点数（后续 `weldVertices` 合并）

### 3.4 资源绑定布局

| 绑定点 | 类型 | 名称 | 用途 |
|--------|------|------|------|
| Image Unit 0 | `readonly uimage3D` | `volumeImage` | 3D 体数据纹理 (r8ui) |
| SSBO Binding 1 | `writeonly vec4[]` | `vertices` | 顶点位置 (xyz) |
| SSBO Binding 2 | `writeonly vec4[]` | `normals` | 顶点法线 (xyz) |
| SSBO Binding 3 | `uint` | `vertexCount` | 原子计数器 |
| SSBO Binding 4 | `readonly int[]` | `edgeTable` | 边查找表 (256) |
| SSBO Binding 5 | `readonly int[]` | `triTableFlat` | 三角形查找表 (4096) |

---

## 4. 文件结构与职责

```
MeshRender/
├── shaders/
│   └── marching_cubes.comp        ← Compute Shader 源码（不含 #version，由 C++ 拼接）
├── gpu_marching_cubes.hpp         ← C++ 封装：编译/上传/Dispatch/读回
├── marching_cubes.hpp             ← CPU 版算法 + 查找表（edgeTable/triTable 原始数据）
├── dicom_mesh_loader.hpp          ← DICOM 加载管线（步骤 4 选择 GPU/CPU）
├── main_mesh_render.cpp           ← 主程序（G 键切换 GPU/CPU）
├── mesh.h                         ← Vertex/MeshData 数据结构
└── MeshRender.vcxproj             ← 项目配置（含 post-build shader 复制）
```

### 文件依赖关系

```
main_mesh_render.cpp
  └─ dicom_mesh_loader.hpp
       ├─ gpu_marching_cubes.hpp     (GPU 路径)
       │    ├─ marching_cubes.hpp    (复用 edgeTable/triTable)
       │    └─ shaders/marching_cubes.comp  (GLSL)
       └─ marching_cubes.hpp         (CPU 回退路径)
```

---

## 5. Compute Shader 详解 (`marching_cubes.comp`)

### 5.1 文件设计原则

`.comp` 文件**不包含** `#version` 指令和查找表数据。这些由 C++ 端在编译前拼接前置：

```cpp
// gpu_marching_cubes.hpp 中的 compileComputeShader()
std::string prefix =
    "#version 430 core\n"
    "#extension GL_ARB_compute_shader : enable\n"
    "#extension GL_ARB_shader_storage_buffer_object : enable\n"
    "#extension GL_ARB_shader_image_load_store : enable\n";

std::string fullSrc = prefix + compSrc;  // 前置 + .comp 源码
```

> 查找表通过 SSBO 上传，不再内联到 GLSL 源码中（详见 [第 8 节踩坑记录](#8-踩坑记录与解决方案)）。

### 5.2 工作组声明

```glsl
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
```

每个工作组 8×8×8 = 512 线程。选择 8 的原因：
- GPU 线程调度以 warp/wavefront 为单位（NVIDIA warp = 32）
- 512 = 16 个 warp，足够隐藏内存延迟
- 8³ 的立方体形状与体素立方体拓扑匹配

### 5.3 资源声明

```glsl
// 3D 体数据纹理（只读 image）
layout(r8ui) readonly uniform uimage3D volumeImage;

// 输出缓冲
layout(std430, binding = 1) writeonly buffer VertexBuffer {
    vec4 vertices[];
};
layout(std430, binding = 2) writeonly buffer NormalBuffer {
    vec4 normals[];
};
layout(std430, binding = 3) buffer CounterBuffer {
    uint vertexCount;
};

// 查找表（通过 SSBO 上传）
layout(std430, binding = 4) readonly buffer EdgeTableBuffer {
    int edgeTable[];
};
layout(std430, binding = 5) readonly buffer TriTableBuffer {
    int triTableFlat[];
};
```

**为什么用 `std430` 布局？**  
`std430` 比 `std140` 更紧凑（数组元素紧密排列无 padding），且与 C++ 端的 `glm::vec4` / `int` 布局完全一致。

**为什么用 `imageLoad` 而不是 `texelFetch`？**  
`imageLoad` 配合 `glBindImageTexture` 绑定的 image unit，可以直接访问 `GL_R8UI` 格式的无符号整数纹理，适合 CT 密度值（0-255）的精确读取。`texelFetch` 返回 float，会有精度损失。

### 5.4 Uniforms

```glsl
uniform int   volumeW, volumeH, volumeD;   // 体数据尺寸
uniform float uIsovalue;                    // 等值面阈值
uniform float uSx, uSy, uSz;               // 间距 × 缩放
uniform float uOx, uOy, uOz;               // 原点偏移
uniform uint  uMaxVertices;                 // 缓冲区容量（溢出保护）
```

### 5.5 辅助函数

#### `safeLoad` — 安全采样

```glsl
uint safeLoad(ivec3 p) {
    if (p.x < 0 || p.x >= volumeW ||
        p.y < 0 || p.y >= volumeH ||
        p.z < 0 || p.z >= volumeD)
        return 0u;
    return imageLoad(volumeImage, p).r;
}
```

越界返回 0（空气密度），避免 `imageLoad` 在边界处的未定义行为。

#### `computeGradient` — 梯度法线

```glsl
vec3 computeGradient(ivec3 p) {
    float gx = float(safeLoad(p + ivec3(1,0,0))) - float(safeLoad(p - ivec3(1,0,0)));
    float gy = float(safeLoad(p + ivec3(0,1,0))) - float(safeLoad(p - ivec3(0,1,0)));
    float gz = float(safeLoad(p + ivec3(0,0,1))) - float(safeLoad(p - ivec3(0,0,1)));
    vec3 n = vec3(-gx, -gy, -gz);  // 负号：法线朝外
    float len = length(n);
    if (len > 1e-6) return n / len;
    return vec3(0.0, 1.0, 0.0);  // 退化法线兜底
}
```

### 5.6 主函数流程

```glsl
void main() {
    ivec3 coord = ivec3(gl_GlobalInvocationID);

    // ① 边界检查
    if (coord.x >= volumeW - 1 || coord.y >= volumeH - 1 || coord.z >= volumeD - 1)
        return;

    // ② 采样 8 角点，计算 cubeConfig
    float val[8];
    int config = 0;
    for (int i = 0; i < 8; i++) {
        val[i] = float(imageLoad(volumeImage, coord + cornerOffset[i]).r);
        if (val[i] < uIsovalue) config |= (1 << i);
    }

    // ③ 快速剔除
    if (config == 0 || config == 255) return;

    // ④ 查 edgeTable，计算每条相交边的顶点位置和法线
    int edges = edgeTable[config];
    if (edges == 0) return;

    vec3 edgeVertex[12];
    vec3 edgeNormal[12];
    bool edgeValid[12];

    for (int e = 0; e < 12; e++) {
        edgeValid[e] = false;
        if ((edges & (1 << e)) == 0) continue;

        // 线性插值求交点
        int a = edgeEndpoints[e].x;
        int b = edgeEndpoints[e].y;
        float t = (uIsovalue - val[a]) / (val[b] - val[a]);

        // 物理空间坐标
        ivec3 ca = coord + cornerOffset[a];
        ivec3 cb = coord + cornerOffset[b];
        edgeVertex[e] = vec3(
            uOx + (ca.x + t * (cb.x - ca.x)) * uSx,
            uOy + (ca.y + t * (cb.y - ca.y)) * uSy,
            uOz + (ca.z + t * (cb.z - ca.z)) * uSz
        );

        // 梯度法线
        ivec3 m = ivec3(vec3(ca) + t * vec3(cb - ca) + 0.5);
        edgeNormal[e] = computeGradient(m);
        edgeValid[e] = true;
    }

    // ⑤ 查 triTableFlat，生成三角形
    for (int i = 0; i < 16; i += 3) {
        int e0 = triTableFlat[config * 16 + i];
        if (e0 == -1) break;
        int e1 = triTableFlat[config * 16 + i + 1];
        int e2 = triTableFlat[config * 16 + i + 2];

        if (!edgeValid[e0] || !edgeValid[e1] || !edgeValid[e2]) continue;

        // ⑥ atomicAdd 分配输出位置
        uint idx = atomicAdd(vertexCount, 3);
        if (idx + 3 > uMaxVertices) return;

        // ⑦ 写入顶点和法线
        vertices[idx]     = vec4(edgeVertex[e0], 0.0);
        vertices[idx + 1] = vec4(edgeVertex[e1], 0.0);
        vertices[idx + 2] = vec4(edgeVertex[e2], 0.0);

        normals[idx]     = vec4(edgeNormal[e0], 0.0);
        normals[idx + 1] = vec4(edgeNormal[e1], 0.0);
        normals[idx + 2] = vec4(edgeNormal[e2], 0.0);
    }
}
```

---

## 6. C++ 封装详解 (`gpu_marching_cubes.hpp`)

### 6.1 命名空间与依赖

```cpp
namespace MeshRender {

namespace detail {
    // 内部工具函数
}

// 主接口
inline MeshData extractIsosurfaceGPU(...);

} // namespace MeshRender
```

依赖：
- `mesh.h` — `Vertex`, `MeshData` 数据结构
- `marching_cubes.hpp` — 复用 `edgeTable[256]` 和 `triTable[256][16]` 查找表数据
- `<glad/glad.h>` — OpenGL 函数加载
- `<glm/glm.hpp>` — `glm::vec4` 用于 SSBO 数据布局

### 6.2 `detail` 命名空间 — 内部工具

#### `readFileContent` — 读取 shader 文件

```cpp
inline std::string readFileContent(const std::string& path) {
    FILE* f = nullptr;
    fopen_s(&f, path.c_str(), "rb");
    if (!f) {
        std::cerr << "[GPU MC] 无法打开文件: " << path << std::endl;
        return "";
    }
    // ... 读取全部内容
}
```

使用 `fopen_s` + 二进制模式读取，避免编码问题。

#### `getEdgeTableData` / `getTriTableData` — 生成查找表数据

```cpp
inline std::vector<int> getEdgeTableData() {
    return std::vector<int>(edgeTable, edgeTable + 256);
}

inline std::vector<int> getTriTableData() {
    std::vector<int> data(4096);
    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 16; j++)
            data[i * 16 + j] = triTable[i][j];
    return data;
}
```

将 `marching_cubes.hpp` 中的静态数组转换为 `std::vector<int>`，用于上传到 SSBO。`triTable` 是二维数组 `[256][16]`，展平为一维 `[4096]`。

#### `compileComputeShader` — 编译链接

```cpp
inline GLuint compileComputeShader(const std::string& shaderDir) {
    // 1. 读取 .comp 文件
    std::string compPath = shaderDir + "/marching_cubes.comp";
    std::string compSrc = readFileContent(compPath);

    // 2. 拼接 #version 前置
    std::string prefix = "#version 430 core\n"
        "#extension GL_ARB_compute_shader : enable\n"
        "#extension GL_ARB_shader_storage_buffer_object : enable\n"
        "#extension GL_ARB_shader_image_load_store : enable\n";
    std::string fullSrc = prefix + compSrc;

    // 3. 编译
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    // ... 错误检查

    // 4. 链接为 program
    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glDeleteShader(shader);
    // ... 错误检查

    return program;
}
```

### 6.3 `extractIsosurfaceGPU` — 主函数

#### 函数签名

```cpp
inline MeshData extractIsosurfaceGPU(
    const std::vector<uint8_t>& buffer,  // 体数据
    int w, int h, int d,                 // 尺寸
    const double spacing[3],             // 物理间距 (mm)
    const double origin[3],              // 原点
    float isovalue,                      // 等值面阈值
    float spacingScale,                  // 间距缩放 (0.01 = 缩小到 cm 级)
    const std::string& shaderDir         // 着色器目录
);
```

#### 执行流程（8 步）

**步骤 1: 编译 Compute Shader（带缓存）**

```cpp
static GLuint s_program = 0;
static std::string s_lastDir;
if (s_program == 0 || s_lastDir != shaderDir) {
    s_program = detail::compileComputeShader(shaderDir);
    s_lastDir = shaderDir;
}
```

使用 `static` 局部变量缓存编译结果，避免每次提取都重新编译。仅当 shader 目录变化时才重新编译。

**步骤 2: 创建 3D 纹理**

```cpp
glGenTextures(1, &volumeTex);
glBindTexture(GL_TEXTURE_3D, volumeTex);
glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
glTexImage3D(GL_TEXTURE_3D, 0, GL_R8UI, w, h, d, 0,
             GL_RED_INTEGER, GL_UNSIGNED_BYTE, buffer.data());
```

- `GL_R8UI`：单通道 8-bit 无符号整数格式，匹配 CT 密度值范围 (0-255)
- `GL_NEAREST`：最近邻过滤，不做插值（精确读取原始值）
- `GL_CLAMP_TO_EDGE`：边界扩展，配合 shader 中的 `safeLoad` 越界检查

**步骤 3: 预分配 SSBO**

```cpp
size_t numVoxels = (size_t)(w - 1) * (h - 1) * (d - 1);
size_t maxVerts = numVoxels * 15;  // 每体素最多 5 三角形 × 3 顶点
const size_t MAX_ALLOC = 12000000;  // 1200 万上限
if (maxVerts > MAX_ALLOC) maxVerts = MAX_ALLOC;
```

5 个 SSBO：

| SSBO | Binding | 大小 | 用途 |
|------|---------|------|------|
| `vboSSBO` | 1 | `maxVerts × 16` bytes | 顶点 (vec4) |
| `nboSSBO` | 2 | `maxVerts × 16` bytes | 法线 (vec4) |
| `counterSSBO` | 3 | 4 bytes | 原子计数器 |
| `edgeTableSSBO` | 4 | `256 × 4` bytes | 边查找表 |
| `triTableSSBO` | 5 | `4096 × 4` bytes | 三角形查找表 |

查找表使用 `GL_STATIC_READ`（上传一次，GPU 只读），输出缓冲使用 `GL_DYNAMIC_COPY`（GPU 写入，CPU 读回）。

**步骤 4: Dispatch**

```cpp
glUseProgram(s_program);

// 绑定 3D 纹理到 image unit 0
glBindImageTexture(0, volumeTex, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R8UI);

// 绑定所有 SSBO
glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vboSSBO);
glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, nboSSBO);
glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, counterSSBO);
glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, edgeTableSSBO);
glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, triTableSSBO);

// 设置 uniforms
glUniform1i(glGetUniformLocation(s_program, "volumeW"), w);
// ... 其他 uniforms

// Dispatch
int groupsX = (w + 6) / 8;  // ceil((w-1)/8)
int groupsY = (h + 6) / 8;
int groupsZ = (d + 6) / 8;
glDispatchCompute(groupsX, groupsY, groupsZ);

// 等待完成
glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT);
glFinish();
```

**`glBindImageTexture` 参数说明：**

| 参数 | 值 | 含义 |
|------|-----|------|
| unit | 0 | Image unit 编号 |
| texture | volumeTex | 纹理 ID |
| level | 0 | mipmap 层级 |
| layered | GL_TRUE | 绑定为 3D（所有层） |
| layer | 0 | 起始层 |
| access | GL_READ_ONLY | 只读 |
| format | GL_R8UI | 内部格式（必须与纹理一致） |

**步骤 5: 读回结果**

```cpp
// 读取顶点计数
uint32_t actualVerts = 0;
glBindBuffer(GL_SHADER_STORAGE_BUFFER, counterSSBO);
glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &actualVerts);

if (actualVerts == 0) {
    std::cerr << "[GPU MC] 未提取到任何顶点！" << std::endl;
    // 清理并返回空 mesh
}

// 读回顶点和法线
std::vector<glm::vec4> vertData(actualVerts);
glBindBuffer(GL_SHADER_STORAGE_BUFFER, vboSSBO);
glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                   actualVerts * sizeof(glm::vec4), vertData.data());

std::vector<glm::vec4> normData(actualVerts);
glBindBuffer(GL_SHADER_STORAGE_BUFFER, nboSSBO);
glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                   actualVerts * sizeof(glm::vec4), normData.data());
```

**步骤 6: 构建 MeshData**

```cpp
mesh.vertices.resize(actualVerts);
mesh.indices.resize(actualVerts);

for (uint32_t i = 0; i < actualVerts; i++) {
    Vertex& v = mesh.vertices[i];
    v.position[0] = vertData[i].x;
    v.position[1] = vertData[i].y;
    v.position[2] = vertData[i].z;
    v.normal[0] = normData[i].x;
    v.normal[1] = normData[i].y;
    v.normal[2] = normData[i].z;
    v.texcoord[0] = vertData[i].x / totalW;  // 归一化纹理坐标
    v.texcoord[1] = vertData[i].y / totalH;
    // tangent/bitangent 留空，后续 computeTangents 计算
    mesh.indices[i] = i;  // 非索引模式
}
```

**步骤 7: 清理 GPU 资源**

```cpp
glDeleteTextures(1, &volumeTex);
glDeleteBuffers(1, &vboSSBO);
glDeleteBuffers(1, &nboSSBO);
glDeleteBuffers(1, &counterSSBO);
glDeleteBuffers(1, &edgeTableSSBO);
glDeleteBuffers(1, &triTableSSBO);
```

每次调用都创建和销毁资源。查找表 SSBO 也可以缓存，但 256+4096 个 int = 17KB，开销可忽略。

**步骤 8: 打印计时信息**

```cpp
std::cout << "  [GPU MC] 提取完成" << std::endl;
std::cout << "    顶点数: " << actualVerts << std::endl;
std::cout << "    三角形数: " << actualVerts / 3 << std::endl;
std::cout << "    耗时: 准备=" << msSetup << "ms"
          << " | 计算=" << msCompute << "ms"
          << " | 读回=" << msReadback << "ms"
          << " | 总计=" << msTotal << "ms" << std::endl;
```

---

## 7. 集成与调用链

### 7.1 DICOM 加载管线 (`dicom_mesh_loader.hpp`)

```
loadDicomAsMesh(folder, isovalue, ..., useGPU=true, shaderDir="shaders")
│
├── [1/5] 扫描 DICOM 序列 → collectSeries(folder)
├── [2/5] 构建 3D 体数据 → buildVolume_none(series)
├── [3/5] 去除床板伪影 → removeBedArtifact(volume, 10)
├── [4/5] Marching Cubes 提取
│   │
│   ├── if (useGPU)
│   │   ├── extractIsosurfaceGPU(...)     ← GPU 路径
│   │   └── if (mesh.vertices.empty())    ← 自动回退
│   │       └── extractIsosurface(...)    ← CPU 回退
│   │
│   └── else
│       └── extractIsosurface(...)        ← CPU 路径
│
├── [5/5] 网格后处理
│   ├── computeNormals() (GPU 路径已有法线，可跳过)
│   ├── computeTangents()
│   ├── laplacianSmooth() (可选)
│   └── computeAABB()
│
└── return mesh
```

关键回退逻辑：

```cpp
if (useGPU) {
    std::cout << "  使用 GPU Compute Shader 提取..." << std::endl;
    mesh = extractIsosurfaceGPU(volume.buffer, ...);
    if (mesh.vertices.empty()) {
        std::cout << "  GPU 提取失败，回退到 CPU..." << std::endl;
        mesh = extractIsosurface(volume.buffer, ...);
    }
} else {
    mesh = extractIsosurface(volume.buffer, ...);
}
```

### 7.2 主程序集成 (`main_mesh_render.cpp`)

#### 全局变量

```cpp
static bool g_useGPU = true;              // 默认使用 GPU
static std::string g_shaderDir = "shaders";  // 着色器目录
```

#### 键盘切换 (G 键)

```cpp
case GLFW_KEY_G:
    g_useGPU = !g_useGPU;
    std::cout << "\n=== Marching Cubes: "
              << (g_useGPU ? "GPU (Compute Shader)" : "CPU")
              << " ===" << std::endl;
    rebuildScene();  // 重新加载网格
    break;
```

#### 调用点

```cpp
// generateCurrentMesh() 中 DICOM case:
MeshData mesh = loadDicomAsMesh(g_dicomFolder, isovalue,
                                 true, 0.01f, true,
                                 g_useGPU, g_shaderDir);
```

### 7.3 项目配置 (`MeshRender.vcxproj`)

#### Post-Build 事件

```xml
<PostBuildEvent>
  <Command>xcopy /Y /I /E "$(ProjectDir)shaders" "$(OutDir)shaders"</Command>
</PostBuildEvent>
```

每次编译后自动将 `shaders/` 目录复制到输出目录（`x64/Debug/shaders/`），确保 `marching_cubes.comp` 可被运行时找到。

#### 文件声明

```xml
<ClInclude Include="gpu_marching_cubes.hpp" />
<None Include="shaders\marching_cubes.comp" />
```

---

## 8. 踩坑记录与解决方案

### 8.1 问题一：Shader 文件找不到

**现象：**

```
[GPU MC] 无法打开文件: shaders/marching_cubes.comp
[GPU MC] 无法读取 compute shader: shaders/marching_cubes.comp
[GPU MC] 着色器编译失败，回退到 CPU
GPU 提取失败，回退到 CPU...
```

**原因：**  
可执行文件在 `x64/Debug/` 目录运行，但 `marching_cubes.comp` 在源码目录的 `shaders/` 子目录中。其他渲染着色器（`.vert`/`.frag`）之前已手动复制到输出目录，但新增的 `.comp` 文件没有。

**解决方案：**  
在 `MeshRender.vcxproj` 的 Debug 和 Release 配置中添加 Post-Build 事件：

```xml
<PostBuildEvent>
  <Command>xcopy /Y /I /E "$(ProjectDir)shaders" "$(OutDir)shaders"</Command>
</PostBuildEvent>
```

这样每次编译后自动同步所有 shader 文件到输出目录。

### 8.2 问题二：GLSL 内联构造函数参数过多 (C1068)

**现象：**

```
0(229) : error C1068: too much data in type constructor
[GPU MC] Compute shader 编译失败
[GPU MC] 着色器编译失败，回退到 CPU
```

**原因：**  
最初的设计是将查找表作为 GLSL 内联构造函数拼接到 shader 源码中：

```glsl
// 由 C++ 生成的 GLSL 代码
const int edgeTable[256] = int[256](0x0, 0x109, 0x203, ...);  // 256 个参数
const int triTableFlat[4096] = int[4096](...);                  // 4096 个参数 ← 超限！
```

NVIDIA 驱动对 GLSL 构造函数的参数数量有上限。`int[4096](...)` 传入了 4096 个参数，远超驱动限制，触发 `C1068`。

**解决方案：**  
将查找表改为通过 SSBO 上传，完全避免 GLSL 内联构造：

**Shader 端：**
```glsl
// 旧方式（失败）：
// const int triTableFlat[4096] = int[4096](...);

// 新方式（成功）：
layout(std430, binding = 4) readonly buffer EdgeTableBuffer {
    int edgeTable[];
};
layout(std430, binding = 5) readonly buffer TriTableBuffer {
    int triTableFlat[];
};
```

**C++ 端：**
```cpp
// 旧方式（失败）：
// std::string fullSrc = prefix + genEdgeTableGLSL() + genTriTableGLSL() + compSrc;

// 新方式（成功）：
auto edgeData = detail::getEdgeTableData();  // std::vector<int>(edgeTable, edgeTable+256)
glGenBuffers(1, &edgeTableSSBO);
glBindBuffer(GL_SHADER_STORAGE_BUFFER, edgeTableSSBO);
glBufferData(GL_SHADER_STORAGE_BUFFER, edgeData.size() * sizeof(int),
             edgeData.data(), GL_STATIC_READ);

auto triData = detail::getTriTableData();  // 展平的 4096 个 int
glGenBuffers(1, &triTableSSBO);
glBindBuffer(GL_SHADER_STORAGE_BUFFER, triTableSSBO);
glBufferData(GL_SHADER_STORAGE_BUFFER, triData.size() * sizeof(int),
             triData.data(), GL_STATIC_READ);

// Dispatch 前绑定
glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, edgeTableSSBO);
glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, triTableSSBO);
```

**经验教训：**  
GLSL 中超过 ~1024 个参数的构造函数可能触发驱动限制。大数组应通过 SSBO/UBO 传递，不要内联到源码。

### 8.3 问题三：Windows 控制台中文乱码

**现象：**  
在 VS Code 集成终端中，`std::cout` 输出的中文显示为乱码。

**解决方案：**  
在 `main()` 开头添加：

```cpp
system("chcp 65001 > nul");  // 设置控制台编码为 UTF-8
```

配合编译选项 `/utf-8`（源码和执行字符集均为 UTF-8），彻底解决乱码问题。

---

## 9. 性能基准

### 9.1 测试环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 3060 |
| 驱动 | 551.61 |
| OpenGL | 4.3.0 |
| 数据 | 512×512×216 CT, 间距 1/1/3.5mm |
| 等值面 | 180 (骨骼) |
| 编译 | VS 2022, Debug, x64 |

### 9.2 GPU 提取结果

```
[GPU MC] 提取完成
    顶点数: 3,570,528
    三角形数: 1,190,176
    耗时: 准备=667.132ms | 计算=281.909ms | 读回=514.601ms | 总计=1463.64ms
```

### 9.3 各阶段耗时分析

```
┌──────────┬──────────────┬───────────┬──────────────────────────────┐
│ 阶段     │ 耗时         │ 占比      │ 说明                         │
├──────────┼──────────────┼───────────┼──────────────────────────────┤
│ 准备     │ 667.132ms    │ 45.6%     │ 编译shader+上传纹理+分配SSBO │
│ GPU 计算 │ 281.909ms    │ 19.3%     │ glDispatchCompute            │
│ 读回     │ 514.601ms    │ 35.2%     │ glGetBufferSubData           │
│ 总计     │ 1463.64ms    │ 100%      │                              │
└──────────┴──────────────┴───────────┴──────────────────────────────┘
```

### 9.4 性能瓶颈分析

1. **准备阶段 (45.6%)**：主要是 3D 纹理上传（512×512×216 = 56MB）和 SSBO 分配（1200万×16×2 = 366MB）。Shader 编译因 `static` 缓存仅首次开销。
2. **GPU 计算 (19.3%)**：实际并行计算仅 282ms，处理 5600 万体素。这是 GPU 的核心优势所在。
3. **读回阶段 (35.2%)**：357 万顶点 × 16 字节 × 2（位置+法线）= 114MB 从 GPU 回传到 CPU。PCIe 带宽限制了速度。

### 9.5 优化建议

| 优化方向 | 预期收益 | 复杂度 |
|----------|----------|--------|
| 缓存 3D 纹理（多次提取复用） | 减少准备阶段 ~300ms | 低 |
| 使用 `glCopyBufferSubData` + PBO 异步读回 | 读回与计算重叠 | 中 |
| 索引化输出（共享顶点） | 减少 60% 顶点数 | 高 |
| 两遍 Pass（先计数再写入） | 精确分配，无浪费 | 中 |
| 查找表 SSBO 缓存 | 减少准备阶段 ~1ms | 低 |

---

## 10. 操作指南

### 10.1 编译

```powershell
$msb = "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\MSBuild.exe"
& $msb "f:\渲染\OpenglRender\MeshRender\MeshRender.vcxproj" /p:Configuration=Debug /p:Platform=x64 /p:PlatformToolset=v143 /t:Rebuild /m /v:minimal
```

编译后 Post-Build 事件自动将 `shaders/` 复制到 `x64/Debug/shaders/`。

### 10.2 运行

```powershell
cd "f:\渲染\OpenglRender\MeshRender\x64\Debug"
.\MeshRender.exe
```

程序启动后自动加载 DICOM 数据（路径在 `g_dicomFolder` 中配置），使用 GPU 提取等值面。

### 10.3 键盘操作

| 按键 | 功能 |
|------|------|
| `G` | 切换 GPU / CPU Marching Cubes |
| `1`-`5` | 切换网格类型（立方体/球体/茶壶/布林/自定义） |
| `6`-`9` | 切换 DICOM 预设（骨骼/软组织/皮肤/自定义阈值） |
| `L` | 切换 LOD 层级 |
| `W` | 线框模式开关 |
| `鼠标左键` | 旋转视角 |
| `鼠标滚轮` | 缩放 |
| `鼠标右键` | 平移 |

### 10.4 诊断输出

成功时控制台输出：

```
[4/5] Marching Cubes 等值面提取...
  等值面阈值: 180
  使用 GPU Compute Shader 提取...
  [GPU MC] 提取完成
    顶点数: 3570528
    三角形数: 1190176
    耗时: 准备=667.132ms | 计算=281.909ms | 读回=514.601ms | 总计=1463.64ms
  顶点数: 3570528
  三角形数: 1190176
```

失败时（自动回退 CPU）：

```
[4/5] Marching Cubes 等值面提取...
  使用 GPU Compute Shader 提取...
  [GPU MC] Compute shader 编译失败:
  0(229) : error C1068: too much data in type constructor
  [GPU MC] 着色器编译失败，回退到 CPU
  GPU 提取失败，回退到 CPU...
  使用 CPU 提取...
```

### 10.5 修改等值面阈值

在 `main_mesh_render.cpp` 中修改预设：

```cpp
static float g_dicomPresets[] = {
    180.0f,   // [6] 骨骼
    120.0f,   // [7] 软组织
    60.0f,    // [8] 皮肤
    150.0f    // [9] 自定义
};
```

或在运行时用 `D` 键自定义阈值。

---

## 11. 扩展方向

### 11.1 索引化输出

当前输出是非索引模式（每个三角形 3 个独立顶点）。可以添加共享顶点逻辑：

```glsl
// 方案：用 edgeVertex 的体素坐标 + 边编号作为哈希键
// 通过 atomicCompSwap 实现 CAS（Compare-And-Swap）去重
uint hashKey = coord.x * 7349 + coord.y * 5237 + coord.z * 3137 + e;
// ... CAS 循环写入共享位置
```

预期减少 50-70% 的顶点数。

### 11.2 两遍 Pass（计数 + 写入）

```
Pass 1: 只计数，不写入 → 得到精确顶点数
Pass 2: 精确分配缓冲区，写入
```

优点：无缓冲区浪费，无溢出风险。缺点：需要两次 Dispatch。

### 11.3 多等值面提取

修改 shader 支持多个等值面阈值，一次 Dispatch 提取骨骼+软组织+皮肤三层表面。

### 11.4 直接渲染（零拷贝）

跳过 CPU 读回，直接将 SSBO 作为 VBO 渲染：

```cpp
// 将 SSBO 直接绑定为 VAO 的 VBO
glBindBuffer(GL_ARRAY_BUFFER, vboSSBO);
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), 0);
```

消除读回阶段（514ms），实现真正的实时交互。

### 11.5 自适应分辨率

对平坦区域（法线变化小）跳过处理，只在曲面区域精细提取。类似八叉树细分但运行在 GPU 上。

---

## 附录：关键数据结构

### Vertex (mesh.h)

```cpp
struct Vertex {
    float position[3];    // 位置
    float normal[3];      // 法线
    float texcoord[2];    // 纹理坐标
    float tangent[3];     // 切线
    float bitangent[3];   // 副切线
};  // 56 bytes
```

### MeshData (mesh.h)

```cpp
struct MeshData {
    std::string name;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    AABB aabb;

    size_t vertexCount() const { return vertices.size(); }
    size_t triangleCount() const { return indices.size() / 3; }
};
```

### 体素角点与边的关系图

```
边编号：
         e8    e9    e10   e11
          │     │     │     │
     v4───┼─────v5────┼─────v6
     /│   │     │     │     ││
   e7│ e4 │    e5     │  e6 │e6
     ││   │     │     │     ││
     v7───┼─────v6────┼─────v6  ← 顶面 (z=1)
     │    │     │     │     │
     │    │     │     │     │
     v0───┼─────v1────┼─────v1  ← 底面 (z=0)
     ││   │     │     │     ││
   e3│ e0 │    e1     │  e2 │e2
     ││   │     │     │     ││
     v3───┼─────v2────┼─────v2

  e0: v0-v1  e1: v1-v2  e2: v2-v3  e3: v3-v0  (底面)
  e4: v4-v5  e5: v5-v6  e6: v6-v7  e7: v7-v4  (顶面)
  e8: v0-v4  e9: v1-v5  e10:v2-v6  e11:v3-v7  (竖边)
```

---

> **文档版本**: 1.0  
> **最后更新**: 2026-08-28  
> **验证状态**: ✅ GPU 提取成功，3,570,528 顶点 / 1,190,176 三角形
