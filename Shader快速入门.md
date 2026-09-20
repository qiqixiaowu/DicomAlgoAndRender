# Shader 快速入门 — 从零到掌握

> 本文档面向 **零基础** 读者，系统讲解 OpenGL Shader（GLSL）的核心概念与编写方法。
> 所有示例均取自本项目的真实着色器代码，学完即可看懂并修改项目中的 shader。

---

## 目录

- [第一章：Shader 是什么](#第一章shader-是什么)
- [第二章：GLSL 语言基础](#第二章glsl-语言基础)
- [第三章：顶点着色器（Vertex Shader）](#第三章顶点着色器vertex-shader)
- [第四章：片元着色器（Fragment Shader）](#第四章片元着色器fragment-shader)
- [第五章：Compute Shader（计算着色器）](#第五章compute-shader计算着色器)
- [第六章：数据传递机制](#第六章数据传递机制)
- [第七章：纹理采样](#第七章纹理采样)
- [第八章：实战拆解——项目着色器逐行解读](#第八章实战拆解项目着色器逐行解读)
- [第九章：在 C++ 中加载和使用 Shader](#第九章在-c-中加载和使用-shader)
- [第十章：常见错误与调试技巧](#第十章常见错误与调试技巧)
- [第十一章：练习题](#第十一章练习题)

---

## 第一章：Shader 是什么

### 1.1 一句话定义

Shader 是**运行在 GPU 上的小程序**，用来控制图形渲染管线的各个阶段。

### 1.2 GPU vs CPU 的根本区别

```
CPU: 4~16 个核心，每个核心很强
     → 擅长复杂逻辑、分支、串行任务

GPU: 数千个核心，每个核心较简单
     → 擅长大量并行的简单计算

例: 渲染 1920×1080 画面 = 200 万像素
    CPU 逐个处理: 很慢
    GPU 并行处理: 每个像素一个线程，同时算完
```

### 1.3 渲染管线中的 Shader 位置

```
┌─────────────────────────────────────────────────────────────┐
│                  OpenGL 渲染管线                              │
│                                                              │
│  顶点数据                                                     │
│     │                                                        │
│     ▼                                                        │
│  ┌──────────┐                                                │
│  │ 顶点着色器  │ ← 你写的 .vert 文件 (每个顶点执行一次)         │
│  │ Vertex    │    职责: 坐标变换、传递数据给下一步              │
│  │ Shader    │                                                │
│  └──────────┘                                                │
│     │                                                        │
│     ▼                                                        │
│  图元装配 (把顶点组成三角形/线段)                                │
│     │                                                        │
│     ▼                                                        │
│  ┌──────────┐                                                │
│  │ 几何着色器  │ ← 可选 (每个图元执行一次)                       │
│  │ Geometry  │    职责: 创建/删除/修改图元                      │
│  │ Shader    │                                                │
│  └──────────┘                                                │
│     │                                                        │
│     ▼                                                        │
│  光栅化 (把三角形变成像素)                                      │
│     │                                                        │
│     ▼                                                        │
│  ┌──────────┐                                                │
│  │ 片元着色器  │ ← 你写的 .frag 文件 (每个像素执行一次)         │
│  │ Fragment  │    职责: 计算最终颜色                           │
│  │ Shader    │                                                │
│  └──────────┘                                                │
│     │                                                        │
│     ▼                                                        │
│  逐像素操作 (深度测试、混合等)                                   │
│     │                                                        │
│     ▼                                                        │
│   帧缓冲 (屏幕上看到的画面)                                     │
└─────────────────────────────────────────────────────────────┘
```

### 1.4 本项目的着色器一览

| 文件 | 类型 | 用途 |
|------|------|------|
| `volume_vert_glsl.vert` | 顶点着色器 | 体渲染：变换立方体顶点 |
| `volume_frag_glsl.frag` | 片元着色器 | 体渲染：光线投射采样 |
| `mpr_vert.vert` | 顶点着色器 | MPR 多平面重建 |
| `mpr_frag.frag` | 片元着色器 | MPR 切面采样 |
| `ct_display_vert.vert` | 顶点着色器 | 全屏四边形显示 |
| `ct_display_frag.frag` | 片元着色器 | 窗宽窗位 + 热力图 |
| `ct_fbp_compute.glsl` | 计算着色器 | CT 滤波反投影重建 |

---

## 第二章：GLSL 语言基础

### 2.1 GLSL 与 C 的关系

GLSL 语法类似 C，但有关键区别：

```
相同点:                          不同点:
- 基本类型 (int, float, bool)    - 没有指针
- if/else/for/while              - 没有递归
- 函数定义和调用                  - 有内置向量/矩阵类型 (vec2, mat4)
- #define 宏                     - 有限定符 (uniform, in, out)
- struct 结构体                   - 每个着色器必须有 main() 函数
                                 - 不支持 char/string
```

### 2.2 基本数据类型

```glsl
// ─── 标量类型 ───
int   i = 42;        // 整数
float f = 3.14;      // 浮点数（GLSL 中浮点字面量必须带小数点）
bool  b = true;      // 布尔
uint  u = 1u;        // 无符号整数（带 u 后缀）

// ─── 向量类型（最常用！）───
vec2  v2  = vec2(1.0, 2.0);           // 2D 向量
vec3  v3  = vec3(1.0, 2.0, 3.0);     // 3D 向量（颜色 RGB、坐标 XYZ）
vec4  v4  = vec4(1.0, 2.0, 3.0, 4.0); // 4D 向量（RGBA、齐次坐标）

ivec3 iv3 = ivec3(0, 1, 2);          // 整数向量
uvec2 uv2 = uvec2(0u, 1u);           // 无符号整数向量
bvec3 bv3 = bvec3(true, false, true); // 布尔向量

// ─── 矩阵类型 ───
mat2 m2;   // 2×2 矩阵
mat3 m3;   // 3×3 矩阵
mat4 m4;   // 4×4 矩阵（最常用：变换矩阵）

// ─── 纹理采样器 ───
sampler2D  tex2D;    // 2D 纹理
sampler3D  tex3D;    // 3D 纹理（体数据）
sampler1D  tex1D;    // 1D 纹理（传递函数）
```

### 2.3 向量的分量访问（Swizzling）

这是 GLSL 最有特色的操作，用 `.xyzw` 或 `.rgba` 或 `.stpq` 访问分量：

```glsl
vec4 color = vec4(1.0, 0.5, 0.3, 1.0);

// 单分量访问
float r = color.r;    // 1.0
float x = color.x;    // 1.0 (等价于 .r)

// 多分量组合（Swizzling）
vec3  rgb = color.rgb;    // (1.0, 0.5, 0.3)
vec2  rg  = color.rg;     // (1.0, 0.5)
vec3  xxx = color.xxx;    // (1.0, 1.0, 1.0) — 可以重复

// 重新排列
vec3  bgr = color.bgr;    // (0.3, 0.5, 1.0) — 顺序可变

// 赋值
color.a = 0.5;            // 只改 alpha
color.rgb = vec3(0.0);    // 只改 RGB

// ⚠️ 注意: .xyzw 和 .rgba 不能混用
// color.rx  ← 错误！
```

### 2.4 内置函数

```glsl
// ─── 数学函数 ───
float a = abs(x);        // 绝对值
float b = sin(x);        // 正弦
float c = cos(x);        // 余弦
float d = pow(x, 2.0);   // 幂
float e = sqrt(x);       // 平方根
float f = exp(x);        // 指数
float g = log(x);        // 对数

// ─── 限制/范围 ───
float h = clamp(x, 0.0, 1.0);  // 限制在 [0, 1]
float i = min(a, b);           // 最小值
float j = max(a, b);           // 最大值

// ─── 插值 ───
float k = mix(a, b, t);   // a*(1-t) + b*t  线性插值
float m = smoothstep(0.0, 1.0, x);  // 平滑阶梯

// ─── 向量函数 ───
float len = length(v);          // 向量长度
float dot_ = dot(a, b);         // 点积
vec3  cross_ = cross(a, b);     // 叉积
vec3  n = normalize(v);         // 归一化
float dist = distance(a, b);   // 两点距离

// ─── 矩阵函数 ───
mat4 inv = inverse(m);          // 矩阵求逆
mat4 tr = transpose(m);         // 矩阵转置
float det = determinant(m);     // 行列式

// ─── 纹理函数 ───
vec4 c = texture(sampler2D, uv);       // 2D 纹理采样
vec4 v = texture(sampler3D, coord);    // 3D 纹理采样
```

### 2.5 控制流

```glsl
// if-else
if (x > 0.5) {
    color = vec3(1.0);
} else {
    color = vec3(0.0);
}

// for 循环（注意: GLSL 要求循环次数在编译时可确定上界）
for (int i = 0; i < 100; i++) {
    sum += data[i];
}

// while 循环
while (t < 1.0) {
    t += 0.01;
}

// 三元运算符
float val = (x > 0.0) ? x : -x;
```

### 2.6 限定符（最重要！）

GLSL 有四种变量限定符，决定了变量的来源和用途：

```glsl
// ─── 1. in: 从上一阶段接收的数据 ───
// 顶点着色器中: 从 CPU 传入的顶点属性
layout(location = 0) in vec3 inPosition;   // 顶点坐标
layout(location = 1) in vec2 inTexCoord;   // 纹理坐标

// 片元着色器中: 从顶点着色器传来的数据（经过插值）
in vec2 vTexCoord;
in vec3 worldPos;

// ─── 2. out: 传递给下一阶段的数据 ───
// 顶点着色器中: 传给片元着色器
out vec2 vTexCoord;
out vec3 worldPos;

// 片元着色器中: 最终输出颜色
out vec4 FragColor;

// ─── 3. uniform: 全局常量，所有顶点/片元共享同一值 ───
// 从 CPU 通过 glUniform* 设置
uniform mat4 projection;       // 投影矩阵
uniform float time;            // 时间
uniform vec3 lightPos;         // 光源位置
uniform sampler3D volume;      // 3D 纹理

// ─── 4. const: 编译时常量 ───
const int MAX_STEPS = 256;
const float PI = 3.14159265;
```

```
数据流总结:

  CPU (C++)                GPU (Shader)
  ┌──────────┐             ┌──────────────────┐
  │ glVertexAttribPtr ──→  │ in (顶点属性)      │
  │                      │  │                   │
  │ glUniform*  ────────→  │ uniform (全局常量) │
  │                      │  │                   │
  │ glBindTexture ──────→  │ uniform sampler    │
  └──────────┘             │                   │
                           │  顶点着色器处理      │
                           │     │ out          │
                           │     ▼ (光栅化插值)   │
                           │  片元着色器处理      │
                           │     │ out          │
                           │     ▼              │
                           │  帧缓冲 (屏幕)      │
                           └──────────────────┘
```

---

## 第三章：顶点着色器（Vertex Shader）

### 3.1 顶点着色器的职责

```
输入: 一个顶点的原始数据（坐标、颜色、法线、纹理坐标等）
输出: 该顶点在屏幕上的位置 (gl_Position) + 任意你想传递的数据

核心工作:
  1. 坐标变换: 模型坐标 → 世界坐标 → 视图坐标 → 裁剪坐标
  2. 传递数据: 把颜色、纹理坐标等传给片元着色器
```

### 3.2 坐标变换流水线

```
模型空间          世界空间          视图空间          裁剪空间         屏幕空间
(局部坐标)  ──→  (全局坐标)  ─→  (相机视角)  ──→  (投影后)  ──→  (像素)
     Model矩阵      View矩阵       Projection矩阵    透视除法+视口变换

gl_Position = projection * view * model * vec4(position, 1.0);

本项目简化为:
gl_Position = projection * modelview * vec4(inPosition, 1.0);
```

### 3.3 最简单的顶点着色器

```glsl
#version 430 core

// 输入: 顶点坐标
layout(location = 0) in vec3 inPosition;

// 输出: 传给片元着色器的纹理坐标
out vec2 vTexCoord;

void main()
{
    // gl_Position 是内置变量，必须设置
    // 这里不做任何变换，直接输出
    gl_Position = vec4(inPosition, 1.0);

    // 把顶点坐标映射到纹理坐标 [0, 1]
    vTexCoord = inPosition.xy * 0.5 + 0.5;
}
```

### 3.4 项目实例：体渲染顶点着色器

来自 `shader/volume_vert_glsl.vert`：

```glsl
#version 430 core

// 输入: 立方体的 8 个顶点
layout (location = 0) in vec3 inPosition;

// Uniform: 从 C++ 传入的变换矩阵
uniform mat4 projection;
uniform mat4 modelview;

// 输出: 传给片元着色器的数据
out vec4 cameraPos;   // 相机位置（世界坐标）
out vec4 vertexPos;   // 顶点位置（模型坐标）
out vec3 worldPos;    // 顶点位置（世界坐标）
out vec3 worldNormal; // 顶点法线

// 立方体 8 个顶点的法线（硬编码）
const vec3 cubeNormals[8] = vec3[](
    vec3(-0.577, -0.577, -0.577),
    vec3( 0.577, -0.577, -0.577),
    vec3( 0.577,  0.577, -0.577),
    vec3(-0.577,  0.577, -0.577),
    vec3(-0.577, -0.577,  0.577),
    vec3( 0.577, -0.577,  0.577),
    vec3( 0.577,  0.577,  0.577),
    vec3(-0.577,  0.577,  0.577)
);

void main()
{
    // ── 核心: 坐标变换 ──
    // 模型坐标 → 裁剪坐标
    gl_Position = projection * modelview * vec4(inPosition, 1.0);

    // ── 计算相机位置 ──
    // modelview 矩阵的逆 × 原点 = 相机在世界空间的位置
    cameraPos = inverse(modelview) * vec4(0.0, 0.0, 0.0, 1.0);

    // ── 传递顶点位置给片元着色器 ──
    vertexPos = vec4(inPosition, 1.0);
    worldPos = inPosition;

    // ── 根据顶点索引选择法线 ──
    int vertexID = gl_VertexID % 8;
    worldNormal = normalize(cubeNormals[vertexID]);
}
```

**逐行解读：**

| 行 | 代码 | 含义 |
|---|------|------|
| 1 | `#version 430 core` | 使用 GLSL 4.30 版本（对应 OpenGL 4.3） |
| 3 | `layout(location = 0) in vec3 inPosition` | 从 VAO 的 0 号属性槽读取顶点坐标 |
| 6-7 | `uniform mat4 projection/modelview` | 从 C++ 通过 `glUniformMatrix4fv` 传入 |
| 9-12 | `out vec4 cameraPos...` | 声明要传给片元着色器的变量 |
| 14-23 | `const vec3 cubeNormals[8]` | 硬编码的立方体法线表 |
| 28 | `gl_Position = projection * modelview * ...` | **最关键的一行**：坐标变换 |
| 31 | `cameraPos = inverse(modelview) * ...` | 反推相机世界坐标 |
| 38 | `gl_VertexID % 8` | 用顶点索引取模来查法线表 |

### 3.5 项目实例：MPR 顶点着色器

来自 `shader/mpr_vert.vert`，更简单：

```glsl
#version 430 core

layout (location = 0) in vec2 inPosition;   // 2D 坐标 [-1, 1]
layout (location = 1) in vec2 inTexCoord;   // 纹理坐标 [0, 1]

out vec2 vTexCoord;

void main()
{
    // 直接输出 2D 坐标，不需要矩阵变换
    // 因为 MPR 是全屏四边形，坐标已经是裁剪空间
    gl_Position = vec4(inPosition, 0.0, 1.0);
    vTexCoord = inTexCoord;
}
```

> **要点：** 如果你的几何体已经是屏幕坐标（如全屏四边形），就不需要投影矩阵变换。

---

## 第四章：片元着色器（Fragment Shader）

### 4.1 片元着色器的职责

```
输入: 光栅化后的片元（像素）数据
      - 来自顶点着色器的 out 变量（已被插值）
      - 内置变量 (gl_FragCoord 等)

输出: 该像素的最终颜色
      - 通过 out 变量输出

每个像素执行一次！
```

### 4.2 最简单的片元着色器

```glsl
#version 430 core

// 输入: 从顶点着色器传来的纹理坐标（已插值）
in vec2 vTexCoord;

// 输出: 像素颜色
out vec4 FragColor;

// Uniform: 2D 纹理
uniform sampler2D tex;

void main()
{
    // 采样纹理，输出颜色
    FragColor = texture(tex, vTexCoord);
}
```

### 4.3 项目实例：CT 显示片元着色器

来自 `shader/ct_display_frag.frag`，这是一个完整的实用着色器：

```glsl
#version 430 core

in vec2 TexCoord;          // 从顶点着色器传来的纹理坐标
out vec4 FragColor;        // 最终输出颜色

uniform sampler2D displayTex;   // 待显示的纹理
uniform float windowCenter;     // 窗位（CT 值中心）
uniform float windowWidth;      // 窗宽（CT 值范围）
uniform int   colorMode;        // 0=灰度, 1=热力图, 2=反色

// 热力图颜色映射函数
vec3 heatmap(float t)
{
    t = clamp(t, 0.0, 1.0);
    vec3 c;
    if (t < 0.25) {
        c = vec3(0.0, 4.0 * t, 1.0);           // 蓝 → 青
    } else if (t < 0.5) {
        c = vec3(0.0, 1.0, 1.0 - 4.0 * (t - 0.25));  // 青 → 绿
    } else if (t < 0.75) {
        c = vec3(4.0 * (t - 0.5), 1.0, 0.0);   // 绿 → 黄
    } else {
        c = vec3(1.0, 1.0 - 4.0 * (t - 0.75), 0.0);  // 黄 → 红
    }
    return c;
}

void main()
{
    // 1. 采样纹理（取 CT 值）
    float val = texture(displayTex, TexCoord).r;

    // 2. 窗宽窗位映射
    //    CT 值范围: -1000(空气) ~ +3000(骨骼)
    //    映射到 [0, 1] 用于显示
    float lower = windowCenter - windowWidth * 0.5;
    float upper = windowCenter + windowWidth * 0.5;
    float mapped = clamp((val - lower) / (upper - lower), 0.0, 1.0);

    // 3. 根据模式选择颜色
    vec3 color;
    if (colorMode == 0) {
        color = vec3(mapped);           // 灰度: R=G=B=mapped
    } else if (colorMode == 1) {
        color = heatmap(mapped);        // 热力图
    } else {
        color = vec3(1.0 - mapped);     // 反色
    }

    FragColor = vec4(color, 1.0);
}
```

**学习要点：**

1. **`texture()` 函数**：采样纹理，返回 `vec4`（RGBA）
2. **`.r` 分量**：CT 数据是单通道浮点纹理，只取 R 分量
3. **`clamp()` 函数**：限制值在范围内，防止溢出
4. **条件分支**：用 `if/else` 选择不同显示模式
5. **自定义函数**：`heatmap()` 是用户自定义的颜色映射函数

### 4.4 项目实例：体渲染片元着色器（核心算法）

来自 `shader/volume_frag_glsl.frag`，这是项目最复杂的着色器。核心逻辑：

```glsl
#version 430 core

// ── Uniform 变量 ──
uniform sampler3D volume;       // 3D 体数据纹理
uniform sampler1D transferFunc; // 1D 传递函数（密度→颜色）
uniform float stepSize = 0.001; // 采样步长
uniform float brightness = 1.5;
uniform float ambient = 0.3;    // 环境光
uniform float diffuse = 0.7;    // 漫反射
uniform float specular = 0.3;   // 镜面反射
uniform float shininess = 32.0; // 高光指数
uniform vec3 lightPos = vec3(2.0, 2.0, 2.0);
uniform vec3 lightColor = vec3(1.0, 1.0, 0.9);

// ── 从顶点着色器传来的数据 ──
in vec4 cameraPos;   // 相机位置
in vec4 vertexPos;   // 顶点位置
in vec3 worldPos;
in vec3 worldNormal;

// ── 输出 ──
out vec4 outColor;

// 射线与 AABB（轴对齐包围盒）相交测试
bool intersectRayAABB(vec3 rayOrigin, vec3 rayDir,
                      vec3 boxMin, vec3 boxMax,
                      out float tEnter, out float tExit)
{
    vec3 invRayDir = 1.0 / rayDir;
    vec3 t1 = (boxMin - rayOrigin) * invRayDir;
    vec3 t2 = (boxMax - rayOrigin) * invRayDir;
    vec3 tMin = min(t1, t2);
    vec3 tMax = max(t1, t2);
    tEnter = max(max(tMin.x, tMin.y), tMin.z);
    tExit  = min(min(tMax.x, tMax.y), tMax.z);
    return tExit >= max(tEnter, 0.0);
}

// 计算梯度（用作法线，用于光照计算）
vec3 computeSimpleGradient(vec3 pos)
{
    const float eps = 0.01;
    vec3 grad;
    grad.x = texture(volume, pos + vec3(eps, 0.0, 0.0)).r
           - texture(volume, pos - vec3(eps, 0.0, 0.0)).r;
    grad.y = texture(volume, pos + vec3(0.0, eps, 0.0)).r
           - texture(volume, pos - vec3(0.0, eps, 0.0)).r;
    grad.z = texture(volume, pos + vec3(0.0, 0.0, eps)).r
           - texture(volume, pos - vec3(0.0, 0.0, eps)).r;
    return normalize(grad);
}

// Phong 光照模型
vec3 applyPhongLighting(vec3 color, vec3 position, vec3 normal, vec3 viewDir)
{
    vec3 lightDir = normalize(lightPos - position);
    vec3 ambientColor = ambient * color;
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuseColor = diff * diffuse * color * lightColor;
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specularColor = spec * specular * lightColor;
    return ambientColor + (diffuseColor + specularColor) * 1.0;
}

void main()
{
    const vec3 boxMin = vec3(0.0);
    const vec3 boxMax = vec3(1.0);

    // ── 1. 构造射线 ──
    vec3 rayOrigin = cameraPos.xyz;
    vec3 rayEnd = vertexPos.xyz;
    vec3 rayDir = normalize(rayEnd - rayOrigin);

    // ── 2. 射线与体数据包围盒求交 ──
    float tEnter, tExit;
    if (!intersectRayAABB(rayOrigin, rayDir, boxMin, boxMax, tEnter, tExit))
        discard;  // 射线没碰到体数据，丢弃该像素

    tEnter = max(tEnter, 0.0);
    vec3 entryPoint = rayOrigin + rayDir * tEnter;
    vec3 exitPoint  = rayOrigin + rayDir * tExit;
    float rayLength = distance(entryPoint, exitPoint);
    int numSteps = int(rayLength / stepSize) + 1;

    // ── 3. 光线投射循环（核心！）──
    vec4 accumulatedColor = vec4(0.0);
    float accumulatedAlpha = 0.0;
    vec3 viewDir = normalize(-rayDir);

    for (int i = 0; i < numSteps; ++i)
    {
        // 提前终止：已经不透明了
        if (accumulatedAlpha > 0.99) break;

        // 计算当前采样点位置
        float t = tEnter + float(i) * stepSize;
        vec3 samplePos = rayOrigin + rayDir * t;

        // 采样体数据
        float density = texture(volume, samplePos).r;

        // 通过传递函数获取颜色和透明度
        vec4 colorSample = texture(transferFunc, density);

        // 跳过透明区域
        if (colorSample.a < 0.01) continue;

        // 计算法线（梯度）
        vec3 normal = computeSimpleGradient(samplePos);

        // 应用光照
        vec3 lit = applyPhongLighting(colorSample.rgb, samplePos, normal, viewDir);

        // 前向合成（Alpha Blending）
        float alpha = colorSample.a * stepSize * 100.0;  // 透明度缩放
        accumulatedColor.rgb += (1.0 - accumulatedAlpha) * lit * alpha;
        accumulatedAlpha += (1.0 - accumulatedAlpha) * alpha;
    }

    outColor = vec4(accumulatedColor.rgb, accumulatedAlpha);
}
```

**这个着色器做了什么（图解）：**

```
        相机
         👁
          │
          │ 射线方向
          │
    ┌─────┼─────────────┐  ← boxMax (1,1,1)
    │     │             │
    │  entryPoint       │
    │     ●──→──→──→──● │  ← 沿射线逐步采样
    │     │   采样点    │ │
    │     │             │
    │     │             │
    └─────┴─────────────┘  ← boxMin (0,0,0)
              exitPoint

每个采样点:
  1. texture(volume, pos) → 密度值
  2. texture(transferFunc, density) → 颜色 + 透明度
  3. computeSimpleGradient(pos) → 法线
  4. applyPhongLighting(...) → 光照后的颜色
  5. Alpha Blending 累加到最终颜色
```

---

## 第五章：Compute Shader（计算着色器）

### 5.1 Compute Shader 的特点

```
顶点/片元着色器: 绑定在图形管线上，输入是顶点/纹理，输出是像素
Compute Shader:  脱离图形管线，纯通用计算，输入/输出都是缓冲区或图像

适用场景:
  - CT 重建（滤波反投影）
  - 图像处理（滤波、分割）
  - 物理模拟
  - 任何大规模并行计算
```

### 5.2 工作组（Work Group）概念

```
Compute Shader 以"工作组"为单位执行:

  ┌─────────────────────────────────────────┐
  │              全局工作空间                   │
  │  ┌───┬───┬───┬───┬───┬───┬───┬───┐     │
  │  │ WG│ WG│ WG│ WG│ WG│ WG│ WG│ WG│     │  ← glDispatchCompute(8, 8, 1)
  │  ├───┼───┼───┼───┼───┼───┼───┼───┤     │
  │  │ WG│ WG│ WG│ WG│ WG│ WG│ WG│ WG│     │
  │  └───┴───┴───┴───┴───┴───┴───┴───┘     │
  └─────────────────────────────────────────┘

  每个工作组内部:
  ┌───┬───┬───┬───┐
  │ T │ T │ T │ T │   ← local_size_x = 4
  ├───┼───┼───┼───┤
  │ T │ T │ T │ T │   ← local_size_y = 4
  └───┴───┴───┴───┘
  每个工作组 4×4=16 个线程

  全局: 8×8 工作组 × 16 线程/组 = 64×64 = 4096 个线程
```

### 5.3 项目实例：CT 滤波反投影

来自 `shader/ct_fbp_compute.glsl`：

```glsl
#version 430 core

// ── 工作组大小: 每组 16×16=256 个线程 ──
layout(local_size_x = 16, local_size_y = 16) in;

// ── 图像绑定（输入/输出）──
layout(binding = 0, r32f) readonly  uniform image2D sinogramTex;  // 输入: 正弦图
layout(binding = 1, r32f) writeonly uniform image2D reconImage;    // 输出: 重建图像

// ── 参数 ──
uniform int   numAngles;       // 投影角度数
uniform int   numDetectors;    // 探测器数
uniform int   outputSize;      // 输出图像尺寸
uniform float angleStart;      // 起始角度
uniform float angleStep;       // 角度步长
uniform float detectorSpacing; // 探测器间距

void main()
{
    // ── 获取当前线程对应的像素坐标 ──
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= outputSize || pixel.y >= outputSize) return;

    float halfOutput = float(outputSize) / 2.0;
    float halfDet    = float(numDetectors) / 2.0;

    // 像素物理坐标（以图像中心为原点）
    float px = float(pixel.x) - halfOutput + 0.5;
    float py = float(pixel.y) - halfOutput + 0.5;

    float sum = 0.0;

    // ── 对每个投影角度累加 ──
    for (int a = 0; a < numAngles; ++a) {
        float theta = angleStart + float(a) * angleStep;
        float cosTheta = cos(theta);
        float sinTheta = sin(theta);

        // 该像素在当前角度下对应的探测器位置
        float t = px * cosTheta + py * sinTheta;
        float detIdx = t / detectorSpacing + halfDet - 0.5;

        // 线性插值采样
        int d0 = int(floor(detIdx));
        int d1 = d0 + 1;
        float frac = detIdx - float(d0);

        float val = 0.0;
        if (d0 >= 0 && d1 < numDetectors) {
            float v0 = imageLoad(sinogramTex, ivec2(d0, a)).r;
            float v1 = imageLoad(sinogramTex, ivec2(d1, a)).r;
            val = mix(v0, v1, frac);
        }

        sum += val;
    }

    // 写入结果
    imageStore(reconImage, pixel, vec4(sum, 0.0, 0.0, 0.0));
}
```

**关键概念：**

| 概念 | 代码 | 含义 |
|------|------|------|
| 工作组大小 | `layout(local_size_x = 16, local_size_y = 16)` | 每组 16×16 线程 |
| 全局线程 ID | `gl_GlobalInvocationID.xy` | 当前线程在全局中的坐标 |
| 图像读取 | `imageLoad(sinogramTex, ivec2(d0, a))` | 读取图像像素 |
| 图像写入 | `imageStore(reconImage, pixel, vec4(...))` | 写入图像像素 |
| 图像格式 | `r32f` | 单通道 32 位浮点 |

---

## 第六章：数据传递机制

### 6.1 顶点属性（Vertex Attributes）

```
C++ 端:
  // 定义顶点数据
  float vertices[] = {
    // 位置        // 纹理坐标
    -1, -1,       0, 0,
     1, -1,       1, 0,
     1,  1,       1, 1,
    -1,  1,       0, 1
  };

  // 设置属性指针
  // location=0: 位置 (2 个 float)
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  // location=1: 纹理坐标 (2 个 float)
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(2*sizeof(float)));
  glEnableVertexAttribArray(1);

Shader 端:
  layout(location = 0) in vec2 inPosition;   // 对应 location=0
  layout(location = 1) in vec2 inTexCoord;   // 对应 location=1
```

### 6.2 Uniform 变量

```cpp
// C++ 端: 设置 uniform 值
GLint loc = glGetUniformLocation(program, "windowCenter");
glUniform1f(loc, 0.5f);

// 设置向量
GLint loc2 = glGetUniformLocation(program, "lightPos");
glUniform3f(loc2, 2.0f, 2.0f, 2.0f);

// 设置矩阵
GLint loc3 = glGetUniformLocation(program, "projection");
glUniformMatrix4fv(loc3, 1, GL_FALSE, &projMatrix[0][0]);
```

```glsl
// Shader 端: 声明 uniform
uniform float windowCenter;
uniform vec3  lightPos;
uniform mat4  projection;
```

### 6.3 顶点→片元数据传递

```
顶点着色器:                    片元着色器:

out vec2 vTexCoord;     →     in vec2 vTexCoord;
out vec3 worldPos;      →     in vec3 worldPos;

// 名称和类型必须完全一致！
// 光栅化阶段会自动对 out 变量进行插值
// 三角形三个顶点的值 → 像素位置的插值值
```

```
插值示意:

  顶点 A (颜色红)          顶点 B (颜色绿)
    ●──────────────────────●
       ╲                  ╱
        ╲    ● 像素      ╱
         ╲  (插值颜色)  ╱
          ╲            ╱
           ●──────────●
        顶点 C (颜色蓝)

  像素颜色 = 重心坐标插值(A红, B绿, C蓝)
```

---

## 第七章：纹理采样

### 7.1 纹理类型

```glsl
// 2D 纹理: 最常见，用于图片、渲染目标
uniform sampler2D myTexture;
vec4 color = texture(myTexture, vec2(u, v));  // uv ∈ [0, 1]

// 3D 纹理: 体数据（CT/MRI）
uniform sampler3D volume;
float density = texture(volume, vec3(x, y, z)).r;  // xyz ∈ [0, 1]

// 1D 纹理: 传递函数（密度→颜色映射表）
uniform sampler1D transferFunc;
vec4 color = texture(transferFunc, density).r;  // density ∈ [0, 1]
```

### 7.2 纹理坐标

```
2D 纹理坐标:

  (0,1) ─────────── (1,1)
   │                   │
   │     纹理空间       │
   │     (u, v)        │
   │                   │
   │                   │
  (0,0) ─────────── (1,0)

  ⚠️ 注意: 纹理坐标的原点在左下角
     而图像坐标的原点通常在左上角
     → 需要翻转 Y 轴
```

### 7.3 纹理过滤

```
放大过滤 (纹理像素 > 屏幕像素):
  GL_NEAREST: 取最近像素（马赛克效果）
  GL_LINEAR:  线性插值（平滑效果）

缩小过滤 (纹理像素 < 屏幕像素):
  GL_NEAREST_MIPMAP_NEAREST: 选最近 mip，取最近像素
  GL_LINEAR_MIPMAP_NEAREST:  选最近 mip，线性插值
  GL_NEAREST_MIPMAP_LINEAR:  两个 mip 混合，最近像素
  GL_LINEAR_MIPMAP_LINEAR:   两个 mip 混合，线性插值（三线性，质量最好）

C++ 设置:
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
```

### 7.4 3D 纹理（体数据）

```cpp
// C++ 端: 创建 3D 纹理
GLuint volumeTex;
glGenTextures(1, &volumeTex);
glBindTexture(GL_TEXTURE_3D, volumeTex);

// 上传体数据
glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F,
             width, height, depth, 0,
             GL_RED, GL_FLOAT, volumeData);

// 过滤设置
glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

// 绑定到着色器
glActiveTexture(GL_TEXTURE0);
glBindTexture(GL_TEXTURE_3D, volumeTex);
glUniform1i(glGetUniformLocation(program, "volume"), 0);  // 0 = TEXTURE0
```

```glsl
// Shader 端
uniform sampler3D volume;

void main()
{
    vec3 coord = vec3(x, y, z);  // [0, 1] 归一化坐标
    float density = texture(volume, coord).r;
}
```

---

## 第八章：实战拆解——项目着色器逐行解读

### 8.1 MPR 着色器完整拆解

**顶点着色器** `mpr_vert.vert`：

```glsl
#version 430 core
// ↑ GLSL 4.30 版本，支持 compute shader、SSBO 等

layout (location = 0) in vec2 inPosition;
// ↑ 从 VAO location=0 读取 2D 坐标
//   值域 [-1, 1]，覆盖整个屏幕

layout (location = 1) in vec2 inTexCoord;
// ↑ 从 VAO location=1 读取纹理坐标
//   值域 [0, 1]

out vec2 vTexCoord;
// ↑ 声明输出变量，传给片元着色器

void main()
{
    gl_Position = vec4(inPosition, 0.0, 1.0);
    // ↑ 直接使用 2D 坐标作为裁剪空间坐标
    //   z=0 (不深度测试), w=1 (齐次坐标)
    //   不需要投影矩阵，因为已经是屏幕空间

    vTexCoord = inTexCoord;
    // ↑ 传递纹理坐标
}
```

**片元着色器** `mpr_frag.frag`（关键部分）：

```glsl
#version 430 core

in vec2 vTexCoord;       // 从顶点着色器传来的纹理坐标
out vec4 outColor;       // 输出颜色

uniform sampler3D volume;          // 3D 体数据
uniform int uSliceAxis;            // 切面方向: 0=轴位, 1=矢状, 2=冠状
uniform float uSlicePosition;      // 切面位置 [0, 1]
uniform float windowLevel;         // 窗位
uniform float windowWidth;         // 窗宽

// 根据 2D 纹理坐标和切面方向，构建 3D 纹理坐标
vec3 buildTexCoord3D(vec2 uv, float slicePos, int axis)
{
    if (axis == 0) {
        // 轴位切面 (Axial): 沿 Z 轴切片，显示 XY 平面
        return vec3(uv.x, uv.y, slicePos);
    } else if (axis == 1) {
        // 矢状切面 (Sagittal): 沿 X 轴切片，显示 YZ 平面
        return vec3(slicePos, uv.y, uv.x);
    } else {
        // 冠状切面 (Coronal): 沿 Y 轴切片，显示 XZ 平面
        return vec3(uv.x, slicePos, uv.y);
    }
}

void main()
{
    // 1. 构建 3D 采样坐标
    vec3 texCoord3D = buildTexCoord3D(vTexCoord, uSlicePosition, uSliceAxis);

    // 2. 采样 3D 体数据
    float density = texture(volume, texCoord3D).r;

    // 3. 窗宽窗位映射
    float windowMin = windowLevel - windowWidth / 2.0;
    float windowMax = windowLevel + windowWidth / 2.0;
    float windowed = clamp((density - windowMin) / windowWidth, 0.0, 1.0);

    // 4. 输出灰度图
    outColor = vec4(vec3(windowed), 1.0);
}
```

```
MPR 原理图:

  3D 体数据 (256×256×128)
  ┌──────────────┐
  │              │
  │   ┌──────┐   │ ← Axial 切面 (Z=固定值)
  │   │      │   │   采样坐标: (u, v, slicePos)
  │   │ 切面  │   │
  │   │      │   │
  │   └──────┘   │
  │              │
  │  ┌──┐        │ ← Sagittal 切面 (X=固定值)
  │  │  │        │   采样坐标: (slicePos, v, u)
  │  │  │        │
  │  │  │        │
  │  └──┘        │
  └──────────────┘

  全屏四边形 (2D):
  (-1,-1) ─────── (1,-1)
    │  ┌─────────┐  │
    │  │  像素    │  │
    │  │  采样    │  │
    │  │  3D纹理  │  │
    │  └─────────┘  │
  (-1, 1) ─────── (1, 1)
```

---

## 第九章：在 C++ 中加载和使用 Shader

### 9.1 完整的 Shader 加载流程

```cpp
#include <glad/glad.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

// ─── 读取着色器文件 ───
std::string readShaderFile(const char* path)
{
    std::ifstream file(path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// ─── 编译着色器 ───
GLuint compileShader(GLenum type, const char* source)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    // 检查编译错误
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader 编译错误:\n" << infoLog << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

// ─── 创建着色器程序 ───
GLuint createShaderProgram(const char* vertPath, const char* fragPath)
{
    // 1. 读取源码
    std::string vertSrc = readShaderFile(vertPath);
    std::string fragSrc = readShaderFile(fragPath);

    // 2. 编译
    GLuint vertShader = compileShader(GL_VERTEX_SHADER, vertSrc.c_str());
    GLuint fragShader = compileShader(GL_FRAGMENT_SHADER, fragSrc.c_str());

    // 3. 链接
    GLuint program = glCreateProgram();
    glAttachShader(program, vertShader);
    glAttachShader(program, fragShader);
    glLinkProgram(program);

    // 检查链接错误
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Shader 链接错误:\n" << infoLog << std::endl;
        return 0;
    }

    // 4. 删除中间对象（已链接到程序中，不再需要）
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    return program;
}

// ─── Compute Shader 单独创建 ───
GLuint createComputeProgram(const char* compPath)
{
    std::string compSrc = readShaderFile(compPath);
    GLuint compShader = compileShader(GL_COMPUTE_SHADER, compSrc.c_str());

    GLuint program = glCreateProgram();
    glAttachShader(program, compShader);
    glLinkProgram(program);

    glDeleteShader(compShader);
    return program;
}
```

### 9.2 使用着色器渲染

```cpp
// ─── 使用顶点+片元着色器 ───
void renderWithShader(GLuint program, GLuint VAO, int vertexCount)
{
    // 1. 激活着色器程序
    glUseProgram(program);

    // 2. 设置 Uniform 变量
    glUniform1f(glGetUniformLocation(program, "windowLevel"), 0.5f);
    glUniform1f(glGetUniformLocation(program, "windowWidth"), 1.0f);
    glUniform1i(glGetUniformLocation(program, "colorMode"), 1);

    // 3. 绑定纹理
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, displayTex);
    glUniform1i(glGetUniformLocation(program, "displayTex"), 0);

    // 4. 绑定 VAO 并绘制
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, vertexCount);
}

// ─── 使用 Compute Shader ───
void runComputeShader(GLuint program, int outputSize)
{
    glUseProgram(program);

    // 设置参数
    glUniform1i(glGetUniformLocation(program, "numAngles"), 180);
    glUniform1i(glGetUniformLocation(program, "outputSize"), outputSize);

    // 绑定图像纹理
    glBindImageTexture(0, sinogramTex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(1, reconImage, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

    // 派发计算（工作组数 = outputSize/16 向上取整）
    int groupsX = (outputSize + 15) / 16;
    int groupsY = (outputSize + 15) / 16;
    glDispatchCompute(groupsX, groupsY, 1);

    // 等待计算完成
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}
```

---

## 第十章：常见错误与调试技巧

### 10.1 编译错误

```glsl
// ❌ 错误: 浮点字面量不带小数点
float x = 1;      // 编译警告/错误
float x = 1.0;    // ✅ 正确

// ❌ 错误: 向量分量混用
vec4 c = vec4(1.0);
float v = c.rx;   // 错误: 不能混用 .rgba 和 .xyzw
float v = c.r;    // ✅ 正确
float v = c.x;    // ✅ 正确

// ❌ 错误: 循环上界不可确定
for (int i = 0; i < count; i++) { }  // count 是 uniform 变量，某些驱动不允许
for (int i = 0; i < 256; i++) { }    // ✅ 常量上界

// ❌ 错误: 递归
void f() { f(); }  // GLSL 不支持递归！
```

### 10.2 链接错误

```
问题: 顶点着色器的 out 变量和片元着色器的 in 变量不匹配

顶点着色器:                    片元着色器:
  out vec2 vTexCoord;    ≠      in vec2 TexCoord;     // 名称不同！
  out vec3 worldPos;     ≠      in vec3 WorldPos;     // 大小写不同！

✅ 解决: 名称和类型必须完全一致
  out vec2 vTexCoord;  →  in vec2 vTexCoord;
```

### 10.3 常见运行时问题

```
问题1: 画面全黑
  原因: Uniform 没设置 / 纹理没绑定 / 着色器没激活
  检查: glUseProgram(program) 是否在绘制前调用？

问题2: 画面全白
  原因: 纹理坐标超出 [0,1] / 窗宽窗位设置错误
  检查: 用 gl_FragCoord 做调试输出

问题3: 纹理采样返回 0
  原因: 纹理单元没绑定 / uniform 值不对
  检查:
    glActiveTexture(GL_TEXTURE0);        // 激活单元 0
    glBindTexture(GL_TEXTURE_2D, tex);   // 绑定纹理
    glUniform1i(loc, 0);                 // 告诉 shader 用单元 0

问题4: Compute Shader 没输出
  原因: 没加内存屏障
  检查: glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
```

### 10.4 调试技巧

```glsl
// 技巧1: 用颜色输出变量值（可视化调试）
void main()
{
    float density = texture(volume, coord).r;
    // 把密度值映射到颜色，直观查看
    outColor = vec4(density, 0.0, 0.0, 1.0);  // 红色通道显示密度
}

// 技巧2: 用 gl_FragCoord 检查像素坐标
void main()
{
    // 把屏幕坐标映射到颜色
    vec2 uv = gl_FragCoord.xy / vec2(1920.0, 1080.0);
    outColor = vec4(uv, 0.0, 1.0);
    // 左下角黑色 → 右上角黄色
}

// 技巧3: 检查法线方向
void main()
{
    // 法线 [-1,1] 映射到颜色 [0,1]
    vec3 normalColor = normal * 0.5 + 0.5;
    outColor = vec4(normalColor, 1.0);
}
```

---

## 第十一章：练习题

### 练习 1：写一个渐变色片元着色器

```glsl
// 要求: 根据像素的 UV 坐标，输出从左下角红色到右上角绿色的渐变
#version 430 core

in vec2 vTexCoord;
out vec4 FragColor;

void main()
{
    // 你的代码在这里
    // 提示: vTexCoord.x 控制红色, vTexCoord.y 控制绿色
}
```

<details>
<summary>参考答案</summary>

```glsl
void main()
{
    FragColor = vec4(vTexCoord.x, vTexCoord.y, 0.0, 1.0);
}
```
</details>

### 练习 2：写一个灰度反转着色器

```glsl
// 要求: 采样 2D 纹理，将灰度反转（黑变白，白变黑）
#version 430 core

in vec2 vTexCoord;
out vec4 FragColor;
uniform sampler2D tex;

void main()
{
    // 你的代码在这里
}
```

<details>
<summary>参考答案</summary>

```glsl
void main()
{
    vec4 color = texture(tex, vTexCoord);
    float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114)); // 灰度转换
    FragColor = vec4(vec3(1.0 - gray), 1.0);  // 反转
}
```
</details>

### 练习 3：修改体渲染步长

打开 `shader/volume_frag_glsl.frag`，找到 `stepSize` 变量：

1. 把 `stepSize` 改大（如 `0.01`），观察画面变化
2. 把 `stepSize` 改小（如 `0.0001`），观察画面变化
3. 思考：步长大小如何影响画质和性能？

### 练习 4：添加新的颜色映射模式

在 `ct_display_frag.frag` 中添加一个新的 `colorMode == 3`，实现"彩虹"颜色映射：

<details>
<summary>提示</summary>

```glsl
vec3 rainbow(float t)
{
    return 0.5 + 0.5 * cos(6.28318 * (t + vec3(0.0, 0.33, 0.67)));
}

// 在 main() 中:
} else if (colorMode == 3) {
    color = rainbow(mapped);
}
```
</details>

---

## 附录：GLSL 版本对照表

| GLSL 版本 | OpenGL 版本 | 关键特性 |
|-----------|------------|---------|
| 1.10 | 2.0 | 基础版本 |
| 1.20 | 2.1 | `in/out` 替代 `attribute/varying` |
| 1.30 | 3.0 | `layout` 限定符 |
| 1.50 | 3.2 | Geometry Shader |
| 3.30 | 3.3 | 整数运算、实例化 |
| 4.00 | 4.0 | Tessellation Shader |
| **4.30** | **4.3** | **Compute Shader、SSBO、Image Load/Store** |
| 4.50 | 4.5 | Indirect draw |

> **本项目使用 `#version 430 core`**，对应 OpenGL 4.3，支持 Compute Shader 和 Image Load/Store。

---

## 附录：本项目着色器架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    项目着色器架构                              │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 体渲染 (Volume Rendering)                            │    │
│  │  volume_vert_glsl.vert  →  volume_frag_glsl.frag    │    │
│  │  volume_vert_optimized  →  volume_frag_optimized    │    │
│  │  职责: 光线投射 + 传递函数 + Phong光照 + Alpha合成    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ MPR 多平面重建                                       │    │
│  │  mpr_vert.vert  →  mpr_frag.frag                    │    │
│  │  职责: 3D纹理切面采样 + 窗宽窗位 + 十字线            │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ CT 显示                                              │    │
│  │  ct_display_vert.vert  →  ct_display_frag.frag      │    │
│  │  职责: 全屏四边形 + 窗宽窗位 + 热力图/灰度/反色      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ CT 重建 (Compute Shader)                             │    │
│  │  ct_fbp_compute.glsl                                 │    │
│  │  职责: 滤波反投影 (FBP) GPU 并行重建                 │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

**相关文档：**
- [光线投射渲染流程详解.md](光线投射渲染流程详解.md) — 体渲染算法详解
- [着色器优化升级指南.md](着色器优化升级指南.md) — 着色器性能优化
- [坐标系统与光线生成详解.md](坐标系统与光线生成详解.md) — 坐标变换
- [核心代码片段说明.md](核心代码片段说明.md) — C++ 端代码

**最后更新：** 2026年9月18日
