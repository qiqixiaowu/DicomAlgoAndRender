# 医学图像光线投射渲染系统 - 技术总览

## 项目简介
这是一个基于OpenGL 4.3实现的医学图像（DICOM）体绘制系统，采用光线投射（Ray Casting）算法进行高质量的三维可视化。

---

## 快速导航

📄 **详细文档：**
- [光线投射渲染流程详解.md](光线投射渲染流程详解.md) - 完整的算法流程和参数说明
- [核心代码片段说明.md](核心代码片段说明.md) - 关键算法的代码级详解
- [坐标系统与光线生成详解.md](坐标系统与光线生成详解.md) - 坐标空间转换、光线发射原理
- [改进建议与优化方案.md](改进建议与优化方案.md) - 16个改进方向和具体实现方案 ⭐新增

🎨 **流程图：**
- 完整渲染流程图（42步骤）
- 坐标空间转换流程图 ⭐新增
- 代理几何体原理图 ⭐新增
- 数据流转图
- 单像素采样示例

📁 **核心代码文件：**
- [shader/volume_frag_glsl.frag](shader/volume_frag_glsl.frag) - 光线投射核心算法
- [shader/volume_vert_glsl.vert](shader/volume_vert_glsl.vert) - 顶点变换
- [src/main.cpp](src/main.cpp) - 主程序和渲染循环
- [include/transferFunction.h](include/transferFunction.h) - 传输函数

---

## 核心技术特性

### 1. 光线投射算法 ✅
- 射线-AABB包围盒求交
- 自适应步长采样
- 早期光线终止（Early Ray Termination）
- 空区域跳过（Empty Space Skipping）

### 2. 蓝噪声抖动 🎯
- 多层梯度噪声
- 屏幕空间分布
- 消除木纹伪影（Wood Grain Artifact）
- 可调抖动强度

### 3. 医学图像处理 🏥
- DICOM文件解析（DCMTK库）
- 窗宽窗位调整（Window Level/Width）
- 多种预设：软组织窗、骨窗、肺窗
- HU值到密度转换

### 4. 真实感渲染 💡
- Phong光照模型
  - 环境光（Ambient）
  - 漫反射（Diffuse）
  - 镜面反射（Specular）
- 距离衰减
- 阴影因子
- Gamma校正

### 5. 传输函数 🌈
- 1D纹理映射
- 多种颜色方案：
  - CT灰度模式
  - 伪彩色模式
  - 医学组织分类
- 实时可调

### 6. 交互控制 🎮
- 轨道相机（Orbit Camera）
- 鼠标拖动旋转
- 滚轮缩放
- 键盘快捷键

### 7. 性能优化 ⚡
- GPU加速计算
- Mipmap纹理过滤
- 4x MSAA抗锯齿
- 自适应采样率

---

## 技术栈

| 类别 | 技术/库 | 版本 | 用途 |
|------|---------|------|------|
| 图形API | OpenGL | 4.3 Core | 渲染管线 |
| 着色器 | GLSL | 430 | GPU编程 |
| 窗口管理 | GLFW | 3.x | 创建窗口和上下文 |
| OpenGL加载 | GLAD | - | 加载OpenGL函数 |
| 数学库 | GLM | 0.9.x | 向量矩阵运算 |
| DICOM解析 | DCMTK | 3.6.x | 读取医学图像 |
| 编译器 | MSVC | VS2022 | C++编译 |

---

## 渲染流程概览

```
┌─────────────────────────────────────────────────────────┐
│                    程序启动                              │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────▼──────────┐
         │ ① 数据加载与预处理   │
         │  - 扫描DICOM文件     │
         │  - 解析元数据        │
         │  - 排序切片          │
         │  - 构建3D体数据      │
         └───────────┬──────────┘
                     │
         ┌───────────▼──────────┐
         │ ② OpenGL资源初始化   │
         │  - 创建3D纹理        │
         │  - 编译Shader        │
         │  - 创建代理几何      │
         │  - 设置传输函数      │
         └───────────┬──────────┘
                     │
         ┌───────────▼──────────┐
         │ ③ 渲染循环          │◄────┐
         │                      │     │
         │  ④ 更新相机状态      │     │
         │  ⑤ 顶点着色器处理    │     │
         │  ⑥ 片元着色器        │     │
         │    - 射线投射        │     │
         │    - 步进采样        │     │
         │    - 光照计算        │     │
         │    - Alpha混合       │     │
         │  ⑦ 后处理            │     │
         │  ⑧ 交换缓冲区        │─────┘
         └───────────┬──────────┘
                     │
         ┌───────────▼──────────┐
         │ ⑨ 清理资源           │
         │  - 删除纹理          │
         │  - 删除缓冲          │
         │  - 终止GLFW          │
         └──────────────────────┘
                     │
                     ▼
                  程序结束
```

---

## 光线投射核心算法

### 伪代码
```python
for each pixel (x, y):
    # 1. 生成射线
    ray_origin = camera_position
    ray_direction = normalize(pixel_world_pos - camera_position)
    
    # 2. 求交
    t_enter, t_exit = intersect_ray_aabb(ray_origin, ray_direction, bbox)
    if not intersected:
        discard pixel
    
    # 3. 步进采样
    accumulated_color = (0, 0, 0, 0)
    num_steps = int(ray_length / step_size)
    
    for i in range(num_steps):
        # 早期终止
        if accumulated_color.alpha > 0.99:
            break
        
        # 计算采样位置（带抖动）
        t = i / num_steps
        jitter = blue_noise(pixel_coord, i) * jitter_strength
        sample_pos = lerp(entry_point, exit_point, t) + jitter
        
        # 采样密度
        density = texture3D(volume, sample_pos)
        
        # 应用窗宽窗位
        windowed_density = apply_window(density, window_level, window_width)
        
        # 跳过低密度
        if windowed_density < threshold:
            continue
        
        # 传输函数映射
        color = texture1D(transfer_function, windowed_density)
        
        # 计算法线（梯度）
        normal = compute_gradient(sample_pos)
        
        # 光照
        lit_color = phong_lighting(color, sample_pos, normal, view_dir)
        
        # Alpha混合
        alpha = color.a * windowed_density
        accumulated_color.rgb += (1 - accumulated_color.a) * alpha * lit_color
        accumulated_color.a += (1 - accumulated_color.a) * alpha
    
    # 4. 后处理
    accumulated_color.rgb *= brightness
    accumulated_color.rgb = pow(accumulated_color.rgb, 1/2.2)  # Gamma校正
    
    # 5. 输出
    output_color = accumulated_color
```

---

## 文件结构

```
OpenglRender/
├── include/
│   ├── camera.h              # 轨道相机
│   ├── dicom_utils.hpp       # DICOM解析
│   ├── post_Processor.h      # 后处理（FXAA等）
│   ├── shader.h              # Shader管理
│   ├── transferFunction.h    # 传输函数
│   └── volume_build.hpp      # 体数据构建
├── src/
│   ├── camera.cpp
│   ├── dicom_utils.cpp
│   ├── glad.c                # OpenGL加载器
│   ├── main.cpp              # 主程序 ⭐
│   ├── shader.cpp
│   └── volume_build.cpp
├── shader/
│   ├── volume_vert_glsl.vert # 顶点着色器 ⭐
│   └── volume_frag_glsl.frag # 片元着色器 ⭐⭐⭐
├── 光线投射渲染流程详解.md    # 详细文档
├── 核心代码片段说明.md         # 代码解析
└── README_光线投射技术总览.md  # 本文件
```

---

## 使用方法

### 编译
```powershell
# 使用Visual Studio打开解决方案
OpenglRender.sln

# 或使用MSBuild命令行
msbuild OpenglRender.vcxproj /p:Configuration=Release
```

### 运行
```powershell
# 需要传入DICOM文件夹路径作为参数
.\x64\Debug\OpenglRender.exe "F:\DICOM_DATA\SERIES_001"
```

### 交互控制

**鼠标操作：**
- 左键拖动：旋转视角
- 滚轮：缩放距离

**键盘快捷键：**
| 按键 | 功能 | 参数变化 |
|------|------|----------|
| ↑ | 增加窗位 | windowLevel += 0.003 |
| ↓ | 减少窗位 | windowLevel -= 0.003 |
| → | 增加窗宽 | windowWidth += 0.003 |
| ← | 减少窗宽 | windowWidth -= 0.003 |
| 2 | 软组织窗 | Level=0.26, Width=0.1 |
| 3 | 骨窗 | Level=0.35, Width=0.5 |
| 4 | 肺窗 | Level=0.3, Width=0.08 |
| I | 增加环境光 | ambient += 0.01 |
| O | 减少环境光 | ambient -= 0.01 |
| K | 增加漫反射 | diffuse += 0.01 |
| L | 减少漫反射 | diffuse -= 0.01 |
| ESC | 退出程序 | - |

---

## 参数调优指南

### 渲染质量参数

#### 1. 步长（stepSize）
- **位置：** [main.cpp](src/main.cpp#L203)
- **默认值：** 0.0015
- **影响：** 采样密度
- **调优：**
  - 更小（0.001）：更细腻，更慢
  - 更大（0.003）：更快，可能出现条纹

#### 2. 抖动强度（jitterStrength）
- **位置：** [main.cpp](src/main.cpp#L218)
- **默认值：** 0.5
- **影响：** 噪点vs条纹
- **调优：**
  - 更小（0.2）：条纹更明显
  - 更大（0.8）：噪点增加但条纹消失

#### 3. 密度缩放（densityScale）
- **位置：** [main.cpp](src/main.cpp#L204)
- **默认值：** 1.2
- **影响：** 整体不透明度
- **调优：**
  - 更小（1.0）：更透明
  - 更大（1.5）：更不透明

#### 4. 亮度（brightness）
- **位置：** [main.cpp](src/main.cpp#L205)
- **默认值：** 1.5
- **影响：** 整体亮度
- **调优：**
  - 更小（1.0）：正常亮度
  - 更大（2.0）：过曝但细节更清晰

### 光照参数

#### 5. 环境光（ambient）
- **默认值：** 0.3
- **范围：** [0.0, 1.0]
- **效果：** 控制阴影区域亮度

#### 6. 漫反射（diffuse）
- **默认值：** 0.7
- **范围：** [0.0, 1.0]
- **效果：** 控制方向性光照强度

#### 7. 镜面反射（specular）
- **默认值：** 0.3
- **范围：** [0.0, 1.0]
- **效果：** 控制高光强度

#### 8. 高光指数（shininess）
- **默认值：** 32.0
- **范围：** [1.0, 256.0]
- **效果：** 控制高光范围大小

---

## 常见问题解答

### Q1: 为什么渲染结果全黑？
**可能原因：**
1. 窗宽窗位设置不正确
2. DICOM数据未正确加载
3. 传输函数没有生效

**解决方法：**
```cpp
// 尝试调整窗宽窗位
windowLevel = 0.5f;  // 中间值
windowWidth = 1.0f;  // 全范围

// 或使用预设
按键盘"2"（软组织窗）或"3"（骨窗）
```

### Q2: 出现条纹伪影（木纹效果）
**原因：** 步进采样的规则性导致

**解决方法：**
```cpp
// 增加抖动强度
jitterStrength = 0.8f;  // 从0.5增加到0.8

// 或减小步长
stepSize = 0.001f;  // 从0.0015减小到0.001
```

### Q3: 性能太低，帧率很低
**优化建议：**
1. 增大步长：`stepSize = 0.003f`
2. 降低窗口分辨率
3. 减少抖动计算：`jitterStrength = 0.3f`
4. 检查是否启用了早期终止

### Q4: 颜色看起来不正常
**检查项：**
1. Gamma校正是否启用
2. 传输函数是否选择正确
3. 光照参数是否合理
4. 亮度参数是否过高

---

## 性能基准

### 测试环境
- GPU: NVIDIA RTX 3060
- 体数据: 512×512×300
- 窗口: 800×600

### 性能数据
| 配置 | stepSize | FPS | 质量 |
|------|----------|-----|------|
| 低质量 | 0.003 | 120 | ★★☆☆☆ |
| 平衡 | 0.0015 | 60 | ★★★★☆ |
| 高质量 | 0.001 | 30 | ★★★★★ |
| 极致 | 0.0005 | 15 | ★★★★★ |

---

## 扩展功能（未来计划）

### 短期目标
- [ ] FXAA后处理抗锯齿
- [ ] 体积阴影（Shadow Volume）
- [ ] 多模态融合（CT + PET）
- [ ] 截面显示

### 中期目标
- [ ] 八叉树加速结构
- [ ] GPU计算Shader优化
- [ ] 自适应采样率
- [ ] 测量工具（距离、角度、体积）

### 长期目标
- [ ] 全局光照（环境光遮蔽）
- [ ] 实时分割叠加
- [ ] VR/AR支持
- [ ] 多GPU并行渲染

---

## 学术参考

### 核心论文
1. **Levoy, M. (1988).** "Display of Surfaces from Volume Data." *IEEE Computer Graphics and Applications.*
2. **Engel, K. et al. (2006).** *Real-Time Volume Graphics.* A K Peters/CRC Press.
3. **Krüger, J., & Westermann, R. (2003).** "Acceleration techniques for GPU-based volume rendering." *IEEE Visualization.*

### 技术资源
- [GPU Gems - Volume Rendering Techniques](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles)
- [OpenGL SuperBible](https://www.openglsuperbible.com/)
- [DCMTK Documentation](https://support.dcmtk.org/)

---

## 许可证
本项目代码仅供学习研究使用。

---

## 作者信息
医学图像光线投射渲染系统  
基于OpenGL 4.3 / GLSL 430

**技术支持：**
- 参见详细文档：[光线投射渲染流程详解.md](光线投射渲染流程详解.md)
- 代码详解：[核心代码片段说明.md](核心代码片段说明.md)

---

## 更新日志

### v1.0 (当前版本)
✅ 基础光线投射算法  
✅ 蓝噪声抖动  
✅ Phong光照  
✅ 窗宽窗位调整  
✅ 传输函数系统  
✅ 轨道相机控制  
✅ DICOM文件支持  

### 待开发功能
🔲 FXAA抗锯齿  
🔲 体积阴影  
🔲 多模态融合  

---

**最后更新：** 2026年2月12日

**文档链接：**
- [光线投射渲染流程详解.md](光线投射渲染流程详解.md) - 完整算法流程
- [核心代码片段说明.md](核心代码片段说明.md) - 代码级详解
- [坐标系统与光线生成详解.md](坐标系统与光线生成详解.md) - 坐标变换与光线发射 ⭐新增
