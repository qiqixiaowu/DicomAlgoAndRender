# MeshRender — 网格生成与优化渲染

基于 OpenGL 4.3 的延迟渲染引擎，集成程序化网格生成、网格优化（QEM 简化、LOD、平滑）和 PBR 光照。

## 功能

### 网格生成
- 立方体、球体、圆环、圆柱、二十面体
- 程序化地形（高度噪声）
- 平面（可细分）
- 自动法线/切线/双切线计算

### 网格优化
- **顶点焊接**：空间哈希去重
- **平滑**：Laplacian、Taubin（保特征）
- **QEM 简化**：基于二次误差度量的边折叠
- **LOD 链**：自动生成多级细节

### 延迟渲染管线
1. **Shadow Pass** — 2048×2048 阴影深度图，PCF 软阴影
2. **Geometry Pass** — G-Buffer（位置/法线/反照率/材质，RGB16F）
3. **SSAO Pass** — 64 核采样 + 模糊
4. **Lighting Pass** — PBR Cook-Torrance BRDF，多光源（方向光/点光/聚光灯）
5. **Skybox Pass** — 程序化天空 + 太阳光晕
6. **Wireframe Pass** — 线框/法线/深度可视化

### 交互
- 轨道相机 + FPS 相机双模式
- 实时网格切换、优化、LOD 切换
- 动态光源动画

## 键盘控制

| 按键 | 功能 |
|------|------|
| `1-6` | 切换网格类型（立方体/球体/圆环/圆柱/地形/二十面体） |
| `F2` | 线框模式开关 |
| `F3` | 法线可视化 |
| `F4` | 深度可视化 |
| `O` | 优化网格（焊接 + Taubin 平滑 + QEM 简化 50%） |
| `L` | LOD 层级切换（0-3） |
| `Tab` | 切换相机模式（轨道/FPS） |
| `Space` | 自动旋转（轨道模式） |
| `P` | 暂停 |
| `R` | 重置相机 |
| `F1` | 统计信息开关 |
| `ESC` | 退出 |
| `WASD` | FPS 移动 |
| `Shift/Space` | FPS 下降/上升 |
| 鼠标拖动 | 旋转视角 |
| 滚轮 | 缩放 |

## 项目结构

```
MeshRender/
├── mesh.h                  # 网格数据结构与程序化生成
├── mesh_optimizer.h        # 网格优化（焊接/平滑/QEM/LOD）
├── gl_mesh.h               # OpenGL GPU 网格资源
├── mesh_camera.h           # 双模式相机
├── deferred_renderer.h     # 延迟渲染管线
├── main_mesh_render.cpp    # 入口程序
├── MeshRender.vcxproj      # VS 项目文件
├── MeshRender.vcxproj.user # 调试工作目录设置
└── shaders/
    ├── mesh_geometry_pass.vert/frag   # G-Buffer 写入
    ├── mesh_lighting_pass.vert/frag   # PBR 光照
    ├── mesh_shadow_depth.vert/frag    # 阴影深度
    ├── mesh_wireframe.vert/frag       # 线框/法线/深度
    ├── mesh_ssao.vert/frag            # SSAO
    ├── mesh_ssao_blur.frag            # SSAO 模糊
    └── mesh_skybox.vert/frag          # 程序化天空
```

## 依赖

- OpenGL 4.3+
- GLFW 3.x
- GLAD
- GLM
- Visual Studio 2022 (v143)

## 构建

1. 打开 `OpenglRender.sln`
2. 选择 `MeshRender` 项目，配置 `Debug|x64` 或 `Release|x64`
3. 生成并运行

## 技术要点

### PBR Cook-Torrance BRDF
```
f = kD * diffuse + kS * specular
kS = F (Fresnel)
kD = (1 - kS) * (1 - metallic)
specular = DFG / (4 · (N·L) · (N·V))
D = GGX NDF
F = Schlick approximation
G = Smith geometry function
```

### QEM 网格简化
对每个顶点计算二次误差矩阵 Q = Σ K_p，边折叠代价 Δ(v) = v^T Q v，选择代价最小的边折叠。

### SSAO
在切线空间生成半球采样核，通过旋转噪声纹理实现每像素不同的采样方向，最终模糊处理。
