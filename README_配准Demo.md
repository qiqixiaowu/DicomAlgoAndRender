# 配准算法 OpenGL Demo

## 项目结构

```
F:\渲染\OpenglRender\
├── RegistrationDemo.vcxproj          ← 新增：Demo 独立项目
├── include\
│   └── registration.hpp              ← 新增：所有配准算法（Header-Only）
└── src\
    └── main_registration.cpp          ← 新增：OpenGL 可视化主程序
```

## 算法实现一览

`registration.hpp` 包含以下完整实现（纯 C++17，无外部依赖）：

| § | 算法 | 对应源码模块 |
|---|------|------------|
| 5 | **刚性配准**（MSE/NMI，多分辨率梯度下降） | `McsfAlgoRegistration` |
| 6 | **Demons 形变配准**（Thirion 1998 光流力） | `BaseDemons` |
| 7 | **Multi-Grid Demons**（粗→细多分辨率） | `MultiGridDemons` |
| 8 | **LCC Demons**（局部互相关，多模态鲁棒） | `LCCDemons` |
| 9 | **B-Spline FFD**（三次样条自由形变，梯度下降） | `MultiGridFFD` |

## 编译方法

### 方法 1：Visual Studio 2022

1. 打开 `F:\渲染\OpenglRender\OpenglRender.sln`
2. 在解决方案资源管理器中右键 **RegistrationDemo** → **设为启动项目**
3. 选择 `Debug x64` 配置
4. 按 **F5** 编译运行

> 依赖库已在项目属性中配置：GLFW3 (`E:\glfw`)，GLAD (`E:\glad`)，GLM (`E:\All\glm`)

### 方法 2：命令行（MSBuild）

```bat
cd F:\渲染\OpenglRender
msbuild RegistrationDemo.vcxproj /p:Configuration=Release /p:Platform=x64
x64\Release\RegistrationDemo.exe
```

## 运行界面

窗口划分为 **2行×3列** 共 6 个面板：

```
┌──────────────┬──────────────┬──────────────┐
│  Reference   │   Moving     │  Rigid Reg   │
│  (参考图像)   │  (运动图像)   │  [1]/[2]键   │
├──────────────┼──────────────┼──────────────┤
│    Demons    │  MG-Demons   │  LCC/FFD     │
│    [3]键     │    [4]键     │  [5]/[6]键   │
└──────────────┴──────────────┴──────────────┘
```

## 键盘控制

| 按键 | 功能 |
|------|------|
| `1` | 运行刚性配准（MSE 度量）|
| `2` | 运行刚性配准（NMI 度量，适合多模态）|
| `3` | 运行 Demons 形变配准 |
| `4` | 运行 Multi-Grid Demons |
| `5` | 运行 LCC Demons（多模态场景更优）|
| `6` | 运行 B-Spline FFD（刚性初始化 + 形变精化）|
| `A` | **顺序运行所有算法**，控制台输出对比表格 |
| `C` | 切换棋盘融合显示模式（评估配准质量）|
| `R` | 重置数据 |
| `Q`/`ESC` | 退出 |

## 测试数据说明

程序自动生成 **Shepp-Logan** 风格的 192×192 医学影像 Phantom，并对其施加已知变换：

- **平移**：tx=8px，ty=5px
- **旋转**：0.08 rad（约 4.6°）
- **局部形变**：正弦形变场，幅度 3.5px

配准成功时，各算法应能恢复接近参考图像的结果（MSE 显著下降）。

## 控制台输出示例

按 `[A]` 运行所有算法后，控制台输出对比表格：

```
┌─────────────────────┬────────────┬────────────┬─────────────┐
│ Algorithm           │   MSE      │   NMI      │  Time (ms)  │
├─────────────────────┼────────────┼────────────┼─────────────┤
│ Moving (before)     │   0.01842  │   1.23456  │     —       │
│ Rigid Registration  │   0.00421  │   1.41234  │        2341 │
│ Demons (Thirion 1998│   0.00312  │   1.45678  │        4521 │
│ Multi-Grid Demons   │   0.00278  │   1.47123  │        3102 │
│ LCC Demons          │   0.00265  │   1.48901  │        5234 │
└─────────────────────┴────────────┴────────────┴─────────────┘
```

## 算法参数调优参考

### 刚性配准
```cpp
MedReg::RigidRegConfig cfg;
cfg.useNMI       = false;  // true=NMI（多模态），false=MSE（同模态）
cfg.numLevels    = 3;      // 多分辨率层数
cfg.maxIterations = 300;   // 每层最大迭代
cfg.maxStep      = 4.0f;   // 初始步长（像素）
cfg.minStep      = 0.005f; // 收敛阈值
cfg.relaxFactor  = 0.95f;  // 步长松弛系数
```

### Demons 形变配准
```cpp
MedReg::DemonsConfig cfg;
cfg.alpha = 1.0f;    // 力归一化系数（越大→力越弱）
cfg.sigma = 3.0f;    // 高斯平滑σ（越大→形变越平滑）
cfg.step  = 0.4f;    // 更新步长（越大→收敛快但可能振荡）
cfg.maxIterations = 150;
```

### LCC Demons
```cpp
MedReg::LCCConfig cfg;
cfg.sigma       = 4.0f;   // 局部窗口σ
cfg.smoothSigma = 3.0f;   // 力场平滑σ
cfg.step        = 0.08f;  // 步长（比 Demons 小）
cfg.relax       = true;   // 自适应步长衰减
```
