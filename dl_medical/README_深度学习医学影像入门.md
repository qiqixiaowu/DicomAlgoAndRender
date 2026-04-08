# 深度学习医学影像入门 —— 分割与配准

> 从零开始，手把手带你入门深度学习在医学影像分割和配准中的应用

## 目录

- [第零章 环境准备](#第零章-环境准备)
- [第一章 深度学习基础速览](#第一章-深度学习基础速览)
- [第二章 医学图像分割](#第二章-医学图像分割)
- [第三章 医学图像配准](#第三章-医学图像配准)
- [第四章 工程代码导航](#第四章-工程代码导航)
- [第五章 实战训练指南](#第五章-实战训练指南)
- [**第七章 血管分割实战 ← 新增**](#第七章-血管分割实战)
- [第六章 进阶路线](#第六章-进阶路线)
- [附录 推荐资源](#附录-推荐资源)

---

## 第零章 环境准备

### 安装

```bash
# 1. 创建 conda 环境 (推荐)
conda create -n dl_medical python=3.12 -y
conda activate dl_medical

# 2. 安装 PyTorch (根据你的 CUDA 版本选择)
# CUDA 12.2:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 仅 CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. 安装其他依赖
cd dl_medical
pip install -r requirements.txt
```

### 验证安装

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 快速体验

```bash
# 运行演示 (无需 GPU, 无需数据集)
python demo.py
```

---

## 第一章 深度学习基础速览

### 1.1 核心概念

**深度学习 = 数据 + 模型 + 损失函数 + 优化器**

```
                  训练循环
                 ┌──────┐
   数据 ──→ 模型 ──→ 预测
                      │
              损失函数 ← 与真值比较
                      │
              反向传播 (计算梯度)
                      │
              优化器 (更新参数) ──→ 回到模型
                 └──────┘
```

| 组件 | 作用 | 类比 |
|------|------|------|
| **数据** | 训练样本 | 课本上的例题 |
| **模型** | 可学习的函数 | 学生的大脑 |
| **损失函数** | 衡量预测与真值的差距 | 考试分数 |
| **优化器** | 根据梯度更新参数 | 学生根据错题改进 |
| **反向传播** | 计算每个参数对 loss 的影响 | 找到哪里出了错 |

### 1.2 卷积神经网络 (CNN)

CNN 是处理图像的核心架构：

```
输入图像 (1, 256, 256)
    │
    ▼
┌───────────────┐
│  卷积层 3×3   │ → 提取局部特征 (边缘, 纹理等)
│  + BN + ReLU  │
└───────┬───────┘
        ▼
┌───────────────┐
│  池化层 2×2   │ → 降低分辨率, 扩大感受野
└───────┬───────┘
        ▼
    ... 重复多次 ...
        ▼
┌───────────────┐
│  全连接层     │ → 分类/回归
└───────────────┘
```

**关键概念**：
- **卷积核 (Kernel)**: 一个小矩阵 (如 3×3)，在图像上滑动提取特征
- **特征图 (Feature Map)**: 卷积核作用后的输出
- **感受野 (Receptive Field)**: 一个输出像素"能看到"的输入区域大小
- **BatchNorm**: 归一化中间层输出，稳定训练
- **ReLU**: 非线性激活函数 `f(x) = max(0, x)`

### 1.3 Tensor (张量)

PyTorch 中一切数据都是 Tensor：

```python
# 标量: shape = ()
a = torch.tensor(3.14)

# 向量: shape = (3,)
b = torch.tensor([1, 2, 3])

# 矩阵: shape = (2, 3)
c = torch.randn(2, 3)

# 图像批次: shape = (batch, channels, height, width) = (B, C, H, W)
images = torch.randn(8, 1, 256, 256)   # 8张灰度256x256图像
```

**医学图像中的 Tensor 约定**:
- `(B, 1, H, W)` — 2D 灰度图像批次
- `(B, 1, D, H, W)` — 3D 体数据批次
- `(B, C, H, W)` — 模型输出 (C = 类别数)

### 1.4 训练循环模板

```python
for epoch in range(num_epochs):
    for images, labels in dataloader:
        # 1. 前向传播
        predictions = model(images)
        
        # 2. 计算损失
        loss = loss_fn(predictions, labels)
        
        # 3. 反向传播
        optimizer.zero_grad()   # 清零上一步的梯度
        loss.backward()         # 计算梯度
        optimizer.step()        # 更新参数
```

**为什么需要 zero_grad()？**
PyTorch 默认累加梯度。如果不清零，梯度会越来越大，训练就崩了。

---

## 第二章 医学图像分割

### 2.1 什么是图像分割？

给每个像素分配一个类别标签：

```
输入图像 (灰度)              分割输出
┌──────────────┐            ┌──────────────┐
│  ░░░░░░░░░░  │            │  00000000000  │
│  ░░░████░░░  │    ──→     │  00011110000  │  0=背景
│  ░░████░░░░  │            │  00111100000  │  1=器官
│  ░░░░░░░░░░  │            │  00000000000  │
└──────────────┘            └──────────────┘
```

**医学影像中的分割任务**：
- 器官分割: 肝脏、肾脏、心脏、肺
- 病变分割: 肿瘤、结节、出血区域
- 结构分割: 脑区、椎体、血管

### 2.2 U-Net 架构

U-Net 是医学图像分割的里程碑网络 (2015, Freiburg)：

```
编码器 (下采样)              解码器 (上采样)
────────────               ────────────
                             
(1,256,256)                  (2,256,256)  ← 输出
     │                           ▲
  DoubleConv                  DoubleConv
  64通道 ─────── skip ───────→ 64通道
     │                           ▲
  MaxPool                    UpConv
     │                           │
  DoubleConv                  DoubleConv
  128通道 ────── skip ──────→ 128通道
     │                           ▲
  MaxPool                    UpConv
     │                           │
  DoubleConv                  DoubleConv
  256通道 ────── skip ──────→ 256通道
     │                           ▲
  MaxPool                    UpConv
     │                           │
  DoubleConv                  DoubleConv
  512通道 ────── skip ──────→ 512通道
     │                           ▲
  MaxPool                    UpConv
     │                           │
     └──→ Bottleneck 1024 ──────┘
```

**三个关键创新**：

#### ① 编码器-解码器结构
- **编码器** (左侧下降): 逐步提取越来越抽象的特征
  - 第1层: 边缘、纹理
  - 第2层: 局部形状
  - 第3层: 器官轮廓
  - 第4层: 全局语义
- **解码器** (右侧上升): 逐步恢复空间分辨率

#### ② 跳跃连接 (Skip Connection)
- 编码器的特征直接拼接到解码器
- **为什么重要？** 下采样会丢失细节 (精确边界等)，跳跃连接把高分辨率细节带回来
- 没有 skip → 分割边界模糊
- 有 skip → 分割边界精确

#### ③ DoubleConv (双卷积块)
```python
DoubleConv = Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU
```
两个连续卷积比单个卷积表达力更强，且感受野更大。

### 2.3 代码解析: U-Net 前向传播

```python
def forward(self, x):
    # ======== 编码路径 ========
    skip_connections = []
    for encoder, pool in zip(self.encoders, self.pools):
        x = encoder(x)              # 提取特征
        skip_connections.append(x)   # 保存用于跳跃连接
        x = pool(x)                 # 下采样 (尺寸减半)
    
    # ======== 瓶颈 ========
    x = self.bottleneck(x)           # 最深层, 特征最抽象
    
    # ======== 解码路径 ========
    for upconv, decoder, skip in zip(...):
        x = upconv(x)               # 上采样 (尺寸翻倍)
        x = cat([skip, x], dim=1)   # 拼接跳跃连接
        x = decoder(x)              # 融合特征
    
    # ======== 输出 ========
    return self.final_conv(x)        # 1×1 卷积 → 类别数
```

### 2.4 损失函数

#### Cross-Entropy Loss
```
L_CE = -Σ y_i · log(p_i)
```
- 标准分类损失
- 像素级计算
- 缺点: 类别不平衡时 (背景>>前景)，模型倾向于预测全是背景

#### Dice Loss
```
Dice = 2 × |A ∩ B| / (|A| + |B|)
L_Dice = 1 - Dice
```
- 直接优化重叠度
- 对类别不平衡天然鲁棒
- 缺点: 梯度可能不稳定 (尤其目标很小时)

#### Dice + CE 组合 (推荐)
```
L = 0.5 × L_Dice + 0.5 × L_CE
```
- CE 提供稳定的像素级梯度
- Dice 关注整体重叠质量
- 工程实践中最常用

### 2.5 评估指标

| 指标 | 公式 | 范围 | 用途 |
|------|------|------|------|
| **Dice** | 2\|A∩B\| / (\|A\|+\|B\|) | [0,1] | 整体重叠度 |
| **IoU** | \|A∩B\| / \|A∪B\| | [0,1] | 更严格的重叠度 |
| **HD95** | 95分位边界距离 | [0,∞) | 边界质量 |
| **Sensitivity** | TP / (TP+FN) | [0,1] | 能找到多少真阳性 |
| **Specificity** | TN / (TN+FP) | [0,1] | 能排除多少真阴性 |

**Dice vs IoU 的关系**: `IoU = Dice / (2 - Dice)`  
IoU 总是 ≤ Dice，更加严格。

---

## 第三章 医学图像配准

### 3.1 什么是图像配准？

找到两幅图像之间的空间对应关系，使它们在空间上对齐：

```
Fixed (固定图像)    Moving (运动图像)    Warped (配准结果)
┌────────────┐    ┌────────────┐    ┌────────────┐
│    ○       │    │        ○   │    │    ○       │
│            │    │            │    │            │
│  □         │    │     □      │    │  □         │
└────────────┘    └────────────┘    └────────────┘
                        │                  ▲
                        │  形变场 φ(x,y)   │
                        └──────────────────┘
                     Moving ∘ φ ≈ Fixed
```

**应用场景**：
- 纵向研究: 同一患者不同时间点的扫描对齐
- 多模态融合: CT 和 MRI 对齐
- 图谱配准: 将个体脑图像配准到标准模板
- 手术导航: 术前术中图像对齐

### 3.2 配准分类

```
配准方法
├── 传统方法
│   ├── 刚性配准: 平移 + 旋转 (6 参数)
│   ├── 仿射配准: + 缩放 + 剪切 (12 参数)
│   └── 可变形配准: 每个像素独立位移 (N×N×2 参数)
│       ├── Demons
│       ├── B-spline
│       └── SyN (ANTs)
│
└── 深度学习方法
    ├── VoxelMorph (2018) ← 我们实现的
    ├── TransMorph (2022, Transformer-based)
    └── ...
```

### 3.3 VoxelMorph 原理

**核心思想**: 用神经网络一次前向传播预测形变场，而不是迭代优化。

```
                    ┌─────────┐
Fixed (B,1,H,W) ───┤         │
                    │ Concat  ├──► Encoder ──► Decoder ──► Flow (B,2,H,W)
Moving (B,1,H,W) ──┤         │                                │
                    └─────────┘                                │
                                                               ▼
                              Moving ──── Spatial Transformer ──── Warped
```

**训练方式: 自监督！**
- 不需要人工标注形变场
- 损失 = 相似性(Fixed, Warped) + λ × 平滑性(Flow)
- 网络学会: "什么样的变形能让两幅图对齐？"

**与传统方法的对比**：

| | 传统 (SyN等) | VoxelMorph |
|---|---|---|
| 推理速度 | 分钟级 (迭代优化) | 毫秒级 (单次前向) |
| 训练需要 | 不需要 | 需要大量图像对 |
| 精度 | 通常较高 | 接近传统方法 |
| 灵活性 | 高 | 需要固定输入尺寸 |

### 3.4 空间变换器 (Spatial Transformer)

配准的关键组件 —— 可微分的图像变形操作：

```python
# 流程:
# 1. 规则网格 (恒等变换)
grid[0, y, x] = y    # y 坐标
grid[1, y, x] = x    # x 坐标

# 2. 加上形变场 → 采样坐标
sample_coords = grid + flow

# 3. 双线性插值采样
output[y, x] = bilinear_sample(input, sample_coords[y, x])
```

**为什么可微分？**

双线性插值公式 (对4个邻近像素加权平均):
```
output = (1-α)(1-β)·Q11 + α(1-β)·Q21 + (1-α)β·Q12 + αβ·Q22
```
其中 α, β 是子像素偏移量，是坐标的连续函数 → 梯度可以传递到形变场 → 可以端到端训练。

### 3.5 配准损失函数

#### 归一化互相关 (NCC)
```
NCC = Σ(I-μ_I)(J-μ_J) / √[Σ(I-μ_I)² · Σ(J-μ_J)²]
```
- 衡量两幅图的结构相似性
- 对亮度/对比度线性变化不变 → 适合多模态
- 使用局部窗口计算 (通过卷积高效实现)

#### 梯度平滑正则化
```
L_smooth = Σ ‖∇φ‖² = Σ (∂φ/∂x)² + (∂φ/∂y)²
```
- 惩罚形变场的梯度 → 保证变形平滑
- 没有正则化 → 可能出现形变场"折叠" (det(J) < 0)
- 折叠 = 变形把空间"翻转"了，物理上不合理

#### 超参数 λ (smooth_weight)
```
L_total = L_similarity + λ × L_smooth
```
- λ 太小: 变形过度，可能折叠
- λ 太大: 变形不足，配准不准
- 典型值: 0.001 ~ 0.1，需要调参

### 3.6 评估配准质量

| 指标 | 含义 | 理想值 |
|------|------|--------|
| NCC / SSIM | 配准后图像相似度 | 越高越好 |
| Dice (标签) | 配准后标签重叠度 | 越高越好 |
| Jacobian det < 0 比例 | 形变场折叠率 | 0% |
| TRE | 标志点配准误差 | 越低越好 |

---

## 第四章 工程代码导航

### 4.1 项目结构

```
dl_medical/
├── README_深度学习医学影像入门.md     ← 你正在读的文档
├── requirements.txt                   ← 依赖列表
│
├── configs/                           ← 配置文件
│   ├── seg_config.yaml                ← 分割训练配置
│   └── reg_config.yaml                ← 配准训练配置
│
├── models/                            ← 模型定义 ★核心★
│   ├── unet2d.py                      ← 2D U-Net (详细注释)
│   ├── unet3d.py                      ← 3D U-Net (体数据)
│   ├── voxelmorph.py                  ← VoxelMorph + 空间变换器
│   └── losses.py                      ← 损失函数集合
│
├── data/                              ← 数据处理
│   ├── synthetic.py                   ← 合成数据生成
│   ├── dataset.py                     ← PyTorch Dataset
│   └── transforms.py                  ← 数据增强
│
├── utils/                             ← 工具函数
│   ├── metrics.py                     ← 评估指标
│   └── visualization.py               ← 可视化
│
├── train_segmentation.py              ← 分割训练脚本
├── train_registration.py              ← 配准训练脚本
└── demo.py                            ← 快速演示 (推荐先运行)
```

### 4.2 建议阅读顺序

```
第 1 步: 运行 demo.py          → 直观感受
第 2 步: 读 models/unet2d.py   → 理解 U-Net 结构
第 3 步: 读 models/losses.py   → 理解损失函数
第 4 步: 读 data/synthetic.py  → 理解数据生成
第 5 步: 读 train_segmentation.py → 理解完整训练流程
第 6 步: 读 models/voxelmorph.py  → 理解配准网络
第 7 步: 修改参数, 自己实验!
```

### 4.3 关键文件详解

#### `models/unet2d.py` — 从这里开始

每个模块都有详细中文注释：
- `DoubleConv`: 基础构建块
- `UNet2D.__init__`: 看编码器、解码器如何构建
- `UNet2D.forward`: 看数据如何流过网络

**动手实验**:
```python
from models.unet2d import UNet2D
import torch

model = UNet2D(in_channels=1, out_channels=2, features=[16, 32, 64])
x = torch.randn(1, 1, 128, 128)
y = model(x)
print(f"输入: {x.shape}")    # (1, 1, 128, 128)
print(f"输出: {y.shape}")    # (1, 2, 128, 128) → 2个类别的概率
```

#### `models/voxelmorph.py` — 配准核心

重点关注：
- `SpatialTransformer`: 可微分变形操作 (配准的灵魂)
- `VoxelMorph2D.forward`: 如何从图像对到形变场

---

## 第五章 实战训练指南

### 5.1 分割训练

```bash
# 使用默认配置
python train_segmentation.py

# 自定义参数
python train_segmentation.py --epochs 30 --batch_size 4 --lr 0.001

# 指定 GPU
python train_segmentation.py --device cuda:0
```

**输出结果**:
```
checkpoints/seg/
├── best_model.pth           ← 最佳模型权重
└── training_history.npz     ← 训练曲线数据
```

### 5.2 配准训练

```bash
python train_registration.py

# 调整正则化权重
# 在 configs/reg_config.yaml 中修改 smooth_weight
```

### 5.3 常见问题与调参

#### Q: Loss 不下降？
- 降低学习率 (0.001 → 0.0001)
- 检查数据是否正确 (可视化几个样本)
- 增加网络容量 (增大 features)

#### Q: 训练 Loss 低但验证 Loss 高？ (过拟合)
- 增加数据增强
- 添加 Dropout / 增大 weight_decay
- 减小网络容量
- 增加训练数据量

#### Q: 分割边界不精确？
- 使用更大的 features (如 [64,128,256,512])
- 添加深监督 (Deep Supervision)
- 后处理: CRF 或形态学操作

#### Q: 配准出现折叠？
- 增大 smooth_weight (0.01 → 0.1)
- 使用 BendingEnergy 替代 Gradient 正则化
- 检查数据变形是否太大

#### Q: 显存不足 (OOM)？
- 减小 batch_size
- 减小 image_size
- 减小 features
- 使用混合精度训练 (AMP)

### 5.4 混合精度训练 (节省显存)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, masks in dataloader:
    with autocast():                    # 自动选择 fp16/fp32
        outputs = model(images)
        loss = criterion(outputs, masks)
    
    optimizer.zero_grad()
    scaler.scale(loss).backward()       # 缩放梯度防止下溢
    scaler.step(optimizer)
    scaler.update()
```

---

## 第六章 进阶路线

### 6.1 分割方向

```
入门                    进阶                      前沿
─────                  ─────                    ─────
U-Net (2D)      →     nnU-Net (自动配置)    →   SAM (通用分割)
                       Attention U-Net          Swin-UNetr
                       V-Net (3D)               nnFormer
                       TransUNet                Universal Model
```

**推荐下一步**: 
1. **nnU-Net** — 自动搜索最佳 U-Net 配置，MICCAI 竞赛常胜将军
2. **MONAI** — NVIDIA 开源的医学影像深度学习框架，集成了大量模型

### 6.2 配准方向

```
入门                    进阶                      前沿
─────                  ─────                    ─────
VoxelMorph (2D)  →    VoxelMorph (3D)      →   TransMorph
                       Diffeomorphic VM          KeyMorph
                       LapIRN (多尺度)           Foundation Models
                       Inverse Consistency
```

### 6.3 数据集推荐

| 数据集 | 类型 | 任务 |
|--------|------|------|
| **ACDC** | 心脏 MRI | 心脏分割 (入门友好) |
| **LiTS** | 腹部 CT | 肝脏+肿瘤分割 |
| **BraTS** | 脑部 MRI | 脑胶质瘤分割 |
| **OASIS** | 脑部 MRI | 脑配准 |
| **NLST** | 胸部 CT | 肺配准 |
| **Learn2Reg** | 多种 | 配准挑战赛 |

### 6.4 真实数据适配

当你从合成数据切换到真实数据时：

```python
# 替换 Dataset 类中的数据加载方式
class RealSegDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # 读取 NIfTI 文件列表
        self.image_paths = sorted(Path(data_dir).glob("images/*.nii.gz"))
        self.label_paths = sorted(Path(data_dir).glob("labels/*.nii.gz"))
        self.transform = transform
    
    def __getitem__(self, idx):
        import nibabel as nib
        # 读取 NIfTI
        image = nib.load(str(self.image_paths[idx])).get_fdata()
        label = nib.load(str(self.label_paths[idx])).get_fdata()
        
        # 预处理
        image = self.normalize(image)  # 归一化到 [0, 1]
        
        # 数据增强 + 转 Tensor
        if self.transform:
            image, label = self.transform(image, label)
        
        return torch.from_numpy(image).unsqueeze(0).float(), \
               torch.from_numpy(label).long()
```

---

## 第七章 血管分割实战

> 使用你在 `E:\traindata` 中的数据，训练 3D U-Net，并在 OpenGL 渲染器中显示结果。

### 7.1 数据目录结构

你当前的数据结构（直接支持，无需修改）：

```
E:\traindata\
├── images\
│   ├── case001\      ← DICOM 序列目录（含 .dcm / .ima 文件）
│   │   ├── 0001.dcm
│   │   └── ...
│   └── case002\
└── labels\
    ├── case001.nii.gz   ← 二值掩码（0=背景，1=血管）
    └── case002.nii.gz
```

若 `images\` 里直接是 NIfTI 文件也支持：

```
images\case001.nii.gz  →  labels\case001.nii.gz
```

### 7.2 完整流程（三步）

#### 步骤 0：安装额外依赖

```bash
pip install SimpleITK scipy
```

#### 步骤 1：验证数据识别

```bash
cd f:\渲染\OpenglRender\dl_medical
python demo_vessel_pipeline.py --mode scan --data_root E:\traindata
```

输出示例：
```
共找到 10 个 case：
  [ 1]  case001   dicom   case001.nii.gz
  [ 2]  case002   dicom   case002.nii.gz
  ...
✓ 数据识别成功！
```

#### 步骤 2：训练模型

```bash
# 标准配置（需要 ≥8GB 显存）
python train_vessel_seg.py --data_root E:\traindata

# 显存不足时（4GB 也能跑）
python train_vessel_seg.py --data_root E:\traindata ^
    --features 16 32 64 ^
    --patch_size 32 64 64 ^
    --batch_size 1

# 训练 100 轮，完成后模型保存到：
#   checkpoints/vessel_seg/best_model.pth
```

训练过程输出：
```
[  1/100] train loss=0.8523 dice=0.1234 | val loss=0.8102 dice=0.1567 | lr=1.0e-04
[  2/100] train loss=0.7841 dice=0.2341 | val loss=0.7503 dice=0.2789
  → 保存最佳模型 val_dice=0.2789
...
[100/100] train loss=0.2103 dice=0.7851 | val loss=0.2341 dice=0.7623
训练完成！最佳验证 Dice = 0.7851
```

#### 步骤 3：对新 CT 序列推理并在渲染器中显示

```bash
# 推理（生成 .raw 掩码文件）
python infer_vessel_seg.py --dicom_dir "E:\Data\CT-Oncology\LIU^AI\CT_WB_1"

# 输出文件：
#   outputs/vessel_result/vessel_mask.raw        ← 供 C++ 使用
#   outputs/vessel_result/vessel_mask_meta.json  ← 维度信息
#   outputs/vessel_result/vessel_overlay.png     ← 三平面预览

# 启动 C++ 渲染器（命令行第 2 参数传入掩码）
x64\Debug\VolumeRenderOptimized\VolumeRenderOptimized.exe "F:\渲染\OpenglRender\dl_medical\outputs\vessel_result\vessel_mask.raw"
```

渲染器启动后，血管自动以**红色**叠加显示，按键控制：

| 按键 | 功能 |
|------|------|
| **F1** | 切换血管叠加显示/隐藏 |
| **F5** | 切换血管边界高亮线 |
| 鼠标右键 | 区域生长（对比参考） |

### 7.3 新增文件一览

| 文件 | 功能 |
|------|------|
| `data/vessel_dataset.py` | 自动扫描 images/+labels/ 目录，3D patch 数据集 |
| `train_vessel_seg.py` | 3D U-Net 训练脚本（AMP + CosineAnnealing） |
| `infer_vessel_seg.py` | 滑动窗口推理 → 导出 `.raw` + 可视化 |
| `demo_vessel_pipeline.py` | 一键式流水线（scan/train/infer/all） |

### 7.4 常见问题

**Q: 训练 Dice 始终在 0.1 左右？**
- 血管体素占比极低（约 1~3%），前几个 epoch Dice 很低是正常现象
- 检查标注文件：`label.max()` 应为 1，不是 255
- 可视化几个切片确认标注正确：`python demo_vessel_pipeline.py --mode scan` 中查看 vessel_ratio

**Q: 推理结果血管断裂/不连续？**
- 增大 `--overlap`（从 0.5 改为 0.75）
- 降低二值化阈值：`--threshold 0.3`
- 训练时增大 `--samples_per_volume` 和 `--epochs`

**Q: 显存 OOM (out of memory)？**
```bash
python train_vessel_seg.py --features 16 32 64 --patch_size 32 64 64 --batch_size 1
```

**Q: C++ 渲染器里血管颜色/透明度不合适？**

在 [src/main_optimized_example.cpp](../src/main_optimized_example.cpp) 的 DL 掩码加载处修改：
```cpp
renderState.segColor   = glm::vec3(1.0f, 0.15f, 0.1f); // 红色（默认）
renderState.segOpacity = 0.85f;                          // 透明度
```

---

## 附录 推荐资源

### 论文 (必读)

1. **U-Net** (2015): "U-Net: Convolutional Networks for Biomedical Image Segmentation"
   - 奠基之作，必读

2. **V-Net** (2016): "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"
   - 3D 分割 + Dice Loss

3. **VoxelMorph** (2019): "VoxelMorph: A Learning Framework for Deformable Medical Image Registration"
   - 深度学习配准开山之作

4. **nnU-Net** (2021): "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation"
   - 工程实践的典范

5. **TransMorph** (2022): "TransMorph: Transformer for unsupervised medical image registration"
   - Transformer + 配准

### 开源框架

| 框架 | 说明 |
|------|------|
| **MONAI** | NVIDIA 官方医学影像 DL 框架 |
| **nnU-Net** | 自动化分割框架 |
| **VoxelMorph** | 官方配准代码库 |
| **TorchIO** | 医学图像数据增强 |
| **ANTsPy** | 传统配准 (SyN) Python 接口 |

### 学习路径建议

```
第 1 周: 运行本项目代码, 理解基本概念
         ↓
第 2 周: 阅读 U-Net 和 VoxelMorph 论文
         ↓
第 3 周: 用 ACDC 数据集训练一个真实的心脏分割模型
         ↓
第 4 周: 学习 MONAI 框架, 用 OASIS 做脑配准
         ↓
第 5+ 周: 尝试 nnU-Net, 参加 MICCAI Challenge
```

---

*本文档配合 `dl_medical/` 工程代码使用。所有代码均有详细中文注释，建议边读文档边看代码。*
