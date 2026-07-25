# CT-PET 刚体配准算法详细文档（基于源码重写）

> **文档定位**：本文档基于对 `algo-registrationgpu` 和 `algo-registration` 源码的逐行阅读，准确记录 CT-PET 刚体配准的完整实现链路，重点覆盖 GPU CUDA 内核实现细节。所有代码片段均来自实际源文件，标注了文件路径和行号。

---

## 目录

1. [系统架构总览](#1-系统架构总览)
2. [核心数据结构](#2-核心数据结构)
3. [GPU 配准入口：ImageRegistrationRigidG](#3-gpu-配准入口imageregistrationrigidg)
4. [变换模型：TransformModelG](#4-变换模型transformmodelg)
5. [NMI 度量：MutualInformationMetricG](#5-nmi-度量mutualinformationmetricg)
6. [CUDA 内核详解](#6-cuda-内核详解)
7. [优化器：OptimizationG](#7-优化器optimizationg)
8. [重采样：ResampleG](#8-重采样resampleg)
9. [完整数据流与 CPU-GPU 交互](#9-完整数据流与-cpu-gpu-交互)
10. [CT-PET 不同尺寸图像的处理机制](#10-ct-pet-不同尺寸图像的处理机制)
11. [关键代码索引](#11-关键代码索引)

---

## 1. 系统架构总览

### 1.1 四层架构

```
┌─────────────────────────────────────────────────────────────────┐
│  应用层 (algo-fusion-app)                                        │
│  FusionRegistrationCT_PET                                        │
│  ├── AutoRegistration()          ← 自动配准入口                  │
│  ├── RegistrationWithPriorMatrix() ← 带先验矩阵配准              │
│  └── RegistrationTool()          ← 核心配准工具（虚函数）         │
│  职责: CT去床板+HU转换, PET百分位窗位, 参数配置, 先验矩阵         │
├─────────────────────────────────────────────────────────────────┤
│  工具层 (McsfAlgoRegFusionCommon)                                │
│  AlgoFusion::DoRigidReg()       ← 刚体配准统一封装               │
│  职责: 4×4矩阵→Versor+Offset, 采样步长计算, CPU/GPU路径选择      │
├─────────────────────────────────────────────────────────────────┤
│  算法层 (algo-registrationgpu)                                   │
│  ImageRegistrationRigidG()       ← GPU刚体配准主函数              │
│  ├── ImageDataG<short>          ← GPU图像数据封装                 │
│  ├── TransformModelG (Rigid)    ← Versor刚体变换模型              │
│  ├── MutualInformationMetricG   ← GPU NMI度量                     │
│  └── OptimizationG              ← GPU优化器                       │
├─────────────────────────────────────────────────────────────────┤
│  CUDA 内核层 (algo-registrationgpu/src/.../*.cu)                 │
│  ├── TransormPointKer           ← 坐标变换+纹理插值               │
│  ├── ComputeSdJHTG_Ker          ← 联合直方图(共享内存+原子操作)   │
│  ├── ComputeRefPDF_Ker          ← 参考图边缘PDF                   │
│  ├── ComputeMovPDFG_Ker         ← 浮动图边缘PDF                   │
│  ├── ComputeJHTTableAndMI_Ker   ← MI值计算                        │
│  ├── ComputeGradient_Ker        ← 梯度计算                        │
│  ├── DerivateRigid_multiMem_Ker ← Versor刚体导数                  │
│  └── Resample16G_Ker            ← 重采样                          │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 核心设计理念

- **多模态配准**：CT（密度 HU 值）和 PET（代谢活性 SUV 值）灰度无线性对应关系，必须使用 **NMI（归一化互信息）** 而非 MSE
- **GPU 优先**：GPU 路径为默认路径，利用 CUDA 纹理内存加速三线性插值，利用共享内存+原子操作加速直方图构建
- **CPU-GPU 混合执行**：采样在 CPU 完成（因 mask 筛选逻辑复杂），NMI 计算和梯度计算在 GPU 完成，Versor 参数更新在 CPU 完成（因四元数乘法逻辑简单）
- **不同尺寸原生支持**：CT 和 PET 保持各自原始尺寸/间距，通过坐标空间映射处理，仅在配准完成后重采样

---

## 2. 核心数据结构

### 2.1 `ImageDataG<T>` —— GPU 图像数据封装

> 文件: `algo-registrationgpu/src/McsfAlgoRegistrationG/include/ImageDataG.h`

```cpp
template<typename T>
class ImageDataG
{
public:
    ImageDataG(const T* pSrc_h, const unsigned pImgSize_h[3], const TFLOAT pImgSpacing_h[3], 
               bool bOwnpDataC = false);

    const T* const_dataG()const;         // GPU 设备指针
    const T* const_dataC()const;         // CPU 主机指针
    const unsigned* const_sizeG()const;  // GPU 上的尺寸数组
    const unsigned* const_sizeC()const;  // CPU 上的尺寸数组
    const TFLOAT* const_spacingG()const; // GPU 上的间距数组
    const TFLOAT* const_spacingC()const; // CPU 上的间距数组
    unsigned PixNum()const;              // 像素总数
    cudaArray* GetCudaArray();           // CUDA 3D 数组（用于纹理绑定）

private:
    unsigned m_pixNum;
    std::shared_ptr<GpuGlobalData<T>> m_pData;       // GPU 设备数据
    std::shared_ptr<GpuGlobalData<unsigned>> m_pSize; // GPU 尺寸
    std::shared_ptr<GpuGlobalData<TFLOAT>> m_pSpacing;// GPU 间距
    cudaArray* m_pArray;                              // CUDA 3D 纹理数组
    bool m_bInArray;
    const T* m_pDataC;                                // CPU 数据指针
    const bool m_ownpDataC;                           // 是否拥有 CPU 数据所有权
    const unsigned* m_pSizeC;                         // CPU 尺寸
    const TFLOAT* m_pSpacingC;                        // CPU 间距
};
```

**关键点**：`ImageDataG` 同时维护 CPU 和 GPU 两份数据指针。构造时从 host 拷贝数据到 device，并创建 `cudaArray` 用于 3D 纹理绑定。

### 2.2 `GpuGlobalData<T>` —— GPU 全局数据容器

> 文件: `algo-registrationgpu/src/McsfAlgoRegistrationG/include/ImageDataG.h`

```cpp
template<typename T>
class GpuGlobalData
{
public:
    GpuGlobalData(unsigned size);                    // 分配 GPU 内存
    GpuGlobalData(const T* pSrc_h, unsigned size);   // 从 host 拷贝到 device
    ~GpuGlobalData();

    void ToCpu(T* pData_h)const;                     // device → host 拷贝
    void SetDataCG(const T* pSrc_h);                 // host → device 拷贝（支持 UVA）
    void ResetNum(const T& num);                     // GPU 上 memset 为指定值
    T* mutable_data();                               // 可写 device 指针
    const T* const_data()const;                      // 只读 device 指针
    unsigned size();                                 // 元素个数

private:
    T* m_data;      // GPU 设备指针
    unsigned m_size; // 元素个数
};
```

**关键点**：这是所有 GPU 数据的基础容器。`SetDataCG` 用于 CPU 计算后将结果同步回 GPU（如 Versor 参数更新后）。

### 2.3 `VersorG` —— 四元数结构体

> 文件: `algo-registrationgpu/src/McsfAlgoRegistrationG/include/RigidCudaHelpG.h`

```cpp
struct VersorG
{
    TFLOAT vw;  // cos(θ/2)
    TFLOAT vx;  // sin(θ/2)·ux
    TFLOAT vy;  // sin(θ/2)·uy
    TFLOAT vz;  // sin(θ/2)·uz
    // 预计算的乘积项（加速 GPU 导数计算）
    TFLOAT vxx, vyy, vzz, vww;    // vw², vx², vy², vz²
    TFLOAT vxy, vxz, vxw;         // vx·vy, vx·vz, vx·vw
    TFLOAT vyz, vyw, vzw;          // vy·vz, vy·vw, vz·vw
};
```

**关键点**：除了四元数的 4 个基本分量，还预计算了 9 个乘积项（`vxx`, `vyy` 等），避免在 CUDA 内核中重复计算乘法，加速 `DerivateRigid_multiMem_Ker` 中的导数计算。

### 2.4 `PARAMETER_t` —— 配准参数

> 文件: `algo-registration/include/algo-registration/McsfAlgoRegistration.h`

CT-PET 配准中的典型参数设置：

| 参数 | 值 | 说明 |
|------|-----|------|
| `m_bAffine` | false | 刚体变换（非仿射） |
| `m_bRotate` | true | 包含旋转 |
| `uiMetricOption` | 1 | NMI（归一化互信息） |
| `NumOfBins` | 50~80 | 直方图 bin 数 |
| `MaxIterationNum` | 200 | 最大迭代次数 |
| `MaxIterationStep` | 3 | 初始步长 |
| `MinIterationStep` | 0.01 | 最小步长（终止条件） |
| `dRelaxationFactor` | 0.9 | 松弛因子（方向反转时缩小步长） |
| `dMagnitudeTolerance` | 0.001 | 梯度模长收敛阈值 |
| `m_dMinEffectiveSampleRatio` | 0.0125 | 最小有效采样比例 (1.25%) |
| `m_Scale` | [1000, 1000, 1000, 0.01, 0.01, 0.01] | 旋转/平移参数缩放 |
| `bMultiResolution` | false | 不使用金字塔多分辨率 |
| `m_bResolutionControl` | true | 使用采样步长控制分辨率 |

---

## 3. GPU 配准入口：ImageRegistrationRigidG

> 文件: `algo-registrationgpu/src/McsfAlgoRegistrationG/src/McsfAlgoRegistrationG.cpp` (L48-L155)

### 3.1 函数签名

```cpp
bool ImageRegistrationRigidG(Mcsf::PARAMETER_t para,
    const short* pReferenceImageData_h,   // CT 数据 (host)
    const short* pMovingImageData_h,      // PET 数据 (host)
    unsigned refWidth, unsigned refHeight, unsigned refSlice,    // CT 尺寸
    unsigned movWidth, unsigned movHeight, unsigned movSlice,    // PET 尺寸（与CT不同！）
    double refSpacingX, double refSpacingY, double refSpacingZ,  // CT 间距
    double movSpacingX, double movSpacingY, double movSpacingZ,  // PET 间距（与CT不同！）
    double* dResultMatrix_h,              // 输出: 3×3 旋转矩阵
    double* dResultOffset_h,              // 输出: 3×1 偏移向量
    double* dResultTranslation_h,         // 输出: 平移（未使用）
    double& dResultMetric,                // 输出: NMI 值
    short* pOutImage_h,                   // 输出: 重采样后的 PET 图像
    IProgress* pProgress)                 // 进度回调
```

### 3.2 执行流程（6 步）

```cpp
bool ImageRegistrationRigidG(...)
{
    // ── 步骤 1: 准备 GPU 图像数据 ──
    unsigned refSize_h[3] = { refWidth, refHeight, refSlice };
    double refSpacing_h[3] = { refSpacingX, refSpacingY, refSpacingZ };
    unsigned movSize_h[3] = { movWidth, movHeight, movSlice };
    double movSpacing_h[3] = { movSpacingX, movSpacingY, movSpacingZ };
    
    // double → float 类型转换（GPU 使用 TFLOAT = float）
    TFLOAT refSpacing_h1[3] = { 0 };
    TFLOAT movSpacing_h1[3] = { 0 };
    TypeCastC<double, TFLOAT>(refSpacing_h, refSpacing_h1, 3);
    TypeCastC<double, TFLOAT>(movSpacing_h, movSpacing_h1, 3);

    // 创建 GPU 图像对象（同时拷贝数据到 device，创建 cudaArray）
    std::shared_ptr<ImageDataG<short>> pRef(
        new ImageDataG<short>(pReferenceImageData_h, refSize_h, refSpacing_h1));
    std::shared_ptr<ImageDataG<short>> pMov(
        new ImageDataG<short>(pMovingImageData_h, movSize_h, movSpacing_h1));

    // ── 步骤 2: 初始化变换模型 ──
    std::shared_ptr<TransformModelG> pModel;
    if (para.GetIfAffine() && para.GetIfRotate())
        pModel.reset(new Affine());           // 12 自由度仿射
    else if (!para.GetIfAffine() && para.GetIfRotate())
        pModel.reset(new Rigid());            // 6 自由度刚体 ← CT-PET 使用此分支
    else
        return false;
    
    // 用 ref 和 mov 的尺寸/间距初始化变换模型（计算初始偏移=中心对齐）
    pModel->InitFromParam(para, refSize_h, refSpacing_h, movSize_h, movSpacing_h);

    // ── 步骤 3: 初始化相似性度量 ──
    std::shared_ptr<ImageMetricG> pMetric;
    if (1 == para.GetMetricOption())
        pMetric.reset(new MutualInformationMetricG(pRef, pMov, pModel,
            para.GetSampleRegionMask(),         // CT 身体区域掩码
            para.GetSampleIntensityLevel(),     // 采样强度下限 (-1024)
            para.GetSampleIntensityLevelMinMax(),
            static_cast<float>(para.GetMinEffectiveSampleRatio()),
            para.GetNumOfBins(),                // 直方图 bin 数
            para.GetRefMinMax(),                // CT 灰度范围（可选）
            para.GetMovMinMax()));              // PET 灰度范围（可选）
    else
        pMetric.reset(new CMeanSquareErrorMetricG(pRef, pMov, pModel, ...));

    // ── 步骤 4: 初始化优化器 ──
    std::shared_ptr<OptimizationG> pOptimizer(
        new OptimizationG(pMetric,
            para.GetMaxIterationNum(),      // 200
            para.GetMagnitudeTolerance(),   // 0.001
            para.GetMaxIterationStep(),     // 3
            para.GetMinIterationStep(),     // 0.01
            para.GetRelaxationFactor(),     // 0.9
            para.GetScale()));              // [1000,1000,1000,0.01,0.01,0.01]

    // ── 步骤 5: 执行配准 ──
    int retValue = 0;
    if (para.GetMultiResolution())
    {
        // 金字塔多分辨率：根据图像尺寸选择降采样级数
        unsigned int mintemp = std::min(refSize_h[0], refSize_h[1]);
        if (mintemp >= 512)
            retValue = pOptimizer->ResumeOptimizationPyramid(2);  // 降采样 2 级
        else if (mintemp >= 128)
            retValue = pOptimizer->ResumeOptimizationPyramid(1);  // 降采样 1 级
    }
    else if (para.GetResolutionControl())
    {
        // 采样步长控制分辨率（CT-PET 使用此路径）
        unsigned pSampleRate[3] = { 0 };
        para.GetSampleRate(pSampleRate);
        retValue = pOptimizer->ResumeOptimization(pSampleRate);
    }
    if (retValue == -1) return false;

    // ── 步骤 6: 提取结果并重采样 ──
    TFLOAT pRotate[9] = { 0 };  // 3×3 旋转矩阵
    TFLOAT pOffset[3] = { 0 };  // 3×1 偏移
    pModel->GetCurState(pRotate, pOffset);
    dResultMetric = pOptimizer->GetMetricInformation();
    
    // float → double 类型转换输出
    TypeCastC<float, double>(pRotate, dResultMatrix_h, 9);
    TypeCastC<float, double>(pOffset, dResultOffset_h, 3);
    
    // 重采样 PET 到 CT 网格（仅在 pOutImage_h 非空时）
    if (NULL != pOutImage_h)
    {
        ResampleG(refSize_h, refSpacing_h1, pMov, pRotate, pOffset, 
                  pOutImage_h, false, LINEAR);
    }
    return true;
}
```

---

## 4. 变换模型：TransformModelG

> 文件: `algo-registrationgpu/src/McsfAlgoRegistrationG/include/TransformModelG.h`
> 文件: `algo-registrationgpu/src/McsfAlgoRegistrationG/src/TransformModelG.cpp`

### 4.1 类继承体系

```cpp
class TransformModelG          // 抽象基类
{
protected:
    unsigned m_transformDim;
    std::shared_ptr<GpuGlobalData<TFLOAT>> m_pRotate;    // GPU 上的 3×3 旋转矩阵
    std::shared_ptr<GpuGlobalData<TFLOAT>> m_pOffset;    // GPU 上的 3×1 偏移
    std::shared_ptr<GpuGlobalData<TFLOAT>> m_pParameter; // GPU 上的参数向量
};

class Affine : public TransformModelG  // 12 自由度: 9 旋转 + 3 平移
{
    // m_transformDim = 12
};

class Rigid : public TransformModelG   // 6 自由度: 3 Versor + 3 平移
{
    // m_transformDim = 6
private:
    VersorG m_versor;                                    // 四元数结构体
    std::shared_ptr<GpuGlobalData<TFLOAT>> m_pRotateVec; // GPU 上的 Versor 向量 [qx,qy,qz]
};
```

### 4.2 Rigid::InitFromParam —— 初始化

```cpp
void Rigid::InitFromParam(const Mcsf::PARAMETER_t& param,
    const unsigned refSize_h[3], const double refSpacing_h[3],
    const unsigned movSize_h[3], const double movSpacing_h[3])
{
    double dRotateVec[3] = { 0 };  // Versor 旋转向量
    double dOffsetVec[3] = { 0 };  // 偏移向量

    // ── 旋转初始化 ──
    if (param.GetIfUsePriorMatrix() && param.GetPriorMatrix() != NULL)
    {
        // 使用先验矩阵的旋转部分（3 个 Versor 分量）
        memcpy(dRotateVec, param.GetPriorMatrix(), sizeof(double) * 3);
    }
    // 否则 dRotateVec 保持 [0,0,0]（无旋转）

    // ── 偏移初始化：中心对齐 ──
    if (param.GetIfUseImagePosition())
    {
        param.GetImagePosition(dOffsetVec);  // 使用指定的图像位置
    }
    else
    {
        // 默认策略：将 PET 中心对齐到 CT 中心
        // offset = movCenter - refCenter
        dOffsetVec[0] = (movSize_h[0]-1)*movSpacing_h[0]/2.0 - (refSize_h[0]-1)*refSpacing_h[0]/2.0;
        dOffsetVec[1] = (movSize_h[1]-1)*movSpacing_h[1]/2.0 - (refSize_h[1]-1)*refSpacing_h[1]/2.0;
        dOffsetVec[2] = (movSize_h[2]-1)*movSpacing_h[2]/2.0 - (refSize_h[2]-1)*refSpacing_h[2]/2.0;
    }

    // double → float 并上传到 GPU
    TFLOAT rotateVec[3], offsetVec[3];
    TypeCastC<double, TFLOAT>(dRotateVec, rotateVec, 3);
    TypeCastC<double, TFLOAT>(dOffsetVec, offsetVec, 3);

    m_pRotate.reset(new GpuGlobalData<TFLOAT>(9));           // 3×3 矩阵（待 SyncVersorMat 填充）
    m_pRotateVec.reset(new GpuGlobalData<TFLOAT>(rotateVec, 3)); // Versor 向量
    m_pOffset.reset(new GpuGlobalData<TFLOAT>(offsetVec, 3));

    // 参数向量 = [qx, qy, qz, tx, ty, tz]（6 维）
    TFLOAT parameter[6] = { 0 };
    memcpy(parameter, rotateVec, 3 * sizeof(TFLOAT));
    memcpy(parameter + 3, offsetVec, 3 * sizeof(TFLOAT));
    m_pParameter.reset(new GpuGlobalData<TFLOAT>(parameter, 6));

    // 从 Versor 向量同步计算 3×3 旋转矩阵和预计算乘积项
    SyncVersorMat(parameter);
}
```

### 4.3 Rigid::SyncVersorMat —— Versor → 旋转矩阵

> 文件: `TransformModelG.cpp` (L283-L339)

```cpp
void Rigid::SyncVersorMat(const TFLOAT* pPara_h)
{
    // 更新偏移
    m_pOffset->SetDataCG(pPara_h + 3);

    // 提取 Versor 向量 [qx, qy, qz]
    TFLOAT axis[3] = { pPara_h[0], pPara_h[1], pPara_h[2] };
    
    // 归一化检查（Versor 模长必须 ≤ 1）
    TFLOAT norm = sqrt(axis[0]² + axis[1]² + axis[2]²);
    TFLOAT epsilon = 1e-10f;
    if (norm >= 1.0 - epsilon)
    {
        for (int i = 0; i < 3; i++)
            axis[i] /= (norm + epsilon * norm);
    }

    // qw = cos(θ/2) = √(1 - |q_xyz|²)
    TFLOAT sinangle2 = sqrt(axis[0]² + axis[1]² + axis[2]²);
    TFLOAT cosangle2 = sqrt(1.0f - sinangle2 * sinangle2);

    m_versor.vx = axis[0];  m_versor.vy = axis[1];
    m_versor.vz = axis[2];  m_versor.vw = cosangle2;

    // 四元数 → 3×3 旋转矩阵
    TFLOAT vw = m_versor.vw, vx = m_versor.vx, vy = m_versor.vy, vz = m_versor.vz;
    TFLOAT xx = vx*vx, yy = vy*vy, zz = vz*vz;
    TFLOAT xy = vx*vy, xz = vx*vz, xw = vx*vw, yz = vy*vz, yw = vy*vw, zw = vz*vw;

    TFLOAT matrix[9] = { 0 };
    matrix[0] = 1.0f - 2.0f * (yy + zz);  // 对角元素
    matrix[4] = 1.0f - 2.0f * (xx + zz);
    matrix[8] = 1.0f - 2.0f * (xx + yy);
    matrix[1] = 2.0f * (xy - zw);          // 非对角元素
    matrix[2] = 2.0f * (xz + yw);
    matrix[3] = 2.0f * (xy + zw);
    matrix[5] = 2.0f * (yz - xw);
    matrix[6] = 2.0f * (xz - yw);
    matrix[7] = 2.0f * (yz + xw);
    m_pRotate->SetDataCG(matrix);  // 上传到 GPU

    // 预计算乘积项（加速 GPU 导数内核）
    m_versor.vxx = xx;  m_versor.vyy = yy;  m_versor.vzz = zz;  m_versor.vww = vw*vw;
    m_versor.vxy = xy;  m_versor.vxz = xz;  m_versor.vxw = xw;
    m_versor.vyz = yz;  m_versor.vyw = yw;  m_versor.vzw = zw;
}
```

### 4.4 Rigid::UpdateParameter —— Versor 四元数乘法更新

> 文件: `TransformModelG.cpp` (L191-L227)

**关键：此函数在 CPU 上执行**（注释："简单工作CPU完成即可"），因为四元数乘法只有 3 个标量运算，GPU 启动内核的开销远大于计算本身。

```cpp
void Rigid::UpdateParameter(TFLOAT factor, const TFLOAT* pGradient_h)
{
    TFLOAT paraDst[6], paraTemp[6];
    m_pParameter->ToCpu(paraTemp);  // GPU → CPU 拷贝当前参数

    // ── 平移参数：直接加法更新 ──
    paraDst[3] = factor * pGradient_h[3] + paraTemp[3];
    paraDst[4] = factor * pGradient_h[4] + paraTemp[4];
    paraDst[5] = factor * pGradient_h[5] + paraTemp[5];

    // ── Versor 旋转参数：四元数乘法更新 ──
    TFLOAT norm = sqrt(paraTemp[0]² + paraTemp[1]² + paraTemp[2]²);
    // 当前旋转四元数
    TFLOAT cx = paraTemp[0], cy = paraTemp[1], cz = paraTemp[2];
    TFLOAT cw = sqrt(1 - norm * norm);

    // 梯度构造增量四元数
    TFLOAT normGrad = sqrt(pGradient_h[0]² + pGradient_h[1]² + pGradient_h[2]²);
    TFLOAT angle = factor * normGrad;
    TFLOAT temp = sin(angle / 2.0f) / normGrad;
    TFLOAT gx = pGradient_h[0] * temp, gy = pGradient_h[1] * temp, gz = pGradient_h[2] * temp;
    TFLOAT gw = cos(angle / 2.0f);

    // 四元数乘法: q_new = q_current × q_delta
    paraDst[0] =  cw*gx - cz*gy + cy*gz + cx*gw;
    paraDst[1] =  cz*gx + cw*gy - cx*gz + cy*gw;
    paraDst[2] = -cy*gx + cx*gy + cw*gz + cz*gw;

    // 上传新参数到 GPU，并同步旋转矩阵
    m_pParameter->SetDataCG(paraDst);  // CPU → GPU
    SyncVersorMat(paraDst);            // 重新计算 3×3 矩阵并上传
}
```

### 4.5 TransformPoint —— 坐标变换（委托给 CUDA 内核）

> 文件: `TransformModelG.cpp` (L14-L19)

```cpp
void TransformModelG::TransformPoint(
    const TFLOAT* pSamplePoints_d,     // GPU 上的采样点坐标 [sampleNum×3]
    unsigned sampleNum,
    const TFLOAT* pRefCenter_d,        // CT 图像中心的物理坐标 [3]
    const cudaArray* pMovArray,        // PET 图像的 CUDA 3D 纹理数组
    const unsigned* pMovSize_d,        // PET 图像尺寸 [3]（GPU）
    const TFLOAT* pMovSpacing_d,       // PET 图像间距 [3]（GPU）
    TFLOAT* pTransformedPoints_d,      // 输出: 变换后的 PET 索引坐标
    short* pTransformdMovImg_d,        // 输出: 插值得到的 PET 灰度值
    unsigned* pSampleOK_d)             // 输出: 采样点是否有效 (0/1)
{
    TransformPointG(pSamplePoints_d, sampleNum, pRefCenter_d, pMovArray,
        pMovSize_d, pMovSpacing_d,
        m_pRotate->const_data(),       // 3×3 旋转矩阵（GPU）
        m_pOffset->const_data(),       // 3×1 偏移（GPU）
        pTransformedPoints_d, pTransformdMovImg_d, pSampleOK_d);
}
```

---

## 5. NMI 度量：MutualInformationMetricG

> 文件: `algo-registrationgpu/src/McsfAlgoRegistrationG/include/MutualInformationMetricG.h`
> 文件: `algo-registrationgpu/src/McsfAlgoRegistrationG/src/MutualInformationMetricG.cpp`

### 5.1 类定义

```cpp
class MutualInformationMetricG : public ImageMetricG
{
private:
    struct BinInfo
    {
        short minValue;    // 图像最小灰度值
        short maxValue;    // 图像最大灰度值
        unsigned binNum;   // bin 数量
        TFLOAT binSize;    // bin 宽度 = (max-min)/(binNum-4)
        TFLOAT normMin;    // 归一化最小值 = min/binSize - 2
        void Init(const short* pImg_d, unsigned len, unsigned binNum, const short* pMinMax = NULL);
    };
    BinInfo m_refBin;  // CT 的 bin 信息
    BinInfo m_movBin;  // PET 的 bin 信息

    std::shared_ptr<GpuGlobalData<TFLOAT>> m_pRefPDF;    // CT 边缘概率密度 [refBinNum]
    std::shared_ptr<GpuGlobalData<TFLOAT>> m_pMovPDF;    // PET 边缘概率密度 [movBinNum]
    std::shared_ptr<GpuGlobalData<TFLOAT>> m_pJointPDF;  // 联合概率密度 [refBinNum × movBinNum]
    std::shared_ptr<GpuGlobalData<TFLOAT>> m_pPDFs;      // 每线程私有直方图 [refBinNum × movBinNum × 2560]

    std::shared_ptr<GpuGlobalData<unsigned>> m_pRefSamplesWindowIndex;  // CT 采样点的 Parzen 窗口索引
    std::shared_ptr<GpuGlobalData<unsigned>> m_pMovSamplesWindowIndex;  // PET 采样点的 Parzen 窗口索引
    std::shared_ptr<GpuGlobalData<TFLOAT>> m_pMovSamplesWindowTerm;     // PET 采样点的 Parzen 窗口项

    std::unique_ptr<GpuGlobalData<TFLOAT>> pJHTTable;  // JHT 查找表 [refBinNum × movBinNum]
    std::unique_ptr<GpuGlobalData<TFLOAT>> pJHTDev;    // 每采样点梯度 [transformDim × sampleNum]
};
```

### 5.2 BinInfo::Init —— 直方图 bin 初始化

```cpp
void MutualInformationMetricG::BinInfo::Init(const short* pImg_d, unsigned len, 
                                              unsigned binNumv, const short* pMinMax)
{
    if (NULL != pMinMax)
    {
        minValue = pMinMax[0];
        maxValue = pMinMax[1];
    }
    else
    {
        minValue = MinG<short>(pImg_d, len);  // GPU 并行求最小值
        maxValue = MaxG<short>(pImg_d, len);  // GPU 并行求最大值
    }
    binNum = binNumv;
    // binSize = (max-min)/(binNum-4)，减 4 是为 Parzen 窗口留边界
    binSize = static_cast<TFLOAT>(maxValue - minValue) / (binNum - 2.0f * 2);
    normMin = minValue / binSize - 2;
}
```

### 5.3 SampleFixedImageDomain —— 采样参考图像域

> **关键：采样在 CPU 上执行**，因为 mask 筛选逻辑在 GPU 上实现复杂。

```cpp
bool MutualInformationMetricG::SampleFixedImageDomain(const unsigned step_h[3])
{
    // 调用基类方法（在 CPU 上执行采样）
    bool ret = ImageMetricG::SampleFixedImageDomain(step_h);
    CHECK_RET(ret);

    // 重置直方图
    m_pRefPDF->ResetNum(0);
    m_pMovPDF->ResetNum(0);
    m_pJointPDF->ResetNum(0);

    // 分配 Parzen 窗口索引数组
    m_pRefSamplesWindowIndex.reset(new GpuGlobalData<unsigned>(m_nSampleNum));
    m_pMovSamplesWindowIndex.reset(new GpuGlobalData<unsigned>(m_nSampleNum));
    m_pMovSamplesWindowTerm.reset(new GpuGlobalData<TFLOAT>(m_nSampleNum));

    // 计算 CT 采样点的 Parzen 窗口索引（GPU 并行）
    ParzenWindowIndex(m_pSampleImgData->const_data(), m_nSampleNum,
        m_refBin.binSize, m_refBin.normMin, m_refBin.binNum,
        m_pRefSamplesWindowIndex->mutable_data());

    pJHTTable.reset(new GpuGlobalData<TFLOAT>(m_refBin.binNum * m_movBin.binNum));
    pJHTDev.reset(new GpuGlobalData<TFLOAT>(m_pModel->GetTransformDim() * m_nSampleNum));
    return true;
}
```

### 5.4 基类 ImageMetricG::SampleFixedImageDomain —— CPU 采样实现

> 文件: `algo-registrationgpu/src/McsfAlgoRegistrationG/src/ImageMetricG.cpp` (L46-L202)

```cpp
void SampleFixedImageDomainG1(const ImageDataG<short>* pRefImg, 
    const ImageDataG<unsigned char>* pRefMask,
    GpuGlobalData<TFLOAT>* pRefCenter,
    short sampleIntensityLevel,
    const unsigned step_h[3], 
    unsigned& sampleNum,
    std::shared_ptr<GpuGlobalData<short>>& pSampleImgData,
    std::shared_ptr<GpuGlobalData<TFLOAT>>& pSamplePoints)
{
    // ★ 在 CPU 上执行（源码注释："暂时用CPU实现，后面看看是否需要用GPU实现"）
    const short* pRefImgC = pRefImg->const_dataC();
    const unsigned char* pRefMaskC = pRefMask->const_dataC();
    const unsigned* pRefSizeC = pRefImg->const_sizeC();
    const TFLOAT* pRefSpacingC = pRefImg->const_spacingC();

    // 第一遍：统计有效采样点数量
    sampleNum = 0;
    for (unsigned k = 0; k < pRefSizeC[2]; k += step_h[2])
        for (unsigned j = 0; j < pRefSizeC[1]; j += step_h[1])
            for (unsigned i = 0; i < pRefSizeC[0]; i += step_h[0])
            {
                unsigned curIndex = k * pRefSizeC[0] * pRefSizeC[1] + j * pRefSizeC[0] + i;
                // 筛选条件：灰度值 > 阈值 AND mask 非零
                if (pRefImgC[curIndex] > sampleIntensityLevel && pRefMaskC[curIndex])
                    ++sampleNum;
            }

    // 第二遍：提取采样点的灰度值和物理坐标
    std::unique_ptr<short[]> pSampleImgC(new short[sampleNum]());
    std::unique_ptr<TFLOAT[]> pSamplePointsC(new TFLOAT[sampleNum * 3]());
    // SoA 布局：X/Y/Z 分别连续存储（利于 GPU 合并访问）
    TFLOAT* pSamplePointsCX = pSamplePointsC.get();
    TFLOAT* pSamplePointsCY = pSamplePointsC.get() + sampleNum;
    TFLOAT* pSamplePointsCZ = pSamplePointsC.get() + sampleNum * 2;

    TFLOAT pRefCenterC[3] = { 0 };
    pRefCenter->ToCpu(pRefCenterC);  // GPU → CPU 拷贝参考图中心

    unsigned index = 0;
    for (unsigned k = 0; k < pRefSizeC[2]; k += step_h[2])
        for (unsigned j = 0; j < pRefSizeC[1]; j += step_h[1])
            for (unsigned i = 0; i < pRefSizeC[0]; i += step_h[0])
            {
                unsigned curIndex = k * pRefSizeC[0] * pRefSizeC[1] + j * pRefSizeC[0] + i;
                if (pRefImgC[curIndex] > sampleIntensityLevel && pRefMaskC[curIndex])
                {
                    pSampleImgC[index] = pRefImgC[curIndex];
                    // 像素索引 → 物理坐标，减去参考图中心
                    pSamplePointsCX[index] = i * pRefSpacingC[0] - pRefCenterC[0];
                    pSamplePointsCY[index] = j * pRefSpacingC[1] - pRefCenterC[1];
                    pSamplePointsCZ[index] = k * pRefSpacingC[2] - pRefCenterC[2];
                    ++index;
                }
            }

    // CPU → GPU 拷贝
    pSampleImgData.reset(new GpuGlobalData<short>(pSampleImgC.get(), sampleNum));
    pSamplePoints.reset(new GpuGlobalData<TFLOAT>(pSamplePointsC.get(), sampleNum * 3));
}
```

**关键设计**：
- 采样点坐标使用 **SoA（Structure of Arrays）** 布局：`[all_x, all_y, all_z]`，而非 AoS `[x0,y0,z0, x1,y1,z1,...]`，利于 GPU 合并内存访问
- 采样点坐标是**以 CT 中心为原点的相对物理坐标**（已减去 `pRefCenter`）
- 两遍遍历：第一遍计数，第二遍提取（因为需要先知道数量才能分配数组）

### 5.5 GetValueAndDerivative —— NMI 值和梯度计算（核心）

> 文件: `MutualInformationMetricG.cpp` (L88-L155)

```cpp
TFLOAT MutualInformationMetricG::GetValueAndDerivative(TFLOAT* pGradient_d)
{
    // ══════════════════════════════════════════════════════
    // 步骤 1: 坐标变换 + 三线性插值（CUDA 内核 TransormPointKer）
    // 输入: CT 采样点物理坐标 → 输出: PET 插值灰度值 + 变换后坐标 + 有效性标记
    // ══════════════════════════════════════════════════════
    m_pMovSampleImgData->ResetNum(0);
    m_pModel->TransformPoint(
        m_pSamplePoints->const_data(),      // CT 采样点坐标（GPU）
        m_nSampleNum,
        m_pRefCenter->const_data(),         // CT 中心（GPU）
        m_pMovImg->GetCudaArray(),          // PET 纹理数组
        m_pMovImg->const_sizeG(),           // PET 尺寸（GPU）
        m_pMovImg->const_spacingG(),        // PET 间距（GPU）
        m_pTransformedPoints->mutable_data(), // 输出: PET 索引坐标
        m_pMovSampleImgData->mutable_data(),  // 输出: PET 插值灰度
        m_pSampleOK->mutable_data());         // 输出: 有效性 (0/1)

    // ══════════════════════════════════════════════════════
    // 步骤 2: 统计有效采样点数量（GPU thrust::reduce）
    // ══════════════════════════════════════════════════════
    unsigned effectiveSampleNum = SumG<unsigned>(m_pSampleOK->const_data(), m_nSampleNum);
    if (effectiveSampleNum < m_nSampleNum * m_minEffectiveSampleRatio)
    {
        return 0;  // 有效采样不足，返回 0（避免虚假 MI 值）
    }

    // ══════════════════════════════════════════════════════
    // 步骤 3: 计算 CT 边缘 PDF（CUDA 内核 ComputeRefPDF_Ker）
    // ══════════════════════════════════════════════════════
    m_pRefPDF->ResetNum(0);
    ComputeRefPDF(m_pRefSamplesWindowIndex->const_data(),
        m_pSampleOK->const_data(), m_nSampleNum, m_refBin.binNum,
        m_pRefPDF->mutable_data());

    // ══════════════════════════════════════════════════════
    // 步骤 4: 计算 PET 采样点的 Parzen 窗口索引（CUDA 内核）
    // ══════════════════════════════════════════════════════
    ParzenWindowIndex(m_pMovSampleImgData->const_data(), m_nSampleNum,
        m_movBin.binSize, m_movBin.normMin, m_movBin.binNum,
        m_pMovSamplesWindowIndex->mutable_data(),
        m_pMovSamplesWindowTerm->mutable_data());

    // ══════════════════════════════════════════════════════
    // 步骤 5: 计算 PET 图像在采样点处的梯度（CUDA 内核 ComputeMovMagG）
    // ══════════════════════════════════════════════════════
    m_pDif->ResetNum(0);
    m_pMag->ResetNum(0);
    ComputeMovMag(m_pTransformedPoints->const_data(),
        m_pSampleOK->const_data(), m_nSampleNum,
        m_pMovImg->GetCudaArray(), m_pMovImg->const_sizeG(), m_pMovImg->const_spacingG(),
        m_pMag->mutable_data());

    // ══════════════════════════════════════════════════════
    // 步骤 6: 计算变换模型的导数（CUDA 内核 DerivateRigid_multiMem_Ker）
    // ══════════════════════════════════════════════════════
    m_pModel->Derivate(m_pSamplePoints->const_data(), NULL,
        m_pMag->const_data(), m_pSampleOK->const_data(), m_nSampleNum,
        m_pDerivatePoints->mutable_data());

    // ══════════════════════════════════════════════════════
    // 步骤 7: 构建联合直方图 + 导数（CUDA 内核 ComputeJHTG_Ker + ReduceJHT_Ker）
    // ══════════════════════════════════════════════════════
    m_pJointPDF->ResetNum(0);
    m_pPDFs->ResetNum(0);
    ComputeJHTDerivate(m_pRefSamplesWindowIndex->const_data(),
        m_refBin.binNum,
        m_pMovSamplesWindowIndex->const_data(),
        m_pMovSamplesWindowTerm->const_data(),
        m_movBin.binNum,
        m_pSampleOK->const_data(), m_nSampleNum,
        m_pDerivatePoints->const_data(),
        m_pJointPDF->mutable_data(),      // 输出: 联合 PDF
        m_pPDFs->mutable_data());          // 输出: 每线程直方图（中间结果）

    // ══════════════════════════════════════════════════════
    // 步骤 8: 计算 PET 边缘 PDF（CUDA 内核 ComputeMovPDFG_Ker）
    // ══════════════════════════════════════════════════════
    m_pMovPDF->ResetNum(0);
    ComputeMovPDF(m_pJointPDF->const_data(), m_refBin.binNum, m_movBin.binNum,
        m_pMovPDF->mutable_data());

    // ══════════════════════════════════════════════════════
    // 步骤 9: 计算 JHT 查找表和 MI 值（CUDA 内核 ComputeJHTTableAndMI_Ker）
    // ══════════════════════════════════════════════════════
    pJHTTable->ResetNum(0);
    TFLOAT MI = ComputeJHTTableAndMI(m_pJointPDF->mutable_data(),
        m_pRefPDF->const_data(), m_pMovPDF->const_data(),
        m_refBin.binNum, m_movBin.binNum,
        m_pModel->GetTransformDim(), pJHTTable->mutable_data());

    // ══════════════════════════════════════════════════════
    // 步骤 10: 计算梯度（CUDA 内核 ComputeGradient_Ker）
    // ══════════════════════════════════════════════════════
    pJHTDev->ResetNum(0);
    ComputeGradient(m_pRefSamplesWindowIndex->const_data(),
        m_refBin.binNum,
        m_pMovSamplesWindowIndex->const_data(),
        m_pMovSamplesWindowTerm->const_data(),
        m_movBin.binNum,
        m_pSampleOK->const_data(), m_nSampleNum,
        m_pModel->GetTransformDim(),
        m_pDerivatePoints->const_data(),
        pJHTTable->mutable_data(),
        pJHTDev->mutable_data());

    // ══════════════════════════════════════════════════════
    // 步骤 11: 梯度求和（GPU thrust::reduce + CPU 拷贝）
    // ══════════════════════════════════════════════════════
    TFLOAT nFactor = 1.0f / (m_movBin.binSize * effectiveSampleNum);
    std::unique_ptr<TFLOAT[]> pGradient_h(new TFLOAT[m_pModel->GetTransformDim()]());
    for (unsigned i = 0; i < m_pModel->GetTransformDim(); ++i)
    {
        // 对所有采样点的梯度贡献求和（GPU 并行 reduce）
        pGradient_h[i] = -SumG<TFLOAT>(pJHTDev->const_data() + i * m_nSampleNum, m_nSampleNum) * nFactor;
    }
    // CPU → GPU 拷贝最终梯度
    checkCudaErrors(cudaMemcpy(pGradient_d, pGradient_h.get(),
        m_pModel->GetTransformDim() * sizeof(TFLOAT), cudaMemcpyHostToDevice));

    return MI;
}
```

---

## 6. CUDA 内核详解

### 6.1 TransormPointKer —— 坐标变换 + 纹理插值

> 文件: `algo-registrationgpu/src/McsfAlgoRegistrationG/src/TransformModelG.cu` (L31-L95)

```cpp
// 常量内存：存储变换参数（所有线程共享，有缓存）
__constant__ TFLOAT g_dMatrix[9];      // 3×3 旋转矩阵
__constant__ TFLOAT g_dOffset[3];      // 3×1 偏移
__constant__ TFLOAT g_dRefCenter[3];   // CT 中心物理坐标
__constant__ unsigned g_dMovSize[3];   // PET 尺寸
__constant__ TFLOAT g_dMovSpacing[3];  // PET 间距

__global__ void TransormPointKer(unsigned sampleNum, 
    const TFLOAT* pSamplePoints,     // CT 采样点坐标 [sampleNum×3, SoA]
    TFLOAT* pTransformedPoints,      // 输出: PET 索引坐标
    short* pTransformedMovImg,       // 输出: PET 插值灰度
    unsigned* pSampleOK,             // 输出: 有效性
    cudaTextureObject_t texObject)   // PET 3D 纹理对象
{
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= sampleNum) return;

    // ── 1. 取出 CT 采样点的相对物理坐标（SoA 布局）──
    const TFLOAT x = pSamplePoints[index];                    // X 分量
    const TFLOAT y = pSamplePoints[index + sampleNum];        // Y 分量
    const TFLOAT z = pSamplePoints[index + sampleNum * 2];    // Z 分量

    // ── 2. 刚体变换：相对物理坐标 → PET 物理坐标 ──
    // dstPoint = R × samplePoint + offset + refCenter
    TFLOAT dstX = g_dMatrix[0]*x + g_dMatrix[1]*y + g_dMatrix[2]*z + g_dOffset[0] + g_dRefCenter[0];
    TFLOAT dstY = g_dMatrix[3]*x + g_dMatrix[4]*y + g_dMatrix[5]*z + g_dOffset[1] + g_dRefCenter[1];
    TFLOAT dstZ = g_dMatrix[6]*x + g_dMatrix[7]*y + g_dMatrix[8]*z + g_dOffset[2] + g_dRefCenter[2];

    // ── 3. PET 物理坐标 → PET 图像索引 ──
    dstX /= g_dMovSpacing[0];
    dstY /= g_dMovSpacing[1];
    dstZ /= g_dMovSpacing[2];

    // ── 4. 边界检查：判断映射点是否在 PET 图像范围内 ──
    if (dstX < 0 || dstY < 0 || dstZ < 0 ||
        dstX > g_dMovSize[0]-1 || dstY > g_dMovSize[1]-1 || dstZ > g_dMovSize[2]-1)
    {
        pSampleOK[index] = 0;  // 超出 PET 范围，标记无效
        return;
    }
    pSampleOK[index] = 1;

    // 保存变换后的 PET 索引坐标（用于后续梯度计算）
    pTransformedPoints[index] = dstX;
    pTransformedPoints[index + sampleNum] = dstY;
    pTransformedPoints[index + sampleNum * 2] = dstZ;

    // ── 5. 三线性插值：从 PET 图像获取灰度值 ──
    int ix0 = static_cast<int>(dstX);
    int iy0 = static_cast<int>(dstY);
    int iz0 = static_cast<int>(dstZ);
    int ix1 = ix0 + 1, iy1 = iy0 + 1, iz1 = iz0 + 1;

    TFLOAT xbias = dstX - ix0;
    TFLOAT ybias = dstY - iy0;
    TFLOAT zbias = dstZ - iz0;

    // 8 个角点的权重
    TFLOAT w0 = (1-xbias)*(1-ybias)*(1-zbias);
    TFLOAT w1 = xbias*(1-ybias)*(1-zbias);
    TFLOAT w2 = (1-xbias)*ybias*(1-zbias);
    TFLOAT w3 = xbias*ybias*(1-zbias);
    TFLOAT w4 = (1-xbias)*(1-ybias)*zbias;
    TFLOAT w5 = xbias*(1-ybias)*zbias;
    TFLOAT w6 = (1-xbias)*ybias*zbias;
    TFLOAT w7 = xbias*ybias*zbias;

    // 使用 CUDA 纹理内存的 tex3D 读取 PET 体素值
    // 纹理内存有硬件缓存，且 addressMode=Clamp 处理边界
    short dvalue =
        tex3D<short>(texObject, ix0, iy0, iz0) * w0 +
        tex3D<short>(texObject, ix1, iy0, iz0) * w1 +
        tex3D<short>(texObject, ix0, iy1, iz0) * w2 +
        tex3D<short>(texObject, ix1, iy1, iz0) * w3 +
        tex3D<short>(texObject, ix0, iy0, iz1) * w4 +
        tex3D<short>(texObject, ix1, iy0, iz1) * w5 +
        tex3D<short>(texObject, ix0, iy1, iz1) * w6 +
        tex3D<short>(texObject, ix1, iy1, iz1) * w7;

    pTransformedMovImg[index] = dvalue;
}
```

**纹理对象创建**（`TransformPointG` 函数中）：

```cpp
void TransformPointG(...)
{
    // 创建 CUDA 纹理对象
    cudaResourceDesc texRes;
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = const_cast<cudaArray*>(pMovArray);  // PET 图像数据

    cudaTextureDesc texDescr;
    texDescr.normalizedCoords = false;        // 使用非归一化坐标（直接用像素索引）
    texDescr.filterMode = cudaFilterModePoint; // 点采样（手动做三线性插值）
    texDescr.addressMode[0] = cudaAddressModeClamp;  // 边界 clamp
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    cudaTextureObject_t texObject = 0;
    cudaCreateTextureObject(&texObject, &texRes, &texDescr, NULL);

    // 将变换参数拷贝到常量内存（有缓存，所有线程共享）
    cudaMemcpyToSymbol(g_dMatrix, pRotate_d, sizeof(TFLOAT)*9, 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(g_dOffset, pOffset_d, sizeof(TFLOAT)*3, 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(g_dRefCenter, pRefCenter_d, sizeof(TFLOAT)*3, 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(g_dMovSpacing, pMovSpacing_d, sizeof(TFLOAT)*3, 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(g_dMovSize, pMovSize_d, sizeof(unsigned)*3, 0, cudaMemcpyDeviceToDevice);

    // 启动内核
    const unsigned threadsPerBlock = std::min(sampleNum, 512u);
    const unsigned blockNum = (sampleNum + threadsPerBlock - 1) / threadsPerBlock;
    TransormPointKer<<<blockNum, threadsPerBlock>>>(sampleNum, pSamplePoints_d,
        pTransformedPoints_d, pTransformdMovImg_d, pSampleOK_d, texObject);
}
```

### 6.2 ParzenWindowIndex_Ker —— Parzen 窗口索引计算

> 文件: `MutualInformationMetricG.cu` (L10-L40)

```cpp
__global__ void ParzenWindowIndex_Ker(const short* pData_d, unsigned len,
    TFLOAT binSize, TFLOAT normMin, unsigned binNum,
    unsigned* pIndex_d, TFLOAT* pTerm_d)
{
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;

    short imgValue = pData_d[index];
    // 归一化窗口项: dWindowTerm = imgValue / binSize - normMin
    TFLOAT dWindowTerm = static_cast<TFLOAT>(imgValue) / binSize - normMin;
    // 窗口索引: floor(dWindowTerm)
    unsigned iWindowIndex = static_cast<unsigned>(floor(dWindowTerm));

    // 边界 clamp（为三次 B 样条留 2 个 bin 的边界）
    if (iWindowIndex < 2) {
        iWindowIndex = 2;
        dWindowTerm = 2;
    } else if (iWindowIndex > binNum - 3) {
        iWindowIndex = binNum - 3;
        dWindowTerm = binNum - 3;
    }

    pIndex_d[index] = iWindowIndex;
    if (NULL != pTerm_d)
        pTerm_d[index] = dWindowTerm;
}
```

### 6.3 B 样条核函数 Evaluate

> 文件: `MutualInformationMetricG.cu` (L45-L80)

```cpp
__device__ TFLOAT Evaluate(const unsigned iOrder, const TFLOAT dDistance)
{
    const TFLOAT dAbs = abs(dDistance);
    const TFLOAT dSqr = dDistance * dDistance;
    TFLOAT dResult;
    switch (iOrder)
    {
    case 2:  // 二次 B 样条（用于导数计算）
        if (dAbs < 0.5)
            dResult = 0.75 - dSqr;
        else if (dAbs < 1.5)
            dResult = ((9.0 - 12.0*dAbs) + 4.0*dSqr) / 8.0;
        else
            dResult = 0.0;
        break;
    default: // 三次 B 样条（用于联合直方图平滑）
        if (dAbs < 1.0)
            dResult = ((4.0 - 6.0*dSqr) + 3.0*dSqr*dAbs) / 6.0;
        else if (dAbs < 2.0)
            dResult = ((8.0 - 12.0*dAbs) - dSqr*dAbs) / 6.0 + dSqr;
        else
            dResult = 0.0;
    }
    return dResult;
}
```

### 6.4 ComputeSdJHTG_Ker —— 联合直方图（共享内存 + 原子操作）

> 文件: `MutualInformationMetricG.cu` (L83-L135)

```cpp
extern __shared__ TFLOAT g_dSharedJHTPDF[];  // 动态共享内存

__global__ void ComputeSdJHTG_Ker(
    const unsigned* pRefIndex_d, unsigned refBinNum,
    const unsigned* pMovIndex_d, const TFLOAT* pMovIndexTerm_d,
    unsigned movBinNum, const unsigned* pSampleOk_d,
    unsigned sampleNum, TFLOAT* pSd_d, TFLOAT* pJHT_d)
{
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned bins2 = refBinNum * movBinNum;

    // ── 1. 初始化共享内存直方图为 0 ──
    for (unsigned i = threadIdx.x; i < bins2; i += blockDim.x)
        g_dSharedJHTPDF[i] = 0;
    __syncthreads();

    if (idx >= sampleNum) return;

    // ── 2. 遍历采样点，用三次 B 样条构建联合直方图 ──
    unsigned b = blockDim.x * gridDim.x;
    for (int i = idx; i < sampleNum; i += b) {
        if (pSampleOk_d[i] != 1) continue;

        unsigned refIndex = pRefIndex_d[i];
        unsigned movIndex = pMovIndex_d[i];
        TFLOAT movTerm = pMovIndexTerm_d[i];
        TFLOAT curSd = 0.f;

        // 三次 B 样条：每个采样点贡献到相邻 4 个 bin
        for (int j = -1; j <= 2; ++j)
        {
            unsigned curMovIndex = movIndex + j;
            TFLOAT movParzenWindowArg = curMovIndex - movTerm;
            TFLOAT jhtValue = Evaluate(3, movParzenWindowArg);  // 三次 B 样条权重
            unsigned index = refIndex * movBinNum + curMovIndex;
            
            // 原子加到共享内存直方图
            atomicAdd(&(g_dSharedJHTPDF[index]), jhtValue);

            // 同时计算 Sd（二次 B 样条导数差分）
            curSd += Evaluate(2, movParzenWindowArg + 0.5f) - Evaluate(2, movParzenWindowArg - 0.5f);
        }
        pSd_d[i] = curSd;
    }

    // ── 3. 将共享内存直方图累加到全局内存 ──
    __syncthreads();
    for (unsigned i = threadIdx.x; i < bins2; i += blockDim.x)
        atomicAdd(&(pJHT_d[i]), g_dSharedJHTPDF[i]);
}
```

**调用后归一化**：

```cpp
void ComputeSdJHTG(...)
{
    // 启动内核（共享内存大小 = refBinNum * movBinNum * sizeof(TFLOAT)）
    ComputeSdJHTG_Ker<<<blockNum, threadsPerBlock, refBinNum*movBinNum*sizeof(TFLOAT)>>>(...);
    
    // 归一化：JHT /= sum(JHT)
    TFLOAT sumJHT = SumG<TFLOAT>(pJHT_d, refBinNum * movBinNum);
    DivCG<TFLOAT>(pJHT_d, refBinNum * movBinNum, sumJHT);
}
```

### 6.5 ComputeRefPDF_Ker —— CT 边缘 PDF

> 文件: `MutualInformationMetricG.cu` (L260-L300)

```cpp
extern __shared__ TFLOAT g_dSharedRefPDF[];

__global__ void ComputeRefPDF_Ker(
    const unsigned* pRefSamplesWindowIndex_d,
    const unsigned* pSampleOK_d, unsigned sampleNum,
    unsigned binNum, TFLOAT* pRefPDF_d)
{
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned tidx = threadIdx.x;

    // 初始化共享内存
    for (unsigned i = tidx; i < binNum; i += blockDim.x)
        g_dSharedRefPDF[i] = 0.f;
    __syncthreads();

    // CT 用零阶 B 样条（盒函数）：直接计数
    if (idx < sampleNum && pSampleOK_d[idx])
    {
        unsigned ss = pRefSamplesWindowIndex_d[idx];
        atomicAdd(&(g_dSharedRefPDF[ss]), 1.f);
    }
    __syncthreads();

    // 共享内存 → 全局内存
    for (unsigned i = tidx; i < binNum; i += blockDim.x)
        atomicAdd(&(pRefPDF_d[i]), g_dSharedRefPDF[i]);
}

// 调用后归一化: RefPDF /= sum(RefPDF)
```

### 6.6 ComputeMovPDFG_Ker —— PET 边缘 PDF

> 文件: `MutualInformationMetricG.cu` (L305-L330)

```cpp
__global__ void ComputeMovPDFG_Ker(const TFLOAT *pJointPDF_d,
    unsigned refBinNum, unsigned movBinNum, TFLOAT *pMovPDF_d)
{
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= movBinNum) return;

    // 对联合 PDF 的 ref 维度求和: MovPDF[j] = Σ_i JointPDF[i*movBinNum + j]
    TFLOAT dsum = 0.0;
    for (unsigned p = 0; p < refBinNum; ++p)
        dsum += pJointPDF_d[p * movBinNum + idx];
    pMovPDF_d[idx] = dsum;
}
```

### 6.7 ComputeJHTTableAndMI_Ker —— MI 值计算

> 文件: `MutualInformationMetricG.cu` (L400-L445)

```cpp
__global__ void ComputeJHTTableAndMI_Ker(
    TFLOAT* pJointPDF_d, const TFLOAT* pRefPDF_d, const TFLOAT* pMovPDF_d,
    unsigned refBinNum, unsigned movBinNum, unsigned tranformDim,
    TFLOAT* pJHTTable_d)
{
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;  // mov bin index
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;  // ref bin index
    if (x >= movBinNum || y >= refBinNum) return;

    const TFLOAT refValue = pRefPDF_d[y];
    const TFLOAT movValue = pMovPDF_d[x];
    const unsigned jhtIndex = y * movBinNum + x;
    const TFLOAT jhtValue = pJointPDF_d[jhtIndex];

    if (jhtValue > 1e-16 && movValue > 1e-16)
    {
        TFLOAT ratio = jhtValue / movValue;
        // JHT 查找表: log(p(i,j) / p_B(j))，用于后续梯度计算
        pJHTTable_d[jhtIndex] = log(ratio);
        // MI 贡献: p(i,j) * log(p(i,j) / (p_A(i) * p_B(j)))
        if (refValue > 1e-16)
            pJointPDF_d[jhtIndex] = jhtValue * log(ratio / refValue);
        else
            pJointPDF_d[jhtIndex] = 0;
    }
    else
    {
        pJHTTable_d[jhtIndex] = 0;
        pJointPDF_d[jhtIndex] = 0;
    }
}

// MI = -Σ p(i,j) * log(p(i,j) / (p_A(i) * p_B(j)))
// 注意：返回负值，因为优化器做梯度下降（MI 要最大化，等价于最小化 -MI）
TFLOAT ComputeJHTTableAndMI(...)
{
    ComputeJHTTableAndMI_Ker<<<blocks, threads>>>(...);
    TFLOAT mi = -SumG<TFLOAT>(pJointPDF_d, binTotal);  // GPU reduce 求和
    return mi;
}
```

### 6.8 ComputeGradient_Ker —— 梯度计算

> 文件: `MutualInformationMetricG.cu` (L460-L500)

```cpp
__global__ void ComputeGradient_Ker(
    const unsigned* pRefIndex_d, unsigned refBinNum,
    const unsigned* pMovIndex_d, const TFLOAT* pMovIndexTerm_d,
    unsigned movBinNum, const unsigned* pSampleOk_d,
    unsigned sampleNum, unsigned transformDim,
    const TFLOAT* pDerivatePoints_d, TFLOAT* pJHTTable_d, TFLOAT* pJHTDev)
{
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sampleNum || pSampleOk_d[idx] != 1) return;

    unsigned refIndex = pRefIndex_d[idx];
    unsigned movIndex = pMovIndex_d[idx];
    TFLOAT movTerm = pMovIndexTerm_d[idx];

    // 遍历 4 个相邻 bin，累加梯度贡献
    for (int i = -1; i <= 2; ++i)
    {
        unsigned curMovIndex = movIndex + i;
        TFLOAT movParzenWindowArg = curMovIndex - movTerm;
        unsigned jhtIndex = refIndex * movBinNum + curMovIndex;
        TFLOAT jhtTableValue = pJHTTable_d[jhtIndex];  // log(p(i,j)/p_B(j))
        
        // 二次 B 样条导数差分
        TFLOAT curSd = Evaluate(2, movParzenWindowArg + 0.5f) 
                     - Evaluate(2, movParzenWindowArg - 0.5f);
        
        // 对每个变换参数维度累加
        for (unsigned j = 0; j < transformDim; ++j)
        {
            TFLOAT t = -curSd * pDerivatePoints_d[j * sampleNum + idx];
            pJHTDev[j * sampleNum + idx] += t * jhtTableValue;
        }
    }
}
```

### 6.9 DerivateRigid_multiMem_Ker —— Versor 刚体导数

> 文件: `TransformModelG.cu` (L198-L245)

```cpp
__global__ void DerivateRigid_multiMem_Ker(
    const TFLOAT* pSamplePoints_d, TFLOAT* pDif_d,
    const TFLOAT* pMag_d, const unsigned* pSampleOK_d,
    unsigned sampleNum, VersorG versor, TFLOAT* pDerivatePoints_d)
{
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= sampleNum) return;

    // 取出采样点坐标和图像梯度
    const TFLOAT px = pSamplePoints_d[index];
    const TFLOAT py = pSamplePoints_d[index + sampleNum];
    const TFLOAT pz = pSamplePoints_d[index + sampleNum * 2];
    const TFLOAT magx = pMag_d[index];
    const TFLOAT magy = pMag_d[index + sampleNum];
    const TFLOAT magz = pMag_d[index + sampleNum * 2];
    const unsigned sampleOK = pSampleOK_d[index];

    TFLOAT dif = (NULL == pDif_d) ? 1.f : (pDif_d[index] *= sampleOK, pDif_d[index]);

    // ── Versor 旋转部分的导数（3 个分量）──
    // 使用预计算的 versor 乘积项（vxx, vyy, vxy 等）加速计算
    // 公式来自 Versor 刚体变换的解析雅可比矩阵
    pDerivatePoints_d[index] = sampleOK * (2.0*dif/versor.vw * 
        (((versor.vyw + versor.vxz)*py + (versor.vzw - versor.vxy)*pz) * magx +
         ((versor.vyw - versor.vxz)*px - 2*versor.vxw*py + (versor.vxx - versor.vww)*pz) * magy +
         ((versor.vzw + versor.vxy)*px + (versor.vww - versor.vxx)*py - 2*versor.vxw*pz) * magz));
    
    pDerivatePoints_d[index + sampleNum] = sampleOK * (2.0*dif/versor.vw * 
        ((-2*versor.vyw*px + (versor.vxw + versor.vyz)*py + (versor.vww - versor.vyy)*pz) * magx +
         ((versor.vxw - versor.vyz)*px + (versor.vzw + versor.vxy)*pz) * magy +
         ((versor.vyy - versor.vww)*px + (versor.vzw - versor.vxy)*py - 2*versor.vyw*pz) * magz));
    
    pDerivatePoints_d[index + 2*sampleNum] = sampleOK * (2.0*dif/versor.vw * 
        ((-2*versor.vzw*px + (versor.vzz - versor.vww)*py + (versor.vxw - versor.vyz)*pz) * magx +
         ((versor.vww - versor.vzz)*px - 2*versor.vzw*py + (versor.vyw + versor.vxz)*pz) * magy +
         ((versor.vxw + versor.vyz)*px + (versor.vyw - versor.vxz)*py) * magz));

    // ── 平移部分的导数（3 个分量，单位矩阵）──
    pDerivatePoints_d[index + 3*sampleNum] = dif * magx * sampleOK;
    pDerivatePoints_d[index + 4*sampleNum] = dif * magy * sampleOK;
    pDerivatePoints_d[index + 5*sampleNum] = dif * magz * sampleOK;
}
```

---

## 7. 优化器：OptimizationG

> 文件: `algo-registrationgpu/src/McsfAlgoRegistrationG/include/OptimizationG.h`
> 文件: `algo-registrationgpu/src/McsfAlgoRegistrationG/src/OptimizationG.cpp`

### 7.1 类定义

```cpp
class OptimizationG
{
protected:
    std::shared_ptr<ImageMetricG> m_pMetric;   // 度量（NMI 或 MSE）
    const unsigned m_maxIterNum;                // 最大迭代次数 (200)
    const TFLOAT m_magTolerance;                // 梯度模长收敛阈值 (0.001)
    const TFLOAT m_maxIterStep;                 // 初始步长 (3)
    const TFLOAT m_minIterStep;                 // 最小步长 (0.01)
    const TFLOAT m_relaxFactor;                 // 松弛因子 (0.9)
    std::unique_ptr<TFLOAT[]> m_pScaleC;        // 参数缩放 [1000,1000,1000,0.01,0.01,0.01]

    TFLOAT m_curStepLength;                     // 当前步长
    unsigned m_curIterNum;                       // 当前迭代次数
    unsigned m_curTransformDim;                  // 变换维度 (6 for Rigid)
    TFLOAT m_curMetricValue;                     // 当前度量值

    std::unique_ptr<TFLOAT[]> m_pGradientC;       // 当前梯度（CPU 副本）
    std::unique_ptr<TFLOAT[]> m_pPreviousGradientC; // 前一次梯度（CPU 副本）
    std::shared_ptr<GpuGlobalData<TFLOAT>> m_pGradient;          // 当前梯度（GPU）
    std::shared_ptr<GpuGlobalData<TFLOAT>> m_pPreviousGradient;  // 前一次梯度（GPU）
};
```

### 7.2 ResumeOptimization —— 主优化循环

> 文件: `OptimizationG.cpp` (L73-L170)

```cpp
int OptimizationG::ResumeOptimization(const unsigned step_h[3])
{
    int iReturnValue = 0;

    // ── 1. 采样参考图像域（CPU 执行，结果传到 GPU）──
    bool ret = m_pMetric->SampleFixedImageDomain(step_h);
    if (!ret) return -1;

    // ── 2. 分配梯度存储 ──
    m_pGradientC.reset(new TFLOAT[m_curTransformDim]());
    m_pPreviousGradientC.reset(new TFLOAT[m_curTransformDim]());
    m_pGradient.reset(new GpuGlobalData<TFLOAT>(m_curTransformDim));
    m_pPreviousGradient.reset(new GpuGlobalData<TFLOAT>(m_curTransformDim));

    // ── 3. 主迭代循环 ──
    while (true)
    {
        // 保存前一次梯度（CPU 拷贝）
        memcpy(m_pPreviousGradientC.get(), m_pGradientC.get(),
               m_curTransformDim * sizeof(TFLOAT));

        // ── 3a. 计算 NMI 值和梯度（GPU 执行）──
        m_curMetricValue = m_pMetric->GetValueAndDerivative(m_pGradient->mutable_data());

        // 梯度从 GPU 拷贝到 CPU（用于参数更新）
        m_pGradient->ToCpu(m_pGradientC.get());

        // ── 3b. 梯度缩放 + 模长计算 + 方向检测 ──
        TFLOAT gradientMagnitude = 0;
        TFLOAT scalarProduct = 0;
        for (unsigned i = 0; i < m_curTransformDim; ++i)
        {
            m_pGradientC[i] /= m_pScaleC[i];           // 参数缩放
            gradientMagnitude += m_pGradientC[i] * m_pGradientC[i]; // 模长
            scalarProduct += (m_pGradientC[i] * m_pPreviousGradientC[i]); // 点积
        }
        gradientMagnitude = sqrt(gradientMagnitude);

        // ── 3c. 收敛判断：梯度足够小 ──
        if (gradientMagnitude < m_magTolerance)
            break;

        // ── 3d. 方向反转检测：缩小步长 ──
        if (scalarProduct < 0)
            m_curStepLength *= m_relaxFactor;

        // ── 3e. 步长终止判断 ──
        if (m_curStepLength < m_minIterStep)
            break;

        // ── 3f. 最大迭代次数判断 ──
        if (++m_curIterNum > m_maxIterNum)
        {
            iReturnValue = 1;
            break;
        }

        // ── 3g. 参数更新（CPU 执行 Versor 乘法，结果传回 GPU）──
        const TFLOAT factor = -1.0f * m_curStepLength / gradientMagnitude;
        m_pMetric->GetModel()->UpdateParameter(factor, m_pGradientC.get());

        // ── 3h. 进度条更新 ──
        // ...（省略进度管理代码）
    }

    return iReturnValue;
}
```

### 7.3 ResumeOptimizationPyramid —— 多分辨率优化

```cpp
int OptimizationG::ResumeOptimizationPyramid(unsigned startExp)
{
    int iReturnValue = 0;
    unsigned step_h[3] = { 0 };

    // 从粗到细：step = 2^i
    for (int i = static_cast<int>(startExp); i >= 0; --i)
    {
        step_h[0] = static_cast<unsigned>(pow(2.0, i));
        step_h[1] = static_cast<unsigned>(pow(2.0, i));
        step_h[2] = static_cast<unsigned>(pow(2.0, i));

        // 步长随分辨率递增：粗分辨率用大步长
        m_curStepLength = m_maxIterStep * pow(0.2f, int(startExp - i));
        m_curIterNum = 0;

        // 非第一级时，先用上一级梯度更新一次参数
        if (i != static_cast<int>(startExp))
        {
            const TFLOAT gradientMagnitude = L2NormG<TFLOAT>(m_pGradient->const_data(), m_pGradient->size());
            const TFLOAT factor = -1.0f * m_curStepLength / gradientMagnitude;
            m_pMetric->GetModel()->UpdateParameter(factor, m_pGradientC.get());
        }

        // 在当前分辨率下优化
        iReturnValue = ResumeOptimization(step_h);
    }
    return iReturnValue;
}
```

---

## 8. 重采样：ResampleG

> 文件: `algo-registrationgpu/src/McsfAlgoRegistrationG/src/ResampleFunc.cu` (L746-L760, L65-L148)

### 8.1 ResampleG —— 入口

```cpp
void ResampleG(const unsigned* pRefSize_h, const TFLOAT* pRefSpacing_h,
    const std::shared_ptr<ImageDataG<short>>& pMov,
    const TFLOAT* pRotateMat_h, const TFLOAT* pOffset_h,
    short* pOutImg_h, bool bMovBoundExtend, InterpolateType interpolator)
{
    // 将 3×3 旋转矩阵 + 偏移 → 4×4 齐次矩阵
    TFLOAT pMat16_h[16] = { 0 };
    TFLOAT cx = pRefSpacing_h[0] * (pRefSize_h[0] - 1) / 2;  // CT 中心 X
    TFLOAT cy = pRefSpacing_h[1] * (pRefSize_h[1] - 1) / 2;  // CT 中心 Y
    TFLOAT cz = pRefSpacing_h[2] * (pRefSize_h[2] - 1) / 2;  // CT 中心 Z
    AffineMat3x3_4x4(pRotateMat_h, pOffset_h, cx, cy, cz, pMat16_h);

    Resample16G(pRefSize_h, pRefSpacing_h, pMov, pMat16_h, pOutImg_h, bMovBoundExtend, interpolator);
}
```

### 8.2 Resample16G_Ker —— 重采样内核

```cpp
__global__ void Resample16G_Ker(const short* pMov_d, short minValue, short maxValue,
    short* pNewMov_d, bool bMovBoundExtend, InterpolateType interpolator,
    cudaTextureObject_t texObject)
{
    // CT 网格坐标
    const unsigned ix = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned iy = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned iz = blockDim.z * blockIdx.z + threadIdx.z;
    if (ix > g_dRefSize[0]-1 || iy > g_dRefSize[1]-1 || iz > g_dRefSize[2]-1) return;

    // ── 1. CT 像素索引 → CT 物理坐标 ──
    TFLOAT3 refPhyCor;
    refPhyCor.x = ix * g_dRefSpacing[0];
    refPhyCor.y = iy * g_dRefSpacing[1];
    refPhyCor.z = iz * g_dRefSpacing[2];

    // ── 2. 用 4×4 矩阵变换: CT 物理坐标 → PET 物理坐标 → PET 索引 ──
    TFLOAT3 movIndexCor;
    movIndexCor.x = (g_dMatrix16[0]*refPhyCor.x + g_dMatrix16[1]*refPhyCor.y + 
                     g_dMatrix16[2]*refPhyCor.z + g_dMatrix16[3]) / g_dMovSpacing[0];
    movIndexCor.y = (g_dMatrix16[4]*refPhyCor.x + g_dMatrix16[5]*refPhyCor.y + 
                     g_dMatrix16[6]*refPhyCor.z + g_dMatrix16[7]) / g_dMovSpacing[1];
    movIndexCor.z = (g_dMatrix16[8]*refPhyCor.x + g_dMatrix16[9]*refPhyCor.y + 
                     g_dMatrix16[10]*refPhyCor.z + g_dMatrix16[11]) / g_dMovSpacing[2];

    const unsigned index = iz * g_dRefSize[0] * g_dRefSize[1] + iy * g_dRefSize[0] + ix;

    // ── 3. 边界处理 ──
    if (!bMovBoundExtend)
    {
        if (movIndexCor.x < 0 || movIndexCor.y < 0 || movIndexCor.z < 0 ||
            movIndexCor.x > g_dMovSize[0]-1 || movIndexCor.y > g_dMovSize[1]-1 || 
            movIndexCor.z > g_dMovSize[2]-1)
        {
            pNewMov_d[index] = minValue;  // 超出 PET 范围，填充最小值
            return;
        }
    }

    // ── 4. 三线性插值（与 TransormPointKer 相同的 8 点加权）──
    int ix0 = static_cast<int>(movIndexCor.x);
    int iy0 = static_cast<int>(movIndexCor.y);
    int iz0 = static_cast<int>(movIndexCor.z);
    // ... 8 个 tex3D 采样 + 加权求和 ...
    pNewMov_d[index] = v;
}
```

**关键区别**：重采样内核遍历 **CT 网格**（目标网格），对每个 CT 体素计算对应的 PET 索引并插值。而 `TransormPointKer` 遍历 **CT 采样点**（稀疏的，按步长采样）。

---

## 9. 完整数据流与 CPU-GPU 交互

### 9.1 数据流全景图

```
┌──────────────────────────────────────────────────────────────────────┐
│                          CPU                                         │
│                                                                      │
│  1. CT/PET 原始数据 (host)                                           │
│     │                                                                │
│     ▼                                                                │
│  2. ImageDataG 构造 → 数据拷贝到 GPU, 创建 cudaArray                  │
│     │                                                                │
│     │                                                                │
│  3. SampleFixedImageDomainG1 (CPU 执行)                              │
│     ├── 遍历 CT 网格, mask 筛选                                      │
│     ├── 像素索引 → 物理坐标 (减去 CT 中心)                            │
│     └── 输出: pSampleImgData + pSamplePoints (CPU 数组)              │
│     │                                                                │
│     ▼  cudaMemcpy (CPU → GPU)                                       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          GPU                                         │
│                                                                      │
│  4. ParzenWindowIndex (CT 采样点)                                    │
│     └── CUDA 内核: 计算每个采样点的 bin 索引                          │
│                                                                      │
│  ┌── 迭代循环 ──────────────────────────────────────────────────┐    │
│  │                                                              │    │
│  │  5. TransormPointKer (CUDA 内核)                             │    │
│  │     ├── CT 物理坐标 → 刚体变换 → PET 物理坐标                 │    │
│  │     ├── PET 物理坐标 / PET spacing → PET 索引                 │    │
│  │     ├── 边界检查 (PET 尺寸)                                   │    │
│  │     └── tex3D 三线性插值 → PET 灰度值                         │    │
│  │                                                              │    │
│  │  6. SumG (thrust::reduce): 统计有效采样数                     │    │
│  │                                                              │    │
│  │  7. ComputeRefPDF_Ker: CT 边缘 PDF (共享内存 + atomicAdd)    │    │
│  │                                                              │    │
│  │  8. ParzenWindowIndex (PET 采样点)                           │    │
│  │                                                              │    │
│  │  9. ComputeMovMagG: PET 图像梯度 (tex3D)                     │    │
│  │                                                              │    │
│  │  10. DerivateRigid_multiMem_Ker: Versor 刚体导数              │    │
│  │                                                              │    │
│  │  11. ComputeJHTG_Ker + ReduceJHT_Ker: 联合直方图              │    │
│  │      (每线程私有直方图 → 全局归约)                             │    │
│  │                                                              │    │
│  │  12. ComputeMovPDFG_Ker: PET 边缘 PDF                        │    │
│  │                                                              │    │
│  │  13. ComputeJHTTableAndMI_Ker: MI 值 + JHT 查找表             │    │
│  │                                                              │    │
│  │  14. ComputeGradient_Ker: 梯度计算                            │    │
│  │                                                              │    │
│  │  15. SumG: 梯度求和 (thrust::reduce)                         │    │
│  │                                                              │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                │                                     │
│                                ▼  cudaMemcpy (GPU → CPU)            │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          CPU                                         │
│                                                                      │
│  16. 梯度缩放: gradient[i] /= scale[i]                               │
│  17. 梯度模长计算 + 方向检测                                          │
│  18. Versor 四元数乘法更新参数 (Rigid::UpdateParameter)               │
│  19. SyncVersorMat: Versor → 3×3 旋转矩阵 + 预计算乘积项              │
│  20. cudaMemcpy (CPU → GPU): 新参数上传                               │
│                                                                      │
│  → 返回步骤 5 (下一轮迭代)                                            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.2 CPU-GPU 数据传输模式

| 传输方向 | 时机 | 数据 | 大小 |
|----------|------|------|------|
| CPU → GPU | 初始化 | CT/PET 图像数据 | refPixNum×2 + movPixNum×2 字节 |
| CPU → GPU | 采样后 | 采样点坐标 + 灰度值 | sampleNum×(3+1)×4 字节 |
| CPU → GPU | 每次迭代 | Versor 参数 + 旋转矩阵 | (6+9+3)×4 字节 |
| GPU → CPU | 每次迭代 | 梯度向量 | 6×4 字节 |
| GPU → CPU | 迭代结束 | 旋转矩阵 + 偏移 + MI 值 | (9+3+1)×4 字节 |
| GPU → CPU | 重采样后 | 重采样图像 | refPixNum×2 字节 |

**关键观察**：每次迭代的 CPU↔GPU 传输量很小（几十字节），不会成为瓶颈。主要传输开销在初始化（图像数据上传）和结束（重采样图像下载）。

---

## 10. CT-PET 不同尺寸图像的处理机制

### 10.1 核心原理

CT 和 PET **不需要**一样大。算法通过**坐标空间映射**处理不同尺寸/间距的图像：

1. **CT 采样**：在 CT 的图像域上按步长采样，得到 CT 像素索引 `(i, j, k)`
2. **CT 索引 → 物理坐标**：`physicalCoord = index × CT_spacing - CT_center`
3. **刚体变换**：`transformedCoord = R × physicalCoord + offset + CT_center`
4. **PET 物理坐标 → PET 索引**：`petIndex = transformedCoord / PET_spacing`
5. **边界检查**：如果 `petIndex` 超出 PET 尺寸范围，标记为无效
6. **三线性插值**：在 PET 图像上用 `tex3D` 插值获取灰度值

### 10.2 初始中心对齐

```cpp
// Rigid::InitFromParam 中的初始偏移计算
dOffsetVec[0] = (movSize_h[0]-1)*movSpacing_h[0]/2.0 - (refSize_h[0]-1)*refSpacing_h[0]/2.0;
dOffsetVec[1] = (movSize_h[1]-1)*movSpacing_h[1]/2.0 - (refSize_h[1]-1)*refSpacing_h[1]/2.0;
dOffsetVec[2] = (movSize_h[2]-1)*movSpacing_h[2]/2.0 - (refSize_h[2]-1)*refSpacing_h[2]/2.0;
```

这确保初始状态下 PET 中心与 CT 中心在物理空间中对齐。

### 10.3 配准完成后的重采样

只有配准**完全结束**后，才调用 `ResampleG` 将 PET 重采样到 CT 网格：

```cpp
ResampleG(refSize_h, refSpacing_h1, pMov, pRotate, pOffset, pOutImage_h, false, LINEAR);
```

重采样内核遍历 CT 网格的每个体素，用最终的 4×4 变换矩阵将 CT 坐标映射到 PET 空间，三线性插值得到 PET 灰度值。

---

## 11. 关键代码索引

| 功能 | 文件路径 | 关键函数/类 |
|------|----------|-------------|
| GPU 配准入口 | `algo-registrationgpu/src/McsfAlgoRegistrationG/src/McsfAlgoRegistrationG.cpp` | `ImageRegistrationRigidG` (L48) |
| GPU 图像数据 | `algo-registrationgpu/src/McsfAlgoRegistrationG/include/ImageDataG.h` | `ImageDataG<T>`, `GpuGlobalData<T>` |
| Versor 结构体 | `algo-registrationgpu/src/McsfAlgoRegistrationG/include/RigidCudaHelpG.h` | `VersorG` (L95) |
| 变换模型基类 | `algo-registrationgpu/src/McsfAlgoRegistrationG/include/TransformModelG.h` | `TransformModelG`, `Rigid`, `Affine` |
| 变换模型实现 | `algo-registrationgpu/src/McsfAlgoRegistrationG/src/TransformModelG.cpp` | `Rigid::InitFromParam` (L139), `Rigid::UpdateParameter` (L191), `SyncVersorMat` (L283) |
| 坐标变换 CUDA 内核 | `algo-registrationgpu/src/McsfAlgoRegistrationG/src/TransformModelG.cu` | `TransormPointKer` (L31), `TransformPointG` (L97) |
| Versor 导数 CUDA 内核 | `algo-registrationgpu/src/McsfAlgoRegistrationG/src/TransformModelG.cu` | `DerivateRigid_multiMem_Ker` (L198) |
| NMI 度量头文件 | `algo-registrationgpu/src/McsfAlgoRegistrationG/include/MutualInformationMetricG.h` | `MutualInformationMetricG`, `BinInfo` |
| NMI 度量实现 | `algo-registrationgpu/src/McsfAlgoRegistrationG/src/MutualInformationMetricG.cpp` | `GetValueAndDerivative` (L88), `SampleFixedImageDomain` (L35), `BinInfo::Init` (L165) |
| NMI CUDA 内核 | `algo-registrationgpu/src/McsfAlgoRegistrationG/src/MutualInformationMetricG.cu` | `ParzenWindowIndex_Ker` (L10), `Evaluate` (L45), `ComputeSdJHTG_Ker` (L83), `ComputeRefPDF_Ker` (L260), `ComputeMovPDFG_Ker` (L305), `ComputeJHTTableAndMI_Ker` (L400), `ComputeGradient_Ker` (L460) |
| 度量基类 | `algo-registrationgpu/src/McsfAlgoRegistrationG/include/ImageMetricG.h` | `ImageMetricG` |
| 度量基类实现 | `algo-registrationgpu/src/McsfAlgoRegistrationG/src/ImageMetricG.cpp` | `SampleFixedImageDomainG1` (L46), `SampleFixedImageDomainG2` (L120), `ComputeMovMag` (L209) |
| 优化器头文件 | `algo-registrationgpu/src/McsfAlgoRegistrationG/include/OptimizationG.h` | `OptimizationG` |
| 优化器实现 | `algo-registrationgpu/src/McsfAlgoRegistrationG/src/OptimizationG.cpp` | `ResumeOptimization` (L73), `ResumeOptimizationPyramid` (L55) |
| 重采样 CUDA 内核 | `algo-registrationgpu/src/McsfAlgoRegistrationG/src/ResampleFunc.cu` | `Resample16G_Ker` (L65), `ResampleG` (L746), `Resample16G` (L152) |
| CUDA 接口声明 | `algo-registrationgpu/src/McsfAlgoRegistrationG/include/RigidCUInterface.h` | 所有 CUDA 函数声明 |
| GPU 内存估算 | `algo-registrationgpu/src/McsfAlgoRegistrationG/src/McsfAlgoRegistrationG.cpp` | `GetGPUPeakMemUsed_sub` (L165) |
| PET 窗位裁剪 | `algo-registrationgpu/src/McsfAlgoRegistrationG/src/McsfAlgoRegistrationG.cpp` | `GetWinValue` (L22) |

---

## 附录 A：NMI 数学公式

### 互信息

$$
MI(A, B) = \sum_{i,j} p_{AB}(i,j) \log \frac{p_{AB}(i,j)}{p_A(i) \cdot p_B(j)}
$$

其中：
- $p_{AB}(i,j)$ —— 联合概率密度（联合直方图归一化）
- $p_A(i) = \sum_j p_{AB}(i,j)$ —— 参考图（CT）边缘概率密度
- $p_B(j) = \sum_i p_{AB}(i,j)$ —— 浮动图（PET）边缘概率密度

### Parzen 窗口

- **CT（参考图）**：零阶 B 样条（盒函数），直接计数
- **PET（浮动图）**：三次 B 样条，在相邻 4 个 bin 间分配权重

$$
\beta^3(x) = \begin{cases}
\frac{4 - 6x^2 + 3|x|^3}{6}, & |x| < 1 \\
\frac{8 - 12|x| - x^2|x|}{6} + x^2, & 1 \le |x| < 2 \\
0, & |x| \ge 2
\end{cases}
$$

### 梯度

$$
\frac{\partial MI}{\partial T_k} = -\sum_{i} \text{Sd}(i) \cdot \log\frac{p_{AB}(i, j_i)}{p_B(j_i)} \cdot \frac{\partial T_k}{\partial \text{point}_i}
$$

其中 $\text{Sd}$ 是二次 B 样条的导数差分：

$$
\text{Sd}(x) = \beta^2(x + 0.5) - \beta^2(x - 0.5)
$$

---

## 附录 B：CT-PET 配准面试要点

### Q1: 为什么 CT-PET 用 NMI 而不是 MSE？

CT 反映组织密度（HU 值，-1024~3000），PET 反映代谢活性（SUV 值，0~数万）。两者灰度值没有线性对应关系，MSE 无法度量其相似性。NMI 基于统计相关性，能捕捉两幅图像灰度分布的统计依赖关系，是跨模态配准的标准选择。

### Q2: GPU 版本中哪些步骤在 CPU 执行？为什么？

1. **采样（SampleFixedImageDomainG1）**：在 CPU 执行，因为 mask 筛选逻辑在 GPU 上实现复杂（分支密集），且采样只需执行一次
2. **Versor 参数更新（UpdateParameter）**：在 CPU 执行，因为四元数乘法只有 3 个标量运算，启动 CUDA 内核的开销远大于计算本身
3. **梯度缩放和收敛判断**：在 CPU 执行，因为只有 6 个标量的简单运算

### Q3: 为什么采样点坐标用 SoA 布局？

SoA（Structure of Arrays）布局 `[all_x, all_y, all_z]` 使得同一维度的坐标在内存中连续存储，GPU 线程访问相邻元素时可以利用合并内存访问（coalesced memory access），大幅提高带宽利用率。AoS 布局 `[x0,y0,z0, x1,y1,z1,...]` 会导致跨步访问。

### Q4: CUDA 纹理内存的优势？

1. **硬件缓存**：纹理内存有专用的纹理缓存，对空间局部性好的访问模式（如三线性插值的 8 个相邻体素）效率极高
2. **边界处理**：`addressMode=Clamp` 自动处理越界访问，无需手动判断
3. **插值加速**：虽然代码中手动实现了三线性插值（`filterMode=Point`），但纹理硬件仍提供缓存加速

### Q5: 联合直方图构建为什么用共享内存 + 原子操作？

联合直方图的大小为 `refBinNum × movBinNum`（约 50×50=2500 个元素），可以放入 GPU 共享内存（每个 block 的共享内存通常有 48KB）。使用共享内存 + `atomicAdd` 构建直方图比直接在全局内存上原子操作快得多，因为共享内存的延迟远低于全局内存。每个 block 先在共享内存中构建局部直方图，再通过 `atomicAdd` 合并到全局内存。

### Q6: 为什么有每线程私有直方图（m_pPDFs）？

`ComputeJHTDerivateG` 中使用了 2560 个线程的私有直方图（`m_pPDFs`，大小为 `refBinNum × movBinNum × 2560`）。这是因为 `atomicAdd` 虽然快，但在高竞争时仍有性能损失。每线程私有直方图完全避免了原子操作，最后通过 `ReduceJHT_Ker` 归约合并。这是一种以空间换时间的策略。

### Q7: Versor 相比欧拉角的优势？

1. **无万向锁**：欧拉角在特定角度组合下会丢失一个自由度
2. **全局无奇异性**：任意旋转都有唯一的四元数表示（不考虑符号）
3. **更新自然**：四元数乘法直接对应旋转的复合，无需三角函数
4. **预计算加速**：VersorG 结构体预计算了 9 个乘积项，避免 GPU 内核中重复计算

### Q8: CT 和 PET 尺寸不同时如何处理？

CT 和 PET 保持各自原始尺寸/间距。算法在 CT 图像域上采样，通过坐标空间映射（CT索引→物理坐标→刚体变换→PET物理坐标→PET索引）和三线性插值，直接从原始尺寸的 PET 图像上获取对应灰度值。超出 PET 边界的采样点被标记为无效并跳过。配准完成后才将 PET 重采样到 CT 网格。
