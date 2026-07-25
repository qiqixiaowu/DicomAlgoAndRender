# algo-fusion-app 配准应用层详细文档

> **文档定位**：本文档基于对 `algo-fusion-app/src/McsfAlgoRegAppFusion/` 源码的逐行阅读，详细记录融合配准应用层的完整实现，包括类继承体系、各模态配准流程、预处理策略、先验矩阵计算、以及与底层 GPU/CPU 配准引擎的衔接。

---

## 目录

1. [整体架构](#1-整体架构)
2. [类继承体系](#2-类继承体系)
3. [FusionRegistrationBase 基类](#3-fusionregistrationbase-基类)
4. [工具层 AlgoFusion::DoRigidReg](#4-工具层-algofusiondorigidreg)
5. [FusionRegistrationCT_PET 详解](#5-fusionregistrationct_pet-详解)
6. [FusionRegistrationCT_CT 详解](#6-fusionregistrationct_ct-详解)
7. [FusionRegistrationCT_MR 详解](#7-fusionregistrationct_mr-详解)
8. [FusionRegistrationMR_MR 详解](#8-fusionregistrationmr_mr-详解)
9. [FusionRegistrationMR_PET 详解](#9-fusionregistrationmr_pet-详解)
10. [FusionRegistrationPET_PET 详解](#10-fusionregistrationpet_pet-详解)
11. [辅助函数与预处理](#11-辅助函数与预处理)
12. [各模态参数对比](#12-各模态参数对比)

---

## 1. 整体架构

```
┌──────────────────────────────────────────────────────────────────────┐
│  应用层入口 (algo-fusion-app)                                        │
│                                                                      │
│  FusionRegistrationBase (抽象基类)                                    │
│  ├── AutoRegistration()          ← 自动配准（含预处理+先验+配准）     │
│  ├── RegistrationWithPriorMatrix() ← 带先验矩阵配准                   │
│  ├── RegistrationWithPointPair() ← 点对配准                          │
│  └── RegistrationTool()          ← 纯虚函数，子类实现                 │
│                                                                      │
│  子类:                                                               │
│  ├── FusionRegistrationCT_CT   (CT-CT 同模态, 去床板+骨骼检测)       │
│  ├── FusionRegistrationCT_PET  (CT-PET 跨模态, HU裁剪+百分位窗位)    │
│  ├── FusionRegistrationCT_MR   (CT-MR 跨模态, 脑部两阶段配准)        │
│  ├── FusionRegistrationMR_MR   (MR-MR 同模态, 灰度自适应采样)        │
│  ├── FusionRegistrationMR_PET  (MR-PET 跨模态)                       │
│  └── FusionRegistrationPET_PET (PET-PET, 部位识别Z轴对齐)            │
├──────────────────────────────────────────────────────────────────────┤
│  工具层 (McsfAlgoRegFusionCommon)                                    │
│  AlgoFusion::DoRigidReg()       ← 统一刚体配准封装                    │
│  ├── ConvertTransformToMatrixOffset() ← 4×4矩阵→Versor+Offset       │
│  ├── AffineMat3x3_4x4()        ← 3×3+偏移→4×4齐次矩阵               │
│  ├── ComputeSampleStep()       ← 采样步长计算                        │
│  └── AutoSampleGrayLevel()     ← MR灰度自适应阈值                    │
├──────────────────────────────────────────────────────────────────────┤
│  算法层 (algo-registrationgpu / algo-registration)                   │
│  ├── ImageRegistrationRigidG() ← GPU刚体配准                         │
│  └── ImageRegistrationRigid()  ← CPU刚体配准                         │
└──────────────────────────────────────────────────────────────────────┘
```

### 核心设计模式

- **模板方法模式**：基类 `AutoRegistration`/`RegistrationWithPriorMatrix` 定义配准流程骨架，子类通过虚函数 `RegistrationTool` 实现模态特定的预处理和参数配置
- **策略模式**：不同模态对（CT-CT, CT-PET, CT-MR...）使用不同的预处理策略和参数集
- **桥接模式**：应用层通过 `DoRigidReg` 统一接口桥接 GPU/CPU 两条配准路径

---

## 2. 类继承体系

> 文件: `algo-fusion-app/include/algo-fusion-app/McsfAlgoRegistrationApplicationFusion.h`

```cpp
// 抽象基类
class FusionRegistrationBase
{
public:
    virtual bool AutoRegistration(...);
    virtual bool RegistrationWithPointPair(...);
    virtual bool RegistrationWithPriorMatrix(...);
    
protected:
    virtual bool RegistrationTool(...) = 0;  // 纯虚函数
    virtual bool PreJudge(...);               // 先验矩阵计算
    
    bool m_bBrainData;      // 是否脑部数据
    bool m_bGPUCheck;       // GPU内存检查模式
    double m_gpuUseBytes;   // GPU峰值内存
    double m_cpuUseBytes;   // CPU峰值内存
};

// CT-CT 配准
class FusionRegistrationCT_CT : public FusionRegistrationBase
{
    // 重写 AutoRegistration, RegistrationWithPriorMatrix, RegistrationTool, PreJudge
private:
    unsigned char* m_pRefBedMask;   // CT参考图床板掩码
    unsigned char* m_pMovBedMask;   // CT浮动图床板掩码
    void* m_pBoneDetectTool;        // 骨骼检测工具（深度学习模型）
};

// CT-PET 配准 (refImage=CT, movImage=PET)
class FusionRegistrationCT_PET : public FusionRegistrationBase
{
    // 重写 AutoRegistration, RegistrationWithPriorMatrix, RegistrationTool
private:
    unsigned char* m_pCTBedMask;    // CT床板掩码
};

// CT-MR 配准 (refImage=CT, movImage=MR)
class FusionRegistrationCT_MR : public FusionRegistrationBase
{
    // 重写 RegistrationTool, PreJudge
private:
    std::shared_ptr<unsigned char>& m_pCTBedMask;
};

// MR-MR 配准
class FusionRegistrationMR_MR : public FusionRegistrationBase
{
    // 重写 RegistrationTool, PreJudge
};

// MR-PET 配准 (refImage=MR, movImage=PET)
class FusionRegistrationMR_PET : public FusionRegistrationBase
{
    // 重写 RegistrationTool
};

// PET-PET 配准
class FusionRegistrationPET_PET : public FusionRegistrationBase
{
    // 重写 AutoRegistration, RegistrationTool
};
```

---

## 3. FusionRegistrationBase 基类

> 文件: `algo-fusion-app/src/McsfAlgoRegAppFusion/src/McsfAlgoRegistrationApplicationFusion.cpp`

### 3.1 AutoRegistration —— 自动配准入口

```cpp
bool FusionRegistrationBase::AutoRegistration(
    const CVolumeDataInfo& refImage,
    const CVolumeDataInfo& movImage,
    bool bIsSameFOR,              // 是否同一坐标系(FOR)
    double resultMatrix[16],      // 输出: 4×4变换矩阵
    short* pOutImage, bool bUseGPU,
    IProgress* pProgress, bool bBrainData)
{
    bUseGPU = true;  // ★ 强制使用GPU
    m_bBrainData = bBrainData;
    
    bool bSameCoord = IsSamePos(refImage, movImage);  // 检查扫描方位是否一致
    
    if (bIsSameFOR)
    {
        // ── 路径A: 同一坐标系 → 坐标配准获取先验矩阵 ──
        double pPriorMatrix[16] = {0};
        CoordinateRegistration(
            refImage.uiSize, refImage.dSpacing, refImage.dPosition, refImage.dOrientation,
            movImage.uiSize, movImage.dSpacing, movImage.dPosition, movImage.dOrientation,
            pPriorMatrix);
        
        ret = RegistrationWithPriorMatrix(refImage, movImage, pPriorMatrix, 
                                          resultMatrix, pOutImage, bUseGPU, pProgress, bBrainData);
    }
    else
    {
        if (!bSameCoord)
        {
            // ── 路径B1: 不同扫描方位 → 转轴位+盲配准 ──
            // 1. 将ref和mov都重采样到轴位
            CVolumeDataInfo newRef, newMov;
            double axi2ref[16], axi2mov[16], axi2ref_i[16];
            TransformToAxi(refImage, newRef, nullptr, nullptr, axi2ref);
            TransformToAxi(movImage, newMov, nullptr, nullptr, axi2mov);
            Mcsf::Invert(axi2ref, axi2ref_i, 4, 4);
            
            // 2. 中心对齐作为先验
            std::unique_ptr<double[]> pPriorMatrix(new double[16]());
            PreJudge(newRef, newMov, pPriorMatrix);
            
            // 3. 在轴位空间配准
            double regMat[16] = {0};
            ret = RegistrationWithPriorMatrix(newRef, newMov, pPriorMatrix.get(),
                                              regMat, nullptr, bUseGPU, pProgress, bBrainData);
            
            // 4. 组合变换: result = axi2mov × regMat × axi2ref⁻¹
            double temp[16];
            Mcsf::AlgoFusion::MatrixMul(axi2mov, regMat, 4, 4, 4, 4, temp);
            Mcsf::AlgoFusion::MatrixMul(temp, axi2ref_i, 4, 4, 4, 4, resultMatrix);
            
            // 5. 用最终矩阵重采样原始movImage
            if (pOutImage != nullptr) {
                ResampleForRegistrationMatrix16(3, movImage.pImage, ...resultMatrix, pOutImage);
            }
        }
        else
        {
            // ── 路径B2: 同扫描方位 → 直接配准 ──
            std::unique_ptr<double[]> pPriorMat(new double[16]());
            PreJudge(refImage, movImage, pPriorMat);
            ret = RegistrationWithPriorMatrix(refImage, movImage, pPriorMat.get(),
                                              resultMatrix, pOutImage, bUseGPU, pProgress, bBrainData);
        }
    }
    return ret;
}
```

### 3.2 RegistrationWithPriorMatrix —— 带先验矩阵配准

```cpp
bool FusionRegistrationBase::RegistrationWithPriorMatrix(
    const CVolumeDataInfo& refImage, const CVolumeDataInfo& movImage,
    const double pPriorMatrix[16], double resultMatrix[16],
    short* pOutImage, bool bUseGPU, IProgress* pProgress, bool bBrainData)
{
    bUseGPU = true;
    m_bBrainData = bBrainData;
    
    bool bSameCoord = IsSamePos(refImage, movImage);
    
    if (!bSameCoord)
    {
        // ── 方位不同: 先用先验矩阵重采样mov到ref方位，再配准 ──
        short* pMovNew = new short[refPixNum]();
        ResampleForRegistrationMatrix16(3, movImage.pImage, refSize, movSize,
            refSpacing, movSpacing, pPriorMatrix, pMovNew);
        
        CVolumeDataInfo movNew;
        movNew.SetSize(refImage.uiSize);
        movNew.SetSpacing(refImage.dSpacing);
        movNew.pImage = pMovNew;
        
        double regMat[16] = {0};
        ret = RegistrationTool(refImage, movNew, nullptr, false, regMat, nullptr, bUseGPU, pProgress);
        
        // 最终矩阵 = 先验矩阵 × 配准矩阵
        Mcsf::AlgoFusion::MatrixMul(pPriorMatrix, regMat, 4, 4, 4, 4, resultMatrix);
        delete[] pMovNew;
    }
    else
    {
        // ── 方位相同: 直接配准 ──
        ret = RegistrationTool(refImage, movImage, pPriorMatrix, false, 
                               resultMatrix, nullptr, bUseGPU, pProgress);
    }
    
    // 重采样输出图像
    if (pOutImage != nullptr) {
        if (bUseGPU)
            ResampleForRegistrationMatrix16G(3, movImage.pImage, refSize, movSize, 
                                            refSpacing, movSpacing, resultMatrix, pOutImage);
        else
            ResampleForRegistrationMatrix16(3, movImage.pImage, refSize, movSize,
                                           refSpacing, movSpacing, resultMatrix, pOutImage);
    }
    return ret;
}
```

### 3.3 RegistrationWithPointPair —— 点对配准

```cpp
bool FusionRegistrationBase::RegistrationWithPointPair(
    const CVolumeDataInfo& refImage, const CVolumeDataInfo& movImage,
    const unsigned refPointIndex[3], const unsigned movPointIndex[3],
    double resultMatrix[16], short* pOutImage, bool bUseGPU, IProgress* pProgress)
{
    // 用点对计算初始偏移
    double offx = movPointIndex[0]*movImage.dSpacing[0] - refPointIndex[0]*refImage.dSpacing[0];
    double offy = movPointIndex[1]*movImage.dSpacing[1] - refPointIndex[1]*refImage.dSpacing[1];
    double offz = movPointIndex[2]*movImage.dSpacing[2] - refPointIndex[2]*refImage.dSpacing[2];
    
    // 构造纯平移先验矩阵
    double priorMatrix[16] = {
        1,0,0,offx,
        0,1,0,offy,
        0,0,1,offz,
        0,0,0,1
    };
    
    return RegistrationTool(refImage, movImage, priorMatrix, false, 
                            resultMatrix, pOutImage, bUseGPU, pProgress);
}
```

### 3.4 PreJudge —— 默认先验矩阵（中心对齐）

```cpp
bool FusionRegistrationBase::PreJudge(
    const CVolumeDataInfo& refImage, const CVolumeDataInfo& movImage,
    std::unique_ptr<double[]>& pPriorMatrix)
{
    // 默认策略：等中心对齐
    // c_dif = movCenter - refCenter = (movSize*movSpacing - refSize*refSpacing) / 2
    double c_dif[3] = {0};
    c_dif[0] = (movImage.dSpacing[0]*movImage.uiSize[0] - refImage.dSpacing[0]*refImage.uiSize[0]) / 2;
    c_dif[1] = (movImage.dSpacing[1]*movImage.uiSize[1] - refImage.dSpacing[1]*refImage.uiSize[1]) / 2;
    c_dif[2] = (movImage.dSpacing[2]*movImage.uiSize[2] - refImage.dSpacing[2]*refImage.uiSize[2]) / 2;
    
    // 构造单位旋转 + 中心偏移的 4×4 矩阵
    pPriorMatrix[0]=1; pPriorMatrix[5]=1; pPriorMatrix[10]=1; pPriorMatrix[15]=1;
    pPriorMatrix[3]=c_dif[0]; pPriorMatrix[7]=c_dif[1]; pPriorMatrix[11]=c_dif[2];
    return true;
}
```

---

## 4. 工具层 AlgoFusion::DoRigidReg

> 文件: `algo-fusion-app/src/McsfAlgoRegAppFusion/src/McsfAlgoRegFusionCommon.cpp` (L318-L420)

这是连接应用层和算法层的关键桥梁：

```cpp
bool DoRigidReg(PARAMETER_t param,
    const short* pRef, const unsigned refSize[3], const double refSpacing[3],
    const short* pMov, const unsigned movSize[3], const double movSpacing[3],
    const double* pPriorMat, double* retMat,
    short* pOut, bool bUseGPU,
    IProgress* pProgress, double progressStart, double progressEnd)
{
    // ── 1. 配置进度条 ──
    if (pProgress != nullptr) {
        ProgressManagement progressBar;
        double regStart = progressStart + (progressEnd - progressStart) * 0.2;
        double regEnd   = progressStart + (progressEnd - progressStart) * 0.9;
        progressBar.init(pProgress, true, regStart, regEnd, 15, 2);
        param.SetProgressBar(progressBar);
    }
    
    // ── 2. 先验矩阵转换: 4×4 → 3×3旋转矩阵 + 3×1偏移 ──
    double dRotate[9] = {0};
    double dOffset[3] = {0};
    if (pPriorMat != nullptr) {
        ConvertTransformToMatrixOffset(
            const_cast<double*>(pPriorMat),
            param.GetIfAffine(),  // false (刚体)
            param.Getdim(),       // 3
            refSize, refSpacing,
            dRotate, dOffset);
        
        param.SetIfUsePriorMatrix(true);
        param.SetPriorMatrix(dRotate);          // Versor旋转向量
        param.SetIfUseImagePosition(true);
        param.SetImagePosition(dOffset);        // 初始偏移
    }
    
    // ── 3. 执行配准 ──
    double pRotate[9], pOffset[3], pTranslation[3], dMetric;
    bool ret;
    if (bUseGPU) {
        ret = ImageRegistrationRigidG(param,       // GPU路径
            pRef, pMov,
            refSize[0], refSize[1], refSize[2],
            movSize[0], movSize[1], movSize[2],
            refSpacing[0], refSpacing[1], refSpacing[2],
            movSpacing[0], movSpacing[1], movSpacing[2],
            pRotate, pOffset, pTranslation, dMetric, pOut);
    } else {
        ret = ImageRegistrationRigid(param,        // CPU路径
            pRef, pMov, ...);
    }
    
    // ── 4. 结果转换: 3×3+偏移 → 4×4齐次矩阵 ──
    double centerX = refSpacing[0] * (refSize[0] - 1) / 2;
    double centerY = refSpacing[1] * (refSize[1] - 1) / 2;
    double centerZ = refSpacing[2] * (refSize[2] - 1) / 2;
    AffineMat3x3_4x4(pRotate, pOffset, centerX, centerY, centerZ, retMat);
    
    return ret;
}
```

### 4.1 ConvertTransformToMatrixOffset —— 4×4矩阵 → Versor + Offset

> 文件: `McsfAlgoRegFusionCommon.cpp` (L130-L200)

对于刚体变换（3D），将 4×4 矩阵的旋转部分转换为 Versor 四元数：

```cpp
// 从3×3旋转矩阵提取Versor四元数
double tr = TransformMatrix[0] + TransformMatrix[5] + TransformMatrix[10];  // 迹

if (tr > 0) {
    double S = 2 * sqrt(tr + 1.0);
    qw = 0.25 * S;
    qx = (TransformMatrix[9] - TransformMatrix[6]) / S;
    qy = (TransformMatrix[2] - TransformMatrix[8]) / S;
    qz = (TransformMatrix[4] - TransformMatrix[1]) / S;
} else if (TransformMatrix[0] > TransformMatrix[5] && ...) {
    // 按对角线最大元素选择分支，避免数值不稳定
    ...
}

// 归一化
double norm = sqrt(qx² + qy² + qz²);
if (norm >= 1.0 - epsilon) {
    qx /= (norm + epsilon * norm);
    qy /= (norm + epsilon * norm);
    qz /= (norm + epsilon * norm);
}

// 输出: matrix[0..2] = [qx, qy, qz] (Versor向量)
// 输出: offset[0..2] = 平移（已转换到以refCenter为原点的坐标系）
offset[0] = TransformMatrix[3] - RefCenter[0] 
          + TransformMatrix[0]*RefCenter[0] + TransformMatrix[1]*RefCenter[1] + TransformMatrix[2]*RefCenter[2];
// ... (y, z 类似)
```

### 4.2 AffineMat3x3_4x4 —— 3×3+Offset → 4×4齐次矩阵

```cpp
// 配准结果的逆变换: 将以refCenter为原点的3×3+偏移 → 4×4齐次矩阵
pMatrix4x4[3]  = cx - R[0]*cx - R[1]*cy - R[2]*cz + offset[0];
pMatrix4x4[7]  = cy - R[3]*cx - R[4]*cy - R[5]*cz + offset[1];
pMatrix4x4[11] = cz - R[6]*cx - R[7]*cy - R[8]*cz + offset[2];
// 其余为旋转矩阵 + [0,0,0,1]
```

### 4.3 ComputeSampleStep —— 采样步长计算

```cpp
unsigned ComputeSampleStep(const unsigned uiSize[3], const double dSpacing[3])
{
    // 取最大间距轴
    double maxSpacing = Max(dSpacing, 3);
    return sample_step_single_axis(maxSpacing);
}

unsigned sample_step_single_axis(double dSpacing)
{
    // 间距截断到 [0.5, 5]
    double clip = min(max(dSpacing, 0.5), 5.0);
    // 线性映射 [0.5, 5] → [4, 1]
    // 间距越小 → 步长越大（因为数据密集，可以跳更多）
    return static_cast<unsigned>(1 + (5.0 - clip) / (5.0 - 0.5) * 3.0 + 0.5);
}
```

---

## 5. FusionRegistrationCT_PET 详解

> 文件: `McsfAlgoRegistrationApplicationFusion.cpp` (L1095-L1280)
> 注释: `refImage=CT, movImage=PET`

### 5.1 AutoRegistration 流程

```
AutoRegistration()
  │
  ├── 1. 大图像降采样（>1500层 → 2×2×2mm间距）
  │     ├── Resample2SpecificSpacing_long(refImage/CT)
  │     └── Resample2SpecificSpacing_long(movImage/PET, 无mask)
  │
  ├── 2. CT床板掩码提取（如果未提供）
  │     └── ExtractBedMask_fast(ref_down_img)
  │
  ├── 3. 先验矩阵计算
  │     ├── bIsSameFOR=true → CoordinateRegistration（坐标配准）
  │     └── bIsSameFOR=false → PreJudge（中心对齐）
  │
  ├── 4. RegistrationWithPriorMatrix()
  │     └── RegistrationTool()  ← 核心预处理+配准
  │
  └── 5. 重采样输出（用原始图像+最终矩阵）
```

### 5.2 RegistrationTool —— CT-PET 核心预处理

```cpp
bool FusionRegistrationCT_PET::RegistrationTool(
    const CVolumeDataInfo& refImage,   // CT
    const CVolumeDataInfo& movImage,   // PET
    const double pPriorMatrix[16],
    bool isAuto, double resultMatrix[16],
    short* pOutImage, bool bUseGPU, IProgress* pProgress)
{
    // ════════════════════════════════════════════
    // CT 预处理: 去床板 + HU转换 + HU裁剪
    // ════════════════════════════════════════════
    const short ct_HU_down = -1024;
    const short ct_HU_up   = 1024;
    
    #pragma omp parallel for
    for (int i = 0; i < refImageLen; ++i) {
        if (m_pCTBedMask[i] != 0) {
            pRefImageNoBed_HU_clip[i] = ct_HU_down;  // 床板区域→-1024
        } else {
            short v = refImage.pImage[i] * refImage.RescaleSlope + refImage.RescaleIntercept;
            v = clamp(v, ct_HU_down, ct_HU_up);       // HU裁剪 [-1024, 1024]
            pRefImageNoBed_HU_clip[i] = v;
        }
    }
    
    // ════════════════════════════════════════════
    // PET 预处理: 百分位窗位裁剪
    // ════════════════════════════════════════════
    // 1. 降采样到10×10×10mm（加速百分位计算）
    Resample2SpecificSpacing(movImage.pImage, movImage.uiSize, movImage.dSpacing,
                             downMovSize, downMovSpacing, pDownMov);
    
    // 2. 第一次百分位: 排除背景（0.01~1.0）
    GetWinValue(pDownMov.get(), downMovLen, 0.01, 1.0, pet_low, pet_up);
    
    // 3. 统计背景比例
    float count = 0;
    for (int i = 0; i < downMovLen; ++i)
        if (pDownMov[i] <= pet_low) ++count;
    float back_ratio = count / downMovLen;
    
    // 4. 第二次百分位: 排除背景后的99%上界
    float pet_up_ratio = back_ratio + (1 - back_ratio) * 0.99f;
    GetWinValue(pDownMov.get(), downMovLen, 0.01, pet_up_ratio, pet_low, pet_up);
    
    // 5. 对原始PET做阈值裁剪
    #pragma omp parallel for
    for (int i = 0; i < movImageLen; ++i) {
        short v = movImage.pImage[i];
        v = clamp(v, pet_low, pet_up);
        pMovImageClip[i] = v;
    }
    
    // ════════════════════════════════════════════
    // CT 身体掩码（用于采样区域限制）
    // ════════════════════════════════════════════
    std::unique_ptr<unsigned char[]> bodyMask(new unsigned char[refImageLen]());
    for (int i = 0; i < refImageLen; ++i) {
        if (m_pCTBedMask[i] == 0)  // 非床板区域 = 身体
            bodyMask[i] = 1;
    }
    
    // ════════════════════════════════════════════
    // 配准参数设置
    // ════════════════════════════════════════════
    PARAMETER_t para;
    para.SetIfAffine(false);                    // 刚体变换
    para.Init(3, 1000, 0.01);                   // dim=3, 旋转缩放=1000, 平移缩放=0.01
    para.SetSampleIntensityLevel(ct_HU_down);   // 采样强度下限 = -1024
    para.SetMaxIterationStep(3);                // 初始步长
    para.SetMinIterationStep(0.01);             // 最小步长
    para.SetRelaxationFactor(0.9);              // 松弛因子
    para.SetMaxIterationNum(200);               // 最大迭代次数
    para.SetMagnitudeTolerance(0.001);          // 收敛阈值
    para.SetMetricOption(1);                    // ★ NMI（归一化互信息）
    para.SetResolutionControl(true);            // 采样步长控制分辨率
    para.SetSampleRate(ComputeSampleStep(refImage.uiSize, refImage.dSpacing));
    para.SetUseMultiThread(true);
    para.SetMinEffectiveSampleRatio(0.0125);    // 最小有效采样比例 1.25%
    para.SetSampleRegionMask(bodyMask.get());   // ★ CT身体掩码
    
    // ════════════════════════════════════════════
    // 执行配准
    // ════════════════════════════════════════════
    bool ret = AlgoFusion::DoRigidReg(para,
        pRefImageNoBed_HU_clip.get(), refImage.uiSize, refImage.dSpacing,
        pMovImageClip.get(), movImage.uiSize, movImage.dSpacing,
        pPriorMatrix, resultMatrix, pOutImage, bUseGPU, pProgress, 0.2, 1);
    
    return ret;
}
```

### 5.3 PET 百分位窗位算法详解

PET 数据的灰度值范围很大（0~数万），直接用于 NMI 直方图会导致 bin 过于稀疏。算法通过两阶段百分位裁剪解决：

1. **第一阶段** `GetWinValue(data, 0.01, 1.0)` —— 获取 1%~100% 百分位，确定背景阈值 `pet_low`
2. **背景比例计算** —— 统计 ≤ `pet_low` 的像素比例 `back_ratio`
3. **第二阶段** `GetWinValue(data, 0.01, back_ratio + (1-back_ratio)*0.99)` —— 在排除背景后的有效数据中取 99% 上界

`GetWinValue` 实现（模板函数）：

```cpp
template<typename T>
void GetWinValue(const T* pSrc, unsigned N, double lowRadio, double upRadio, 
                 T& lowValue, T& upValue)
{
    std::unique_ptr<T[]> pDst(new T[N]());
    memcpy(pDst.get(), pSrc, sizeof(T) * N);
    Mcsf::SortAscend(pDst.get(), N);           // 升序排序
    unsigned lowIdx = static_cast<unsigned>(lowRadio * N);
    unsigned upIdx  = static_cast<unsigned>(upRadio * N);
    lowValue = pDst[lowIdx];
    upValue  = pDst[upIdx - 1];
}
```

---

## 6. FusionRegistrationCT_CT 详解

> 文件: `McsfAlgoRegistrationApplicationFusion.cpp` (L440-L960)

### 6.1 特有功能

CT-CT 配准是同模态配准，有以下特有功能：

1. **骨骼检测（深度学习）**：使用 `BoneLocationRun` 检测肋骨和骨盆区域，生成骨骼 mask 用于局部配准
2. **关键点定位**：使用 `McsfAlgoRegKeyptsLocalize` 检测 14 个解剖关键点（脑室、肺、肾、股骨等），取中位数差异作为先验偏移
3. **ref/mov 自动切换**：经验上小 FOV 作为参考图更稳定

### 6.2 AutoRegistration 流程

```
AutoRegistration()
  │
  ├── 1. 大图像降采样（>1500层 → 2×2×2mm）
  │     ├── Resample2SpecificSpacing_long(refImage, pRefBedMask)
  │     └── Resample2SpecificSpacing_long(movImage, pMovBedMask)
  │
  ├── 2. 床板掩码提取（如果未提供）
  │     ├── ExtractBedMask_fast(ref_down_img)
  │     └── ExtractBedMask_fast(mov_down_img)
  │
  ├── 3. 先验矩阵计算
  │     ├── bIsSameFOR=true → CoordinateRegistration
  │     └── bIsSameFOR=false → 关键点定位 PreJudge:
  │           ├── GetCTKeyPts(refImage) → 14个解剖关键点
  │           ├── GetCTKeyPts(movImage) → 14个解剖关键点
  │           ├── 计算交集器官中心点差异
  │           ├── 排序取中位数
  │           └── 构造先验偏移矩阵
  │
  ├── 4. RegistrationWithPriorMatrix → RegistrationTool → CTCT_Reg
  │
  └── 5. 重采样输出
```

### 6.3 CTCT_Reg —— CT-CT 核心配准函数

```cpp
bool CTCT_Reg(void* m_pBoneDetectTool, 
    const CVolumeDataInfo& refImage, unsigned char* pRefBedMask,
    const CVolumeDataInfo& movImage, unsigned char* pMovBedMask,
    const double pPriorMatrix[16], double resultMatrix[16],
    short* pOutImage, bool bUseGPU, IProgress* pProgress, ...)
{
    // ════════════════════════════════════════════
    // 1. 骨骼区域检测（深度学习模型）
    // ════════════════════════════════════════════
    if (refImage.uiSize[2] * refImage.dSpacing[2] > 500) {
        // Z方向>500mm，检测胸腹部骨骼
        int pBoxes[7*6] = {};          // 7个目标的包围盒
        unsigned char pBoxes_mask[7] = {};
        // {'skull':0, 'sternum':1, 'spine':2, 'rib':3, 
        //  'femur_left':4, 'femur_right':5, 'pelvis':6}
        
        // CT转HU
        #pragma omp parallel for
        for (int i = 0; i < refImageLen; ++i)
            pImgHU[i] = refImage.pImage[i] * RescaleSlope + RescaleIntercept;
        
        Mcsf::BoneLocationRun(pImgHU, piSize, pfSpacing, pRefBedMask,
            m_pBoneDetectTool, pBoxes_mask, pBoxes);
        
        // 如果检测到肋骨/骨盆，设置局部配准mask
        if (pBoxes_mask[3] != 0) {  // 肋骨
            bUseMask = true;
            // 将肋骨包围盒区域设为1
            for (z/y/x in pRibBox范围)
                pRefImgMask[idx] = 1;
        }
        if (pBoxes_mask[6] != 0) {  // 骨盆
            bUseMask = true;
            // 将骨盆包围盒区域设为1
            ...
        }
    }
    
    // ════════════════════════════════════════════
    // 2. CT预处理: 去床板 + HU转换 + 灰度裁剪
    // ════════════════════════════════════════════
    short upRv = 256;    // 上限256（骨骼以下）
    short downRv = grayLevel;  // 下限（骨骼检测时为0）
    
    #pragma omp parallel for
    for (int i = 0; i < refImageLen; ++i) {
        if (pRefBedMask[i] != 0)
            pRefImageNoBedHU[i] = -1024;  // 床板→-1024
        else
            pRefImageNoBedHU[i] = clamp(
                refImage.pImage[i] * Slope + Intercept, downRv, upRv);
    }
    // movImage 同理
    
    // ════════════════════════════════════════════
    // 3. 配准参数
    // ════════════════════════════════════════════
    PARAMETER_t para;
    para.SetIfAffine(false);
    para.Init(3, 1000, 0.01);
    para.SetSampleIntensityLevel(grayLevel);
    para.SetMaxIterationStep(3);
    para.SetMinIterationStep(0.01);
    para.SetRelaxationFactor(0.9);
    para.SetMaxIterationNum(500);           // ★ 500次（比CT-PET的200多）
    para.SetMagnitudeTolerance(0.001);
    para.SetMetricOption(0);                // ★ MSE（均方误差）！同模态用MSE
    para.SetResolutionControl(true);
    para.SetSampleRate(ComputeSampleStep(...));
    para.SetMinEffectiveSampleRatio(0.05);  // ★ 5%（比CT-PET的1.25%高）
    
    if (bUseMask) {
        para.SetIfUseSampleRegionMask(true);
        para.SetSampleRegionMask(pRefImgMask.get());  // 骨骼区域mask
    }
    
    // ════════════════════════════════════════════
    // 4. 执行配准
    // ════════════════════════════════════════════
    ret = AlgoFusion::DoRigidReg(para, pRefImageNoBedHU, refSize, refSpacing,
        pMovImageNoBedHU, movSize, movSpacing, pPriorMatrix, resultMatrix,
        pOutImage, bUseGPU, pProgress, 0.5, 1);
    
    return ret;
}
```

### 6.4 RegistrationTool —— ref/mov 自动切换

```cpp
bool FusionRegistrationCT_CT::RegistrationTool(...)
{
    // 经验: 小FOV作为参考图更稳定
    if (refImage.uiSize[2] * refImage.dSpacing[2] > movImage.uiSize[2] * movImage.dSpacing[2]) {
        // ref比mov大 → 交换ref/mov
        double pPriorInv[16], regMat[16];
        Mcsf::Invert(pPriorMatrix, pPriorInv, 4, 4);
        ret = CTCT_Reg(m_pBoneDetectTool, movImage, m_pMovBedMask, 
                       refImage, m_pRefBedMask, pPriorInv, regMat, ...);
        Mcsf::Invert(regMat, resultMatrix, 4, 4);  // 逆变换还原
    } else {
        ret = CTCT_Reg(m_pBoneDetectTool, refImage, m_pRefBedMask,
                       movImage, m_pMovBedMask, pPriorMatrix, resultMatrix, ...);
    }
    return ret;
}
```

### 6.5 关键点定位 PreJudge

CT-CT 的 `PreJudge` 使用 `McsfAlgoRegKeyptsLocalize` 检测 14 个解剖关键点：

```
关键点索引: 0=脑室, 1-4=其他, 5/6=左/右肺, 7=其他, 8/9=左/右肾, 10/11=左/右股骨, 12-13=其他
```

算法流程：
1. 对 ref 和 mov 分别调用 `GetCTKeyPts` 获取关键点坐标
2. 计算交集器官（两图都检测到的）的中心点差异
3. 优先使用 7 个配准器官 `{0, 5, 6, 8, 9, 10, 11}` 的交集
4. 按差异平方和排序，取中位数对应的偏移
5. 如果优先器官无交集，则使用全部 14 个关键点

---

## 7. FusionRegistrationCT_MR 详解

> 文件: `McsfAlgoRegistrationApplicationFusion.cpp` (L1290-L1480)
> 注释: `refImage=CT, movImage=MR`

### 7.1 特有功能：脑部两阶段配准

CT-MR 配准在脑部数据时使用**两阶段策略**：

```
阶段1: 全局配准（whole-body）
  ├── CT: 去床板 + HU转换
  ├── MR: 原始数据
  ├── 度量: NMI
  ├── 采样强度: -200 (脑部) 或 -1024 (全身)
  ├── 旋转缩放: 300 (脑部) 或 1000 (全身)
  └── 输出: globalMat + pGlobalOut (MR重采样到CT空间)

阶段2: 局部脑部配准（仅 bBrainData=true）
  ├── CT: 脑分割 (CTBrainSegmentation3D) → 仅保留脑组织
  ├── MR: 使用阶段1的重采样结果
  ├── CT脑组织灰度映射: [0, 100] → [0, 255]
  ├── 度量: NMI
  └── 输出: localMat
  最终: resultMatrix = globalMat × localMat
```

### 7.2 RegistrationTool 参数

```cpp
PARAMETER_t para;
para.SetIfAffine(false);
if (m_bBrainData) {
    para.Init(3, 300, 0.01);     // 脑部: 旋转缩放300（更小，因为脑部旋转范围小）
} else {
    para.Init(3, 1000, 0.01);    // 全身: 旋转缩放1000
}
short GrayLevel = m_bBrainData ? (-200) : -1024;  // 脑部采样下限更高
para.SetSampleIntensityLevel(GrayLevel);
para.SetMaxIterationStep(3);
para.SetMinIterationStep(0.01);
para.SetRelaxationFactor(0.95);   // ★ 0.95（比CT-PET的0.9高，更保守）
para.SetMaxIterationNum(500);
para.SetMagnitudeTolerance(0.001);
para.SetMetricOption(1);          // NMI
para.SetResolutionControl(true);
if (m_bBrainData) {
    unsigned sr[3] = {2, 2, 1};   // 脑部高分辨率: 采样步长[2,2,1]
    para.SetSampleRate(sr);
}
para.SetMinEffectiveSampleRatio(0.0125);
```

### 7.3 脑部分割与灰度映射

```cpp
// 阶段2: 脑部局部配准
if (m_bBrainData) {
    // 1. CT脑分割
    char* brainmask = new char[RefImgPixelNumber]();
    Mcsf::CTBrainSegmentation3D(pRefImageNoBed, refSize, refSpacing,
        bottomBoundSliceIndex, topBoundSliceIndex, brainmask, false);
    
    // 2. CT仅保留脑组织，非脑组织→-1024
    for (i = 0; i < RefImgPixelNumber; ++i) {
        if (brainmask[i] == 0)
            pRefImageNoBed[i] = -1024;
        if (brainmask[i] == 1 && pRefImageNoBed[i] < 20)
            pRefImageNoBed[i] = 55;  // 脑组织最低值
    }
    
    // 3. CT灰度映射: [0, 55] → [0, 255]
    for (i = 0; i < RefImgPixelNumber; ++i) {
        if (pRefImageNoBed[i] < 0 || pRefImageNoBed[i] > 100)
            pRefImageNoBed[i] = 0;
        else if (pRefImageNoBed[i] > 55)
            pRefImageNoBed[i] = 255;
        else
            pRefImageNoBed[i] = 255.0 * (pRefImageNoBed[i] - 0) / (55 - 0);
    }
    
    // 4. 局部配准（使用阶段1结果作为mov）
    RigidRegistrationToolkit toolkit_brain(
        pRefImageNoBed, refSize, refSpacing,    // CT脑组织
        pGlobalOut, refSize, refSpacing,         // 阶段1重采样的MR
        bUseGPU);
    ret = toolkit_brain.DoRegistration(para, nullptr, localMat, pLocalOut, pProgress, 0.5, 1);
    
    // 5. 组合变换
    Mcsf::AlgoFusion::MatrixMul(globalMat, localMat, 4, 4, 4, 4, resultMatrix);
}
```

---

## 8. FusionRegistrationMR_MR 详解

> 文件: `McsfAlgoRegistrationApplicationFusion.cpp` (L1490-L1620)

### 8.1 特点

- **同模态配准**：使用 NMI（而非 MSE），因为 MR-MR 可能存在灰度偏移
- **灰度自适应采样**：使用 `AutoSampleGrayLevel` 自动确定采样强度下限
- **关键点定位**：使用 `GetMRKeyPts` 检测 MR 关键点
- **ref/mov 自动切换**：与 CT-CT 类似

### 8.2 MRMR_Reg 参数

```cpp
PARAMETER_t para;
para.SetIfAffine(false);
para.Init(3, 500, 0.01);          // 旋转缩放500（介于CT-CT的1000和脑部CT-MR的300之间）
para.SetMaxIterationStep(2.0);     // ★ 2.0（比其他模态的3.0小）
para.SetMinIterationStep(0.01);
para.SetRelaxationFactor(0.92);    // ★ 0.92
para.SetMagnitudeTolerance(0.001);
para.SetMetricOption(1);           // NMI
para.SetMaxIterationNum(200);
para.SetMinEffectiveSampleRatio(0.125);  // ★ 12.5%（最高！）
para.SetResolutionControl(true);

// 灰度自适应: 取20%背景分位作为采样下限
short grayLevel = 0;
Mcsf::AlgoFusion::AutoSampleGrayLevel(refImage.pImage, refImageLen, grayLevel, 0.2);
para.SetSampleIntensityLevel(grayLevel);
```

### 8.3 AutoSampleGrayLevel —— MR灰度自适应

```cpp
bool AutoSampleGrayLevel(short* pData, int iLength, short& GrayLevel, double backRatio)
{
    // 1. 多线程构建直方图
    // 2. 计算累积分布
    // 3. 取 backRatio (0.2) 分位作为灰度下限
    // 目的: MR图像背景区域灰度值不固定，需要自适应确定有效组织区域
}
```

---

## 9. FusionRegistrationMR_PET 详解

> 文件: `McsfAlgoRegistrationApplicationFusion.cpp` (L1720-L1800)
> 注释: `refImage=MR, movImage=PET`

### 9.1 RegistrationTool

```cpp
bool FusionRegistrationMR_PET::RegistrationTool(...)
{
    // ── PET 预处理: 百分位窗位 ──
    short lowValue, upValue;
    GetWinValue(movImage.pImage, movPixNum, 0.001, 0.995, lowValue, upValue);
    // 裁剪到 [lowValue, upValue]
    
    // ── MR 预处理: 灰度自适应采样下限 ──
    short grayLevel = 0;
    AutoSampleGrayLevel(refImage.pImage, refImageLen, grayLevel, 0.2);
    
    // ── 参数 ──
    PARAMETER_t para;
    para.SetIfAffine(false);
    para.Init(3, 500, 0.01);
    para.SetMaxIterationStep(3.0);
    para.SetRelaxationFactor(0.90);
    para.SetMaxIterationNum(500);
    para.SetMetricOption(1);           // NMI
    para.SetMinEffectiveSampleRatio(0.125);
    
    // ── 采样步长: 自定义映射 ──
    unsigned sr[3] = {0};
    for (unsigned i = 0; i < 3; ++i) {
        double maxSpacing = min(refImage.dSpacing[i], 5.0);
        maxSpacing = max(maxSpacing, 0.5);
        sr[i] = static_cast<unsigned>(1 + (5.0 - maxSpacing) / (5.0 - 0.5) * 3.0 + 0.5);
    }
    para.SetSampleRate(sr);
    
    // ── 配准 ──
    bool ret = AlgoFusion::DoRigidReg(para,
        refImage.pImage, refImage.uiSize, refImage.dSpacing,
        pMovNew.get(), movImage.uiSize, movImage.dSpacing,
        pPriorMatrix, resultMatrix, pOutImage, bUseGPU, pProgress, 0, 1);
    
    return ret;
}
```

---

## 10. FusionRegistrationPET_PET 详解

> 文件: `McsfAlgoRegistrationApplicationFusion.cpp` (L1820-L2050)

### 10.1 特有功能：基于部位识别的 Z 轴对齐

PET-PET 配准不使用 NMI 迭代优化，而是通过**深度学习部位识别**直接计算 Z 轴偏移：

```cpp
bool FusionRegistrationPET_PET::AutoRegistration(...)
{
    // ── 1. 部位识别（深度学习）──
    Mcsf::AlgoIntelliWorkflow::Interface::ImageRecognition ImgRec;
    
    // 对ref和mov分别采样50层，识别每层所属身体部位
    ImgRec.Run(refImage.pImage, "pet", refSize, refSpacing, refPartRatio, ...);
    ImgRec.Run(movImage.pImage, "pet", movSize, movSpacing, movPartRatio, ...);
    
    // 获取每层的部位ID序列（从头到脚）
    // refPartOutput: [0, 0, 1, 1, 2, 2, 3, 3, ...] (部位ID)
    
    // ── 2. 计算每个部位的Z轴位置比例 ──
    // 找连续3层相同部位ID的位置，记录为该部位的Z比例
    for (int i = 0; i < partLen - 3; ++i) {
        if (partOutput[i] == partOutput[i+1] && partOutput[i] == partOutput[i+2]
            && partOutput[i] <= partOutput[i+3]) {
            partRatio[partOutput[i]].second = (i+1) / partLen;
        }
    }
    
    // ── 3. 计算Z轴偏移 ──
    double z_dif = 0.0;
    int overlapNum = 0;
    for (size_t i = 0; i < refPartRatio.size(); ++i) {
        if (refPartRatio[i].second > 0.04 && movPartRatio[i].second > 0.04) {
            // 有交集的部位: 计算Z差异
            z_dif += -(1-refPartRatio[i].second) * refImage.uiSize[2] * refImage.dSpacing[2]
                     +(1-movPartRatio[i].second) * movImage.uiSize[2] * movImage.dSpacing[2];
            ++overlapNum;
        }
    }
    if (overlapNum > 0)
        z_dif /= overlapNum;  // 取平均
    else {
        // 无交集: 根据部位顺序估算距离
        if (refTop > movBottom)      // ref在下方
            z_dif = -refImage.uiSize[2] * refImage.dSpacing[2] - (refTop-movBottom-1)*150;
        else if (refBottom < movTop) // ref在上方
            z_dif =  refImage.uiSize[2] * refImage.dSpacing[2] + (movTop-refBottom-1)*150;
    }
    
    // ── 4. 构造结果矩阵（纯平移）──
    double x_dif = (movImage.dSpacing[0]*movImage.uiSize[0] - refImage.dSpacing[0]*refImage.uiSize[0]) / 2;
    double y_dif = (movImage.dSpacing[1]*movImage.uiSize[1] - refImage.dSpacing[1]*refImage.uiSize[1]) / 2;
    
    resultMatrix = {
        1, 0, 0, x_dif,
        0, 1, 0, y_dif,
        0, 0, 1, z_dif,
        0, 0, 0, 1
    };
    
    // ── 5. 重采样输出 ──
    if (pOutImage != nullptr)
        ResampleForRegistrationMatrix16G(3, movImage.pImage, ..., resultMatrix, pOutImage);
    
    return (ret1 && ret2);
}
```

### 10.2 RegistrationTool

PET-PET 的 `RegistrationTool` 是空实现（直接返回 true），因为所有逻辑都在 `AutoRegistration` 中完成：

```cpp
bool FusionRegistrationPET_PET::RegistrationTool(...)
{
    // 空实现
    return true;
}
```

---

## 11. 辅助函数与预处理

### 11.1 TransformToAxi —— 方位转轴位

> 文件: `McsfAlgoRegistrationApplicationFusion.cpp` (L65-L155)

将矢状面/冠状面图像重采样到轴位面：

```cpp
void TransformToAxi(const CVolumeDataInfo& AnatomicalVolume, 
    CVolumeDataInfo& new_AnatomicalVolume,
    const unsigned char* pMaskIn, unsigned char* pMaskOut,
    double Transform_ToAxi[16])
{
    std::bitset<3> bs;
    GetScanPos(AnatomicalVolume, bs);  // 检测扫描方位
    
    if (bs[2])  // 矢状面 → 轴位
    {
        // 交换尺寸: [W,H,D] → [D,W,H]
        // 交换间距: [Sx,Sy,Sz] → [Sz,Sx,Sy]
        // 变换矩阵:
        // [0  0  1  0]
        // [1  0  0  0]
        // [0 -1  0  (Dimy-1)*Sy]
        // [0  0  0  1]
    }
    else if (bs[0])  // 已是轴位 → 单位矩阵
    {
        // 无变换
    }
    else  // 冠状面 → 轴位
    {
        // 交换尺寸: [W,H,D] → [W,D,H]
        // 变换矩阵:
        // [1  0  0  0]
        // [0  0  1  0]
        // [0 -1  0  (Dimy-1)*Sy]
        // [0  0  0  1]
    }
    
    // 求逆（因为需要从轴位映射回原始方位）
    Mcsf::Invert(Transform_ToAxi, temp, 4, 4);
    memcpy(Transform_ToAxi, temp, 16 * sizeof(double));
    
    // 重采样图像和mask
    ResampleForRegistrationMatrix16G(3, AnatomicalVolume.pImage, ...);
    if (pMaskIn != nullptr)
        ResampleMaskForRegistrationMatrix16G(3, pMaskIn, ...);
}
```

### 11.2 GetScanPos —— 扫描方位检测

```cpp
void GetScanPos(const CVolumeDataInfo& img, std::bitset<3>& bs)
{
    // bs[0]=axial, bs[1]=coronal, bs[2]=sagittal
    // 通过方向矩阵(dOrientation[6])的最大分量判断
    
    if (|orientation[1]| > |orientation[0]| && |orientation[1]| > |orientation[2]|)
        bs.set(2);  // 矢状面
    else if (|orientation[4]| > |orientation[5]| && |orientation[4]| > |orientation[3]|)
        bs.set(0);  // 轴位面
    else
        bs.set(1);  // 冠状面
}
```

### 11.3 ExtractBedMask_fast —— 快速床板提取

```cpp
bool ExtractBedMask_fast(const CVolumeDataInfo& srcImg, 
    std::unique_ptr<unsigned char[]>& pBedMask)
{
    // 1. 降采样到 [2,2,4]mm（加速处理）
    ResampleForRegistrationMatrix16G_long(3, srcImg.pImage, dstSize, srcSize,
        dstSpacing, srcSpacing, I, pDstImg);
    
    // 2. 调用床板去除算法
    BedDataInfo bedInfo;
    bedInfo.pData = pDstImg.get();
    bedInfo.pMask = pDstBed.get();
    bedInfo.Intercept = srcImg.RescaleIntercept;
    bedInfo.Slope = srcImg.RescaleSlope;
    Mcsf::McsfAlgoRemoveBedBoard(bedInfo, 0, 1, 8, true, false);
    
    // 3. 将mask恢复到原始分辨率
    ResampleMaskForRegistrationMatrix16G_long(3, pDstBed, srcSize, dstSize,
        srcSpacing, dstSpacing, I, pBedMask);
    
    return bedRet;
}
```

### 11.4 GetIfDown_fusion —— 大图像判断

```cpp
bool GetIfDown_fusion(const CVolumeDataInfo& img) {
    if (img.uiSize[2] > 1500)  // Z方向超过1500层 → 2m PET-CT
        return true;
    return false;
}
```

### 11.5 CenterAlign —— 中心对齐矩阵

```cpp
void CenterAlign(const CVolumeDataInfo& refImg, const CVolumeDataInfo& movImg, 
                 double retMat[16])
{
    // 1. 转轴位
    CVolumeDataInfo refAxiImg, movAxiImg;
    double axi2ref[16], axi2mov[16];
    TransformToAxi(refImg, refAxiImg, nullptr, nullptr, axi2ref);
    TransformToAxi(movImg, movAxiImg, nullptr, nullptr, axi2mov);
    
    // 2. 在轴位空间计算中心对齐
    double c_mat[16];
    CenterAlign(refAxiImg.uiSize, refAxiImg.dSpacing,
                movAxiImg.uiSize, movAxiImg.dSpacing, c_mat);
    
    // 3. 组合: retMat = axi2mov × c_mat × axi2ref⁻¹
    double axi2ref_i[16];
    Mcsf::Invert(axi2ref, axi2ref_i, 4, 4);
    double temp[16];
    Mcsf::AlgoFusion::MatrixMul(axi2mov, c_mat, 4, 4, 4, 4, temp);
    Mcsf::AlgoFusion::MatrixMul(temp, axi2ref_i, 4, 4, 4, 4, retMat);
}
```

---

## 12. 各模态参数对比

| 参数 | CT-CT | CT-PET | CT-MR (全身) | CT-MR (脑部) | MR-MR | MR-PET | PET-PET |
|------|-------|--------|-------------|-------------|-------|--------|---------|
| **度量** | MSE (0) | NMI (1) | NMI (1) | NMI (1) | NMI (1) | NMI (1) | 部位识别 |
| **旋转缩放** | 1000 | 1000 | 1000 | 300 | 500 | 500 | N/A |
| **平移缩放** | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | N/A |
| **最大迭代** | 500 | 200 | 500 | 500 | 200 | 500 | N/A |
| **初始步长** | 3 | 3 | 3 | 3 | 2.0 | 3.0 | N/A |
| **松弛因子** | 0.9 | 0.9 | 0.95 | 0.95 | 0.92 | 0.90 | N/A |
| **最小有效采样比** | 5% | 1.25% | 1.25% | 1.25% | 12.5% | 12.5% | N/A |
| **CT预处理** | 去床板+HU[-1024,256] | 去床板+HU[-1024,1024] | 去床板+HU | 去床板+HU+脑分割 | N/A | N/A | N/A |
| **PET预处理** | N/A | 百分位窗位 | N/A | N/A | N/A | 百分位窗位 | N/A |
| **MR预处理** | N/A | N/A | 原始 | 原始 | AutoSampleGrayLevel | AutoSampleGrayLevel | N/A |
| **先验矩阵** | 关键点中位数 | 中心对齐 | 关键点中位数 | 关键点中位数 | 关键点中位数 | 中心对齐 | 部位识别 |
| **采样mask** | 骨骼区域(可选) | CT身体区域 | 无 | 无(阶段2:脑组织) | 无 | 无 | N/A |
| **两阶段** | 否 | 否 | 否 | 是(全局+脑部) | 否 | 否 | 否 |
| **ref/mov切换** | 是 | 否 | 否 | 否 | 是 | 否 | 否 |

### 关键设计决策解读

1. **CT-CT 用 MSE，其他跨模态用 NMI**：CT-CT 是同模态，灰度值线性对应，MSE 更精确且计算更快
2. **CT-CT 最小有效采样比 5%**：同模态配准有效采样多，要求更高
3. **MR-MR 最小有效采样比 12.5%**：MR 图像组织对比度低，需要更多有效采样保证可靠性
4. **脑部 CT-MR 旋转缩放 300**：脑部旋转范围小，缩小缩放因子使优化更精细
5. **脑部两阶段配准**：全身配准提供初始对齐，脑部局部配准精细调整（去除非脑组织干扰）
6. **PET-PET 不用 NMI**：PET-PET 配准主要是 Z 轴对齐问题（不同扫描范围），部位识别比 NMI 更直接有效
7. **CT-PET HU 裁剪 [-1024, 1024]**：CT 骨骼（>1024）在 PET 中无对应信号，裁剪避免干扰 NMI 直方图
8. **CT-CT HU 裁剪 [-1024, 256]**：同模态配准关注软组织对比，骨骼（>256）信息对 MSE 贡献过大需抑制
