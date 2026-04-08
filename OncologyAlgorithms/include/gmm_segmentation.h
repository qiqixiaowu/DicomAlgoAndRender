#pragma once
/**
 * @file gmm_segmentation.h
 * @brief 高斯混合模型（GMM）EM 算法肿瘤分割
 *
 * 原理
 * ────
 * 假设 ROI 内体素灰度服从 K 个高斯分量的混合分布：
 *   p(x) = Σ_k  π_k · N(x | μ_k, σ_k²)
 *
 * 用 EM 算法迭代：
 *   E-step：计算每个样本属于各分量的后验概率（responsibility）
 *   M-step：更新权重 π_k、均值 μ_k、方差 σ_k²
 *
 * 种子点最重要的分量（或灰度最高分量）对应前景，
 * 输出 responsibility 最高的前景类体素。
 */

#include "oncology_types.h"

namespace Onc {

class GMMSegmentation {
public:
    struct Params {
        int   numComponents = 2;       ///< 高斯分量数，肿瘤场景通常 2~3
        int   maxIter       = 50;      ///< EM 最大迭代次数
        float convergenceThr= 1e-4f;   ///< 对数似然收敛阈值
        int   roiMargin     = 25;      ///< ROI 扩展边距（体素）
        float minSigma      = 1.0f;    ///< 防止方差退化到 0 的下限
    };

    GMMSegmentation() = default;
    explicit GMMSegmentation(const Params& p) : m_params(p) {}

    /**
     * @brief 执行 GMM 分割
     * @param data      原始图像（short）
     * @param info      图像元信息
     * @param seedPoint 种子点（图像坐标）
     * @param result    [out] 前景体素坐标
     * @return false 表示失败
     */
    bool segment(const short*     data,
                 const ImageInfo& info,
                 const Point3i&   seedPoint,
                 SegmentResult&   result);

private:
    Params m_params;
};

} // namespace Onc
