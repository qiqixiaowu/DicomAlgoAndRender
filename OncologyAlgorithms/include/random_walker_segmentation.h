#pragma once
/**
 * @file random_walker_segmentation.h
 * @brief Random Walker 概率扩散分割
 *
 * 原理（Grady 2006）
 * ──────────────────
 * 将图像体素建模为加权图：边权 w(i,j) = exp(-β·(I_i - I_j)²)
 *
 * 给定前景种子（标签=1）和背景种子（标签=0），
 * 求解拉普拉斯方程：
 *     L_U · x = -B_U^T · b
 * 其中 L_U 为未标记体素的子拉普拉斯矩阵，
 * 解 x 为各体素属于前景的概率。
 * 概率 >= 0.5 判为前景。
 *
 * 实现简化
 * ─────────
 * 精确解法需要稀疏线性求解，本实现使用高斯-赛德尔（Gauss-Seidel）
 * 迭代法，在ROI内迭代至收敛，适用于中小型结节（≤100³体素）。
 */

#include "oncology_types.h"

namespace Onc {

class RandomWalkerSegmentation {
public:
    struct Params {
        float beta    = 90.f;   ///< 边权灵敏度（越大灰度差异越敏感）
        int   maxIter = 200;    ///< Gauss-Seidel 最大迭代次数
        float convThr = 1e-3f;  ///< 收敛阈值
        int   roiMargin = 20;   ///< ROI 边距（体素）
        float fgThreshold = 0.5f; ///< 概率阈值判前景
    };

    RandomWalkerSegmentation() = default;
    explicit RandomWalkerSegmentation(const Params& p) : m_params(p) {}

    /**
     * @brief 执行 Random Walker 分割
     * @param data       原始图像（short）
     * @param info       图像元信息
     * @param ldStart    长轴线段起点（用户交互画线，作为前景种子区域）
     * @param ldEnd      长轴线段终点
     * @param drawDir    绘制方向
     * @param result     [out] 分割结果
     * @return false 表示失败
     */
    bool segment(const short*     data,
                 const ImageInfo& info,
                 const Point3i&   ldStart,
                 const Point3i&   ldEnd,
                 DrawDirection    drawDir,
                 SegmentResult&   result);

private:
    Params m_params;

    /** 从线段采样前景种子点 */
    std::vector<Point3i> sampleLineForeground(
        const Point3i& start, const Point3i& end) const;

    /** 计算边权 */
    float edgeWeight(float vi, float vj) const;
};

} // namespace Onc
