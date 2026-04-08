#pragma once
/**
 * @file kmeans_segmentation.h
 * @brief K-means 聚类肿瘤分割
 *
 * 通过将感兴趣区域内的体素灰度值聚成 2 类（背景 / 前景），
 * 以种子点所在聚类作为肿瘤前景输出。
 * 支持可配置的最大迭代次数和收敛阈值。
 */

#include "oncology_types.h"

namespace Onc {

class KMeansSegmentation {
public:
    struct Params {
        int   numClusters  = 2;       ///< 聚类数（肿瘤场景通常为 2）
        int   maxIter      = 100;     ///< 最大迭代次数
        float convergenceThr = 1e-4f; ///< 中心点移动量收敛阈值
        int   roiMargin    = 20;      ///< 在种子点周围取的感兴趣区域边界（体素）
    };

    KMeansSegmentation() = default;
    explicit KMeansSegmentation(const Params& p) : m_params(p) {}

    /**
     * @brief 执行分割
     * @param data       原始图像（short）
     * @param info       图像元信息
     * @param seedPoint  用户给定的种子点（图像坐标）
     * @param result     [out] 前景体素坐标集合
     * @return false 表示失败
     */
    bool segment(const short*        data,
                 const ImageInfo&    info,
                 const Point3i&      seedPoint,
                 SegmentResult&      result);

private:
    Params m_params;

    // 内部：单次聚类迭代
    bool runKMeans(const std::vector<float>& samples,
                   const std::vector<int>&   indices,
                   const ImageInfo&          info,
                   const Point3i&            seedPoint,
                   SegmentResult&            result);
};

} // namespace Onc
