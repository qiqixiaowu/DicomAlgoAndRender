#pragma once
/**
 * @file lung_nodule_segment_traditional.h
 * @brief CT 肺结节传统分割算法接口
 *
 * 分割流程
 * ─────────
 * 1. 胸腔提取（ChestWallRemoval）
 *    利用阈值（-400 HU）+ 形态学操作提取肺野掩码，
 *    消除胸壁/脊柱干扰。
 *
 * 2. 感兴趣区域（Bounding Box）提取
 *    以种子点为中心，根据结节估计半径取 ROI。
 *
 * 3. Hessian 血管增强
 *    对 ROI 内图像做多尺度 Hessian 滤波，
 *    获得点状增强响应图（hdot），用于区分血管与结节。
 *
 * 4. 初始化分割（InitialSegment）
 *    基于 HU 阈值（-400 HU）+ 连通区域标记出肺组织候选区。
 *
 * 5. 精化（Refine）——多形态结节适配
 *    - 实性结节（Solid）：   Hessian 导引的区域生长
 *    - 贴壁结节（JuxtaWall）：先分离胸壁，再 Active Contour 精化
 *    - 血管附着（JuxtaVessel）：Hessian 管状响应抑制血管，只保留结节
 *    - 磨玻璃（GGO）：        低阈值 + 梯度一致性过滤
 *
 * 6. 凸包平滑
 *    对分割结果取 3D 凸包，消除不规则边界伪影。
 */

#include "oncology_types.h"
#include "hessian_filter.h"
#include <vector>

namespace Onc {

/** 结节形态类型 */
enum class NoduleType {
    Auto,          ///< 自动判断
    Solid,         ///< 实性结节
    JuxtaWall,     ///< 贴壁结节
    JuxtaVessel,   ///< 血管附着结节
    GGO,           ///< 磨玻璃结节
};

struct LungNoduleSegParams {
    NoduleType  noduleType     = NoduleType::Auto;
    float       lungHUThresh   = -400.f; ///< 肺野 HU 阈值
    float       noduleHUMin    = -100.f; ///< 结节最低 HU
    float       noduleHUMax    = 400.f;  ///< 结节最高 HU
    int         roiRadiusMM    = 30;     ///< ROI 半径（mm）
    float       hessianScale1  = 1.5f;   ///< Hessian 最小尺度
    float       hessianScale2  = 3.0f;   ///< Hessian 最大尺度
    bool        enableConvexHull = true; ///< 是否做凸包平滑后处理
};

class LungNoduleSegmentTraditional {
public:
    explicit LungNoduleSegmentTraditional(const LungNoduleSegParams& p = {});

    /**
     * @brief 执行肺结节分割
     * @param data       CT 图像（short，HU + 1024 或标准 HU）
     * @param info       图像元信息
     * @param seedPoint  用户点击的种子点（结节内部）
     * @param result     [out] 分割结果体素集合
     * @param progress   进度回调
     * @return false 表示失败
     */
    bool segment(const short*      data,
                 const ImageInfo&  info,
                 const Point3i&    seedPoint,
                 SegmentResult&    result,
                 ProgressCallback  progress = nullptr);

private:
    LungNoduleSegParams m_params;

    // ── 各阶段函数 ─────────────────────────────────

    /** 提取肺野掩码（0=胸壁/外部，1=肺野） */
    bool extractLungMask(const short*      data,
                         const ImageInfo&  info,
                         std::vector<uint8_t>& lungMask);

    /** 在 ROI 内用阈值+连通标记初始化分割 */
    bool initialSegment(const short*            data,
                        const ImageInfo&        info,
                        const std::vector<uint8_t>& lungMask,
                        const Point3i&          seedPoint,
                        SegmentResult&          coarseResult);

    /** Hessian 引导的区域生长精化 */
    bool refineWithHessian(const short*           data,
                           const ImageInfo&       info,
                           const SegmentResult&   coarse,
                           const Point3i&         seedPoint,
                           SegmentResult&         refined);

    /** 贴壁结节精化（分离胸壁后再精化） */
    bool refineJuxtaWall(const short*           data,
                         const ImageInfo&       info,
                         const std::vector<uint8_t>& lungMask,
                         const SegmentResult&   coarse,
                         const Point3i&         seedPoint,
                         SegmentResult&         refined);

    /** 血管附着结节精化（用 Hessian 管状响应过滤血管） */
    bool refineJuxtaVessel(const short*           data,
                           const ImageInfo&       info,
                           const SegmentResult&   coarse,
                           const Point3i&         seedPoint,
                           SegmentResult&         refined);

    /** 判断结节类型（Auto 模式） */
    NoduleType detectNoduleType(const short*      data,
                                const ImageInfo&  info,
                                const Point3i&    seed,
                                const std::vector<uint8_t>& lungMask) const;

    /** 3D 凸包平滑 */
    void applyConvexHull(SegmentResult& inout, const ImageInfo& info);
};

} // namespace Onc
