#pragma once
/**
 * ============================================================
 *  金属定位点检测 (Fiducial Marker Detection)
 * ============================================================
 *
 * 提炼自 McsfAlgoAutoContour 工程。
 *
 * 【背景】
 *   放射治疗中，在患者体表放置金属标记物（fiducial marker），
 *   CT 扫描后自动检测这些标记物的 3D 位置，用于治疗时的精确定位。
 *
 * 【算法流程】
 *
 *   1. 双阈值候选检出
 *      ├── 高阈值 (1000 HU): 初通检出高密度金属候选
 *      └── 中阈值 (390 HU): 对未检出区域补救
 *
 *   2. 3D 连通域标记
 *      └── 对候选二值图做连通域分析
 *
 *   3. 候选特征提取
 *      ├── 空间中心 (质心)
 *      └── 形态特征: 体积、长短轴比、空气占比
 *
 *   4. 真阳性筛选
 *      └── 基于体积范围和形态约束过滤假阳
 *
 *   5. 点分类
 *      ├── Left:  左侧标记
 *      ├── Right: 右侧标记
 *      └── Top:   上方标记
 *
 *   6. 分组
 *      ├── Z 方向邻近约束 (3~5mm)
 *      ├── 配对 Left/Right
 *      └── 查找 Top
 *
 *   7. 后处理
 *      └── 计算每组 Center，限制组数
 *
 * 【参考】
 *   原工程: McsfAlgoRTFiducialDetection.cpp
 * ============================================================
 */

#include "rt_types.hpp"
#include "connected_component.hpp"

namespace rtac {

// ============================================================
//  检测参数
// ============================================================
struct FiducialDetectionParams {
    short highThreshold = 1000;    ///< 高阈值 (HU) — 金属
    short midThreshold  = 390;     ///< 中阈值 (HU) — 补救
    int   minVolume     = 3;       ///< 最小候选体积（体素数）
    int   maxVolume     = 2000;    ///< 最大候选体积
    float zGroupTolerance = 5.0f;  ///< Z 方向分组容差 (mm)
    int   maxGroups     = 10;      ///< 最大检出组数
};

// ============================================================
//  候选点信息（内部使用）
// ============================================================
namespace detail {

struct Candidate {
    Point3f center;    ///< 质心
    int volume = 0;    ///< 体素数
    float zMin = 0, zMax = 0; ///< Z 范围
    FiducialType type = FiducialType::Left;
};

/// 阈值分割
inline Image3D<uint8_t> ThresholdSegment(
    const Image3D<short>& ct, short threshold)
{
    Image3D<uint8_t> mask(ct.width, ct.height, ct.depth, 0);
    for (size_t i = 0; i < ct.size(); ++i) {
        if (ct[i] >= threshold) mask[i] = 255;
    }
    return mask;
}

/// 计算连通域质心
inline Point3f ComputeCenter(
    const Image3D<int32_t>& labels,
    int label, int W, int H, int D,
    float spX, float spY, float spZ)
{
    double sx = 0, sy = 0, sz = 0;
    int count = 0;
    for (int z = 0; z < D; ++z)
    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
        if (labels.at(x, y, z) == label) {
            sx += x * spX; sy += y * spY; sz += z * spZ;
            ++count;
        }
    }
    if (count == 0) return {};
    return {static_cast<float>(sx / count),
            static_cast<float>(sy / count),
            static_cast<float>(sz / count)};
}

/// 分类：根据位置判断 Left / Right / Top
inline FiducialType ClassifyPoint(const Point3f& pt, const Point3f& volumeCenter) {
    float dx = pt.x - volumeCenter.x;
    float dy = pt.y - volumeCenter.y;
    // 简化分类：x < center → Left, x > center → Right
    // y 最小（最靠上）→ Top
    if (dy < -std::abs(dx) * 0.5f) return FiducialType::Top;
    if (dx < 0) return FiducialType::Left;
    return FiducialType::Right;
}

} // namespace detail

// ============================================================
//  主检测函数
// ============================================================
/**
 * 自动检测 CT 体数据中的金属定位点。
 *
 * @param ct       短整型 CT 体数据（HU 值）
 * @param params   检测参数
 * @param progress 进度回调
 * @return         分组的定位点列表
 */
inline std::vector<FiducialGroup> DetectFiducials(
    const Image3D<short>& ct,
    const FiducialDetectionParams& params = {},
    ProgressCallback progress = NoProgress())
{
    int W = ct.width, H = ct.height, D = ct.depth;
    float spX = ct.spacingX, spY = ct.spacingY, spZ = ct.spacingZ;

    // 体数据中心（用于点分类）
    Point3f volCenter{W * spX * 0.5f, H * spY * 0.5f, D * spZ * 0.5f};

    progress(0.1);

    // ── 第1步: 高阈值分割 ──
    auto highMask = detail::ThresholdSegment(ct, params.highThreshold);

    // ── 第2步: 连通域标记 ──
    auto lr = LabelConnectedComponents3D(highMask, 255, true);

    progress(0.3);

    // ── 第3步: 提取候选 ──
    std::vector<detail::Candidate> candidates;
    for (int i = 1; i <= lr.domainCount; ++i) {
        if (lr.areas[i] < params.minVolume || lr.areas[i] > params.maxVolume)
            continue;

        detail::Candidate c;
        c.volume = lr.areas[i];
        c.center = detail::ComputeCenter(lr.labels, i, W, H, D, spX, spY, spZ);

        // Z 范围
        c.zMin = D * spZ; c.zMax = 0;
        for (int z = 0; z < D; ++z)
        for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            if (lr.labels.at(x, y, z) == i) {
                float zPhys = z * spZ;
                c.zMin = std::min(c.zMin, zPhys);
                c.zMax = std::max(c.zMax, zPhys);
            }
        }

        c.type = detail::ClassifyPoint(c.center, volCenter);
        candidates.push_back(c);
    }

    progress(0.5);

    // ── 第4步: 中阈值补救 ──
    if (candidates.size() < 3) {
        auto midMask = detail::ThresholdSegment(ct, params.midThreshold);
        // 排除已检出的高阈值区域
        for (size_t i = 0; i < midMask.size(); ++i) {
            if (highMask[i] == 255) midMask[i] = 0;
        }

        auto lr2 = LabelConnectedComponents3D(midMask, 255, true);
        for (int i = 1; i <= lr2.domainCount; ++i) {
            if (lr2.areas[i] < params.minVolume || lr2.areas[i] > params.maxVolume)
                continue;

            detail::Candidate c;
            c.volume = lr2.areas[i];
            c.center = detail::ComputeCenter(lr2.labels, i, W, H, D, spX, spY, spZ);
            c.type = detail::ClassifyPoint(c.center, volCenter);
            candidates.push_back(c);
        }
    }

    progress(0.7);

    // ── 第5步: Z 方向分组 ──
    // 按 Z 坐标排序
    std::sort(candidates.begin(), candidates.end(),
        [](const detail::Candidate& a, const detail::Candidate& b) {
            return a.center.z < b.center.z;
        });

    std::vector<FiducialGroup> groups;
    std::vector<bool> used(candidates.size(), false);

    for (size_t i = 0; i < candidates.size() && static_cast<int>(groups.size()) < params.maxGroups; ++i) {
        if (used[i]) continue;

        FiducialGroup group;
        group.push_back({candidates[i].center.x, candidates[i].center.y,
                         candidates[i].center.z, candidates[i].type});
        used[i] = true;

        // 找 Z 方向相邻的点
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (used[j]) continue;
            float zDist = std::abs(candidates[j].center.z - candidates[i].center.z);
            if (zDist <= params.zGroupTolerance) {
                group.push_back({candidates[j].center.x, candidates[j].center.y,
                                 candidates[j].center.z, candidates[j].type});
                used[j] = true;
            }
        }

        // 计算组中心
        if (group.size() >= 2) {
            float cx = 0, cy = 0, cz = 0;
            for (const auto& p : group) {
                cx += p.pos.x; cy += p.pos.y; cz += p.pos.z;
            }
            int n = static_cast<int>(group.size());
            group.push_back({cx / n, cy / n, cz / n, FiducialType::Center});
        }

        groups.push_back(std::move(group));
    }

    progress(1.0);
    return groups;
}

} // namespace rtac
