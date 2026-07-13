#pragma once
/**
 * ============================================================
 *  CT ResetMatch — 切片匹配
 * ============================================================
 *
 * 提炼自 McsfAlgoAutoContour 工程。
 *
 * 【背景】
 *   放射治疗中，将计划 CT 某一层图像与每日重建 CT 体数据匹配，
 *   找到最相似的 Z 切片位置，用于验证患者摆位是否一致。
 *
 * 【算法原理】
 *   1. 对计划 CT 层和重建 CT 各层做阈值分割（300 HU）提取骨结构
 *   2. 对骨结构做投影：
 *      - 行投影 rowProj[y] = Σ_x mask(x,y)
 *      - 列投影 colProj[x] = Σ_y mask(x,y)
 *   3. 计算计划层投影与重建 CT 各层投影的归一化互相关（NCC）
 *   4. NCC 最大的 Z 切片 = 最佳匹配
 *
 * 【归一化互相关 (NCC)】
 *   NCC(a, b) = Σ(a_i - ā)(b_i - b̄) / [sqrt(Σ(a_i-ā)²) * sqrt(Σ(b_i-b̄)²)]
 *   范围: [-1, 1]，1 = 完全相关
 *
 * 【参考】
 *   原工程: McsfAlgoAutoContourBasicFunction.cpp — ResetMatch
 * ============================================================
 */

#include "rt_types.hpp"

namespace rtac {

// ============================================================
//  参数
// ============================================================
struct ResetMatchParams {
    short boneThreshold = 300;     ///< 骨结构阈值 (HU)
    int   searchRange   = -1;      ///< 搜索范围（层数，-1 = 全范围）
    bool  useRowProj    = true;    ///< 使用行投影
    bool  useColProj    = true;    ///< 使用列投影
};

// ============================================================
//  结果
// ============================================================
struct ResetMatchResult {
    int   bestSlice = -1;           ///< 最佳匹配切片号
    float bestNCC   = -1.f;         ///< 最佳 NCC 值
    std::vector<float> nccCurve;    ///< 每层的 NCC 值
};

// ============================================================
//  辅助函数
// ============================================================
namespace detail {

/// 计算一层骨结构的行投影和列投影
inline void ComputeProjections(
    const short* slice, int W, int H, short threshold,
    std::vector<float>& rowProj,
    std::vector<float>& colProj)
{
    rowProj.assign(H, 0.f);
    colProj.assign(W, 0.f);

    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
        if (slice[y * W + x] >= threshold) {
            rowProj[y] += 1.f;
            colProj[x] += 1.f;
        }
    }
}

/// 归一化互相关
inline float NormalizedCrossCorrelation(
    const std::vector<float>& a,
    const std::vector<float>& b)
{
    if (a.size() != b.size() || a.empty()) return 0.f;

    int n = static_cast<int>(a.size());
    double meanA = 0, meanB = 0;
    for (int i = 0; i < n; ++i) { meanA += a[i]; meanB += b[i]; }
    meanA /= n; meanB /= n;

    double num = 0, denA = 0, denB = 0;
    for (int i = 0; i < n; ++i) {
        double da = a[i] - meanA;
        double db = b[i] - meanB;
        num += da * db;
        denA += da * da;
        denB += db * db;
    }

    double den = std::sqrt(denA * denB);
    return (den > 1e-12) ? static_cast<float>(num / den) : 0.f;
}

/// 合并行投影和列投影为单一向量
inline std::vector<float> CombineProjections(
    const std::vector<float>& rowProj,
    const std::vector<float>& colProj,
    bool useRow, bool useCol)
{
    std::vector<float> combined;
    if (useRow)
        combined.insert(combined.end(), rowProj.begin(), rowProj.end());
    if (useCol)
        combined.insert(combined.end(), colProj.begin(), colProj.end());
    return combined;
}

} // namespace detail

// ============================================================
//  主接口
// ============================================================
/**
 * 将计划 CT 单层与重建 CT 体数据匹配，找最相似的切片。
 *
 * @param planSlice   计划 CT 单层（short 数据，W×H）
 * @param W, H        层面尺寸
 * @param reconVolume 重建 CT 体数据
 * @param params      匹配参数
 * @return            匹配结果
 */
inline ResetMatchResult ResetMatch(
    const short* planSlice, int W, int H,
    const Image3D<short>& reconVolume,
    const ResetMatchParams& params = {})
{
    ResetMatchResult result;
    int D = reconVolume.depth;

    // 计划层投影
    std::vector<float> planRow, planCol;
    detail::ComputeProjections(planSlice, W, H, params.boneThreshold, planRow, planCol);
    auto planProj = detail::CombineProjections(planRow, planCol,
                                                params.useRowProj, params.useColProj);

    // 搜索范围
    int searchStart = 0, searchEnd = D;
    if (params.searchRange > 0 && params.searchRange < D) {
        int mid = D / 2;
        searchStart = std::max(0, mid - params.searchRange / 2);
        searchEnd   = std::min(D, mid + params.searchRange / 2);
    }

    result.nccCurve.resize(D, 0.f);
    result.bestNCC = -1.f;
    result.bestSlice = searchStart;

    for (int z = searchStart; z < searchEnd; ++z) {
        // 重建层投影
        std::vector<float> reconRow, reconCol;
        detail::ComputeProjections(
            reconVolume.data.data() + z * static_cast<size_t>(W) * H,
            W, H, params.boneThreshold, reconRow, reconCol);
        auto reconProj = detail::CombineProjections(reconRow, reconCol,
                                                     params.useRowProj, params.useColProj);

        float ncc = detail::NormalizedCrossCorrelation(planProj, reconProj);
        result.nccCurve[z] = ncc;

        if (ncc > result.bestNCC) {
            result.bestNCC = ncc;
            result.bestSlice = z;
        }
    }

    return result;
}

/**
 * 便捷版本：输入为 Image2D 和 Image3D。
 */
inline ResetMatchResult ResetMatch(
    const Image2D<short>& planSlice,
    const Image3D<short>& reconVolume,
    const ResetMatchParams& params = {})
{
    return ResetMatch(planSlice.data.data(), planSlice.width, planSlice.height,
                      reconVolume, params);
}

} // namespace rtac
