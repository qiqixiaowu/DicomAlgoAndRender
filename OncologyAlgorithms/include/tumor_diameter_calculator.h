#pragma once
/**
 * @file tumor_diameter_calculator.h
 * @brief 肿瘤径线计算与统计参数接口
 *
 * 提供三类核心功能：
 *  1. CalculateTumorDiameters  —— 2D最大截面长径/垂直径 + 3D最长径
 *  2. CalculateStatistics      —— 体积(mm³)、均值、标准差、最大/最小值
 *  3. （跨时间点传播辅助函数见 CrossTimePointUtils）
 *
 * 算法原理
 * ─────────
 * 2D 最大截面：
 *   遍历所有 Z 切片，在包含体素数最多的横断位切片上，
 *   对轮廓点集执行旋转卡壳(Rotating Calipers)算法，
 *   找到最大跨度方向（长径）及其垂直方向（短径）。
 *
 * 3D 最长径：
 *   从分割集合中随机采样若干点对，计算欧氏距离，
 *   取最大值作为 3D 最长径的近似。对于精确计算可使用
 *   凸包 + 旋转卡壳，但耗时较长。超过 MaxCalTimeMs 时返回 -1。
 */

#include "oncology_types.h"
#include <chrono>
#include <map>

namespace Onc {

// ─────────────────────────────────────────────
//  结果结构体
// ─────────────────────────────────────────────

/** 肿瘤径线计算结果 */
struct TumorDiameterResult {
    float longestDiameter3D_cm   = -1.f; ///< 3D最长径，单位cm；超时则为-1
    float longestDiameter2D_cm   = 0.f;  ///< 最大截面层长径，单位cm
    float perpendicularDiameter_cm = 0.f;///< 最大截面层垂直径，单位cm

    Point3f ldStart;   ///< 长径起点（图像体素坐标）
    Point3f ldEnd;     ///< 长径终点
    Point3f pdStart;   ///< 垂直径起点
    Point3f pdEnd;     ///< 垂直径终点
};

/** 统计学参数结果 */
struct TumorStatistics {
    float volume_mm3 = 0.f; ///< 体积，mm³
    float mean       = 0.f; ///< 平均灰度/SUV
    float stdDev     = 0.f; ///< 标准差
    float minValue   = 0.f; ///< 最小灰度/SUV
    float maxValue   = 0.f; ///< 最大灰度/SUV
};

// ─────────────────────────────────────────────
//  接口函数
// ─────────────────────────────────────────────

/**
 * @brief 计算肿瘤 2D 最大截面长径/垂直径，以及 3D 最长径
 *
 * @param segPoints  分割结果体素坐标集合（图像坐标系）
 * @param info       图像维度与间距信息
 * @param result     [out] 计算结果
 * @param maxCalTimeMs  3D最长径计算最大允许时间（毫秒），超时则 longestDiameter3D_cm=-1
 * @param progress   进度回调（可为 nullptr）
 * @return false 表示分割结果为空或参数非法
 */
bool CalculateTumorDiameters(
    const SegmentResult&     segPoints,
    const ImageInfo&         info,
    TumorDiameterResult&     result,
    double                   maxCalTimeMs = 30000.0,
    ProgressCallback         progress     = nullptr);

/**
 * @brief 计算肿瘤统计学参数（体积、灰度均值、标准差等）
 *
 * @tparam T      图像数据类型（short / unsigned short / float）
 * @param data    原始图像数据指针（线性排布，z轴慢轴）
 * @param segPoints  分割体素集合
 * @param info    图像元信息
 * @param stats   [out] 统计结果
 * @return false 表示参数非法
 */
template <typename T>
bool CalculateStatistics(
    const T*             data,
    const SegmentResult& segPoints,
    const ImageInfo&     info,
    TumorStatistics&     stats);

// ─────────────────────────────────────────────
//  VOI 综合统计（一站式）
// ─────────────────────────────────────────────

/**
 * @brief 单 ROI 一站式 VOI 统计
 *
 * 参照参考工程 CalculateStatisticInfo_Impl，一次性计算：
 *  - 灰度统计（体素数、均值、标准差、极值、AABB）
 *  - 径线测量（2D长径/垂直径、3D最长径）
 *  - SUV 统计（当 suvInfo != nullptr 时）
 *
 * @tparam T            图像像素类型（short / unsigned short / float）
 * @param data          原始图像数据指针
 * @param info          图像元信息
 * @param segPoints     分割体素坐标集合
 * @param stat          [out] 综合统计结果
 * @param suvInfo       PET SUV 映射参数（CT 可传 nullptr 跳过 SUV 统计）
 * @param calcDiameters 是否计算径线（默认 true，关闭可提速）
 * @param maxCalTimeMs  3D 最长径计算超时（毫秒）
 * @return false 表示参数非法或分割为空
 */
template <typename T>
bool CalculateVOIStatistic(
    const T*             data,
    const ImageInfo&     info,
    const SegmentResult& segPoints,
    VOIStatistic&        stat,
    const SUVInfo*       suvInfo         = nullptr,
    bool                 calcDiameters   = true,
    double               maxCalTimeMs    = 30000.0);

/**
 * @brief 多标签 VOI 统计（从 mask 中同时处理多个 label）
 *
 * 参照参考工程的多 label 遍历逻辑：
 * 在一次扫描中完成所有 label 的灰度统计和坐标收集，
 * 然后为每个 label 分别计算径线和 SUV 统计。
 *
 * @tparam T            图像像素类型
 * @param data          原始图像数据指针
 * @param info          图像元信息
 * @param maskData      标签掩码（与图像等尺寸，0=背景）
 * @param labels        要统计的标签列表
 * @param suvInfo       PET SUV 映射参数（可为 nullptr）
 * @param calcDiameters 是否计算径线
 * @param maxCalTimeMs  3D 最长径计算超时
 * @return label → VOIStatistic 映射
 */
template <typename T>
std::map<uint8_t, VOIStatistic> CalculateMultiLabelVOIStatistics(
    const T*                     data,
    const ImageInfo&             info,
    const uint8_t*               maskData,
    const std::vector<uint8_t>&  labels,
    const SUVInfo*               suvInfo         = nullptr,
    bool                         calcDiameters   = true,
    double                       maxCalTimeMs    = 30000.0);

/**
 * @brief 计算 SUVpeak（1 ml 球体加权平均最大值）
 *
 * 算法：对分割区域的每个体素，以该体素为中心构建约 1 ml 的球形 VOI
 *（半径 ≈ 6.2 mm），对落入球内的体素做 SUV 值加权平均，
 * 取所有体素中最大的平均值作为 SUVpeak。
 *
 * @tparam T           图像像素类型
 * @param data         原始图像数据指针
 * @param info         图像元信息
 * @param segPoints    分割体素坐标集合
 * @param suvInfo      SUV 映射参数
 * @return SUVpeak 值；若计算失败返回 0
 */
template <typename T>
double CalculateSUVpeak(
    const T*             data,
    const ImageInfo&     info,
    const SegmentResult& segPoints,
    const SUVInfo&       suvInfo);

} // namespace Onc

// ── 模板实现放在头文件内 ──────────────────────────
#include <numeric>
#include <cmath>
#include <map>

namespace Onc {

template <typename T>
bool CalculateStatistics(
    const T*             data,
    const SegmentResult& segPoints,
    const ImageInfo&     info,
    TumorStatistics&     stats)
{
    if (!data || segPoints.empty()) return false;
    if (info.dim[0] <= 0 || info.dim[1] <= 0 || info.dim[2] <= 0) return false;
    if (info.spacing[0] <= 0 || info.spacing[1] <= 0 || info.spacing[2] <= 0) return false;

    // 体积 = 体素数 × 单体素体积
    stats.volume_mm3 = static_cast<float>(segPoints.size())
                     * static_cast<float>(info.spacing[0])
                     * static_cast<float>(info.spacing[1])
                     * static_cast<float>(info.spacing[2]);

    // 采集各体素灰度值
    std::vector<float> values;
    values.reserve(segPoints.size());
    for (const auto& p : segPoints) {
        if (!info.inBounds(p.x, p.y, p.z)) continue;
        int idx = info.linearIndex(p.x, p.y, p.z);
        values.push_back(static_cast<float>(data[idx]));
    }
    if (values.empty()) return false;

    float sum = 0.f, sumSq = 0.f;
    float minV =  std::numeric_limits<float>::max();
    float maxV = -std::numeric_limits<float>::max();
    for (float v : values) {
        sum   += v;
        sumSq += v * v;
        minV   = std::min(minV, v);
        maxV   = std::max(maxV, v);
    }
    float n      = static_cast<float>(values.size());
    stats.mean   = sum / n;
    stats.minValue = minV;
    stats.maxValue = maxV;

    float variance = sumSq / n - stats.mean * stats.mean;
    stats.stdDev   = (variance > 0.f) ? std::sqrt(variance) : 0.f;
    return true;
}

// ──────────────────────────────────────────────────────
//  SUVpeak 计算
// ──────────────────────────────────────────────────────

template <typename T>
double CalculateSUVpeak(
    const T*             data,
    const ImageInfo&     info,
    const SegmentResult& segPoints,
    const SUVInfo&       suvInfo)
{
    if (!data || segPoints.empty() || !suvInfo.hasValidSUV)
        return 0.0;

    // 1 ml 球体半径: V = 4/3 * pi * r^3 = 1000 mm^3
    // r = (1000 * 3 / (4 * pi))^(1/3) ≈ 6.2035 mm
    constexpr double SPHERE_VOLUME_ML = 1.0;
    const double radius_mm = std::cbrt(SPHERE_VOLUME_ML * 1000.0 * 3.0 / (4.0 * 3.14159265358979323846));

    // 球内偏移量预计算（相对于中心体素的整数偏移）
    int rx = static_cast<int>(std::ceil(radius_mm / info.spacing[0]));
    int ry = static_cast<int>(std::ceil(radius_mm / info.spacing[1]));
    int rz = static_cast<int>(std::ceil(radius_mm / info.spacing[2]));
    double r2_mm = radius_mm * radius_mm;

    struct SphereOffset { int dx, dy, dz; double contribute; };
    std::vector<SphereOffset> kernel;
    for (int dz = -rz; dz <= rz; ++dz)
    for (int dy = -ry; dy <= ry; ++dy)
    for (int dx = -rx; dx <= rx; ++dx) {
        double px = dx * info.spacing[0];
        double py = dy * info.spacing[1];
        double pz = dz * info.spacing[2];
        double d2 = px*px + py*py + pz*pz;
        if (d2 <= r2_mm)
            kernel.push_back({dx, dy, dz, 1.0});
    }
    if (kernel.empty()) return 0.0;

    // 对分割区域每个体素计算球内平均 SUV
    double peakSUV = -std::numeric_limits<double>::max();

    // 采样以避免超大病灶耗时过长
    constexpr int MAX_SAMPLE = 5000;
    int step = std::max<int>(1, static_cast<int>(segPoints.size()) / MAX_SAMPLE);

    for (int si = 0; si < static_cast<int>(segPoints.size()); si += step) {
        const auto& c = segPoints[si];
        double wSum = 0.0, vSum = 0.0;
        for (const auto& k : kernel) {
            int nx = c.x + k.dx, ny = c.y + k.dy, nz = c.z + k.dz;
            if (!info.inBounds(nx, ny, nz)) continue;
            double rawVal = static_cast<double>(data[info.linearIndex(nx, ny, nz)]);
            double suvVal = suvInfo.rawToSUV(rawVal);
            vSum += suvVal * k.contribute;
            wSum += k.contribute;
        }
        if (wSum > 0.0) {
            double avg = vSum / wSum;
            if (avg > peakSUV) peakSUV = avg;
        }
    }
    return (peakSUV > -1e30) ? peakSUV : 0.0;
}

// ──────────────────────────────────────────────────────
//  单 ROI 一站式 VOI 统计
// ──────────────────────────────────────────────────────

template <typename T>
bool CalculateVOIStatistic(
    const T*             data,
    const ImageInfo&     info,
    const SegmentResult& segPoints,
    VOIStatistic&        stat,
    const SUVInfo*       suvInfo,
    bool                 calcDiameters,
    double               maxCalTimeMs)
{
    if (!data || segPoints.empty()) return false;
    if (info.dim[0] <= 0 || info.dim[1] <= 0 || info.dim[2] <= 0) return false;

    stat = VOIStatistic{};
    size_t layerSize = static_cast<size_t>(info.dim[0]) * info.dim[1];

    double suvSum = 0.0;
    double suvMin =  std::numeric_limits<double>::max();
    double suvMax = -std::numeric_limits<double>::max();

    for (const auto& p : segPoints) {
        if (!info.inBounds(p.x, p.y, p.z)) continue;

        size_t idx = static_cast<size_t>(p.z) * layerSize
                   + static_cast<size_t>(p.y) * info.dim[0] + p.x;
        double val = static_cast<double>(data[idx]);

        stat.elemCount++;
        // 极值和累加
        if (static_cast<int>(val) > stat.maxDcm) {
            stat.maxDcm = static_cast<int>(val);
            stat.maxSlicePositionIndex = idx;
        }
        if (static_cast<int>(val) < stat.minDcm)
            stat.minDcm = static_cast<int>(val);

        stat.sumDcm       += val;
        stat.squareSumDcm += val * val;

        // AABB
        stat.iMinRange[0] = std::min(stat.iMinRange[0], p.x);
        stat.iMinRange[1] = std::min(stat.iMinRange[1], p.y);
        stat.iMinRange[2] = std::min(stat.iMinRange[2], p.z);
        stat.iMaxRange[0] = std::max(stat.iMaxRange[0], p.x);
        stat.iMaxRange[1] = std::max(stat.iMaxRange[1], p.y);
        stat.iMaxRange[2] = std::max(stat.iMaxRange[2], p.z);

        // SUV 统计
        if (suvInfo && suvInfo->hasValidSUV) {
            double sv = suvInfo->rawToSUV(val);
            suvSum += sv;
            suvMin = std::min(suvMin, sv);
            suvMax = std::max(suvMax, sv);
        }
    }

    if (stat.elemCount == 0) return false;

    // 派生统计
    stat.finalize();
    stat.volume_mm3 = static_cast<double>(stat.elemCount)
                    * info.spacing[0] * info.spacing[1] * info.spacing[2];

    // SUV
    if (suvInfo && suvInfo->hasValidSUV) {
        stat.suvMax  = suvMax;
        stat.suvMin  = suvMin;
        stat.suvMean = suvSum / stat.elemCount;
        stat.suvPeak = CalculateSUVpeak(data, info, segPoints, *suvInfo);
    }

    // 径线
    if (calcDiameters) {
        TumorDiameterResult diamResult;
        if (CalculateTumorDiameters(segPoints, info, diamResult, maxCalTimeMs)) {
            stat.longAxis_mm    = diamResult.longestDiameter2D_cm * 10.0;
            stat.shortAxis_mm   = diamResult.perpendicularDiameter_cm * 10.0;
            stat.maxDiameter_mm = (diamResult.longestDiameter3D_cm >= 0.f)
                                ? diamResult.longestDiameter3D_cm * 10.0 : -1.0;
            stat.ptLongAxisStart  = diamResult.ldStart;
            stat.ptLongAxisEnd    = diamResult.ldEnd;
            stat.ptShortAxisStart = diamResult.pdStart;
            stat.ptShortAxisEnd   = diamResult.pdEnd;
        }
    }
    return true;
}

// ──────────────────────────────────────────────────────
//  多标签 VOI 统计
// ──────────────────────────────────────────────────────

template <typename T>
std::map<uint8_t, VOIStatistic> CalculateMultiLabelVOIStatistics(
    const T*                     data,
    const ImageInfo&             info,
    const uint8_t*               maskData,
    const std::vector<uint8_t>&  labels,
    const SUVInfo*               suvInfo,
    bool                         calcDiameters,
    double                       maxCalTimeMs)
{
    std::map<uint8_t, VOIStatistic> result;
    std::map<uint8_t, std::vector<Point3i>> labelPoints;

    if (!data || !maskData || labels.empty()) return result;
    if (info.dim[0] <= 0 || info.dim[1] <= 0 || info.dim[2] <= 0) return result;

    // 快速标签查找表
    bool isNeeded[256] = {};
    for (uint8_t lb : labels) {
        isNeeded[lb] = true;
        result[lb]   = VOIStatistic{};
    }

    // 单趟扫描：收集所有标签的灰度统计 + 体素坐标
    size_t layerSize = static_cast<size_t>(info.dim[0]) * info.dim[1];

    for (int z = 0; z < info.dim[2]; ++z) {
        size_t zOff = static_cast<size_t>(z) * layerSize;
        for (int y = 0; y < info.dim[1]; ++y) {
            size_t yOff = static_cast<size_t>(y) * info.dim[0];
            for (int x = 0; x < info.dim[0]; ++x) {
                size_t idx = zOff + yOff + x;
                uint8_t label = maskData[idx];
                if (!isNeeded[label]) continue;

                auto& stat = result[label];
                double val = static_cast<double>(data[idx]);

                stat.elemCount++;
                if (static_cast<int>(val) > stat.maxDcm) {
                    stat.maxDcm = static_cast<int>(val);
                    stat.maxSlicePositionIndex = idx;
                }
                if (static_cast<int>(val) < stat.minDcm)
                    stat.minDcm = static_cast<int>(val);

                stat.sumDcm       += val;
                stat.squareSumDcm += val * val;

                stat.iMinRange[0] = std::min(stat.iMinRange[0], x);
                stat.iMinRange[1] = std::min(stat.iMinRange[1], y);
                stat.iMinRange[2] = std::min(stat.iMinRange[2], z);
                stat.iMaxRange[0] = std::max(stat.iMaxRange[0], x);
                stat.iMaxRange[1] = std::max(stat.iMaxRange[1], y);
                stat.iMaxRange[2] = std::max(stat.iMaxRange[2], z);

                labelPoints[label].push_back({x, y, z});
            }
        }
    }

    // 逐标签计算派生统计、SUV、径线
    for (uint8_t lb : labels) {
        auto& stat = result[lb];
        if (stat.elemCount == 0) continue;

        stat.finalize();
        stat.volume_mm3 = static_cast<double>(stat.elemCount)
                        * info.spacing[0] * info.spacing[1] * info.spacing[2];

        const auto& pts = labelPoints[lb];

        // SUV
        if (suvInfo && suvInfo->hasValidSUV) {
            double sMin =  std::numeric_limits<double>::max();
            double sMax = -std::numeric_limits<double>::max();
            double sSum = 0.0;
            for (const auto& p : pts) {
                double raw = static_cast<double>(data[info.linearIndex(p.x, p.y, p.z)]);
                double sv  = suvInfo->rawToSUV(raw);
                sSum += sv;
                sMin = std::min(sMin, sv);
                sMax = std::max(sMax, sv);
            }
            stat.suvMax  = sMax;
            stat.suvMin  = sMin;
            stat.suvMean = sSum / stat.elemCount;
            stat.suvPeak = CalculateSUVpeak(data, info, pts, *suvInfo);
        }

        // 径线
        if (calcDiameters) {
            TumorDiameterResult diamResult;
            if (CalculateTumorDiameters(pts, info, diamResult, maxCalTimeMs)) {
                stat.longAxis_mm    = diamResult.longestDiameter2D_cm * 10.0;
                stat.shortAxis_mm   = diamResult.perpendicularDiameter_cm * 10.0;
                stat.maxDiameter_mm = (diamResult.longestDiameter3D_cm >= 0.f)
                                    ? diamResult.longestDiameter3D_cm * 10.0 : -1.0;
                stat.ptLongAxisStart  = diamResult.ldStart;
                stat.ptLongAxisEnd    = diamResult.ldEnd;
                stat.ptShortAxisStart = diamResult.pdStart;
                stat.ptShortAxisEnd   = diamResult.pdEnd;
            }
        }
    }
    return result;
}

} // namespace Onc
