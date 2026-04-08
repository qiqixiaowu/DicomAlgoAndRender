/**
 * @file pet_lesion_segment.cpp
 * @brief PET 分子影像病灶分割实现（Fixed / Percent / Adaptive）
 *
 * 核心辅助函数 RegionGrow3D:
 *   从种子点出发，BFS 区域生长，纳入原始灰度 >= rawThreshold 的体素。
 *
 * Adaptive 模式权重调整:
 *   目标：使分割体积 ≈ 基于 (50%SUVmax) 时的体积 的某比例
 *   通过二分搜索找到对应权重；若首次调用则从 0.5 开始。
 */

#include "pet_lesion_segment.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <queue>
#include <cassert>
#include <limits>

namespace Onc {

// ─────────────────────────────────────────────
//  内部工具：读取任意类型的原始灰度值
// ─────────────────────────────────────────────
static float ReadRaw(const void* data, const std::string& imgType, int idx)
{
    if (imgType == "MET_SHORT" || imgType.empty())
        return static_cast<float>(reinterpret_cast<const short*>(data)[idx]);
    else if (imgType == "MET_USHORT")
        return static_cast<float>(reinterpret_cast<const unsigned short*>(data)[idx]);
    else if (imgType == "MET_FLOAT")
        return reinterpret_cast<const float*>(data)[idx];
    // 默认 unsigned short
    return static_cast<float>(reinterpret_cast<const unsigned short*>(data)[idx]);
}

// ─────────────────────────────────────────────
//  3D BFS 区域生长
// ─────────────────────────────────────────────
static void RegionGrow3D(
    const void*      data,
    const std::string& imgType,
    const ImageInfo& info,
    int              seedX, int seedY, int seedZ,
    float            rawThreshold,
    SegmentResult&   out)
{
    out.clear();
    if (!info.inBounds(seedX, seedY, seedZ)) return;
    int seedIdx = info.linearIndex(seedX, seedY, seedZ);
    if (ReadRaw(data, imgType, seedIdx) < rawThreshold) return;

    const int dx6[] = {1,-1,0,0,0,0};
    const int dy6[] = {0,0,1,-1,0,0};
    const int dz6[] = {0,0,0,0,1,-1};

    int total = info.totalVoxels();
    std::vector<uint8_t> visited(total, 0);
    std::queue<Point3i> q;
    q.push({seedX, seedY, seedZ});
    visited[seedIdx] = 1;

    while (!q.empty()) {
        auto cur = q.front(); q.pop();
        out.push_back(cur);
        for (int d = 0; d < 6; ++d) {
            int nx = cur.x + dx6[d];
            int ny = cur.y + dy6[d];
            int nz = cur.z + dz6[d];
            if (!info.inBounds(nx, ny, nz)) continue;
            int ni = info.linearIndex(nx, ny, nz);
            if (visited[ni]) continue;
            if (ReadRaw(data, imgType, ni) < rawThreshold) continue;
            visited[ni] = 1;
            q.push({nx, ny, nz});
        }
    }
}

// VOI 内区域生长（只在 VOI 区域内扩展）
static void RegionGrowInVOI(
    const void*          data,
    const std::string&   imgType,
    const ImageInfo&     info,
    const SegmentResult& voiCoords,
    float                rawThreshold,
    int                  outMaxCoord[3],
    SegmentResult&       out)
{
    out.clear();
    if (voiCoords.empty()) return;

    // 建 VOI 掩码，同时找 SUVmax 体素
    int total = info.totalVoxels();
    std::vector<uint8_t> voiMask(total, 0);
    float maxRaw = -1.f;
    int   maxIdx = info.linearIndex(voiCoords[0].x, voiCoords[0].y, voiCoords[0].z);

    for (const auto& p : voiCoords) {
        if (!info.inBounds(p.x, p.y, p.z)) continue;
        int idx = info.linearIndex(p.x, p.y, p.z);
        voiMask[idx] = 1;
        float v = ReadRaw(data, imgType, idx);
        if (v > maxRaw) { maxRaw = v; maxIdx = idx; }
    }

    // 还原最大值坐标
    int mz = maxIdx / info.sliceSize();
    int mr = maxIdx % info.sliceSize();
    int my = mr / info.dim[0];
    int mx = mr % info.dim[0];
    outMaxCoord[0] = mx; outMaxCoord[1] = my; outMaxCoord[2] = mz;

    RegionGrow3D(data, imgType, info, mx, my, mz, rawThreshold, out);

    // 过滤只保留 VOI 内的体素
    SegmentResult filtered;
    filtered.reserve(out.size());
    for (const auto& p : out) {
        if (!info.inBounds(p.x, p.y, p.z)) continue;
        if (voiMask[info.linearIndex(p.x, p.y, p.z)])
            filtered.push_back(p);
    }
    out = std::move(filtered);
}

// ─────────────────────────────────────────────
//  寻找种子点邻域 SUVmax 对应的 raw 值
// ─────────────────────────────────────────────
static float FindLocalMaxRaw(
    const void* data, const std::string& imgType,
    const ImageInfo& info, int sx, int sy, int sz, int searchR = 5)
{
    float maxRaw = -std::numeric_limits<float>::max();
    int x0 = std::max(0, sx-searchR), x1 = std::min(info.dim[0]-1, sx+searchR);
    int y0 = std::max(0, sy-searchR), y1 = std::min(info.dim[1]-1, sy+searchR);
    int z0 = std::max(0, sz-searchR), z1 = std::min(info.dim[2]-1, sz+searchR);
    for (int z = z0; z <= z1; ++z)
    for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x) {
        float v = ReadRaw(data, imgType, info.linearIndex(x,y,z));
        if (v > maxRaw) maxRaw = v;
    }
    return maxRaw;
}

// ═════════════════════════════════════════════
//  Fixed 模式
// ═════════════════════════════════════════════
bool PETSegmentFixed(
    const SUVInfo&    suvInfo,
    const void*       data,
    const ImageInfo&  info,
    const int         seedPoint[3],
    double            fixedSUV,
    PETSegmentResult& result)
{
    if (!data) return false;
    double rawThresh = suvInfo.hasValidSUV
        ? suvInfo.suvToRaw(fixedSUV)
        : fixedSUV; // 无 SUV 时直接用原始值
    result.usedThreshold = rawThresh;
    RegionGrow3D(data, suvInfo.ImgType, info,
                 seedPoint[0], seedPoint[1], seedPoint[2],
                 static_cast<float>(rawThresh), result.voxels);
    return !result.voxels.empty();
}

bool PETSegmentFixedWithVOI(
    const SUVInfo&       suvInfo,
    const void*          data,
    const ImageInfo&     info,
    const SegmentResult& voiCoords,
    double               fixedSUV,
    PETSegmentResult&    result)
{
    if (!data) return false;
    double rawThresh = suvInfo.hasValidSUV ? suvInfo.suvToRaw(fixedSUV) : fixedSUV;
    result.usedThreshold = rawThresh;
    RegionGrowInVOI(data, suvInfo.ImgType, info, voiCoords,
                    static_cast<float>(rawThresh), result.maxCoord, result.voxels);
    return !result.voxels.empty();
}

// ═════════════════════════════════════════════
//  Percent 模式
// ═════════════════════════════════════════════
bool PETSegmentPercent(
    const SUVInfo&    suvInfo,
    const void*       data,
    const ImageInfo&  info,
    const int         seedPoint[3],
    double            percentOfMax,
    PETSegmentResult& result)
{
    if (!data) return false;
    float rawMax = FindLocalMaxRaw(data, suvInfo.ImgType, info,
                                   seedPoint[0], seedPoint[1], seedPoint[2]);
    float rawThresh = rawMax * static_cast<float>(percentOfMax);
    result.usedThreshold = rawThresh;
    RegionGrow3D(data, suvInfo.ImgType, info,
                 seedPoint[0], seedPoint[1], seedPoint[2],
                 rawThresh, result.voxels);
    return !result.voxels.empty();
}

bool PETSegmentPercentWithVOI(
    const SUVInfo&       suvInfo,
    const void*          data,
    const ImageInfo&     info,
    const SegmentResult& voiCoords,
    double               percentOfMax,
    PETSegmentResult&    result)
{
    if (!data || voiCoords.empty()) return false;

    // 先在 VOI 内找全局最大值
    float maxRaw = -std::numeric_limits<float>::max();
    for (const auto& p : voiCoords) {
        if (!info.inBounds(p.x, p.y, p.z)) continue;
        float v = ReadRaw(data, suvInfo.ImgType, info.linearIndex(p.x,p.y,p.z));
        maxRaw = std::max(maxRaw, v);
    }
    float rawThresh = maxRaw * static_cast<float>(percentOfMax);
    result.usedThreshold = rawThresh;
    RegionGrowInVOI(data, suvInfo.ImgType, info, voiCoords,
                    rawThresh, result.maxCoord, result.voxels);
    return !result.voxels.empty();
}

// ═════════════════════════════════════════════
//  Adaptive 模式
//  原理：二分搜索 weight ∈ [0.1, 0.9]，
//        目标函数：分割结果体积在 [参考最小体积, 参考最大体积] 范围内
//        且分割体积随 weight 减小而增大，具有单调性。
// ═════════════════════════════════════════════
static bool AdaptiveCore(
    const SUVInfo&    suvInfo,
    const void*       data,
    const ImageInfo&  info,
    int               sx, int sy, int sz,
    double&           weight,
    PETSegmentResult& result)
{
    float rawMax = FindLocalMaxRaw(data, suvInfo.ImgType, info, sx, sy, sz);

    // 参考：50%SUVmax 时的体积作为基准
    float rawRef50  = rawMax * 0.5f;
    SegmentResult ref50;
    RegionGrow3D(data, suvInfo.ImgType, info, sx, sy, sz, rawRef50, ref50);

    // 目标：找到 weight 使分割体积 ≈ ref50.size() * 0.9（略保守）
    size_t targetVol = static_cast<size_t>(ref50.size() * 0.9);
    if (targetVol < 5) targetVol = 5;

    // 二分搜索 weight ∈ [0.1, 0.9]
    double wLow = 0.1, wHigh = 0.9;
    for (int iter = 0; iter < 15; ++iter) {
        double wMid = (wLow + wHigh) * 0.5;
        float rawThr = rawMax * static_cast<float>(wMid);
        SegmentResult tmp;
        RegionGrow3D(data, suvInfo.ImgType, info, sx, sy, sz, rawThr, tmp);
        if (tmp.size() > targetVol) wLow  = wMid;
        else                        wHigh = wMid;
    }
    weight = (wLow + wHigh) * 0.5;

    float finalThr = rawMax * static_cast<float>(weight);
    result.usedThreshold = finalThr;
    RegionGrow3D(data, suvInfo.ImgType, info, sx, sy, sz, finalThr, result.voxels);
    return !result.voxels.empty();
}

bool PETSegmentAdaptive(
    const SUVInfo&    suvInfo,
    const void*       data,
    const ImageInfo&  info,
    const int         seedPoint[3],
    double&           weight,
    PETSegmentResult& result)
{
    if (!data) return false;
    return AdaptiveCore(suvInfo, data, info,
                        seedPoint[0], seedPoint[1], seedPoint[2],
                        weight, result);
}

bool PETSegmentAdaptiveWithVOI(
    const SUVInfo&       suvInfo,
    const void*          data,
    const ImageInfo&     info,
    const SegmentResult& voiCoords,
    double&              weight,
    PETSegmentResult&    result)
{
    if (!data || voiCoords.empty()) return false;

    // 找 VOI 内 SUVmax 体素
    float maxRaw = -std::numeric_limits<float>::max();
    Point3i maxPt = voiCoords[0];
    for (const auto& p : voiCoords) {
        if (!info.inBounds(p.x, p.y, p.z)) continue;
        float v = ReadRaw(data, suvInfo.ImgType, info.linearIndex(p.x,p.y,p.z));
        if (v > maxRaw) { maxRaw = v; maxPt = p; }
    }
    result.maxCoord[0] = maxPt.x;
    result.maxCoord[1] = maxPt.y;
    result.maxCoord[2] = maxPt.z;
    return AdaptiveCore(suvInfo, data, info, maxPt.x, maxPt.y, maxPt.z, weight, result);
}

// ═════════════════════════════════════════════
//  Hover 模式（实时轻量预览，简化版自适应）
// ═════════════════════════════════════════════
bool PETSegmentAdaptiveHover(
    const SUVInfo&    suvInfo,
    const void*       data,
    const ImageInfo&  info,
    const int         hoverPoint[3],
    double&           weight,
    PETSegmentResult& result)
{
    // Hover 模式：直接用当前 weight 快速分割，不做二分搜索（保证实时性）
    if (!data) return false;
    if (!info.inBounds(hoverPoint[0], hoverPoint[1], hoverPoint[2])) return false;

    float rawMax = FindLocalMaxRaw(data, suvInfo.ImgType, info,
                                   hoverPoint[0], hoverPoint[1], hoverPoint[2], 3);
    float rawThr = rawMax * static_cast<float>(weight);
    result.usedThreshold = rawThr;
    RegionGrow3D(data, suvInfo.ImgType, info,
                 hoverPoint[0], hoverPoint[1], hoverPoint[2],
                 rawThr, result.voxels);
    return true; // Hover 允许"空分割"（不报错，前端直接清除预览）
}

} // namespace Onc
