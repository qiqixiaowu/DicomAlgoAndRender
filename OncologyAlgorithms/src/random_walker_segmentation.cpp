/**
 * @file random_walker_segmentation.cpp
 * @brief Random Walker 分割实现（Gauss-Seidel 迭代近似求解）
 */

#include "random_walker_segmentation.h"
#include <cmath>
#include <cassert>
#include <algorithm>
#include <queue>
#include <cstring>

namespace Onc {

// ─────────────────────────────────────────────
//  线段采样工具
// ─────────────────────────────────────────────
std::vector<Point3i> RandomWalkerSegmentation::sampleLineForeground(
    const Point3i& s, const Point3i& e) const
{
    std::vector<Point3i> pts;
    int steps = std::max({std::abs(e.x-s.x), std::abs(e.y-s.y), std::abs(e.z-s.z)});
    if (steps == 0) { pts.push_back(s); return pts; }
    for (int i = 0; i <= steps; ++i) {
        float t = static_cast<float>(i) / steps;
        int x = static_cast<int>(std::round(s.x + t*(e.x-s.x)));
        int y = static_cast<int>(std::round(s.y + t*(e.y-s.y)));
        int z = static_cast<int>(std::round(s.z + t*(e.z-s.z)));
        pts.push_back({x, y, z});
    }
    return pts;
}

// ─────────────────────────────────────────────
//  边权
// ─────────────────────────────────────────────
float RandomWalkerSegmentation::edgeWeight(float vi, float vj) const {
    float diff = vi - vj;
    return std::exp(-m_params.beta * diff * diff / (65536.f * 65536.f));
}

// ─────────────────────────────────────────────
//  主分割逻辑
// ─────────────────────────────────────────────
bool RandomWalkerSegmentation::segment(
    const short*     data,
    const ImageInfo& info,
    const Point3i&   ldStart,
    const Point3i&   ldEnd,
    DrawDirection    /*drawDir*/,
    SegmentResult&   result)
{
    if (!data) return false;

    // ── 确定 ROI ────────────────────────────────
    int mg = m_params.roiMargin;
    int rxMin = std::max(0, std::min(ldStart.x, ldEnd.x) - mg);
    int rxMax = std::min(info.dim[0]-1, std::max(ldStart.x, ldEnd.x) + mg);
    int ryMin = std::max(0, std::min(ldStart.y, ldEnd.y) - mg);
    int ryMax = std::min(info.dim[1]-1, std::max(ldStart.y, ldEnd.y) + mg);
    int rzMin = std::max(0, std::min(ldStart.z, ldEnd.z) - mg);
    int rzMax = std::min(info.dim[2]-1, std::max(ldStart.z, ldEnd.z) + mg);

    int rW = rxMax - rxMin + 1;
    int rH = ryMax - ryMin + 1;
    int rD = rzMax - rzMin + 1;
    int rN = rW * rH * rD;
    if (rN <= 0) return false;

    // ── 建立局部索引映射 ────────────────────────
    auto localIdx = [&](int x, int y, int z) -> int {
        return (z - rzMin)*rW*rH + (y - ryMin)*rW + (x - rxMin);
    };
    auto localInBounds = [&](int x, int y, int z) -> bool {
        return x >= rxMin && x <= rxMax
            && y >= ryMin && y <= ryMax
            && z >= rzMin && z <= rzMax;
    };

    // 0 = 未标记，1 = 前景种子，-1 = 背景种子
    std::vector<int8_t>  seedLabel(rN, 0);
    std::vector<float>   prob(rN, 0.5f);     // 前景概率
    std::vector<float>   localData(rN);

    for (int z = rzMin; z <= rzMax; ++z)
    for (int y = ryMin; y <= ryMax; ++y)
    for (int x = rxMin; x <= rxMax; ++x) {
        localData[localIdx(x,y,z)] = static_cast<float>(
            data[info.linearIndex(x,y,z)]);
    }

    // ── 设定前景种子（长径线段上各点） ────────────
    auto fgSeeds = sampleLineForeground(ldStart, ldEnd);
    for (const auto& p : fgSeeds) {
        if (localInBounds(p.x, p.y, p.z)) {
            seedLabel[localIdx(p.x,p.y,p.z)] = 1;
            prob      [localIdx(p.x,p.y,p.z)] = 1.f;
        }
    }
    // ── 背景种子（ROI 边界体素） ────────────────
    for (int z = rzMin; z <= rzMax; ++z)
    for (int y = ryMin; y <= ryMax; ++y)
    for (int x = rxMin; x <= rxMax; ++x) {
        bool border = (x==rxMin || x==rxMax || y==ryMin || y==ryMax || z==rzMin || z==rzMax);
        if (border) {
            int li = localIdx(x,y,z);
            if (seedLabel[li] == 0) {
                seedLabel[li] = -1;
                prob[li]     = 0.f;
            }
        }
    }

    // ── Gauss-Seidel 迭代 ─────────────────────
    const int dx6[] = {1,-1,0,0,0,0};
    const int dy6[] = {0,0,1,-1,0,0};
    const int dz6[] = {0,0,0,0,1,-1};

    for (int iter = 0; iter < m_params.maxIter; ++iter) {
        float maxChange = 0.f;
        for (int z = rzMin; z <= rzMax; ++z)
        for (int y = ryMin; y <= ryMax; ++y)
        for (int x = rxMin; x <= rxMax; ++x) {
            int li = localIdx(x,y,z);
            if (seedLabel[li] != 0) continue;  // 种子点固定

            float vi = localData[li];
            float wSum = 0.f, pSum = 0.f;
            for (int d = 0; d < 6; ++d) {
                int nx = x + dx6[d], ny = y + dy6[d], nz = z + dz6[d];
                if (!localInBounds(nx,ny,nz)) continue;
                int nli = localIdx(nx,ny,nz);
                float w = edgeWeight(vi, localData[nli]);
                wSum += w;
                pSum += w * prob[nli];
            }
            float newP = (wSum > 1e-8f) ? pSum / wSum : 0.5f;
            maxChange = std::max(maxChange, std::abs(newP - prob[li]));
            prob[li] = newP;
        }
        if (maxChange < m_params.convThr) break;
    }

    // ── 阈值输出前景 ─────────────────────────
    result.clear();
    for (int z = rzMin; z <= rzMax; ++z)
    for (int y = ryMin; y <= ryMax; ++y)
    for (int x = rxMin; x <= rxMax; ++x) {
        if (prob[localIdx(x,y,z)] >= m_params.fgThreshold)
            result.push_back({x, y, z});
    }
    return !result.empty();
}

} // namespace Onc
