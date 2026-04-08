/**
 * @file kmeans_segmentation.cpp
 * @brief K-means 聚类肿瘤分割实现
 *
 * 流程：
 *  1. 在种子点周围提取 ROI 区域
 *  2. 收集 ROI 内所有体素灰度值为 1D 特征
 *  3. 用均匀采样初始化 k 个聚类中心
 *  4. EM 迭代：分配标签 → 更新中心，直到收敛或达到上限
 *  5. 取种子点所属聚类的所有体素作为分割结果
 *  6. 对结果做连通域过滤，只保留包含种子点的最大连通域
 */

#include "kmeans_segmentation.h"
#include <queue>
#include <cmath>
#include <cassert>
#include <limits>
#include <numeric>

namespace Onc {

// ─────────────────────────────────────────────
//  辅助：3D 6邻域连通域过滤
// ─────────────────────────────────────────────
static SegmentResult ConnectedComponentFilter(
    const SegmentResult& input,
    const ImageInfo&     info,
    const Point3i&       seed)
{
    if (input.empty()) return {};

    // 建立体素集合 -> 掩码
    std::vector<uint8_t> mask(info.totalVoxels(), 0);
    for (const auto& p : input) {
        if (info.inBounds(p.x, p.y, p.z))
            mask[info.linearIndex(p.x, p.y, p.z)] = 1;
    }

    int seedIdx = info.linearIndex(seed.x, seed.y, seed.z);
    if (mask[seedIdx] == 0) return input; // 种子在前景外，直接返回原集合

    // BFS 从种子点扩展
    const int dx6[] = {1,-1,0,0,0,0};
    const int dy6[] = {0,0,1,-1,0,0};
    const int dz6[] = {0,0,0,0,1,-1};

    std::vector<uint8_t> visited(info.totalVoxels(), 0);
    std::queue<Point3i> q;
    q.push(seed);
    visited[seedIdx] = 1;
    SegmentResult connected;

    while (!q.empty()) {
        auto cur = q.front(); q.pop();
        connected.push_back(cur);
        for (int d = 0; d < 6; ++d) {
            int nx = cur.x + dx6[d];
            int ny = cur.y + dy6[d];
            int nz = cur.z + dz6[d];
            if (!info.inBounds(nx, ny, nz)) continue;
            int ni = info.linearIndex(nx, ny, nz);
            if (mask[ni] == 0 || visited[ni]) continue;
            visited[ni] = 1;
            q.push({nx, ny, nz});
        }
    }
    return connected;
}

// ─────────────────────────────────────────────
//  K-means 主实现
// ─────────────────────────────────────────────

bool KMeansSegmentation::runKMeans(
    const std::vector<float>& samples,
    const std::vector<int>&   indices,
    const ImageInfo&          info,
    const Point3i&            seedPoint,
    SegmentResult&            result)
{
    int N = static_cast<int>(samples.size());
    int K = m_params.numClusters;
    if (N < K) return false;

    // 均匀采样初始化中心
    std::vector<float> centers(K);
    for (int k = 0; k < K; ++k) {
        int idx = k * (N / K);
        centers[k] = samples[idx];
    }

    std::vector<int> labels(N, 0);

    for (int iter = 0; iter < m_params.maxIter; ++iter) {
        // 分配步骤
        for (int i = 0; i < N; ++i) {
            float bestDist = std::numeric_limits<float>::max();
            int   bestK    = 0;
            for (int k = 0; k < K; ++k) {
                float d = std::abs(samples[i] - centers[k]);
                if (d < bestDist) { bestDist = d; bestK = k; }
            }
            labels[i] = bestK;
        }

        // 更新步骤
        float totalMove = 0.f;
        for (int k = 0; k < K; ++k) {
            float sum = 0.f; int cnt = 0;
            for (int i = 0; i < N; ++i) {
                if (labels[i] == k) { sum += samples[i]; ++cnt; }
            }
            if (cnt > 0) {
                float newCenter = sum / cnt;
                totalMove += std::abs(newCenter - centers[k]);
                centers[k] = newCenter;
            }
        }
        if (totalMove < m_params.convergenceThr) break;
    }

    // 确定种子点所属聚类
    int sIdx = info.linearIndex(seedPoint.x, seedPoint.y, seedPoint.z);
    // 在 indices 中查找种子体素
    int seedCluster = -1;
    for (int i = 0; i < N; ++i) {
        if (indices[i] == sIdx) { seedCluster = labels[i]; break; }
    }
    if (seedCluster < 0) {
        // 种子不在 ROI 内：取灰度最高的聚类
        float maxCenter = -std::numeric_limits<float>::max();
        for (int k = 0; k < K; ++k) {
            if (centers[k] > maxCenter) { maxCenter = centers[k]; seedCluster = k; }
        }
    }

    // 收集前景体素
    result.clear();
    for (int i = 0; i < N; ++i) {
        if (labels[i] == seedCluster) {
            int idx = indices[i];
            int z = idx / info.sliceSize();
            int rem = idx % info.sliceSize();
            int y = rem / info.dim[0];
            int x = rem % info.dim[0];
            result.push_back({x, y, z});
        }
    }
    return !result.empty();
}

bool KMeansSegmentation::segment(
    const short*     data,
    const ImageInfo& info,
    const Point3i&   seedPoint,
    SegmentResult&   result)
{
    if (!data) return false;
    if (!info.inBounds(seedPoint.x, seedPoint.y, seedPoint.z)) return false;

    int m = m_params.roiMargin;
    int x0 = std::max(0, seedPoint.x - m), x1 = std::min(info.dim[0]-1, seedPoint.x + m);
    int y0 = std::max(0, seedPoint.y - m), y1 = std::min(info.dim[1]-1, seedPoint.y + m);
    int z0 = std::max(0, seedPoint.z - m), z1 = std::min(info.dim[2]-1, seedPoint.z + m);

    std::vector<float> samples;
    std::vector<int>   indices;
    samples.reserve((x1-x0+1)*(y1-y0+1)*(z1-z0+1));

    for (int z = z0; z <= z1; ++z)
    for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x) {
        int idx = info.linearIndex(x, y, z);
        samples.push_back(static_cast<float>(data[idx]));
        indices.push_back(idx);
    }

    if (!runKMeans(samples, indices, info, seedPoint, result)) return false;

    // 连通域过滤，只保留含种子点的连通分量
    result = ConnectedComponentFilter(result, info, seedPoint);
    return !result.empty();
}

} // namespace Onc
