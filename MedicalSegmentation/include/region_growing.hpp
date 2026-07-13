#pragma once
/**
 * ============================================================
 *  区域生长分割 (Region Growing) — 增强版
 * ============================================================
 *
 * 【算法原理】
 *   从一个或多个种子点出发，使用 BFS（广度优先搜索）逐步检查相邻像素，
 *   若满足生长准则则纳入区域，直到没有更多满足条件的像素。
 *
 *   基本流程：
 *     1. 将种子点加入队列 Q，标记为已访问
 *     2. 从 Q 中取出一个像素 p
 *     3. 检查 p 的所有邻居 q：
 *        - 若 q 未访问 且 criterion(q, seedMean) == true
 *        - 将 q 标记为区域内，加入 Q
 *     4. 重复 2-3 直到 Q 为空
 *
 * 【生长准则】
 *   提供三种可选准则（与 AlgoSegmentationConan 对齐）：
 *
 *   1. 阈值准则（Threshold）：
 *      |I(q)| ∈ [low, high]
 *      即像素灰度在指定范围内
 *
 *   2. 差值准则（Difference）：
 *      |I(q) - meanSeed| ≤ tolerance
 *      即与种子均值的差在容差内
 *
 *   3. 比值准则（Ratio）：
 *      |I(q) / meanSeed - 1| ≤ tolerance
 *      即与种子均值的比值偏差在容差内
 *
 * 【增强特性】
 *   - 支持 2D（4/8连通）和 3D（6/26连通）
 *   - 支持自定义生长准则函数
 *   - 支持最大体素数限制（防止过度生长）
 *   - 支持多种子点
 *   - 统计输出（体素数、均值、范围、边界框）
 *
 * 【参考】
 *   Adams R, Bischof L. "Seeded region growing." IEEE TPAMI, 1994.
 * ============================================================
 */

#include "image2d.hpp"

namespace medseg {

// ============================================================
// 生长准则类型
// ============================================================
enum class GrowCriterion {
    Threshold,   // 灰度阈值范围
    Difference,  // 与种子均值差
    Ratio        // 与种子均值比
};

// ============================================================
// 参数
// ============================================================
struct RegionGrowParams {
    GrowCriterion criterion = GrowCriterion::Difference;
    float tolerance = 15.0f;   // 差值/比值容差
    float lowThresh = 0.0f;    // 阈值准则下限
    float highThresh = 255.0f; // 阈值准则上限
    int   connectivity = 4;    // 2D: 4 或 8；3D: 6 或 26
    size_t maxVoxels  = 0;     // 最大生长量（0=不限制）
};

// ============================================================
// 结果
// ============================================================
struct RegionGrowResult {
    Image2D<uint8_t> mask;
    uint64_t voxelCount = 0;
    float meanIntensity = 0.0f;
    float minIntensity  = 1e30f;
    float maxIntensity  = -1e30f;
};

struct RegionGrowResult3D {
    Image3D<uint8_t> mask;
    uint64_t voxelCount = 0;
    float meanIntensity = 0.0f;
    float minIntensity  = 1e30f;
    float maxIntensity  = -1e30f;
};

// ============================================================
// 2D 区域生长
// ============================================================
class RegionGrowing2D {
public:
    /**
     * 从种子点执行区域生长
     * @param image  灰度图像 (float)
     * @param seeds  种子点列表 {(x, y), ...}
     * @param params 参数
     * @return       分割结果
     */
    RegionGrowResult segment(
        const Image2D<float>& image,
        const std::vector<std::pair<int,int>>& seeds,
        const RegionGrowParams& params = {});

    /**
     * 自定义准则函数版本
     * @param criterion  bool(float neighborVal, float seedMean) 函数
     */
    RegionGrowResult segment(
        const Image2D<float>& image,
        const std::vector<std::pair<int,int>>& seeds,
        std::function<bool(float, float)> criterion,
        int connectivity = 4,
        size_t maxVoxels = 0);
};

// ============================================================
// 3D 区域生长
// ============================================================
class RegionGrowing3D {
public:
    RegionGrowResult3D segment(
        const Image3D<float>& volume,
        const std::vector<std::array<int,3>>& seeds,
        const RegionGrowParams& params = {});
};

// ============================================================
//                      实现
// ============================================================

inline RegionGrowResult RegionGrowing2D::segment(
    const Image2D<float>& image,
    const std::vector<std::pair<int,int>>& seeds,
    const RegionGrowParams& params)
{
    // 根据准则类型构造判断函数
    std::function<bool(float, float)> criterion;

    switch (params.criterion) {
    case GrowCriterion::Threshold:
        criterion = [&](float val, float) {
            return val >= params.lowThresh && val <= params.highThresh;
        };
        break;
    case GrowCriterion::Difference:
        criterion = [&](float val, float mean) {
            return std::abs(val - mean) <= params.tolerance;
        };
        break;
    case GrowCriterion::Ratio:
        criterion = [&](float val, float mean) {
            if (std::abs(mean) < 1e-8f) return std::abs(val) < params.tolerance;
            return std::abs(val / mean - 1.0f) <= params.tolerance;
        };
        break;
    }

    return segment(image, seeds, criterion, params.connectivity, params.maxVoxels);
}

inline RegionGrowResult RegionGrowing2D::segment(
    const Image2D<float>& image,
    const std::vector<std::pair<int,int>>& seeds,
    std::function<bool(float, float)> criterion,
    int connectivity,
    size_t maxVoxels)
{
    int W = image.width, H = image.height;
    RegionGrowResult result;
    result.mask = Image2D<uint8_t>(W, H, 0);

    if (seeds.empty()) return result;

    // 计算种子均值
    double seedSum = 0.0;
    for (auto& [sx, sy] : seeds) {
        seedSum += image.at(sx, sy);
    }
    float seedMean = static_cast<float>(seedSum / seeds.size());

    // 邻域偏移
    std::vector<std::pair<int,int>> offsets;
    if (connectivity >= 8) {
        offsets = {{-1,-1},{0,-1},{1,-1},{-1,0},{1,0},{-1,1},{0,1},{1,1}};
    } else {
        offsets = {{0,-1},{-1,0},{1,0},{0,1}};
    }

    // BFS
    std::queue<int> queue;
    std::vector<bool> visited(W * H, false);
    double totalSum = 0.0;

    for (auto& [sx, sy] : seeds) {
        if (!image.inBounds(sx, sy)) continue;
        int idx = sy * W + sx;
        if (!visited[idx]) {
            visited[idx] = true;
            queue.push(idx);
            result.mask[idx] = 255;
            result.voxelCount++;
            float v = image[idx];
            totalSum += v;
            result.minIntensity = std::min(result.minIntensity, v);
            result.maxIntensity = std::max(result.maxIntensity, v);
        }
    }

    while (!queue.empty()) {
        if (maxVoxels > 0 && result.voxelCount >= maxVoxels) break;

        int idx = queue.front(); queue.pop();
        int x = idx % W, y = idx / W;

        for (auto& [dx, dy] : offsets) {
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
            int nidx = ny * W + nx;
            if (visited[nidx]) continue;

            float val = image[nidx];
            if (criterion(val, seedMean)) {
                visited[nidx] = true;
                queue.push(nidx);
                result.mask[nidx] = 255;
                result.voxelCount++;
                totalSum += val;
                result.minIntensity = std::min(result.minIntensity, val);
                result.maxIntensity = std::max(result.maxIntensity, val);

                if (maxVoxels > 0 && result.voxelCount >= maxVoxels) break;
            }
        }
    }

    result.meanIntensity = (result.voxelCount > 0)
        ? static_cast<float>(totalSum / result.voxelCount) : 0.0f;

    return result;
}

inline RegionGrowResult3D RegionGrowing3D::segment(
    const Image3D<float>& volume,
    const std::vector<std::array<int,3>>& seeds,
    const RegionGrowParams& params)
{
    int W = volume.width, H = volume.height, D = volume.depth;
    size_t N = volume.size();

    RegionGrowResult3D result;
    result.mask = Image3D<uint8_t>(W, H, D, 0);

    if (seeds.empty()) return result;

    // 种子均值
    double seedSum = 0.0;
    for (auto& s : seeds) seedSum += volume.at(s[0], s[1], s[2]);
    float seedMean = static_cast<float>(seedSum / seeds.size());

    // 准则函数
    std::function<bool(float, float)> criterion;
    switch (params.criterion) {
    case GrowCriterion::Threshold:
        criterion = [&](float val, float) {
            return val >= params.lowThresh && val <= params.highThresh;
        };
        break;
    case GrowCriterion::Difference:
        criterion = [&](float val, float mean) {
            return std::abs(val - mean) <= params.tolerance;
        };
        break;
    case GrowCriterion::Ratio:
        criterion = [&](float val, float mean) {
            if (std::abs(mean) < 1e-8f) return std::abs(val) < params.tolerance;
            return std::abs(val / mean - 1.0f) <= params.tolerance;
        };
        break;
    }

    // 6连通偏移
    std::vector<std::array<int,3>> offsets6 = {
        {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}
    };
    // 26连通
    std::vector<std::array<int,3>> offsets26;
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx)
        if (dx || dy || dz) offsets26.push_back({dx, dy, dz});

    auto& offsets = (params.connectivity >= 26) ? offsets26 : offsets6;

    // BFS
    std::vector<bool> visited(N, false);
    std::queue<int> queue;
    double totalSum = 0.0;

    for (auto& s : seeds) {
        if (!volume.inBounds(s[0], s[1], s[2])) continue;
        int idx = volume.index(s[0], s[1], s[2]);
        if (!visited[idx]) {
            visited[idx] = true;
            queue.push(idx);
            result.mask[idx] = 255;
            result.voxelCount++;
            float v = volume[idx];
            totalSum += v;
            result.minIntensity = std::min(result.minIntensity, v);
            result.maxIntensity = std::max(result.maxIntensity, v);
        }
    }

    while (!queue.empty()) {
        if (params.maxVoxels > 0 && result.voxelCount >= params.maxVoxels) break;

        int idx = queue.front(); queue.pop();
        int z = idx / (W * H);
        int rem = idx % (W * H);
        int y = rem / W;
        int x = rem % W;

        for (auto& off : offsets) {
            int nx = x + off[0], ny = y + off[1], nz = z + off[2];
            if (!volume.inBounds(nx, ny, nz)) continue;
            int nidx = volume.index(nx, ny, nz);
            if (visited[nidx]) continue;

            float val = volume[nidx];
            if (criterion(val, seedMean)) {
                visited[nidx] = true;
                queue.push(nidx);
                result.mask[nidx] = 255;
                result.voxelCount++;
                totalSum += val;
                result.minIntensity = std::min(result.minIntensity, val);
                result.maxIntensity = std::max(result.maxIntensity, val);

                if (params.maxVoxels > 0 && result.voxelCount >= params.maxVoxels) break;
            }
        }
    }

    result.meanIntensity = (result.voxelCount > 0)
        ? static_cast<float>(totalSum / result.voxelCount) : 0.0f;

    return result;
}

} // namespace medseg
