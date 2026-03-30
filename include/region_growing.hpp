#pragma once

/**
 * ============================================================
 * 三维区域生长算法 (3D Region Growing Segmentation)
 * ============================================================
 *
 * 算法原理：
 *   区域生长是一种基于局部相似性的图像分割方法。
 *   从用户指定的"种子点"出发，使用BFS（广度优先搜索）逐步将
 *   与种子点强度相近的邻接体素归并入分割区域，直至没有满足
 *   条件的体素可以加入为止。
 *
 * 生长准则（Intensity-based）：
 *   一个体素 v 被加入区域，当且仅当：
 *     |density(v) - seedIntensity| <= tolerance
 *   其中 tolerance 由用户调节。
 *
 * 连通性：
 *   支持 6连通（面邻居）和 26连通（含边/角邻居）两种模式。
 *   医学图像通常使用 6连通，结果更保守、不易穿透间隙。
 *
 * 输入：
 *   - volume: 体素数据（uint8_t, 归一化到 [0,255]）
 *   - width/height/depth: 体积尺寸
 *   - seed: 种子点体素坐标 (x, y, z)
 *   - tolerance: 强度容差（0~255范围内的绝对值，默认15）
 *   - connectivity: 6 或 26
 *   - maxVoxels: 最大生长体素数（防止无限扩张，0=不限制）
 *
 * 输出：
 *   - RegionGrowingResult: 包含二值掩码（与原体积同尺寸）及统计信息
 *     mask[i] = 255 表示该体素属于分割区域
 *     mask[i] = 0   表示该体素不属于分割区域
 * ============================================================
 */

#include <vector>
#include <queue>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <iostream>

#include <array>    // 必须
 
// ============================================================
// 结果结构体
// ============================================================
struct RegionGrowingResult {
    std::vector<uint8_t> mask;    // 二值掩码（0 或 255）
    uint32_t width  = 0;
    uint32_t height = 0;
    uint32_t depth  = 0;

    uint64_t voxelCount   = 0;     // 分割到的体素总数
    float    meanIntensity = 0.0f; // 分割区域平均强度（[0,1]归一化）
    float    minIntensity  = 1.0f;
    float    maxIntensity  = 0.0f;
    double   elapsedMs     = 0.0;  // 算法耗时（毫秒）

    bool empty() const { return voxelCount == 0; }

    // 返回归一化的体素坐标范围（[0,1]）
    void getBoundingBox(float& x0, float& y0, float& z0,
                        float& x1, float& y1, float& z1) const {
        // 简单遍历掩码找AABB（仅用于调试）
        int minX = width, minY = height, minZ = depth;
        int maxX = -1,    maxY = -1,     maxZ = -1;

        for (uint32_t z = 0; z < depth; ++z)
        for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x) {
            if (mask[z * height * width + y * width + x]) {
                if ((int)x < minX) minX = x;
                if ((int)y < minY) minY = y;
                if ((int)z < minZ) minZ = z;
                if ((int)x > maxX) maxX = x;
                if ((int)y > maxY) maxY = y;
                if ((int)z > maxZ) maxZ = z;
            }
        }

        x0 = (float)minX / (float)(width  - 1);
        y0 = (float)minY / (float)(height - 1);
        z0 = (float)minZ / (float)(depth  - 1);
        x1 = (float)maxX / (float)(width  - 1);
        y1 = (float)maxY / (float)(height - 1);
        z1 = (float)maxZ / (float)(depth  - 1);
    }
};

// ============================================================
// 连通性枚举
// ============================================================
enum class Connectivity {
    CONNECT_6  = 6,   // 面邻居（医学图像推荐）
    CONNECT_26 = 26   // 面+边+角邻居
};

// ============================================================
// 辅助：生成26邻域偏移量
// ============================================================
static std::vector<std::array<int,3>> buildNeighborOffsets(Connectivity conn)
{
    std::vector<std::array<int,3>> offsets;
    offsets.reserve(26);

    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0 && dz == 0) continue;

        int dist = std::abs(dx) + std::abs(dy) + std::abs(dz);

        if (conn == Connectivity::CONNECT_6 && dist != 1) continue;
        // 26连通：全部纳入

        offsets.push_back({ {dx, dy, dz} });
    }
    return offsets;
}

// ============================================================
// 核心算法：三维区域生长（BFS）
// ============================================================
/**
 * @param volumeData   体素数据（uint8_t，每体素占1字节，范围[0,255]）
 * @param width/height/depth  体积三维尺寸
 * @param seedX/Y/Z    种子点体素索引坐标
 * @param tolerance    强度容差（uint8_t量纲，0~255）。
 *                     建议初始值：10~30（CT骨骼），5~15（软组织）
 * @param connectivity 连通性选择（默认6连通）
 * @param maxVoxels    最大生长体素数（0表示不限制）
 */
inline RegionGrowingResult regionGrow3D(
    const std::vector<uint8_t>& volumeData,
    uint32_t width, uint32_t height, uint32_t depth,
    int seedX, int seedY, int seedZ,
    uint8_t tolerance      = 20,
    Connectivity connectivity = Connectivity::CONNECT_6,
    uint64_t maxVoxels     = 0)
{
    auto startTime = std::chrono::high_resolution_clock::now();

    RegionGrowingResult result;
    result.width  = width;
    result.height = height;
    result.depth  = depth;
    result.mask.assign((size_t)width * height * depth, 0);

    // 边界检查
    if (seedX < 0 || seedX >= (int)width  ||
        seedY < 0 || seedY >= (int)height ||
        seedZ < 0 || seedZ >= (int)depth) {
        std::cerr << "[RegionGrow] 种子点超出体积范围: ("
                  << seedX << ", " << seedY << ", " << seedZ << ")\n";
        return result;
    }

    // 线性索引辅助函数
    auto idx = [&](int x, int y, int z) -> size_t {
        return (size_t)z * height * width + (size_t)y * width + (size_t)x;
    };

    // 种子点强度
    uint8_t seedIntensity = volumeData[idx(seedX, seedY, seedZ)];

    // 接受范围
    int lo = (int)seedIntensity - (int)tolerance;
    int hi = (int)seedIntensity + (int)tolerance;

    // 统计信息初始化
    double intensitySum = 0.0;
    result.minIntensity = (float)seedIntensity / 255.0f;
    result.maxIntensity = (float)seedIntensity / 255.0f;

    // visited标记（与mask共用，先作为访问标记）
    // mask[i]=1 表示已访问，最后统一改为255
    result.mask[idx(seedX, seedY, seedZ)] = 1;

    // BFS队列
    struct Voxel { int x, y, z; };
    std::queue<Voxel> queue;
    queue.push({seedX, seedY, seedZ});

    // 邻域偏移
    auto offsets = buildNeighborOffsets(connectivity);

    while (!queue.empty()) {
        Voxel cur = queue.front();
        queue.pop();

        uint8_t curIntensity = volumeData[idx(cur.x, cur.y, cur.z)];
        float   normIntensity = (float)curIntensity / 255.0f;

        // 更新统计
        intensitySum += normIntensity;
        result.voxelCount++;
        if (normIntensity < result.minIntensity) result.minIntensity = normIntensity;
        if (normIntensity > result.maxIntensity) result.maxIntensity = normIntensity;

        // 最大体素数限制
        if (maxVoxels > 0 && result.voxelCount >= maxVoxels) break;

        // 遍历邻域
        for (const auto& off : offsets) {
            int nx = cur.x + off[0];
            int ny = cur.y + off[1];
            int nz = cur.z + off[2];

            // 边界检查
            if (nx < 0 || nx >= (int)width  ||
                ny < 0 || ny >= (int)height ||
                nz < 0 || nz >= (int)depth) continue;

            size_t ni = idx(nx, ny, nz);

            // 已访问则跳过
            if (result.mask[ni]) continue;

            // 强度相似性判断
            int neighborIntensity = (int)volumeData[ni];
            if (neighborIntensity >= lo && neighborIntensity <= hi) {
                result.mask[ni] = 1;   // 标记为已访问
                queue.push({nx, ny, nz});
            }
        }
    }

    // 将访问标记 1 → 255（用于纹理：0=背景，255=分割区域）
    for (auto& v : result.mask) {
        if (v == 1) v = 255;
    }

    // 计算平均强度
    result.meanIntensity = result.voxelCount > 0
        ? (float)(intensitySum / (double)result.voxelCount)
        : 0.0f;

    auto endTime = std::chrono::high_resolution_clock::now();
    result.elapsedMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    std::cout << "[RegionGrow] 完成! 体素数=" << result.voxelCount
              << ", 均值=" << result.meanIntensity
              << ", 耗时=" << result.elapsedMs << "ms\n";
    return result;
}

// ============================================================
// 扩展：多种子区域生长
// ============================================================
/**
 * 从多个种子点同时生长，各种子点共享同一容差。
 * 最终掩码为所有种子点扩展区域的并集。
 */
inline RegionGrowingResult regionGrow3DMultiSeed(
    const std::vector<uint8_t>& volumeData,
    uint32_t width, uint32_t height, uint32_t depth,
    const std::vector<std::array<int,3>>& seeds,
    uint8_t tolerance         = 20,
    Connectivity connectivity = Connectivity::CONNECT_6,
    uint64_t maxVoxels        = 0)
{
    if (seeds.empty()) {
        RegionGrowingResult r;
        r.width = width; r.height = height; r.depth = depth;
        r.mask.assign((size_t)width * height * depth, 0);
        return r;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    RegionGrowingResult result;
    result.width  = width;
    result.height = height;
    result.depth  = depth;
    result.mask.assign((size_t)width * height * depth, 0);

    auto idx = [&](int x, int y, int z) -> size_t {
        return (size_t)z * height * width + (size_t)y * width + (size_t)x;
    };

    // 计算所有种子点的强度均值作为参考
    double seedMean = 0.0;
    int validSeeds = 0;
    for (const auto& s : seeds) {
        if (s[0] >= 0 && s[0] < (int)width &&
            s[1] >= 0 && s[1] < (int)height &&
            s[2] >= 0 && s[2] < (int)depth) {
            seedMean += (double)volumeData[idx(s[0], s[1], s[2])];
            validSeeds++;
        }
    }
    if (validSeeds == 0) return result;
    seedMean /= validSeeds;

    int lo = (int)(seedMean - tolerance);
    int hi = (int)(seedMean + tolerance);

    double intensitySum = 0.0;
    result.minIntensity = 1.0f;
    result.maxIntensity = 0.0f;

    struct Voxel { int x, y, z; };
    std::queue<Voxel> queue;

    // 将所有有效种子加入队列
    for (const auto& s : seeds) {
        if (s[0] < 0 || s[0] >= (int)width  ||
            s[1] < 0 || s[1] >= (int)height ||
            s[2] < 0 || s[2] >= (int)depth) continue;

        size_t si = idx(s[0], s[1], s[2]);
        if (!result.mask[si]) {
            result.mask[si] = 1;
            queue.push({s[0], s[1], s[2]});
        }
    }

    auto offsets = buildNeighborOffsets(connectivity);

    while (!queue.empty()) {
        Voxel cur = queue.front();
        queue.pop();

        float normIntensity = (float)volumeData[idx(cur.x, cur.y, cur.z)] / 255.0f;
        intensitySum += normIntensity;
        result.voxelCount++;
        if (normIntensity < result.minIntensity) result.minIntensity = normIntensity;
        if (normIntensity > result.maxIntensity) result.maxIntensity = normIntensity;

        if (maxVoxels > 0 && result.voxelCount >= maxVoxels) break;

        for (const auto& off : offsets) {
            int nx = cur.x + off[0];
            int ny = cur.y + off[1];
            int nz = cur.z + off[2];

            if (nx < 0 || nx >= (int)width  ||
                ny < 0 || ny >= (int)height ||
                nz < 0 || nz >= (int)depth) continue;

            size_t ni = idx(nx, ny, nz);
            if (result.mask[ni]) continue;

            int neighborIntensity = (int)volumeData[ni];
            if (neighborIntensity >= lo && neighborIntensity <= hi) {
                result.mask[ni] = 1;
                queue.push({nx, ny, nz});
            }
        }
    }

    for (auto& v : result.mask) {
        if (v == 1) v = 255;
    }

    result.meanIntensity = result.voxelCount > 0
        ? (float)(intensitySum / (double)result.voxelCount)
        : 0.0f;

    auto endTime = std::chrono::high_resolution_clock::now();
    result.elapsedMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    std::cout << "[RegionGrow] 多种子完成! 体素数=" << result.voxelCount
              << ", 均值=" << result.meanIntensity
              << ", 耗时=" << result.elapsedMs << "ms\n";
    return result;
}
