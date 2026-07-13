#pragma once
/**
 * ============================================================
 *  形态学操作 (Morphology Operations)
 * ============================================================
 *
 * 提炼自 McsfAlgoAutoContour 工程中的形态学算法。
 *
 * 【2D 操作】
 *   - Erosion2D:  单像素腐蚀（4邻域）
 *   - Dilation2D: 单像素膨胀（4邻域）
 *   - Opening2D:  开运算 = 先腐蚀后膨胀
 *   - Closing2D:  闭运算 = 先膨胀后腐蚀
 *   - FillHoles2D: 孔洞填充（floodfill 外部后取反）
 *
 * 【3D 操作】
 *   - Erosion3D:  球形结构元素腐蚀
 *   - Dilation3D: 球形结构元素膨胀
 *   - Closing3D:  3D 闭运算
 *   - GenerateBallMask: 生成球形结构元素
 *
 * 【参考】
 *   原工程: McsfAlgoAutoContourCommon.cpp — Erosion2D, Dilation2D,
 *            ImageErosion, ImageDilation, ImageClose,
 *            GenerateBallMask, FillHoles2D
 * ============================================================
 */

#include "rt_types.hpp"

namespace rtac {

// ============================================================
//  2D 腐蚀 — 4邻域
// ============================================================
/**
 * 对二值掩膜进行单像素腐蚀。前景 = maskID，其余为背景。
 * 原理：仅当上下左右 4 邻居均为前景时，该像素保留为前景。
 *
 * @param mask     输入掩膜（直接原地修改）
 * @param maskID   前景标记值
 * @return         腐蚀后的掩膜（同一块内存）
 */
inline void Erosion2D(Image2D<uint8_t>& mask, uint8_t maskID = 255) {
    int W = mask.width, H = mask.height;
    Image2D<uint8_t> tmp(W, H, 0);

    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            if (mask.at(x, y) != maskID) continue;
            // 4邻域全为前景才保留
            if (mask.at(x-1, y) == maskID &&
                mask.at(x+1, y) == maskID &&
                mask.at(x, y-1) == maskID &&
                mask.at(x, y+1) == maskID) {
                tmp.at(x, y) = maskID;
            }
        }
    }
    mask = std::move(tmp);
}

// ============================================================
//  2D 膨胀 — 4邻域
// ============================================================
/**
 * 对二值掩膜进行单像素膨胀。
 * 原理：若上下左右 4 邻居中有任一为前景，则该像素设为前景。
 */
inline void Dilation2D(Image2D<uint8_t>& mask, uint8_t maskID = 255) {
    int W = mask.width, H = mask.height;
    Image2D<uint8_t> tmp = mask; // 复制

    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            if (mask.at(x, y) == maskID) continue;
            if (mask.at(x-1, y) == maskID ||
                mask.at(x+1, y) == maskID ||
                mask.at(x, y-1) == maskID ||
                mask.at(x, y+1) == maskID) {
                tmp.at(x, y) = maskID;
            }
        }
    }
    mask = std::move(tmp);
}

// ============================================================
//  2D 开运算 = 先腐蚀后膨胀（去除小突起）
// ============================================================
inline void Opening2D(Image2D<uint8_t>& mask, int iterations = 1, uint8_t maskID = 255) {
    for (int i = 0; i < iterations; ++i) Erosion2D(mask, maskID);
    for (int i = 0; i < iterations; ++i) Dilation2D(mask, maskID);
}

// ============================================================
//  2D 闭运算 = 先膨胀后腐蚀（填充小孔）
// ============================================================
inline void Closing2D(Image2D<uint8_t>& mask, int iterations = 1, uint8_t maskID = 255) {
    for (int i = 0; i < iterations; ++i) Dilation2D(mask, maskID);
    for (int i = 0; i < iterations; ++i) Erosion2D(mask, maskID);
}

// ============================================================
//  2D 孔洞填充
// ============================================================
/**
 * 从图像边界开始 floodfill 标记所有外部区域，
 * 然后未被标记的非前景区域即为孔洞，统一填为前景。
 *
 * 原理对应原工程 FillHoles2D
 */
inline void FillHoles2D(Image2D<uint8_t>& mask, uint8_t maskID = 255) {
    int W = mask.width, H = mask.height;
    std::vector<bool> external(mask.size(), false);
    std::queue<int> queue;

    // 从四条边界开始标记
    auto tryPush = [&](int x, int y) {
        if (x < 0 || x >= W || y < 0 || y >= H) return;
        int idx = y * W + x;
        if (!external[idx] && mask[idx] != maskID) {
            external[idx] = true;
            queue.push(idx);
        }
    };

    for (int x = 0; x < W; ++x) { tryPush(x, 0); tryPush(x, H - 1); }
    for (int y = 0; y < H; ++y) { tryPush(0, y); tryPush(W - 1, y); }

    // BFS
    while (!queue.empty()) {
        int idx = queue.front(); queue.pop();
        int x = idx % W, y = idx / W;
        tryPush(x - 1, y); tryPush(x + 1, y);
        tryPush(x, y - 1); tryPush(x, y + 1);
    }

    // 未被标记为外部 且 不是前景 → 孔洞，填充
    for (size_t i = 0; i < mask.size(); ++i) {
        if (!external[i] && mask[i] != maskID) {
            mask[i] = maskID;
        }
    }
}

// ============================================================
//  生成球形结构元素
// ============================================================
/**
 * 生成离散球形结构元素（3D bool mask），
 * 用于 3D 膨胀/腐蚀。
 *
 * @param radius 半径（体素单位）
 * @return Image3D<uint8_t> 尺寸为 (2r+1)^3
 */
inline Image3D<uint8_t> GenerateBallMask(int radius) {
    int side = 2 * radius + 1;
    Image3D<uint8_t> ball(side, side, side, 0);
    float r2 = static_cast<float>(radius * radius);

    for (int z = 0; z < side; ++z)
    for (int y = 0; y < side; ++y)
    for (int x = 0; x < side; ++x) {
        float dx = static_cast<float>(x - radius);
        float dy = static_cast<float>(y - radius);
        float dz = static_cast<float>(z - radius);
        if (dx * dx + dy * dy + dz * dz <= r2 + 0.5f)
            ball.at(x, y, z) = 1;
    }
    return ball;
}

// ============================================================
//  3D 膨胀 — 球形结构元素
// ============================================================
/**
 * @param volume  输入掩膜（原地修改）
 * @param radius  结构元素半径
 * @param maskID  前景标记
 */
inline void Dilation3D(Image3D<uint8_t>& volume, int radius, uint8_t maskID = 255) {
    auto ball = GenerateBallMask(radius);
    int W = volume.width, H = volume.height, D = volume.depth;
    Image3D<uint8_t> result(W, H, D, 0);

    for (int z = 0; z < D; ++z)
    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
        if (volume.at(x, y, z) != maskID) continue;
        // 将球内对应位置都设为前景
        for (int dz = -radius; dz <= radius; ++dz)
        for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
            if (ball.at(dx + radius, dy + radius, dz + radius) == 0) continue;
            int nx = x + dx, ny = y + dy, nz = z + dz;
            if (volume.inBounds(nx, ny, nz))
                result.at(nx, ny, nz) = maskID;
        }
    }
    volume = std::move(result);
}

// ============================================================
//  3D 腐蚀 — 球形结构元素
// ============================================================
inline void Erosion3D(Image3D<uint8_t>& volume, int radius, uint8_t maskID = 255) {
    auto ball = GenerateBallMask(radius);
    int W = volume.width, H = volume.height, D = volume.depth;
    Image3D<uint8_t> result(W, H, D, 0);

    for (int z = 0; z < D; ++z)
    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
        if (volume.at(x, y, z) != maskID) continue;
        // 检查球内所有位置是否都为前景
        bool allForeground = true;
        for (int dz = -radius; dz <= radius && allForeground; ++dz)
        for (int dy = -radius; dy <= radius && allForeground; ++dy)
        for (int dx = -radius; dx <= radius && allForeground; ++dx) {
            if (ball.at(dx + radius, dy + radius, dz + radius) == 0) continue;
            int nx = x + dx, ny = y + dy, nz = z + dz;
            if (!volume.inBounds(nx, ny, nz) || volume.at(nx, ny, nz) != maskID)
                allForeground = false;
        }
        if (allForeground)
            result.at(x, y, z) = maskID;
    }
    volume = std::move(result);
}

// ============================================================
//  3D 闭运算 = 先膨胀后腐蚀
// ============================================================
inline void Closing3D(Image3D<uint8_t>& volume, int radius, uint8_t maskID = 255) {
    Dilation3D(volume, radius, maskID);
    Erosion3D(volume, radius, maskID);
}

} // namespace rtac
