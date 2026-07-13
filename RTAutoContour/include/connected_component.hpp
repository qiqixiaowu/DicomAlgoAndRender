#pragma once
/**
 * ============================================================
 *  连通域分析 (Connected Component Analysis)
 * ============================================================
 *
 * 提炼自 McsfAlgoAutoContour 工程:
 *   - MaskConnectedDomain (2D/3D)
 *   - GetContourFromSingleDomain
 *   - regionprops
 *
 * 【算法原理】
 *   2D: 使用 BFS 遍历，为每个连通域分配递增标签 (1, 2, 3, ...)
 *   3D: 同理，使用 6 连通或 26 连通
 *   区域属性: 遍历标签图统计面积、质心、包围盒；
 *             计算协方差矩阵的特征值得到主轴、偏心率
 *
 * 【提供能力】
 *   - LabelConnectedComponents2D: 2D 连通域标记
 *   - LabelConnectedComponents3D: 3D 连通域标记
 *   - ComputeRegionProps: 单个域的区域属性
 *   - ExtractBoundary2D: 提取单个连通域有序边界
 *   - KeepLargestComponent2D/3D: 仅保留最大连通域
 *
 * 【参考】
 *   原工程: AutoContourCommon::MaskConnectedDomain,
 *           GetContourFromSingleDomain, regionprops 系列
 * ============================================================
 */

#include "rt_types.hpp"

namespace rtac {

// ============================================================
//  2D 连通域标记
// ============================================================
/**
 * 将二值掩膜中的各连通域标记为 1, 2, 3, ...
 *
 * @param mask        输入二值掩膜 (maskID = 前景)
 * @param maskID      前景值
 * @param use8Connect 是否使用 8 连通（false = 4 连通）
 * @return            标签图（0 = 背景，1~N = 各连通域），以及域数量
 */
struct LabelResult2D {
    Image2D<int32_t> labels;
    int domainCount = 0;          ///< 连通域总数
    std::vector<int> areas;       ///< 每个域的面积 (下标从1开始)
};

inline LabelResult2D LabelConnectedComponents2D(
    const Image2D<uint8_t>& mask,
    uint8_t maskID = 255,
    bool use8Connect = false)
{
    int W = mask.width, H = mask.height;
    LabelResult2D result;
    result.labels = Image2D<int32_t>(W, H, 0);
    result.areas.push_back(0); // 占位，下标 0 不用

    // 邻域偏移
    std::vector<std::pair<int,int>> offsets;
    if (use8Connect)
        offsets = {{-1,-1},{0,-1},{1,-1},{-1,0},{1,0},{-1,1},{0,1},{1,1}};
    else
        offsets = {{0,-1},{-1,0},{1,0},{0,1}};

    int currentLabel = 0;
    std::queue<int> queue;

    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
        int idx = y * W + x;
        if (mask[idx] != maskID || result.labels[idx] != 0) continue;

        // 新连通域
        ++currentLabel;
        result.labels[idx] = currentLabel;
        queue.push(idx);
        int area = 0;

        while (!queue.empty()) {
            int ci = queue.front(); queue.pop();
            ++area;
            int cx = ci % W, cy = ci / W;

            for (auto& [dx, dy] : offsets) {
                int nx = cx + dx, ny = cy + dy;
                if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                int ni = ny * W + nx;
                if (mask[ni] == maskID && result.labels[ni] == 0) {
                    result.labels[ni] = currentLabel;
                    queue.push(ni);
                }
            }
        }
        result.areas.push_back(area);
    }

    result.domainCount = currentLabel;
    return result;
}

// ============================================================
//  3D 连通域标记
// ============================================================
struct LabelResult3D {
    Image3D<int32_t> labels;
    int domainCount = 0;
    std::vector<int> areas;
};

inline LabelResult3D LabelConnectedComponents3D(
    const Image3D<uint8_t>& mask,
    uint8_t maskID = 255,
    bool use26Connect = false)
{
    int W = mask.width, H = mask.height, D = mask.depth;
    LabelResult3D result;
    result.labels = Image3D<int32_t>(W, H, D, 0);
    result.areas.push_back(0);

    // 生成邻域偏移
    std::vector<std::array<int,3>> offsets;
    if (use26Connect) {
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            if (dx || dy || dz) offsets.push_back({dx, dy, dz});
    } else {
        offsets = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    }

    int currentLabel = 0;
    std::queue<int> queue;

    for (int z = 0; z < D; ++z)
    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
        int idx = mask.index(x, y, z);
        if (mask[idx] != maskID || result.labels[idx] != 0) continue;

        ++currentLabel;
        result.labels[idx] = currentLabel;
        queue.push(idx);
        int area = 0;

        while (!queue.empty()) {
            int ci = queue.front(); queue.pop();
            ++area;
            int cz = ci / static_cast<int>(mask.sliceSize());
            int rem = ci % static_cast<int>(mask.sliceSize());
            int cy = rem / W, cx = rem % W;

            for (auto& off : offsets) {
                int nx = cx + off[0], ny = cy + off[1], nz = cz + off[2];
                if (!mask.inBounds(nx, ny, nz)) continue;
                int ni = mask.index(nx, ny, nz);
                if (mask[ni] == maskID && result.labels[ni] == 0) {
                    result.labels[ni] = currentLabel;
                    queue.push(ni);
                }
            }
        }
        result.areas.push_back(area);
    }

    result.domainCount = currentLabel;
    return result;
}

// ============================================================
//  计算单个连通域的区域属性
// ============================================================
/**
 * 对标签图中 label == targetLabel 的区域计算属性。
 *
 * 属性包括：面积、质心、包围盒、主轴长度、偏心率、紧致度
 *
 * 原理：
 *   - 质心 = 坐标均值
 *   - 主轴/短轴 = 二阶中心矩矩阵的特征值 → sqrt(4*eigenvalue/area)
 *   - 偏心率 = sqrt(1 - (短轴/主轴)²)
 */
inline RegionProps ComputeRegionProps(
    const Image2D<int32_t>& labels,
    int targetLabel)
{
    int W = labels.width, H = labels.height;
    RegionProps rp;

    double sumX = 0, sumY = 0;
    int count = 0;
    int x0 = W, y0 = H, x1 = 0, y1 = 0;

    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
        if (labels.at(x, y) != targetLabel) continue;
        sumX += x;
        sumY += y;
        ++count;
        x0 = std::min(x0, x); y0 = std::min(y0, y);
        x1 = std::max(x1, x); y1 = std::max(y1, y);
    }

    if (count == 0) return rp;

    rp.area = count;
    rp.centroidX = static_cast<float>(sumX / count);
    rp.centroidY = static_cast<float>(sumY / count);
    rp.bboxX0 = x0; rp.bboxY0 = y0;
    rp.bboxX1 = x1; rp.bboxY1 = y1;

    // 二阶中心矩
    double mu20 = 0, mu02 = 0, mu11 = 0;
    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
        if (labels.at(x, y) != targetLabel) continue;
        double dx = x - rp.centroidX;
        double dy = y - rp.centroidY;
        mu20 += dx * dx;
        mu02 += dy * dy;
        mu11 += dx * dy;
    }
    mu20 /= count;
    mu02 /= count;
    mu11 /= count;

    // 特征值 = 0.5*(mu20+mu02) ± 0.5*sqrt((mu20-mu02)^2 + 4*mu11^2)
    double diff = mu20 - mu02;
    double disc = std::sqrt(diff * diff + 4.0 * mu11 * mu11);
    double lambda1 = 0.5 * (mu20 + mu02 + disc); // 大
    double lambda2 = 0.5 * (mu20 + mu02 - disc); // 小
    lambda2 = std::max(lambda2, 0.0);

    rp.majorAxisLen = static_cast<float>(4.0 * std::sqrt(lambda1));
    rp.minorAxisLen = static_cast<float>(4.0 * std::sqrt(lambda2));

    if (rp.majorAxisLen > 1e-6f)
        rp.eccentricity = std::sqrt(1.0f - (rp.minorAxisLen * rp.minorAxisLen)
                                           / (rp.majorAxisLen * rp.majorAxisLen));
    else
        rp.eccentricity = 0.f;

    // solidity 需要凸包面积（简化：用包围盒面积近似）
    float bboxArea = static_cast<float>((x1 - x0 + 1) * (y1 - y0 + 1));
    rp.solidity = (bboxArea > 0) ? (static_cast<float>(count) / bboxArea) : 0.f;

    return rp;
}

// ============================================================
//  仅保留最大连通域（2D）
// ============================================================
inline void KeepLargestComponent2D(
    Image2D<uint8_t>& mask,
    uint8_t maskID = 255,
    bool use8Connect = false)
{
    auto lr = LabelConnectedComponents2D(mask, maskID, use8Connect);
    if (lr.domainCount == 0) return;

    // 找最大域
    int bestLabel = 1;
    int bestArea = 0;
    for (int i = 1; i <= lr.domainCount; ++i) {
        if (lr.areas[i] > bestArea) {
            bestArea = lr.areas[i];
            bestLabel = i;
        }
    }

    // 仅保留最大域
    for (size_t i = 0; i < mask.size(); ++i) {
        mask[i] = (lr.labels[i] == bestLabel) ? maskID : 0;
    }
}

// ============================================================
//  仅保留最大连通域（3D）
// ============================================================
inline void KeepLargestComponent3D(
    Image3D<uint8_t>& mask,
    uint8_t maskID = 255,
    bool use26Connect = false)
{
    auto lr = LabelConnectedComponents3D(mask, maskID, use26Connect);
    if (lr.domainCount == 0) return;

    int bestLabel = 1, bestArea = 0;
    for (int i = 1; i <= lr.domainCount; ++i) {
        if (lr.areas[i] > bestArea) {
            bestArea = lr.areas[i];
            bestLabel = i;
        }
    }

    for (size_t i = 0; i < mask.size(); ++i) {
        mask[i] = (lr.labels[i] == bestLabel) ? maskID : 0;
    }
}

// ============================================================
//  提取 2D 有序边界轮廓
// ============================================================
/**
 * 从二值掩膜中提取某个连通域的有序边界点，使用 8 邻域追踪。
 *
 * 算法（Moore Boundary Tracing）：
 *   1. 扫描找到第一个前景像素作为起始点
 *   2. 从起始方向开始，逆时针搜索下一个前景邻居
 *   3. 移动到该邻居，调整搜索方向
 *   4. 重复直到回到起始点
 *
 * 对应原工程 GetContourFromSingleDomain
 */
inline std::vector<Point2i> ExtractBoundary2D(
    const Image2D<uint8_t>& mask,
    uint8_t maskID = 255)
{
    int W = mask.width, H = mask.height;
    std::vector<Point2i> boundary;

    // 找起始点（最上最左的前景像素）
    int startX = -1, startY = -1;
    for (int y = 0; y < H && startX < 0; ++y)
    for (int x = 0; x < W && startX < 0; ++x) {
        if (mask.at(x, y) == maskID) {
            startX = x; startY = y;
        }
    }
    if (startX < 0) return boundary;

    // Moore 邻域 8 方向（顺时针）：右、右下、下、左下、左、左上、上、右上
    const int dx8[] = { 1, 1, 0,-1,-1,-1, 0, 1};
    const int dy8[] = { 0, 1, 1, 1, 0,-1,-1,-1};

    auto isForeground = [&](int x, int y) -> bool {
        if (x < 0 || x >= W || y < 0 || y >= H) return false;
        return mask.at(x, y) == maskID;
    };

    int cx = startX, cy = startY;
    int dir = 7; // 从"上"方向开始（因为起始点上方必然是背景）
    boundary.push_back({cx, cy});

    do {
        // 从 (dir + 5) % 8 开始，即"回退"方向后偏移
        int startDir = (dir + 5) % 8;
        bool found = false;

        for (int i = 0; i < 8; ++i) {
            int d = (startDir + i) % 8;
            int nx = cx + dx8[d], ny = cy + dy8[d];
            if (isForeground(nx, ny)) {
                cx = nx; cy = ny;
                dir = d;
                found = true;
                break;
            }
        }

        if (!found) break;

        if (cx == startX && cy == startY) break;
        boundary.push_back({cx, cy});

        // 安全限制
        if (boundary.size() > mask.size()) break;

    } while (true);

    return boundary;
}

} // namespace rtac
