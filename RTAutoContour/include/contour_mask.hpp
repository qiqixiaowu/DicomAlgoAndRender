#pragma once
/**
 * ============================================================
 *  轮廓与掩膜互转 (Contour ↔ Mask Conversion)
 * ============================================================
 *
 * 提炼自 McsfAlgoAutoContour 工程核心功能。
 *
 * 在放射治疗计划系统中，医生勾画的轮廓（一组有序 2D 点）
 * 和体积掩膜（二值 3D 图像）需要频繁互转：
 *   - 勾画轮廓 → 生成掩膜（用于剂量计算）
 *   - 分割掩膜 → 提取轮廓（用于显示和编辑）
 *
 * 【轮廓 → 掩膜 (ContourToMask)】
 *   1. 将轮廓点用样条/线性插值连成闭合边缘
 *   2. 在边缘掩膜上从外部做 floodfill 标记外部区域
 *   3. 未被标记的区域即为内部 → 填充为前景
 *   可选: 5x 超采样后再缩回，获得更精确的亚像素边界
 *
 * 【掩膜 → 轮廓 (MaskToContour)】
 *   1. 对掩膜做单像素腐蚀获得边界像素
 *   2. 使用 Moore Boundary Tracing（8邻域追踪）提取有序轮廓点
 *   3. 可选样条平滑
 *
 * 【参考】
 *   原工程: McsfAlgoAutoContourBasicFunction.cpp —
 *           GetMaskFromContoursnew, GetContoursFromMasknew
 * ============================================================
 */

#include "rt_types.hpp"
#include "morphology.hpp"
#include "connected_component.hpp"

namespace rtac {

// ============================================================
//  曲线插值工具
// ============================================================

/**
 * 对轮廓点进行线性插值加密，使相邻点间距 ≤ 1 像素。
 * 对应原工程的直线插值。
 */
inline Contour2D InterpolateLinear(const Contour2D& contour) {
    if (contour.size() < 2) return contour;
    Contour2D result;
    result.reserve(contour.size() * 3);

    for (size_t i = 0; i < contour.size(); ++i) {
        const auto& p0 = contour[i];
        const auto& p1 = contour[(i + 1) % contour.size()];
        result.push_back(p0);

        float dist = p0.distanceTo(p1);
        int steps = static_cast<int>(std::ceil(dist));
        for (int s = 1; s < steps; ++s) {
            float t = static_cast<float>(s) / steps;
            result.push_back({p0.x + t * (p1.x - p0.x),
                              p0.y + t * (p1.y - p0.y)});
        }
    }
    return result;
}

/**
 * Catmull-Rom 样条细分（三轮迭代），对应原工程的曲线插值。
 *
 * 每轮在相邻点之间插入新点：
 *   P_new = 0.5 * (P_i + P_{i+1})
 * 然后做一次平均平滑。
 */
inline Contour2D InterpolateSpline(const Contour2D& contour, int subdivisions = 3) {
    if (contour.size() < 3) return contour;
    Contour2D current = contour;

    for (int iter = 0; iter < subdivisions; ++iter) {
        size_t n = current.size();
        Contour2D refined;
        refined.reserve(n * 2);

        for (size_t i = 0; i < n; ++i) {
            size_t prev = (i + n - 1) % n;
            size_t next = (i + 1) % n;
            // 原始点
            refined.push_back(current[i]);
            // 插入中点（Catmull-Rom 风格: 加权平均）
            Point2f mid;
            mid.x = 0.5f * (current[i].x + current[next].x);
            mid.y = 0.5f * (current[i].y + current[next].y);
            refined.push_back(mid);
        }

        // 平滑通道
        Contour2D smoothed(refined.size());
        size_t m = refined.size();
        for (size_t i = 0; i < m; ++i) {
            size_t prev = (i + m - 1) % m;
            size_t next = (i + 1) % m;
            smoothed[i].x = 0.25f * refined[prev].x + 0.5f * refined[i].x + 0.25f * refined[next].x;
            smoothed[i].y = 0.25f * refined[prev].y + 0.5f * refined[i].y + 0.25f * refined[next].y;
        }
        current = std::move(smoothed);
    }
    return current;
}

// ============================================================
//  轮廓 → 闭合边缘掩膜
// ============================================================
/**
 * 将轮廓点光栅化为闭合边缘掩膜（边界线为前景）。
 * 对应 GetClosedEdgesFromContour。
 *
 * @param contour    2D 轮廓点（有序，闭合）
 * @param width      图像宽度
 * @param height     图像高度
 * @param useSpline  true=样条插值, false=线性插值
 * @param edgeID     边缘标记值
 * @return           边缘掩膜
 */
inline Image2D<uint8_t> ContourToEdgeMask(
    const Contour2D& contour,
    int width, int height,
    bool useSpline = false,
    uint8_t edgeID = 255)
{
    Image2D<uint8_t> edge(width, height, 0);

    // 插值加密
    Contour2D dense = useSpline ? InterpolateSpline(contour) : InterpolateLinear(contour);

    // 光栅化：将每个点四舍五入到像素网格
    for (const auto& p : dense) {
        int px = static_cast<int>(std::round(p.x));
        int py = static_cast<int>(std::round(p.y));
        if (px >= 0 && px < width && py >= 0 && py < height) {
            edge.at(px, py) = edgeID;
        }
    }

    return edge;
}

// ============================================================
//  轮廓 → 填充掩膜
// ============================================================
/**
 * 将闭合轮廓转换为填充掩膜。
 *
 * 算法流程：
 *   1. 光栅化轮廓为闭合边缘
 *   2. 从图像边界做 floodfill 标记所有外部区域
 *   3. 非外部 且 非边缘 的区域 → 内部
 *   4. 边缘 + 内部 = 最终掩膜
 *
 * @param contour    2D 轮廓（有序闭合点集）
 * @param width      掩膜宽度
 * @param height     掩膜高度
 * @param useSpline  样条插值
 * @param maskID     前景标记值
 * @param supersample 超采样倍率（1=不超采样，5=5x超采样）
 */
inline Image2D<uint8_t> ContourToMask(
    const Contour2D& contour,
    int width, int height,
    bool useSpline = false,
    uint8_t maskID = 255,
    int supersample = 1)
{
    if (contour.size() < 3) return Image2D<uint8_t>(width, height, 0);

    int ss = std::max(1, supersample);
    int ssW = width * ss, ssH = height * ss;

    // 缩放轮廓到超采样分辨率
    Contour2D ssContour;
    ssContour.reserve(contour.size());
    for (const auto& p : contour) {
        ssContour.push_back({p.x * ss, p.y * ss});
    }

    // 生成边缘
    Image2D<uint8_t> edgeMask = ContourToEdgeMask(ssContour, ssW, ssH, useSpline, 1);

    // Floodfill 标记外部
    std::vector<bool> external(edgeMask.size(), false);
    std::queue<int> queue;

    auto tryPush = [&](int x, int y) {
        if (x < 0 || x >= ssW || y < 0 || y >= ssH) return;
        int idx = y * ssW + x;
        if (!external[idx] && edgeMask[idx] == 0) {
            external[idx] = true;
            queue.push(idx);
        }
    };

    // 从四条边界开始
    for (int x = 0; x < ssW; ++x) { tryPush(x, 0); tryPush(x, ssH - 1); }
    for (int y = 0; y < ssH; ++y) { tryPush(0, y); tryPush(ssW - 1, y); }

    while (!queue.empty()) {
        int idx = queue.front(); queue.pop();
        int x = idx % ssW, y = idx / ssW;
        tryPush(x - 1, y); tryPush(x + 1, y);
        tryPush(x, y - 1); tryPush(x, y + 1);
    }

    // 超采样掩膜：非外部即为前景
    Image2D<uint8_t> ssMask(ssW, ssH, 0);
    for (size_t i = 0; i < ssMask.size(); ++i) {
        if (!external[i]) ssMask[i] = 1;
    }

    if (ss == 1) {
        // 直接输出
        Image2D<uint8_t> result(width, height, 0);
        for (size_t i = 0; i < result.size(); ++i)
            result[i] = ssMask[i] ? maskID : 0;
        return result;
    }

    // 降采样：超采样区域投票（>50% 前景则为前景）
    Image2D<uint8_t> result(width, height, 0);
    int halfVote = (ss * ss) / 2;
    for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
        int count = 0;
        for (int dy = 0; dy < ss; ++dy)
        for (int dx = 0; dx < ss; ++dx) {
            if (ssMask.at(x * ss + dx, y * ss + dy)) ++count;
        }
        result.at(x, y) = (count > halfVote) ? maskID : 0;
    }
    return result;
}

// ============================================================
//  掩膜 → 轮廓
// ============================================================
/**
 * 从二值掩膜中提取有序轮廓点。
 *
 * 算法：
 *   1. 对掩膜做单像素腐蚀，得到内部区域
 *   2. 原始 - 腐蚀 = 边界像素
 *   3. 使用 Moore Boundary Tracing 提取有序边界
 *   4. 可选样条平滑
 *
 * @param mask       二值掩膜
 * @param maskID     前景值
 * @param smooth     是否样条平滑
 * @return           多个轮廓（可能有多个连通域）
 */
inline std::vector<Contour2D> MaskToContours(
    const Image2D<uint8_t>& mask,
    uint8_t maskID = 255,
    bool smooth = false)
{
    std::vector<Contour2D> result;

    // 先做连通域标记
    auto lr = LabelConnectedComponents2D(mask, maskID, true);

    for (int label = 1; label <= lr.domainCount; ++label) {
        if (lr.areas[label] < 3) continue; // 太小的域跳过

        // 从标签图提取该域的掩膜
        Image2D<uint8_t> domainMask(mask.width, mask.height, 0);
        for (size_t i = 0; i < mask.size(); ++i) {
            if (lr.labels[i] == label) domainMask[i] = maskID;
        }

        // 提取边界
        auto boundary = ExtractBoundary2D(domainMask, maskID);
        if (boundary.size() < 3) continue;

        // 转换为浮点轮廓
        Contour2D contour;
        contour.reserve(boundary.size());
        for (const auto& p : boundary) {
            contour.push_back({static_cast<float>(p.x), static_cast<float>(p.y)});
        }

        if (smooth && contour.size() >= 4) {
            contour = InterpolateSpline(contour, 1);
        }

        result.push_back(std::move(contour));
    }

    return result;
}

// ============================================================
//  逐层体积轮廓转掩膜
// ============================================================
/**
 * 将每层的 2D 轮廓列表转换为 3D 掩膜。
 *
 * @param contoursBySlice 每层的轮廓（外层 index = z 层号）
 * @param width, height, depth  体数据尺寸
 * @param maskID     前景值
 */
inline Image3D<uint8_t> ContoursToVolumeMask(
    const std::vector<ContourSlice>& contoursBySlice,
    int width, int height, int depth,
    bool useSpline = false,
    uint8_t maskID = 255)
{
    Image3D<uint8_t> volume(width, height, depth, 0);

    int nSlices = std::min(depth, static_cast<int>(contoursBySlice.size()));
    for (int z = 0; z < nSlices; ++z) {
        // 每层可能有多个轮廓
        for (const auto& contour : contoursBySlice[z]) {
            auto sliceMask = ContourToMask(contour, width, height, useSpline, maskID);
            // XOR 合并（处理嵌套轮廓/孔洞）
            for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                if (sliceMask.at(x, y) == maskID) {
                    auto& v = volume.at(x, y, z);
                    v = (v == maskID) ? 0 : maskID; // XOR
                }
            }
        }
    }
    return volume;
}

} // namespace rtac
