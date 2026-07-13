#pragma once
/**
 * ============================================================
 *  两阶段分割框架 (Coarse-to-Fine Two-Stage Segmentation)
 * ============================================================
 *
 * 【算法原理】
 *   直接对全分辨率大图像做精细分割计算量大、显存占用高。
 *   两阶段策略先用低分辨率快速定位目标区域，再在目标区域的
 *   高分辨率裁剪子图上做精细分割，兼顾速度与精度。
 *
 *   [Stage 1: Coarse Segmentation]
 *     原图 → 降采样（低分辨率）→ 粗分割 → 二值化 → 最大连通域 → BBox
 *
 *   [Stage 2: Fine Segmentation]
 *     原图 → 按 BBox 裁剪（高分辨率子区域）→ 精细分割 → 回映射到原图
 *
 * 【空间映射公式】
 *   粗分割在分辨率 (W₁, H₁) 下得到 BBox = (x₁, y₁, x₂, y₂)
 *   映射到原始分辨率 (W₀, H₀)：
 *
 *     x₀ = x₁ · (W₀ / W₁)
 *     y₀ = y₁ · (H₀ / H₁)
 *
 *   精细分割在原始分辨率的 BBox 区域上操作，结果直接回填。
 *
 * 【BBox 扩展策略】
 *   为避免截断目标边缘，通常将检测到的 BBox 按比例/固定像素扩展：
 *     expanded_bbox = bbox + margin
 *   margin 可以是绝对像素值或 bbox 尺寸的百分比。
 *
 * 【连通域分析】
 *   使用 flood fill 提取最大连通域：
 *     1. 遍历所有前景像素
 *     2. 对每个未标记的前景像素做 BFS/DFS flood fill
 *     3. 记录每个连通域的大小
 *     4. 保留最大的连通域（去除噪声区域）
 *
 * 【这里的实现】
 *   提供框架层，"粗分割"和"精分割"以回调方式注入。
 *   内置连通域分析、BBox 计算、图像裁剪/回填等工具函数。
 *
 * 【参考】
 *   Liu Z, et al. "Liver and liver tumor segmentation with a coarse-to-fine
 *   approach." Medical Image Analysis, 2017.
 * ============================================================
 */

#include "image2d.hpp"
#include <functional>
#include <vector>
#include <stack>
#include <algorithm>

namespace medseg {

// ============================================================
// 参数
// ============================================================
struct TwoStageParams {
    float coarseScale    = 0.25f;  // 粗分割降采样比例（0.25 = 缩小4倍）
    float bboxMargin     = 0.1f;   // BBox 扩展比例（每边扩展 10%）
    float threshold      = 0.5f;   // 二值化阈值
    bool  keepLargest    = true;   // 是否只保留最大连通域
};

// ============================================================
// 边界框
// ============================================================
struct BBox {
    int x0, y0, x1, y1;

    int width()  const { return x1 - x0; }
    int height() const { return y1 - y0; }
    bool valid() const { return x1 > x0 && y1 > y0; }
};

/**
 * 分割回调函数类型
 * 输入: 图像 (float)
 * 输出: 概率图 (float, 0~1 范围)
 */
using SegmentCallback = std::function<Image2D<float>(const Image2D<float>& image)>;

// ============================================================
// 两阶段分割器
// ============================================================
class TwoStageSegmentor {
public:
    /**
     * 执行两阶段分割
     * @param image        原始图像 (float)
     * @param coarseFunc   粗分割回调（输入缩小图 → 输出概率图）
     * @param fineFunc     精分割回调（输入裁剪区域 → 输出概率图）
     * @param params       参数
     * @return             最终分割掩码 (0/255)
     */
    Image2D<uint8_t> segment(
        const Image2D<float>& image,
        SegmentCallback coarseFunc,
        SegmentCallback fineFunc,
        const TwoStageParams& params = {});

    // ========== 工具函数 ==========

    /** 双线性缩放图像 */
    static Image2D<float> resize(const Image2D<float>& src, int newW, int newH);

    /** 从概率图提取 BBox（大于阈值的像素的外接矩形） */
    static BBox computeBBox(const Image2D<float>& probMap, float threshold = 0.5f);

    /** 扩展 BBox（按比例，限制在图像范围内） */
    static BBox expandBBox(const BBox& bbox, float margin, int imgW, int imgH);

    /** 提取最大连通域 */
    static Image2D<uint8_t> largestConnectedComponent(const Image2D<uint8_t>& mask);

    /** 裁剪图像 */
    static Image2D<float> crop(const Image2D<float>& src, const BBox& bbox);

    /** 将裁剪结果回填到原图 */
    static void pasteback(const Image2D<uint8_t>& patch, const BBox& bbox,
                          Image2D<uint8_t>& target);
};

// ============================================================
//                      实现
// ============================================================

inline Image2D<float> TwoStageSegmentor::resize(
    const Image2D<float>& src, int newW, int newH)
{
    Image2D<float> dst(newW, newH);
    float scaleX = static_cast<float>(src.width) / newW;
    float scaleY = static_cast<float>(src.height) / newH;

    for (int y = 0; y < newH; ++y) {
        for (int x = 0; x < newW; ++x) {
            float sx = (x + 0.5f) * scaleX - 0.5f;
            float sy = (y + 0.5f) * scaleY - 0.5f;

            int x0 = static_cast<int>(std::floor(sx));
            int y0 = static_cast<int>(std::floor(sy));
            float dx = sx - x0, dy = sy - y0;

            float v00 = src.atClamped(x0, y0);
            float v10 = src.atClamped(x0+1, y0);
            float v01 = src.atClamped(x0, y0+1);
            float v11 = src.atClamped(x0+1, y0+1);

            dst.at(x, y) = v00*(1-dx)*(1-dy) + v10*dx*(1-dy)
                         + v01*(1-dx)*dy + v11*dx*dy;
        }
    }
    return dst;
}

inline BBox TwoStageSegmentor::computeBBox(
    const Image2D<float>& probMap, float threshold)
{
    int W = probMap.width, H = probMap.height;
    BBox bbox = {W, H, 0, 0};

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (probMap.at(x, y) >= threshold) {
                bbox.x0 = std::min(bbox.x0, x);
                bbox.y0 = std::min(bbox.y0, y);
                bbox.x1 = std::max(bbox.x1, x + 1);
                bbox.y1 = std::max(bbox.y1, y + 1);
            }
        }
    }
    return bbox;
}

inline BBox TwoStageSegmentor::expandBBox(
    const BBox& bbox, float margin, int imgW, int imgH)
{
    int mw = static_cast<int>(bbox.width() * margin);
    int mh = static_cast<int>(bbox.height() * margin);
    return {
        std::max(0, bbox.x0 - mw),
        std::max(0, bbox.y0 - mh),
        std::min(imgW, bbox.x1 + mw),
        std::min(imgH, bbox.y1 + mh)
    };
}

inline Image2D<uint8_t> TwoStageSegmentor::largestConnectedComponent(
    const Image2D<uint8_t>& mask)
{
    int W = mask.width, H = mask.height;
    std::vector<int> labels(W * H, -1);
    int numLabels = 0;
    std::vector<int> sizes;

    // Flood fill 标记连通域
    for (int i = 0; i < W * H; ++i) {
        if (mask[i] == 0 || labels[i] >= 0) continue;

        int labelId = numLabels++;
        sizes.push_back(0);

        std::stack<int> stk;
        stk.push(i);
        labels[i] = labelId;

        while (!stk.empty()) {
            int idx = stk.top(); stk.pop();
            sizes[labelId]++;

            int x = idx % W, y = idx / W;
            // 4-连通
            int dx[] = {1, -1, 0, 0};
            int dy[] = {0, 0, 1, -1};
            for (int d = 0; d < 4; ++d) {
                int nx = x + dx[d], ny = y + dy[d];
                if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                int nidx = ny * W + nx;
                if (mask[nidx] > 0 && labels[nidx] < 0) {
                    labels[nidx] = labelId;
                    stk.push(nidx);
                }
            }
        }
    }

    // 找最大连通域
    if (sizes.empty()) return Image2D<uint8_t>(W, H, 0);

    int largestId = static_cast<int>(
        std::max_element(sizes.begin(), sizes.end()) - sizes.begin());

    Image2D<uint8_t> result(W, H, 0);
    for (int i = 0; i < W * H; ++i) {
        if (labels[i] == largestId) result[i] = 255;
    }
    return result;
}

inline Image2D<float> TwoStageSegmentor::crop(
    const Image2D<float>& src, const BBox& bbox)
{
    int cropW = bbox.width(), cropH = bbox.height();
    Image2D<float> dst(cropW, cropH);
    for (int y = 0; y < cropH; ++y) {
        for (int x = 0; x < cropW; ++x) {
            dst.at(x, y) = src.atClamped(bbox.x0 + x, bbox.y0 + y);
        }
    }
    return dst;
}

inline void TwoStageSegmentor::pasteback(
    const Image2D<uint8_t>& patch, const BBox& bbox, Image2D<uint8_t>& target)
{
    for (int y = 0; y < patch.height; ++y) {
        for (int x = 0; x < patch.width; ++x) {
            int tx = bbox.x0 + x, ty = bbox.y0 + y;
            if (target.inBounds(tx, ty)) {
                target.at(tx, ty) = patch.at(x, y);
            }
        }
    }
}

inline Image2D<uint8_t> TwoStageSegmentor::segment(
    const Image2D<float>& image,
    SegmentCallback coarseFunc,
    SegmentCallback fineFunc,
    const TwoStageParams& params)
{
    int W = image.width, H = image.height;

    // ====== Stage 1: 粗分割 ======
    // 1a. 降采样
    int coarseW = static_cast<int>(W * params.coarseScale);
    int coarseH = static_cast<int>(H * params.coarseScale);
    coarseW = std::max(coarseW, 1);
    coarseH = std::max(coarseH, 1);
    auto coarseImg = resize(image, coarseW, coarseH);

    // 1b. 粗分割推理
    auto coarseProb = coarseFunc(coarseImg);

    // 1c. 在粗分辨率下找 BBox
    BBox coarseBBox = computeBBox(coarseProb, params.threshold);
    if (!coarseBBox.valid()) {
        // 粗分割未检测到目标，返回全零掩码
        return Image2D<uint8_t>(W, H, 0);
    }

    // 1d. 映射 BBox 到原始分辨率
    float scaleX = static_cast<float>(W) / coarseW;
    float scaleY = static_cast<float>(H) / coarseH;
    BBox origBBox = {
        static_cast<int>(coarseBBox.x0 * scaleX),
        static_cast<int>(coarseBBox.y0 * scaleY),
        static_cast<int>(std::ceil(coarseBBox.x1 * scaleX)),
        static_cast<int>(std::ceil(coarseBBox.y1 * scaleY))
    };

    // 1e. 扩展 BBox
    origBBox = expandBBox(origBBox, params.bboxMargin, W, H);

    // ====== Stage 2: 精细分割 ======
    // 2a. 裁剪高分辨率子区域
    auto fineImg = crop(image, origBBox);

    // 2b. 精细分割推理
    auto fineProb = fineFunc(fineImg);

    // 2c. 二值化
    Image2D<uint8_t> fineMask(fineProb.width, fineProb.height, 0);
    for (int i = 0; i < fineProb.size(); ++i) {
        fineMask[i] = (fineProb[i] >= params.threshold) ? 255 : 0;
    }

    // 2d. 最大连通域过滤
    if (params.keepLargest) {
        fineMask = largestConnectedComponent(fineMask);
    }

    // 2e. 回填到全图
    Image2D<uint8_t> result(W, H, 0);
    pasteback(fineMask, origBBox, result);

    return result;
}

} // namespace medseg
