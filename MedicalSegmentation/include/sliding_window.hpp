#pragma once
/**
 * ============================================================
 *  滑动窗口推理框架 (Sliding Window Inference)
 * ============================================================
 *
 * 【算法原理】
 *   当输入图像尺寸远大于网络接受的 patch 尺寸时，需要将大图切分为
 *   小的 patch 逐个推理，然后将结果拼接回原图尺寸。
 *
 *   基本流程：
 *     1. 预处理：重采样到目标 spacing、归一化
 *     2. Patch 切分：按步长（stride）生成覆盖整个图像的 patch 坐标
 *     3. 逐 Patch 推理：调用网络/分割器对每个 patch 执行推理
 *     4. 结果融合：将各 patch 的概率/标签映射回原图并融合
 *     5. 后处理：阈值化、连通域过滤、恢复原始分辨率
 *
 * 【关键技术：高斯加权融合】
 *   相邻 patch 之间通常有重叠区域（overlap），简单的直接拼接
 *   会在边界产生不连续伪影。
 *
 *   解决方案：对每个 patch 的输出乘以 2D/3D 高斯权重，然后
 *   在重叠区域做加权平均：
 *
 *     weight(x,y) = exp(-((x-cx)² + (y-cy)²) / (2·σ²))
 *
 *     result(x,y) = Σ_patch [ output_patch(x,y) · weight(x,y) ]
 *                   / Σ_patch [ weight(x,y) ]
 *
 *   这样边缘区域贡献小、中心区域贡献大，过渡自然平滑。
 *
 * 【Patch 计算公式】
 *   给定图像尺寸 S = (Sx, Sy)，patch 尺寸 P = (Px, Py)，
 *   重叠比例 r：
 *
 *     stride = P · (1 - r)
 *     Nx = ceil((Sx - Px) / stride_x) + 1
 *     Ny = ceil((Sy - Py) / stride_y) + 1
 *
 *   若图像不够被整除，需要 padding（通常用边界值或零填充）。
 *
 * 【这里的实现】
 *   提供一个 CPU 版本的滑窗框架，"推理函数" 以回调方式注入，
 *   可以接入任何分割器（传统算法或深度学习推理引擎）。
 *
 * 【参考】
 *   Isensee F, et al. "nnU-Net: a self-configuring method for deep
 *   learning-based biomedical image segmentation." Nature Methods, 2021.
 * ============================================================
 */

#include "image2d.hpp"
#include <functional>
#include <cmath>
#include <vector>
#include <algorithm>

namespace medseg {

// ============================================================
// 参数
// ============================================================
struct SlidingWindowParams {
    int   patchWidth    = 128;   // Patch 宽度
    int   patchHeight   = 128;   // Patch 高度
    float overlapRatio  = 0.5f;  // 重叠比例 (0~1)
    bool  useGaussianWeighting = true;  // 是否使用高斯加权融合
    float gaussianSigmaRatio = 0.125f;  // σ = patchSize * ratio
    int   padMode       = 0;     // 0=零填充, 1=边界复制
};

// ============================================================
// Patch 坐标描述
// ============================================================
struct PatchCoord {
    int x, y;   // patch 左上角在原图中的坐标
    int w, h;   // patch 尺寸
};

/**
 * 推理回调函数类型
 * 输入: patch 图像 (float, patchW × patchH)
 * 输出: 推理结果概率图 (float, patchW × patchH × numClasses)
 *       按 [class0_pixel0, class1_pixel0, ..., class0_pixel1, ...] 排列
 */
using InferenceCallback = std::function<std::vector<float>(
    const Image2D<float>& patch, int numClasses)>;

// ============================================================
// 滑动窗口推理器
// ============================================================
class SlidingWindowInference {
public:
    /**
     * 执行滑动窗口推理
     * @param image       输入图像
     * @param inferFunc   推理回调函数
     * @param numClasses  类别数
     * @param params      参数
     * @return            每个像素的类别概率 (W×H×numClasses)
     */
    std::vector<float> infer(
        const Image2D<float>& image,
        InferenceCallback inferFunc,
        int numClasses,
        const SlidingWindowParams& params = {});

    /**
     * 从概率图取 argmax 得到标签
     */
    static Image2D<uint8_t> argmax(
        const std::vector<float>& probs, int width, int height, int numClasses);

    /**
     * 计算 patch 坐标列表
     */
    static std::vector<PatchCoord> computePatches(
        int imgW, int imgH, int patchW, int patchH, float overlapRatio);

private:
    // 生成 2D 高斯权重图
    static Image2D<float> gaussianWeight(int w, int h, float sigmaRatio);

    // 提取 patch（带 padding）
    static Image2D<float> extractPatch(
        const Image2D<float>& image, int px, int py, int pw, int ph, int padMode);
};

// ============================================================
//                      实现
// ============================================================

inline std::vector<PatchCoord> SlidingWindowInference::computePatches(
    int imgW, int imgH, int patchW, int patchH, float overlapRatio)
{
    float strideX = patchW * (1.0f - overlapRatio);
    float strideY = patchH * (1.0f - overlapRatio);
    strideX = std::max(strideX, 1.0f);
    strideY = std::max(strideY, 1.0f);

    int nx = static_cast<int>(std::ceil((imgW - patchW) / strideX)) + 1;
    int ny = static_cast<int>(std::ceil((imgH - patchH) / strideY)) + 1;

    std::vector<PatchCoord> patches;
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            int px = static_cast<int>(ix * strideX);
            int py = static_cast<int>(iy * strideY);
            // 确保最后一个 patch 不超出边界
            px = std::min(px, std::max(0, imgW - patchW));
            py = std::min(py, std::max(0, imgH - patchH));
            patches.push_back({px, py, patchW, patchH});
        }
    }
    return patches;
}

inline Image2D<float> SlidingWindowInference::gaussianWeight(int w, int h, float sigmaRatio) {
    Image2D<float> weight(w, h);
    float sigmaX = w * sigmaRatio;
    float sigmaY = h * sigmaRatio;
    float cx = (w - 1) * 0.5f;
    float cy = (h - 1) * 0.5f;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float dx = (x - cx) / sigmaX;
            float dy = (y - cy) / sigmaY;
            weight.at(x, y) = std::exp(-0.5f * (dx * dx + dy * dy));
        }
    }
    return weight;
}

inline Image2D<float> SlidingWindowInference::extractPatch(
    const Image2D<float>& image, int px, int py, int pw, int ph, int padMode)
{
    Image2D<float> patch(pw, ph);
    for (int y = 0; y < ph; ++y) {
        for (int x = 0; x < pw; ++x) {
            int ix = px + x, iy = py + y;
            if (padMode == 1) {
                // 边界复制
                patch.at(x, y) = image.atClamped(ix, iy);
            } else {
                // 零填充
                if (image.inBounds(ix, iy))
                    patch.at(x, y) = image.at(ix, iy);
                else
                    patch.at(x, y) = 0.0f;
            }
        }
    }
    return patch;
}

inline std::vector<float> SlidingWindowInference::infer(
    const Image2D<float>& image,
    InferenceCallback inferFunc,
    int numClasses,
    const SlidingWindowParams& params)
{
    int W = image.width, H = image.height;
    int N = W * H;

    // 1. 计算 patch 坐标
    auto patches = computePatches(W, H, params.patchWidth, params.patchHeight, params.overlapRatio);

    // 2. 准备高斯权重
    Image2D<float> gweight;
    if (params.useGaussianWeighting) {
        gweight = gaussianWeight(params.patchWidth, params.patchHeight, params.gaussianSigmaRatio);
    }

    // 3. 累积缓冲区
    std::vector<float> accumProbs(N * numClasses, 0.0f);
    std::vector<float> accumWeights(N, 0.0f);

    // 4. 逐 patch 推理并融合
    for (auto& pc : patches) {
        // 提取 patch
        auto patch = extractPatch(image, pc.x, pc.y, pc.w, pc.h, params.padMode);

        // 推理
        auto patchResult = inferFunc(patch, numClasses);

        // 融合到全局
        for (int py = 0; py < pc.h; ++py) {
            for (int px = 0; px < pc.w; ++px) {
                int gx = pc.x + px, gy = pc.y + py;
                if (gx < 0 || gx >= W || gy < 0 || gy >= H) continue;

                int gidx = gy * W + gx;
                int pidx = py * pc.w + px;

                float w = params.useGaussianWeighting ? gweight.at(px, py) : 1.0f;

                for (int c = 0; c < numClasses; ++c) {
                    accumProbs[gidx * numClasses + c] += w * patchResult[pidx * numClasses + c];
                }
                accumWeights[gidx] += w;
            }
        }
    }

    // 5. 归一化
    for (int i = 0; i < N; ++i) {
        if (accumWeights[i] > 1e-8f) {
            for (int c = 0; c < numClasses; ++c) {
                accumProbs[i * numClasses + c] /= accumWeights[i];
            }
        }
    }

    return accumProbs;
}

inline Image2D<uint8_t> SlidingWindowInference::argmax(
    const std::vector<float>& probs, int width, int height, int numClasses)
{
    Image2D<uint8_t> result(width, height, 0);
    int N = width * height;
    for (int i = 0; i < N; ++i) {
        int bestClass = 0;
        float bestProb = probs[i * numClasses];
        for (int c = 1; c < numClasses; ++c) {
            if (probs[i * numClasses + c] > bestProb) {
                bestProb = probs[i * numClasses + c];
                bestClass = c;
            }
        }
        result[i] = static_cast<uint8_t>(bestClass);
    }
    return result;
}

} // namespace medseg
