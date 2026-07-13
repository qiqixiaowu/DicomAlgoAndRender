/**
 * ============================================================
 *  医学图像分割算法演示 (Medical Image Segmentation Demo)
 * ============================================================
 *
 * 本程序演示从 AlgoSegmentationConan 工程提炼出的 7 种核心分割算法：
 *
 *   1. Chan-Vese Level Set    — 基于区域统计的水平集
 *   2. DRLSE                  — 距离正则化水平集
 *   3. Snake + GVF            — 参数化活动轮廓 + 梯度向量流
 *   4. Graph Cut              — 基于图割的交互式分割
 *   5. Region Growing         — 区域生长
 *   6. Dense CRF              — 全连接条件随机场后处理
 *   7. Sliding Window         — 滑动窗口推理框架
 *   8. Two-Stage              — 粗到精两阶段分割
 *
 * 创建合成测试图像进行演示，输出 PGM 格式结果图像。
 * ============================================================
 */

#include <iostream>
#include <cmath>
#include <chrono>

#include "../include/image2d.hpp"
#include "../include/chan_vese.hpp"
#include "../include/drlse.hpp"
#include "../include/snake_gvf.hpp"
#include "../include/graph_cut.hpp"
#include "../include/region_growing.hpp"
#include "../include/dense_crf.hpp"
#include "../include/sliding_window.hpp"
#include "../include/two_stage.hpp"

using namespace medseg;

// 计时辅助
class Timer {
    std::chrono::high_resolution_clock::time_point t0;
public:
    Timer() : t0(std::chrono::high_resolution_clock::now()) {}
    double elapsedMs() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// ============================================================
// 创建合成测试图像
// ============================================================
/**
 * 生成用于测试的合成医学图像：
 * - 暗背景 (灰度 ~30)
 * - 亮圆形目标区域 (灰度 ~180)，模拟器官截面
 * - 加入高斯噪声
 */
Image2D<float> createSyntheticImage(int W, int H) {
    Image2D<float> img(W, H);
    float cx = W / 2.0f, cy = H / 2.0f;
    float r1 = std::min(W, H) * 0.3f;  // 大目标半径
    float r2 = std::min(W, H) * 0.12f; // 小目标半径

    // 简单的伪随机生成器（可预测结果）
    uint32_t seed = 12345;
    auto pseudoRand = [&seed]() -> float {
        seed = seed * 1103515245 + 12345;
        return static_cast<float>((seed >> 16) & 0x7FFF) / 32767.0f;
    };

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float dx = x - cx, dy = y - cy;
            float dist = std::sqrt(dx * dx + dy * dy);

            float val = 30.0f; // 背景
            if (dist < r1) {
                val = 180.0f; // 大目标
            }
            // 加一个偏移小目标
            float dx2 = x - cx * 0.5f, dy2 = y - cy * 0.5f;
            if (std::sqrt(dx2*dx2 + dy2*dy2) < r2) {
                val = 160.0f;
            }

            // 加高斯噪声
            val += (pseudoRand() - 0.5f) * 20.0f;
            img.at(x, y) = std::max(0.0f, std::min(255.0f, val));
        }
    }
    return img;
}

// 创建圆形掩码（初始化用）
Image2D<uint8_t> createCircleMask(int W, int H, float cx, float cy, float r) {
    Image2D<uint8_t> mask(W, H, 0);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float dx = x - cx, dy = y - cy;
            if (std::sqrt(dx*dx + dy*dy) < r) {
                mask.at(x, y) = 255;
            }
        }
    }
    return mask;
}

// ============================================================
// Demo 1: Chan-Vese Level Set
// ============================================================
void demoChanVese(const Image2D<float>& image) {
    std::cout << "\n========== [1] Chan-Vese Level Set ==========\n";
    std::cout << "原理: 基于区域统计的水平集演化，最小化内外均值拟合能量\n";
    std::cout << "公式: dφ/dt = δ(φ)[μκ - λ₁(I-c₁)² + λ₂(I-c₂)²]\n\n";

    // 初始掩码（比目标大一些的圆）
    auto initMask = createCircleMask(
        image.width, image.height,
        image.width/2.0f, image.height/2.0f,
        std::min(image.width, image.height) * 0.4f);

    ChanVeseParams params;
    params.iterations = 200;
    params.mu = 0.2f;
    params.useLocalMean = false;

    Timer timer;
    ChanVese2D cv;
    auto result = cv.segment(image, initMask, params);
    double ms = timer.elapsedMs();

    // 统计
    int fgCount = 0;
    for (int i = 0; i < result.mask.size(); ++i)
        if (result.mask[i] > 0) fgCount++;

    std::cout << "  迭代次数: " << params.iterations << "\n";
    std::cout << "  前景像素: " << fgCount << " / " << result.mask.size() << "\n";
    std::cout << "  耗时: " << ms << " ms\n";

    savePGM("result_chanvese.pgm", result.mask);
    std::cout << "  结果已保存: result_chanvese.pgm\n";
}

// ============================================================
// Demo 2: DRLSE
// ============================================================
void demoDRLSE(const Image2D<float>& image) {
    std::cout << "\n========== [2] DRLSE (Distance Regularized Level Set) ==========\n";
    std::cout << "原理: 在 Level Set 能量中加入距离正则项，无需重初始化\n";
    std::cout << "公式: dφ/dt = μ·div(dp(|∇φ|)·∇φ) + λ·δ(φ)·div(g∇φ/|∇φ|) + α·g·δ(φ)\n\n";

    // 初始掩码接近目标大小但略小，膨胀到边界
    auto initMask = createCircleMask(
        image.width, image.height,
        image.width/2.0f, image.height/2.0f,
        std::min(image.width, image.height) * 0.25f);

    DRLSEParams params;
    params.iterations = 500;
    params.mu = 0.01f;
    params.alpha = -15.0f;
    params.lambda = 1.0f;
    params.epsilon = 1.5f;
    params.sigma = 3.0f;
    params.timestep = 1.0f;
    params.doubleWell = false;

    Timer timer;
    DRLSE2D drlse;
    auto result = drlse.segment(image, initMask, params);
    double ms = timer.elapsedMs();

    int fgCount = 0;
    for (int i = 0; i < result.size(); ++i) if (result[i] > 0) fgCount++;

    std::cout << "  前景像素: " << fgCount << "\n";
    std::cout << "  耗时: " << ms << " ms\n";
    savePGM("result_drlse.pgm", result);
    std::cout << "  结果已保存: result_drlse.pgm\n";
}

// ============================================================
// Demo 3: Snake + GVF
// ============================================================
void demoSnakeGVF(const Image2D<float>& image) {
    std::cout << "\n========== [3] Snake + GVF ==========\n";
    std::cout << "原理: 参数化轮廓点在内力(弹性+刚度)和外力(GVF场)驱动下演化\n";
    std::cout << "GVF: u←u+μ∇²u-|∇f|²(u-fx), v←v+μ∇²v-|∇f|²(v-fy)\n\n";

    // 初始化为绕目标的圆形轮廓
    auto initContour = SnakeContour::circleContour(
        image.width/2.0f, image.height/2.0f,
        std::min(image.width, image.height) * 0.35f, 80);

    SnakeParams params;
    params.iterations = 150;
    params.gvfIterations = 60;

    Timer timer;
    SnakeContour snake;
    auto contour = snake.segment(image, initContour, params);
    double ms = timer.elapsedMs();

    auto mask = SnakeContour::contourToMask(contour, image.width, image.height);

    int fgCount = 0;
    for (int i = 0; i < mask.size(); ++i) if (mask[i] > 0) fgCount++;

    std::cout << "  轮廓点数: " << contour.size() << "\n";
    std::cout << "  前景像素: " << fgCount << "\n";
    std::cout << "  耗时: " << ms << " ms\n";
    savePGM("result_snake.pgm", mask);
    std::cout << "  结果已保存: result_snake.pgm\n";
}

// ============================================================
// Demo 4: Graph Cut
// ============================================================
void demoGraphCut(const Image2D<float>& image) {
    std::cout << "\n========== [4] Graph Cut ==========\n";
    std::cout << "原理: 像素为节点、灰度差异为边权，最小割=最优分割\n";
    std::cout << "Unary: -log(P(I|label)+e), Pairwise: exp(-(Ip-Iq)^2/2s^2)\n\n";

    // 缩小图像以加速 BFS 最大流（Edmonds-Karp 在大图上较慢）
    int gcW = 64, gcH = 64;
    auto gcImg = TwoStageSegmentor::resize(image, gcW, gcH);
    int W = gcW, H = gcH;

    // 创建种子：中心区域为前景种子，四角为背景种子
    Image2D<uint8_t> fgSeeds(W, H, 0), bgSeeds(W, H, 0);
    float cx = W / 2.0f, cy = H / 2.0f;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float dx = x - cx, dy = y - cy;
            float dist = std::sqrt(dx*dx + dy*dy);
            // 中心小圆为前景种子
            if (dist < std::min(W, H) * 0.1f)
                fgSeeds.at(x, y) = 255;
            // 边缘为背景种子
            if (x < 5 || x >= W - 5 || y < 5 || y >= H - 5)
                bgSeeds.at(x, y) = 255;
        }
    }

    GraphCutParams params;
    params.lambda = 10.0f;
    params.sigma = 30.0f;

    Timer timer;
    GraphCutSegmentor gc;
    auto mask = gc.segment(gcImg, fgSeeds, bgSeeds, params);
    double ms = timer.elapsedMs();

    int fgCount = 0;
    for (int i = 0; i < mask.size(); ++i) if (mask[i] > 0) fgCount++;

    std::cout << "  前景像素: " << fgCount << "\n";
    std::cout << "  耗时: " << ms << " ms\n";
    savePGM("result_graphcut.pgm", mask);
    std::cout << "  结果已保存: result_graphcut.pgm\n";
}

// ============================================================
// Demo 5: Region Growing
// ============================================================
void demoRegionGrowing(const Image2D<float>& image) {
    std::cout << "\n========== [5] Region Growing ==========\n";
    std::cout << "原理: 从种子点BFS生长，按相似度准则逐像素扩展区域\n";
    std::cout << "准则: |I(q) - meanSeed| ≤ tolerance\n\n";

    // 种子点选在目标中心
    std::vector<std::pair<int,int>> seeds = {
        {image.width / 2, image.height / 2}
    };

    RegionGrowParams params;
    params.criterion = GrowCriterion::Difference;
    params.tolerance = 40.0f;
    params.connectivity = 8;

    Timer timer;
    RegionGrowing2D rg;
    auto result = rg.segment(image, seeds, params);
    double ms = timer.elapsedMs();

    std::cout << "  种子位置: (" << seeds[0].first << ", " << seeds[0].second << ")\n";
    std::cout << "  生长体素: " << result.voxelCount << "\n";
    std::cout << "  均值: " << result.meanIntensity << "\n";
    std::cout << "  耗时: " << ms << " ms\n";
    savePGM("result_regiongrow.pgm", result.mask);
    std::cout << "  结果已保存: result_regiongrow.pgm\n";
}

// ============================================================
// Demo 6: Dense CRF
// ============================================================
void demoDenseCRF(const Image2D<float>& image) {
    std::cout << "\n========== [6] Dense CRF ==========\n";
    std::cout << "原理: 均值场近似推理，优化 unary + pairwise 能量\n";
    std::cout << "核心: Q <- softmax(-U - Sum_k pairwise_k(Q))\n\n";

    // 使用缩小图像加速 CRF 演示（Permutohedral Lattice 在大图上较慢）
    int W = image.width / 2, H = image.height / 2;
    auto smallImg = TwoStageSegmentor::resize(image, W, H);
    Image2D<uint8_t> initLabels(W, H, 0);
    float cx = W / 2.0f, cy = H / 2.0f;
    float r = std::min(W, H) * 0.3f;

    uint32_t noiseSeed = 67890;
    auto prand = [&noiseSeed]() -> float {
        noiseSeed = noiseSeed * 1103515245 + 12345;
        return static_cast<float>((noiseSeed >> 16) & 0x7FFF) / 32767.0f;
    };

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float dx = x - cx, dy = y - cy;
            bool inCircle = std::sqrt(dx*dx + dy*dy) < r;
            // 翻转 10% 像素标签来模拟噪声初始分割
            bool flip = (prand() < 0.1f);
            initLabels.at(x, y) = (inCircle ^ flip) ? 1 : 0;
        }
    }

    // 生成 unary 概率
    auto unaryProbs = DenseCRF2D::labelToUnary(initLabels, 2, 0.7f);

    DenseCRFParams params;
    params.iterations = 5;

    Timer timer;
    DenseCRF2D crf;
    auto result = crf.inference(smallImg, unaryProbs, 2, params);
    double ms = timer.elapsedMs();

    // 转换为 0/255 掩码
    Image2D<uint8_t> mask(W, H, 0);
    for (int i = 0; i < W * H; ++i) mask[i] = (result[i] == 1) ? 255 : 0;

    int fgCount = 0;
    for (int i = 0; i < mask.size(); ++i) if (mask[i] > 0) fgCount++;

    std::cout << "  CRF 迭代: " << params.iterations << "\n";
    std::cout << "  前景像素: " << fgCount << "\n";
    std::cout << "  耗时: " << ms << " ms\n";
    savePGM("result_densecrf.pgm", mask);
    std::cout << "  结果已保存: result_densecrf.pgm\n";
}

// ============================================================
// Demo 7: Sliding Window
// ============================================================
void demoSlidingWindow(const Image2D<float>& image) {
    std::cout << "\n========== [7] Sliding Window Inference ==========\n";
    std::cout << "原理: 大图切分为重叠小块推理，高斯加权融合\n";
    std::cout << "公式: result(x,y) = Σ[output·w(x,y)] / Σ[w(x,y)]\n\n";

    // 模拟推理回调：对 patch 做简单阈值分割模拟
    auto mockInference = [](const Image2D<float>& patch, int numClasses) -> std::vector<float> {
        int N = patch.size();
        std::vector<float> probs(N * numClasses);
        for (int i = 0; i < N; ++i) {
            float p_fg = (patch[i] > 100.0f) ? 0.9f : 0.1f;
            probs[i * numClasses + 0] = 1.0f - p_fg; // P(bg)
            probs[i * numClasses + 1] = p_fg;         // P(fg)
        }
        return probs;
    };

    SlidingWindowParams params;
    params.patchWidth = 64;
    params.patchHeight = 64;
    params.overlapRatio = 0.5f;
    params.useGaussianWeighting = true;

    Timer timer;
    SlidingWindowInference sw;
    auto probs = sw.infer(image, mockInference, 2, params);
    auto mask = SlidingWindowInference::argmax(probs, image.width, image.height, 2);
    double ms = timer.elapsedMs();

    // 转 0/255
    Image2D<uint8_t> visMask(image.width, image.height, 0);
    for (int i = 0; i < mask.size(); ++i) visMask[i] = (mask[i] == 1) ? 255 : 0;

    auto patches = SlidingWindowInference::computePatches(
        image.width, image.height, params.patchWidth, params.patchHeight, params.overlapRatio);

    std::cout << "  Patch 数: " << patches.size() << " (" << params.patchWidth << "x" << params.patchHeight << ")\n";
    std::cout << "  重叠率: " << params.overlapRatio * 100 << "%\n";
    std::cout << "  Gaussian weighting: " << (params.useGaussianWeighting ? "Yes" : "No") << "\n";
    std::cout << "  耗时: " << ms << " ms\n";
    savePGM("result_slidingwindow.pgm", visMask);
    std::cout << "  结果已保存: result_slidingwindow.pgm\n";
}

// ============================================================
// Demo 8: Two-Stage
// ============================================================
void demoTwoStage(const Image2D<float>& image) {
    std::cout << "\n========== [8] Two-Stage (Coarse-to-Fine) ==========\n";
    std::cout << "原理: 低分辨率粗分割定位 → 高分辨率精细分割\n";
    std::cout << "映射: BBox_orig = BBox_coarse × (W/W_coarse)\n\n";

    // 粗分割回调（简单阈值）
    auto coarseFunc = [](const Image2D<float>& img) -> Image2D<float> {
        Image2D<float> prob(img.width, img.height, 0.0f);
        for (int i = 0; i < img.size(); ++i) {
            prob[i] = (img[i] > 80.0f) ? 0.8f : 0.2f;
        }
        return prob;
    };

    // 精分割回调（基于灰度的软分割）
    auto fineFunc = [](const Image2D<float>& img) -> Image2D<float> {
        Image2D<float> prob(img.width, img.height, 0.0f);
        for (int i = 0; i < img.size(); ++i) {
            // sigmoid-like 概率
            float x = (img[i] - 100.0f) / 30.0f;
            prob[i] = 1.0f / (1.0f + std::exp(-x));
        }
        return prob;
    };

    TwoStageParams params;
    params.coarseScale = 0.25f;
    params.bboxMargin = 0.15f;

    Timer timer;
    TwoStageSegmentor ts;
    auto mask = ts.segment(image, coarseFunc, fineFunc, params);
    double ms = timer.elapsedMs();

    int fgCount = 0;
    for (int i = 0; i < mask.size(); ++i) if (mask[i] > 0) fgCount++;

    std::cout << "  粗分割缩放: " << params.coarseScale << "x\n";
    std::cout << "  BBox 扩展: " << params.bboxMargin * 100 << "%\n";
    std::cout << "  前景像素: " << fgCount << "\n";
    std::cout << "  耗时: " << ms << " ms\n";
    savePGM("result_twostage.pgm", mask);
    std::cout << "  结果已保存: result_twostage.pgm\n";
}

// ============================================================
// Main
// ============================================================
int main() {
    std::cout << "============================================================\n";
    std::cout << "  医学图像分割算法演示\n";
    std::cout << "  Medical Image Segmentation Algorithm Demo\n";
    std::cout << "  提炼自 AlgoSegmentationConan 工程\n";
    std::cout << "============================================================\n";

    // 创建合成测试图像
    const int W = 256, H = 256;
    auto image = createSyntheticImage(W, H);
    savePGM("input_synthetic.pgm", image);
    std::cout << "\n合成测试图像: " << W << "x" << H << " (已保存 input_synthetic.pgm)\n";

    // 运行所有演示
    demoChanVese(image);
    demoDRLSE(image);
    demoSnakeGVF(image);
    demoGraphCut(image);
    demoRegionGrowing(image);
    demoDenseCRF(image);
    demoSlidingWindow(image);
    demoTwoStage(image);

    std::cout << "\n============================================================\n";
    std::cout << "  所有演示完成！结果保存为 PGM 图像文件。\n";
    std::cout << "============================================================\n";

    return 0;
}
