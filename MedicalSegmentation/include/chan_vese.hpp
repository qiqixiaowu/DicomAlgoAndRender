#pragma once
/**
 * ============================================================
 *  Chan-Vese Level Set 分割 (2D)
 * ============================================================
 *
 * 【算法原理】
 *   Chan-Vese 模型是一种基于区域的活动轮廓方法，不依赖图像梯度，
 *   适用于边缘模糊、噪声较大的医学图像。
 *
 *   能量泛函：
 *     E(φ) = μ·∫|∇H(φ)|dx                    // 曲线长度正则项
 *          + λ₁·∫(I - c₁)²·H(φ)dx             // 内部区域拟合项
 *          + λ₂·∫(I - c₂)²·(1-H(φ))dx         // 外部区域拟合项
 *
 *   其中：
 *     φ:   水平集函数（level set function），零水平集 {φ=0} 即分割轮廓
 *     H:   Heaviside 阶跃函数
 *     c₁:  前景（φ>0区域）的均值灰度
 *     c₂:  背景（φ<0区域）的均值灰度
 *     μ:   曲率正则系数
 *     λ₁,λ₂: 区域拟合权重
 *
 *   对 φ 求变分得到演化方程：
 *     ∂φ/∂t = δ(φ) [ μ·κ - λ₁·(I-c₁)² + λ₂·(I-c₂)² ]
 *
 *   其中 κ 是 φ 的曲率：
 *     κ = div(∇φ/|∇φ|) = (φ_xx·φ_y² + φ_yy·φ_x² - 2·φ_x·φ_y·φ_xy) / (φ_x²+φ_y²)^{3/2}
 *
 * 【关键技术】
 *   1. 窄带 (Narrow Band)：仅在 |φ| < band_width 的像素处做演化计算，
 *      大幅减少计算量（从 O(N²) 降至 O(N·边界长度)）
 *   2. Sussman 重初始化：演化过程中 φ 会偏离符号距离函数性质，
 *      通过求解 |∇ψ|=1 的 Hamilton-Jacobi 方程重新修正
 *   3. CFL 条件自适应步长：Δt = 0.45 / max(|Δφ|) 保证数值稳定
 *   4. 全局/局部均值力：全局均值适用于均匀图像，局部均值适用于
 *      灰度非均匀图像（如MR中的偏置场伪影）
 *
 * 【接口说明】
 *   输入：
 *     - image:        灰度图像 (float, 值域任意)
 *     - initialMask:  初始前景掩码 (uint8_t, >0 表示前景)
 *     - params:       算法参数（迭代次数、曲率系数、局部窗口半径等）
 *   输出：
 *     - 分割结果掩码 (uint8_t, 255=前景, 0=背景)
 *     - 最终的水平集函数 (可选)
 *
 * 【参考文献】
 *   Chan T, Vese L. "Active contours without edges." IEEE TIP, 2001.
 *   Li C, et al. "Level set evolution without re-initialization." CVPR, 2005.
 * ============================================================
 */

#include "image2d.hpp"

namespace medseg {

// ============================================================
// 参数
// ============================================================
struct ChanVeseParams {
    int   iterations     = 300;    // 最大迭代次数
    float mu             = 0.2f;   // 曲率正则系数
    float lambda1        = 1.0f;   // 内部拟合权重
    float lambda2        = 1.0f;   // 外部拟合权重
    float bandWidth      = 1.2f;   // 窄带半宽
    int   reinitInterval = 5;      // 每隔多少次迭代做一次 Sussman 重初始化
    int   reinitIters    = 2;      // Sussman 重初始化迭代次数
    bool  useLocalMean   = false;  // true=局部均值力, false=全局均值力
    int   localRadius    = 5;      // 局部窗口半径（仅 useLocalMean=true 时生效）
};

// ============================================================
// 结果
// ============================================================
struct ChanVeseResult {
    Image2D<uint8_t>  mask;        // 分割掩码 (0/255)
    Image2D<float>    phi;         // 最终水平集函数
};

// ============================================================
// Chan-Vese 2D Level Set 分割器
// ============================================================
class ChanVese2D {
public:
    /**
     * 执行分割
     * @param image       输入灰度图（float，值域任意）
     * @param initialMask 初始前景掩码（>0 为前景）
     * @param params      算法参数
     * @return 分割结果
     */
    ChanVeseResult segment(const Image2D<float>& image,
                           const Image2D<uint8_t>& initialMask,
                           const ChanVeseParams& params = {});

private:
    int W = 0, H = 0;
    const float EPS = 1e-10f;

    // 从掩码初始化为符号距离函数 (SDF)
    // φ = bwdist(BG) - bwdist(FG) + 1_{FG} - 0.5
    void initSDF(const Image2D<uint8_t>& mask, Image2D<float>& phi);

    // 距离变换（近似：逐行逐列扫描）
    Image2D<float> distanceTransform(const Image2D<uint8_t>& binaryImg);

    // 计算 2D 曲率
    // κ = (φ_xx·φ_y² + φ_yy·φ_x² - 2·φ_x·φ_y·φ_xy) / (|∇φ|³ + ε) · |∇φ|
    float curvature(const Image2D<float>& phi, int x, int y);

    // 提取窄带像素索引（|φ| < bandWidth 的所有位置）
    std::vector<int> narrowBand(const Image2D<float>& phi, float bandWidth);

    // 计算全局前景/背景均值
    void globalMeans(const Image2D<float>& image, const Image2D<float>& phi,
                     float& c1, float& c2);

    // 计算局部均值力
    float localMeanForce(const Image2D<float>& image, const Image2D<float>& phi,
                         int x, int y, int radius);

    // Sussman 重初始化：修正 φ 为符号距离函数
    // 求解 ∂ψ/∂t + S(φ)(|∇ψ|-1) = 0
    void sussmanReinit(Image2D<float>& phi, int iters);

    // Sussman 符号函数：S(φ) = φ / √(φ² + 1)
    float sussmanSign(float phi);
};

// ============================================================
//                      实现
// ============================================================

inline ChanVeseResult ChanVese2D::segment(
    const Image2D<float>& image,
    const Image2D<uint8_t>& initialMask,
    const ChanVeseParams& params)
{
    W = image.width;
    H = image.height;
    assert(W == initialMask.width && H == initialMask.height);

    // 1. 初始化水平集函数
    Image2D<float> phi(W, H);
    initSDF(initialMask, phi);

    // 2. 迭代演化
    for (int iter = 0; iter < params.iterations; ++iter) {
        // 2a. 提取窄带
        auto band = narrowBand(phi, params.bandWidth);
        if (band.empty()) break;

        // 2b. 计算全局均值（即使用局部模式，也需要作为参考）
        float c1, c2;
        globalMeans(image, phi, c1, c2);

        // 2c. 对窄带内每个像素计算 Δφ
        std::vector<float> dphi(band.size(), 0.0f);
        float maxDphi = 0.0f;

        for (size_t i = 0; i < band.size(); ++i) {
            int idx = band[i];
            int x = idx % W, y = idx / W;

            // 曲率项
            float kappa = curvature(phi, x, y);

            // 区域力
            float force;
            if (params.useLocalMean) {
                force = localMeanForce(image, phi, x, y, params.localRadius);
            } else {
                float I = image[idx];
                // F = -λ₁(I-c₁)² + λ₂(I-c₂)²
                force = -params.lambda1 * (I - c1) * (I - c1)
                        + params.lambda2 * (I - c2) * (I - c2);
            }

            dphi[i] = params.mu * kappa + force;
            maxDphi = std::max(maxDphi, std::abs(dphi[i]));
        }

        // 2d. CFL 自适应步长
        float dt = 0.45f / (maxDphi + EPS);

        // 2e. 更新 φ
        for (size_t i = 0; i < band.size(); ++i) {
            phi[band[i]] += dt * dphi[i];
        }

        // 2f. 定期 Sussman 重初始化
        if (params.reinitInterval > 0 && (iter + 1) % params.reinitInterval == 0) {
            sussmanReinit(phi, params.reinitIters);
        }
    }

    // 3. 从 φ 提取掩码：φ > 0 → 前景
    ChanVeseResult result;
    result.phi = phi;
    result.mask = Image2D<uint8_t>(W, H, 0);
    for (int i = 0; i < W * H; ++i) {
        result.mask[i] = (phi[i] > 0) ? 255 : 0;
    }
    return result;
}

inline void ChanVese2D::initSDF(const Image2D<uint8_t>& mask, Image2D<float>& phi) {
    // 构造前景和背景二值图
    Image2D<uint8_t> fg(W, H, 0), bg(W, H, 0);
    for (int i = 0; i < W * H; ++i) {
        if (mask[i] > 0) fg[i] = 255;
        else              bg[i] = 255;
    }

    // 分别做距离变换
    auto distFG = distanceTransform(fg);
    auto distBG = distanceTransform(bg);

    // φ = dist(BG) - dist(FG) + 1_{FG} - 0.5
    for (int i = 0; i < W * H; ++i) {
        float isFG = (mask[i] > 0) ? 1.0f : 0.0f;
        phi[i] = distBG[i] - distFG[i] + isFG - 0.5f;
    }
}

inline Image2D<float> ChanVese2D::distanceTransform(const Image2D<uint8_t>& binary) {
    // 简化的欧几里得距离变换（两遍扫描近似）
    const float INF = static_cast<float>(W + H);
    Image2D<float> dist(W, H, INF);

    // 零值像素距离为0
    for (int i = 0; i < W * H; ++i) {
        if (binary[i] > 0) dist[i] = 0.0f;
    }

    // 前向扫描
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (x > 0) dist.at(x, y) = std::min(dist.at(x, y), dist.at(x-1, y) + 1.0f);
            if (y > 0) dist.at(x, y) = std::min(dist.at(x, y), dist.at(x, y-1) + 1.0f);
        }
    }
    // 反向扫描
    for (int y = H - 1; y >= 0; --y) {
        for (int x = W - 1; x >= 0; --x) {
            if (x < W-1) dist.at(x, y) = std::min(dist.at(x, y), dist.at(x+1, y) + 1.0f);
            if (y < H-1) dist.at(x, y) = std::min(dist.at(x, y), dist.at(x, y+1) + 1.0f);
        }
    }
    return dist;
}

inline float ChanVese2D::curvature(const Image2D<float>& phi, int x, int y) {
    // 中心差分计算一阶和二阶导数
    float phiC = phi.atClamped(x, y);
    float phiL = phi.atClamped(x-1, y);
    float phiR = phi.atClamped(x+1, y);
    float phiU = phi.atClamped(x, y-1);
    float phiD = phi.atClamped(x, y+1);

    float fx = (phiR - phiL) * 0.5f;
    float fy = (phiD - phiU) * 0.5f;
    float fxx = phiR + phiL - 2.0f * phiC;
    float fyy = phiD + phiU - 2.0f * phiC;

    float phiRU = phi.atClamped(x+1, y-1);
    float phiLU = phi.atClamped(x-1, y-1);
    float phiRD = phi.atClamped(x+1, y+1);
    float phiLD = phi.atClamped(x-1, y+1);
    float fxy = (phiRD - phiRU - phiLD + phiLU) * 0.25f;

    float gradSq = fx * fx + fy * fy;
    float gradMag = std::sqrt(gradSq);
    float denom = std::pow(gradSq, 1.5f) + EPS;

    // κ = (fx²·fyy + fy²·fxx - 2·fx·fy·fxy) / (|∇φ|³ + ε) · |∇φ|
    float kappa = (fx*fx*fyy + fy*fy*fxx - 2.0f*fx*fy*fxy) / denom * gradMag;
    return kappa;
}

inline std::vector<int> ChanVese2D::narrowBand(const Image2D<float>& phi, float bandWidth) {
    std::vector<int> indices;
    for (int i = 0; i < W * H; ++i) {
        if (std::abs(phi[i]) <= bandWidth) {
            indices.push_back(i);
        }
    }
    return indices;
}

inline void ChanVese2D::globalMeans(const Image2D<float>& image, const Image2D<float>& phi,
                                     float& c1, float& c2) {
    double sum1 = 0, sum2 = 0;
    int cnt1 = 0, cnt2 = 0;
    for (int i = 0; i < W * H; ++i) {
        if (phi[i] > 0) { sum1 += image[i]; cnt1++; }
        else             { sum2 += image[i]; cnt2++; }
    }
    c1 = (cnt1 > 0) ? static_cast<float>(sum1 / cnt1) : 0.0f;
    c2 = (cnt2 > 0) ? static_cast<float>(sum2 / cnt2) : 0.0f;
}

inline float ChanVese2D::localMeanForce(
    const Image2D<float>& image, const Image2D<float>& phi,
    int x, int y, int radius)
{
    // 在局部窗口内计算前景/背景均值
    double sumIn = 0, sumOut = 0;
    int cntIn = 0, cntOut = 0;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = x + dx, ny = y + dy;
            if (!image.inBounds(nx, ny)) continue;
            float I = image.at(nx, ny);
            if (phi.at(nx, ny) > 0) { sumIn += I; cntIn++; }
            else                      { sumOut += I; cntOut++; }
        }
    }

    float mIn  = (cntIn  > 0) ? static_cast<float>(sumIn  / cntIn)  : 0.0f;
    float mOut = (cntOut > 0) ? static_cast<float>(sumOut / cntOut) : 0.0f;
    float I = image.at(x, y);

    // 局部 Chan-Vese 力
    return -(I - mIn) * (I - mIn) + (I - mOut) * (I - mOut);
}

inline void ChanVese2D::sussmanReinit(Image2D<float>& phi, int iters) {
    float dt = 0.5f;
    Image2D<float> phiOld(W, H);

    for (int iter = 0; iter < iters; ++iter) {
        phiOld = phi;

        for (int y = 1; y < H - 1; ++y) {
            for (int x = 1; x < W - 1; ++x) {
                float p = phiOld.at(x, y);

                // Godunov 上风差分
                float a = p - phiOld.at(x-1, y);   // D⁻_x
                float b = phiOld.at(x+1, y) - p;   // D⁺_x
                float c = p - phiOld.at(x, y-1);   // D⁻_y
                float d = phiOld.at(x, y+1) - p;   // D⁺_y

                float S = sussmanSign(p);
                float G = 0.0f;

                if (S > 0) {
                    float ap = std::max(a, 0.0f), bn = std::min(b, 0.0f);
                    float cp = std::max(c, 0.0f), dn = std::min(d, 0.0f);
                    G = std::sqrt(std::max(ap*ap, bn*bn) + std::max(cp*cp, dn*dn)) - 1.0f;
                } else {
                    float an = std::min(a, 0.0f), bp = std::max(b, 0.0f);
                    float cn = std::min(c, 0.0f), dp = std::max(d, 0.0f);
                    G = std::sqrt(std::max(an*an, bp*bp) + std::max(cn*cn, dp*dp)) - 1.0f;
                }

                phi.at(x, y) = p - dt * S * G;
            }
        }
    }
}

inline float ChanVese2D::sussmanSign(float p) {
    return p / std::sqrt(p * p + 1.0f);
}

} // namespace medseg
