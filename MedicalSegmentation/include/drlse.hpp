#pragma once
/**
 * ============================================================
 *  DRLSE (Distance Regularized Level Set Evolution) — 3D
 * ============================================================
 *
 * 【算法原理】
 *   DRLSE 是 Li Chunming 等人提出的改进 Level Set 方法。
 *   传统 Level Set 在演化过程中需要频繁重初始化以保持 SDF 性质，
 *   DRLSE 在能量中引入距离正则项，自动维持 level set 的规整性，
 *   完全消除或大幅减少重初始化的需要。
 *
 *   DRLSE 能量：
 *     E(φ) = μ·R(φ) + λ·E_ext(φ) + α·A(φ)
 *
 *   其中：
 *     R(φ) = ∫ p(|∇φ|) dx              // 距离正则项
 *     E_ext(φ) = ∫ g·δ(φ)·|∇φ| dx       // 边缘加权长度项
 *     A(φ) = ∫ g·H(-φ) dx               // 面积加速项
 *
 *   距离正则势函数 p(s)：
 *     Double-well：p(s) = { (s-1)²/4,         s ≤ 1
 *                           (1/(2π))·(1-cos(2πs)), s > 1 }
 *     使得 |∇φ| → 1（符号距离函数性质）
 *
 *     Single-well：p(s) = (s-1)²/2
 *     同样驱动 |∇φ| → 1
 *
 *   边缘停止函数 g：
 *     g(I) = 1 / (1 + |∇(G_σ * I)|²)
 *     在边缘处 g → 0，平滑区域 g → 1
 *
 *   正则化 Dirac 函数：
 *     δ_ε(φ) = (1 + cos(πφ/ε)) / (2ε),  |φ| ≤ ε
 *     δ_ε(φ) = 0,                         |φ| > ε
 *
 * 【演化方程】
 *     ∂φ/∂t = μ·div(d_p(|∇φ|)·∇φ) + λ·δ(φ)·div(g·∇φ/|∇φ|) + α·g·δ(φ)
 *
 *   其中 d_p(s) = p'(s)/s 是正则化梯度权重。
 *
 * 【优势】
 *   - 无需重初始化（与传统 level set 相比大幅简化）
 *   - 初始化灵活（可以用简单的二值函数，不必是 SDF）
 *   - 数值稳定性好
 *
 * 【参考文献】
 *   Li C, et al. "Distance regularized level set evolution and its
 *   application to image segmentation." IEEE TIP, 2010.
 * ============================================================
 */

#include "image2d.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace medseg {

// ============================================================
// 参数
// ============================================================
struct DRLSEParams {
    int   iterations    = 300;    // 迭代次数
    float mu            = 0.2f;   // 距离正则化系数（μ/Δt）
    float lambda        = 5.0f;   // 边缘加权长度项系数
    float alpha         = -3.0f;  // 面积项系数（负=膨胀，正=收缩）
    float epsilon       = 1.5f;   // Dirac 函数带宽
    float sigma         = 1.5f;   // 高斯平滑 σ
    float timestep      = 1.0f;   // 时间步长
    bool  doubleWell    = true;   // true=双势阱, false=单势阱
};

// ============================================================
// DRLSE 2D 分割器
// ============================================================
class DRLSE2D {
public:
    /**
     * 执行 DRLSE 分割
     * @param image       灰度图像 (float)
     * @param initialMask 初始前景掩码 (>0 为前景)
     * @param params      参数
     * @return            分割掩码 (0/255)
     */
    Image2D<uint8_t> segment(const Image2D<float>& image,
                              const Image2D<uint8_t>& initialMask,
                              const DRLSEParams& params = {});

private:
    int W = 0, H = 0;
    const float EPS = 1e-10f;
    const float PI  = 3.14159265358979f;

    // 高斯平滑
    Image2D<float> gaussianSmooth(const Image2D<float>& img, float sigma);

    // 计算边缘停止函数 g = 1/(1+|∇(G*I)|²)
    Image2D<float> edgeIndicator(const Image2D<float>& smoothed);

    // 正则化 Dirac 函数
    float dirac(float x, float eps);

    // 双势阱距离正则项
    void distRegDoubleWell(const Image2D<float>& phi, Image2D<float>& reg);

    // 单势阱距离正则项
    void distRegSingleWell(const Image2D<float>& phi, Image2D<float>& reg);

    // Neumann 边界条件
    void neumannBoundary(Image2D<float>& phi);
};

// ============================================================
//                      实现
// ============================================================

inline Image2D<float> DRLSE2D::gaussianSmooth(const Image2D<float>& img, float sigma) {
    int r = static_cast<int>(std::ceil(3.0f * sigma));
    if (r < 1) return img;

    // 构建 1D 高斯核
    std::vector<float> kernel(2 * r + 1);
    float sum = 0;
    for (int i = -r; i <= r; ++i) {
        kernel[i + r] = std::exp(-0.5f * i * i / (sigma * sigma));
        sum += kernel[i + r];
    }
    for (auto& k : kernel) k /= sum;

    // 水平卷积
    Image2D<float> temp(img.width, img.height);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            float val = 0;
            for (int k = -r; k <= r; ++k) {
                val += img.atClamped(x + k, y) * kernel[k + r];
            }
            temp.at(x, y) = val;
        }
    }

    // 垂直卷积
    Image2D<float> result(img.width, img.height);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            float val = 0;
            for (int k = -r; k <= r; ++k) {
                val += temp.atClamped(x, y + k) * kernel[k + r];
            }
            result.at(x, y) = val;
        }
    }
    return result;
}

inline Image2D<float> DRLSE2D::edgeIndicator(const Image2D<float>& smoothed) {
    // g = 1 / (1 + |∇I|²)
    Image2D<float> g(smoothed.width, smoothed.height);
    for (int y = 0; y < smoothed.height; ++y) {
        for (int x = 0; x < smoothed.width; ++x) {
            float gx = (smoothed.atClamped(x+1, y) - smoothed.atClamped(x-1, y)) * 0.5f;
            float gy = (smoothed.atClamped(x, y+1) - smoothed.atClamped(x, y-1)) * 0.5f;
            g.at(x, y) = 1.0f / (1.0f + gx * gx + gy * gy);
        }
    }
    return g;
}

inline float DRLSE2D::dirac(float x, float eps) {
    if (std::abs(x) > eps) return 0.0f;
    return (1.0f + std::cos(PI * x / eps)) / (2.0f * eps);
}

inline void DRLSE2D::distRegDoubleWell(const Image2D<float>& phi, Image2D<float>& reg) {
    // 距离正则项 (double-well potential)
    // d_p(s) = { 1 - 1/s,           s ≤ 1
    //          { sin(2πs)/(2πs),    s > 1 }
    // 然后做 div(d_p(|∇φ|)·∇φ) 的有限差分近似

    reg = Image2D<float>(W, H, 0.0f);
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            float phiC = phi.at(x, y);

            // 前向差分
            float dxP = phi.at(x+1, y) - phiC;
            float dyP = phi.at(x, y+1) - phiC;
            // 后向差分
            float dxM = phiC - phi.at(x-1, y);
            float dyM = phiC - phi.at(x, y-1);
            // 中心差分
            float dxC = (phi.at(x+1, y) - phi.at(x-1, y)) * 0.5f;
            float dyC = (phi.at(x, y+1) - phi.at(x, y-1)) * 0.5f;

            float gradMag = std::sqrt(dxC * dxC + dyC * dyC + EPS);

            // d_p(s) 权重
            float dp;
            if (gradMag <= 1.0f) {
                dp = 1.0f - 1.0f / (gradMag + EPS);
            } else {
                dp = std::sin(2.0f * PI * gradMag) / (2.0f * PI * gradMag + EPS);
            }

            // 近似 div(d_p·∇φ)
            // 拉普拉斯项 + 修正
            float lap = phi.at(x+1, y) + phi.at(x-1, y) + phi.at(x, y+1) + phi.at(x, y-1) - 4.0f * phiC;
            float divNorm = (dxP / std::sqrt(dxP*dxP + dyC*dyC + EPS)) - dxC / (gradMag + EPS)
                          + (dyP / std::sqrt(dxC*dxC + dyP*dyP + EPS)) - dyC / (gradMag + EPS);

            reg.at(x, y) = dp * lap + divNorm;
        }
    }
}

inline void DRLSE2D::distRegSingleWell(const Image2D<float>& phi, Image2D<float>& reg) {
    // 简单拉普拉斯近似
    reg = Image2D<float>(W, H, 0.0f);
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            reg.at(x, y) = phi.at(x+1, y) + phi.at(x-1, y)
                          + phi.at(x, y+1) + phi.at(x, y-1)
                          - 4.0f * phi.at(x, y);
        }
    }
}

inline void DRLSE2D::neumannBoundary(Image2D<float>& phi) {
    // 零梯度边界条件
    for (int x = 0; x < W; ++x) {
        phi.at(x, 0) = phi.at(x, 1);
        phi.at(x, H-1) = phi.at(x, H-2);
    }
    for (int y = 0; y < H; ++y) {
        phi.at(0, y) = phi.at(1, y);
        phi.at(W-1, y) = phi.at(W-2, y);
    }
    // 角点
    phi.at(0, 0) = phi.at(1, 1);
    phi.at(W-1, 0) = phi.at(W-2, 1);
    phi.at(0, H-1) = phi.at(1, H-2);
    phi.at(W-1, H-1) = phi.at(W-2, H-2);
}

inline Image2D<uint8_t> DRLSE2D::segment(
    const Image2D<float>& image,
    const Image2D<uint8_t>& initialMask,
    const DRLSEParams& params)
{
    W = image.width;
    H = image.height;

    // 1. 初始化 φ 为近似符号距离函数 (SDF)
    //    使用两次独立的 Chamfer 距离变换（前景和背景各一次）
    Image2D<float> phi(W, H);
    {
        // 1a. 计算每个像素到最近"异类"像素的距离
        //     前景像素 → 到最近背景像素的距离（正值）
        //     背景像素 → 到最近前景像素的距离（正值）
        //     然后 phi = 前景内:负, 背景内:正

        // 首先标记前景/背景
        Image2D<uint8_t> isFg(W, H);
        for (int i = 0; i < W * H; ++i)
            isFg[i] = (initialMask[i] > 0) ? 1 : 0;

        // 距离图：初始化为 LARGE，边界处（有异类邻域）初始化为 0.5
        const float LARGE = static_cast<float>(W + H);
        Image2D<float> dist(W, H, LARGE);

        // 标记边界像素：与异类邻域相邻的像素
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                uint8_t me = isFg.at(x, y);
                bool onBorder = false;
                for (int dy = -1; dy <= 1 && !onBorder; ++dy) {
                    for (int dx = -1; dx <= 1 && !onBorder; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        int nx = x + dx, ny = y + dy;
                        if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                            if (isFg.at(nx, ny) != me) onBorder = true;
                        }
                    }
                }
                if (onBorder) dist.at(x, y) = 0.5f; // 边界像素距离 0.5
            }
        }

        // 正向 Chamfer 传播
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (x > 0) dist.at(x, y) = std::min(dist.at(x, y), dist.at(x-1, y) + 1.0f);
                if (y > 0) dist.at(x, y) = std::min(dist.at(x, y), dist.at(x, y-1) + 1.0f);
            }
        }
        // 反向 Chamfer 传播
        for (int y = H - 1; y >= 0; --y) {
            for (int x = W - 1; x >= 0; --x) {
                if (x < W-1) dist.at(x, y) = std::min(dist.at(x, y), dist.at(x+1, y) + 1.0f);
                if (y < H-1) dist.at(x, y) = std::min(dist.at(x, y), dist.at(x, y+1) + 1.0f);
            }
        }

        // 赋值 phi：前景为负，背景为正
        for (int i = 0; i < W * H; ++i) {
            phi[i] = (isFg[i] > 0) ? -dist[i] : dist[i];
        }
    }

    // 2. 计算边缘停止函数
    auto smoothed = gaussianSmooth(image, params.sigma);
    auto g = edgeIndicator(smoothed);

    // 预计算 g 的梯度
    Image2D<float> gx(W, H, 0.0f), gy(W, H, 0.0f);
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            gx.at(x, y) = (g.at(x+1, y) - g.at(x-1, y)) * 0.5f;
            gy.at(x, y) = (g.at(x, y+1) - g.at(x, y-1)) * 0.5f;
        }
    }

    // 3. 迭代演化
    float muDt = params.mu / params.timestep; // 正则化权重

    for (int iter = 0; iter < params.iterations; ++iter) {
        // 3a. Neumann 边界
        neumannBoundary(phi);

        // 3b. 距离正则项
        Image2D<float> reg;
        if (params.doubleWell) {
            distRegDoubleWell(phi, reg);
        } else {
            distRegSingleWell(phi, reg);
        }

        // 3c. 边缘长度项 + 面积项
        for (int y = 1; y < H - 1; ++y) {
            for (int x = 1; x < W - 1; ++x) {
                float phiC = phi.at(x, y);
                float d = dirac(phiC, params.epsilon);

                // 梯度
                float dxC = (phi.at(x+1, y) - phi.at(x-1, y)) * 0.5f;
                float dyC = (phi.at(x, y+1) - phi.at(x, y-1)) * 0.5f;
                float gradMag = std::sqrt(dxC * dxC + dyC * dyC + EPS);

                // 曲率项 div(g · ∇φ/|∇φ|)
                // 近似为 g · κ + ∇g · ∇φ/|∇φ|
                float lap = phi.at(x+1, y) + phi.at(x-1, y)
                          + phi.at(x, y+1) + phi.at(x, y-1) - 4.0f * phiC;
                float divTerm = g.at(x, y) * lap / (gradMag + EPS)
                              + (gx.at(x, y) * dxC + gy.at(x, y) * dyC) / (gradMag + EPS);

                // 总更新
                // φ += Δt { μ·R(φ) + λ·δ(φ)·div(g·∇φ/|∇φ|) + α·g·δ(φ) }
                float update = muDt * reg.at(x, y)
                             + params.lambda * d * divTerm
                             + params.alpha * g.at(x, y) * d;

                phi.at(x, y) += params.timestep * update;
            }
        }
    }

    // 4. 提取掩码（φ < 0 → 前景）
    Image2D<uint8_t> mask(W, H, 0);
    for (int i = 0; i < W * H; ++i) {
        mask[i] = (phi[i] < 0) ? 255 : 0;
    }
    return mask;
}

} // namespace medseg
