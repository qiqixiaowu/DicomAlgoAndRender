/**
 * @file hessian_filter.cpp
 * @brief Hessian 血管增强滤波器实现
 *
 * 实现细节：
 *  1. 对每个 σ 尺度做分离高斯平滑（X/Y/Z 三次 1D卷积）
 *  2. 对平滑后图像计算二阶偏导数（有限差分）构建 3×3 Hessian 矩阵
 *  3. 对 Hessian 做 3×3 特征值分解（Jacobi 迭代法）
 *  4. 根据特征值大小排序，代入 Frangi 公式计算响应
 *  5. 多尺度取最大响应（Max over scales）
 */

#include "hessian_filter.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <array>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Onc {

// ─────────────────────────────────────────────
//  1D 高斯核生成
// ─────────────────────────────────────────────
std::vector<float> HessianFilter::makeGaussianKernel(float sigma, int& halfW) {
    halfW = static_cast<int>(std::ceil(3.0f * sigma));
    std::vector<float> kern(2 * halfW + 1);
    float sum = 0.f;
    float s2 = 2.f * sigma * sigma;
    for (int i = -halfW; i <= halfW; ++i) {
        kern[i + halfW] = std::exp(-static_cast<float>(i*i) / s2);
        sum += kern[i + halfW];
    }
    for (auto& v : kern) v /= sum;
    return kern;
}

// ─────────────────────────────────────────────
//  沿指定轴 1D 卷积（axis: 0=X, 1=Y, 2=Z）
// ─────────────────────────────────────────────
void HessianFilter::convolve1D(
    const std::vector<float>& src,
    const ImageInfo&          info,
    const std::vector<float>& kernel,
    int                       halfW,
    int                       axis,
    std::vector<float>&       dst)
{
    int W = info.dim[0], H = info.dim[1], D = info.dim[2];
    dst.assign(src.size(), 0.f);

    int strides[3] = {1, W, W*H};
    int axLen = info.dim[axis];
    int stride = strides[axis];

    for (int z = 0; z < D; ++z)
    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
        int pos[3] = {x, y, z};
        int baseIdx = info.linearIndex(x, y, z);
        float acc = 0.f;
        for (int k = -halfW; k <= halfW; ++k) {
            int c = pos[axis] + k;
            c = std::clamp(c, 0, axLen - 1);
            int ni = baseIdx + (c - pos[axis]) * stride;
            acc += kernel[k + halfW] * src[ni];
        }
        dst[baseIdx] = acc;
    }
}

// ─────────────────────────────────────────────
//  Gaussian 平滑（分离式三轴卷积）
// ─────────────────────────────────────────────
void HessianFilter::gaussianSmooth(
    const short*       input,
    const ImageInfo&   info,
    float              sigma,
    std::vector<float>& output)
{
    int N = info.totalVoxels();
    std::vector<float> tmp1(N), tmp2(N);

    // 转 float
    for (int i = 0; i < N; ++i) tmp1[i] = static_cast<float>(input[i]);

    int halfW; auto kern = makeGaussianKernel(sigma, halfW);
    convolve1D(tmp1, info, kern, halfW, 0, tmp2);  // X
    convolve1D(tmp2, info, kern, halfW, 1, tmp1);  // Y
    convolve1D(tmp1, info, kern, halfW, 2, output);// Z
}

// ─────────────────────────────────────────────
//  3×3 对称矩阵特征值分解（Jacobi 迭代，最多 50次）
//  返回 3 个特征值（无序）
// ─────────────────────────────────────────────
static void Jacobi3x3(float A[3][3], float eigen[3])
{
    // 复制工作矩阵
    float a[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            a[i][j] = A[i][j];

    for (int sweep = 0; sweep < 50; ++sweep) {
        // 找最大非对角元素
        float maxOff = 0.f; int p = 0, q = 1;
        for (int i = 0; i < 3; ++i)
            for (int j = i+1; j < 3; ++j)
                if (std::abs(a[i][j]) > maxOff) {
                    maxOff = std::abs(a[i][j]);
                    p = i; q = j;
                }
        if (maxOff < 1e-10f) break;

        float theta = 0.5f * std::atan2(2.f * a[p][q], a[q][q] - a[p][p]);
        float c = std::cos(theta), s = std::sin(theta);

        // Givens 旋转
        float ap = c*c*a[p][p] - 2.f*s*c*a[p][q] + s*s*a[q][q];
        float aq = s*s*a[p][p] + 2.f*s*c*a[p][q] + c*c*a[q][q];
        a[p][q] = a[q][p] = 0.f;
        a[p][p] = ap; a[q][q] = aq;

        for (int r = 0; r < 3; ++r) {
            if (r == p || r == q) continue;
            float apr = c*a[p][r] - s*a[q][r];
            float aqr = s*a[p][r] + c*a[q][r];
            a[p][r] = a[r][p] = apr;
            a[q][r] = a[r][q] = aqr;
        }
    }
    eigen[0] = a[0][0]; eigen[1] = a[1][1]; eigen[2] = a[2][2];

    // 按绝对值从小到大排序
    if (std::abs(eigen[0]) > std::abs(eigen[1])) std::swap(eigen[0], eigen[1]);
    if (std::abs(eigen[1]) > std::abs(eigen[2])) std::swap(eigen[1], eigen[2]);
    if (std::abs(eigen[0]) > std::abs(eigen[1])) std::swap(eigen[0], eigen[1]);
}

// ─────────────────────────────────────────────
//  Frangi 管状增强公式
// ─────────────────────────────────────────────
static float FrangiVesselness(float ev[3], float alpha, float beta, float gamma, bool dark)
{
    // ev[0] ≤ ev[1] ≤ ev[2] (按绝对值排序，已含符号)
    float l1 = ev[0], l2 = ev[1], l3 = ev[2];

    // 亮管状：λ₂ < 0 且 λ₃ < 0
    if (!dark && (l2 >= 0.f || l3 >= 0.f)) return 0.f;
    // 暗管状：λ₂ > 0 且 λ₃ > 0
    if (dark  && (l2 <= 0.f || l3 <= 0.f)) return 0.f;

    float absL1 = std::abs(l1), absL2 = std::abs(l2), absL3 = std::abs(l3);

    float RA  = (absL3 > 1e-8f) ? absL2 / absL3 : 0.f;
    float RB  = (absL2*absL3 > 1e-16f)
              ? absL1 / std::sqrt(absL2 * absL3)
              : 0.f;
    float S2  = l1*l1 + l2*l2 + l3*l3;

    float v = (1.f - std::exp(-RA*RA / (2.f*alpha*alpha)))
            * std::exp(-RB*RB / (2.f*beta*beta))
            * (1.f - std::exp(-S2 / (2.f*gamma*gamma)));
    return v;
}

// ─────────────────────────────────────────────
//  单尺度 Hessian 特征值
// ─────────────────────────────────────────────
void HessianFilter::computeEigenvalues(
    const std::vector<float>& smooth,
    const ImageInfo&          info,
    float                     sigma,
    std::vector<float>&       ev1,
    std::vector<float>&       ev2,
    std::vector<float>&       ev3)
{
    int W = info.dim[0], H = info.dim[1], D = info.dim[2];
    int N = W * H * D;
    ev1.assign(N, 0.f); ev2.assign(N, 0.f); ev3.assign(N, 0.f);

    float s2 = sigma * sigma; // σ² 用于尺度归一化

    for (int z = 1; z < D-1; ++z)
    for (int y = 1; y < H-1; ++y)
    for (int x = 1; x < W-1; ++x) {
        auto at = [&](int dx, int dy, int dz) -> float {
            return smooth[info.linearIndex(x+dx, y+dy, z+dz)];
        };
        float c = at(0,0,0);
        // 二阶偏导数（有限差分，乘 σ² 进行尺度归一化）
        float Hxx = s2 * (at(1,0,0) - 2.f*c + at(-1,0,0));
        float Hyy = s2 * (at(0,1,0) - 2.f*c + at(0,-1,0));
        float Hzz = s2 * (at(0,0,1) - 2.f*c + at(0,0,-1));
        float Hxy = s2 * 0.25f * (at(1,1,0) - at(-1,1,0) - at(1,-1,0) + at(-1,-1,0));
        float Hxz = s2 * 0.25f * (at(1,0,1) - at(-1,0,1) - at(1,0,-1) + at(-1,0,-1));
        float Hyz = s2 * 0.25f * (at(0,1,1) - at(0,-1,1) - at(0,1,-1) + at(0,-1,-1));

        float A[3][3] = {{Hxx,Hxy,Hxz},{Hxy,Hyy,Hyz},{Hxz,Hyz,Hzz}};
        float ev[3];
        Jacobi3x3(A, ev);

        int idx = info.linearIndex(x, y, z);
        ev1[idx] = ev[0]; ev2[idx] = ev[1]; ev3[idx] = ev[2];
    }
}

// ─────────────────────────────────────────────
//  主接口
// ─────────────────────────────────────────────
void HessianFilter::compute(
    const short*        data,
    const ImageInfo&    info,
    std::vector<float>& hdot,
    std::vector<float>& hline,
    std::vector<float>& hplane)
{
    int N = info.totalVoxels();
    hdot.assign(N, 0.f);
    hline.assign(N, 0.f);
    hplane.assign(N, 0.f);

    for (float sigma : m_params.scales) {
        // 平滑
        std::vector<float> smooth;
        gaussianSmooth(data, info, sigma, smooth);

        // 特征值
        std::vector<float> ev1, ev2, ev3;
        computeEigenvalues(smooth, info, sigma, ev1, ev2, ev3);

        // Frangi 管状增强 → hdot（点）
        for (int i = 0; i < N; ++i) {
            float ev[3] = {ev1[i], ev2[i], ev3[i]};
            float v = FrangiVesselness(ev, m_params.alpha, m_params.beta, m_params.gamma, m_params.dark);
            hdot[i]  = std::max(hdot[i],  v); // 点状（此处 Frangi 本质是管状，用于血管类）
            hline[i] = std::max(hline[i], v); // 线状（同上）

            // 板状：λ₂、λ₃ 同号且 |λ₁|<<|λ₂|
            float absL1 = std::abs(ev[0]), absL2 = std::abs(ev[1]);
            float planeV = (absL2 > 1e-8f) ? (1.f - absL1/absL2) * std::abs(ev[1]) : 0.f;
            hplane[i] = std::max(hplane[i], planeV);
        }
    }
}

} // namespace Onc
