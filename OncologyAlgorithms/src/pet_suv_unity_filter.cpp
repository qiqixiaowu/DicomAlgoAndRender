/**
 * @file pet_suv_unity_filter.cpp
 * @brief PET SUV 跨设备归一化高斯滤波实现
 *
 * 算法核心：3D 可分离高斯卷积
 * ─────────────────────────────
 *
 * 高斯卷积的可分离性：
 *   G_3D(x,y,z) = G_1D(x) × G_1D(y) × G_1D(z)
 *
 * 因此 3D 卷积可分解为三次独立的 1D 卷积：
 *   out = (((src ⊛ Kx) ⊛ Ky) ⊛ Kz)
 *
 * 计算量从 O(N × r³) 降至 O(3 × N × r)，对大图像尤其显著。
 *
 * 边界处理：镜像对称延拓（Mirror Padding）
 *   index = -1  → 使用 index = 1
 *   index = N   → 使用 index = N-2
 *   即：边界处的信号"折叠"回来，避免边缘区域 SUV 被人为压低。
 */

#define MCSF_ALGO_MIONC_SEGMENTATION_EXPORTS
#include "pet_suv_unity_filter.h"

#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>

// ── 常量 ──────────────────────────────────────────────────────────────────────

/// FWHM 到 σ 的换算系数：σ = FWHM / (2√(2ln2)) ≈ FWHM / 2.354820045
static constexpr double FWHM_TO_SIGMA = 1.0 / (2.0 * 1.17741002252); // 1/2√(2ln2)

// ── 内部工具函数 ────────────────────────────────────────────────────────────

/**
 * @brief 生成归一化 1D 高斯核
 *
 * @param sigma_vox  标准差（体素单位），若 <=0 则生成单位冲激响应（[1.0]）
 * @param radius     [out] 核半径，kernel 长度 = 2*radius+1
 * @return           归一化后的高斯核（和为 1.0）
 */
static std::vector<double> MakeGaussianKernel1D(double sigma_vox, int& radius)
{
    if (sigma_vox <= 1e-6) {
        radius = 0;
        return {1.0};
    }

    // 截断半径取 3σ（99.73% 能量），最小为 1
    radius = std::max(1, static_cast<int>(std::ceil(3.0 * sigma_vox)));

    int ksize = 2 * radius + 1;
    std::vector<double> kernel(ksize);
    double sum = 0.0;
    double inv2sig2 = 1.0 / (2.0 * sigma_vox * sigma_vox);

    for (int i = 0; i < ksize; ++i) {
        double x   = static_cast<double>(i - radius);
        kernel[i]  = std::exp(-x * x * inv2sig2);
        sum       += kernel[i];
    }
    for (auto& v : kernel) v /= sum; // 归一化，确保能量守恒
    return kernel;
}

/**
 * @brief 镜像边界索引
 *
 * 将越界索引映射回合法范围：
 *   [-r, -1]   → [r, 1]       （左边界镜像）
 *   [N, N+r-1] → [N-2, N-r-1] （右边界镜像）
 */
static inline int MirrorIndex(int idx, int size)
{
    if (idx < 0)    return -idx;                   // 左镜像
    if (idx >= size) return 2 * size - 2 - idx;    // 右镜像
    return idx;
}

/**
 * @brief 沿指定轴对 3D double 数组做 1D 高斯卷积
 *
 * @param src    输入数组，线性排列 [z * W * H + y * W + x]
 * @param dst    输出数组（可与 src 不同）
 * @param W,H,D  三轴尺寸（X,Y,Z）
 * @param kernel 归一化高斯核
 * @param radius 核半径
 * @param axis   0=X轴, 1=Y轴, 2=Z轴
 */
static void Convolve1D(
    const double*              src,
    double*                    dst,
    int W, int H, int D,
    const std::vector<double>& kernel,
    int                        radius,
    int                        axis)
{
    long long wh = static_cast<long long>(W) * H;

    for (int z = 0; z < D; ++z)
    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
    {
        double sum = 0.0;
        for (int t = -radius; t <= radius; ++t)
        {
            int nx = x, ny = y, nz = z;
            // 沿指定轴偏移，其余轴不变
            switch (axis) {
            case 0: nx = MirrorIndex(x + t, W); break;
            case 1: ny = MirrorIndex(y + t, H); break;
            case 2: nz = MirrorIndex(z + t, D); break;
            }
            sum += kernel[t + radius] * src[static_cast<long long>(nz) * wh
                                          + static_cast<long long>(ny) * W + nx];
        }
        dst[static_cast<long long>(z) * wh + static_cast<long long>(y) * W + x] = sum;
    }
}

// ── 公开接口实现 ──────────────────────────────────────────────────────────────

bool McsfAlgoMMOncUnityFilter(
    const unsigned short* pImage,
    const int             iSize[3],
    const double          dSpacing[3],
    const double&         FWHM,
    double*               pOutImage,
    IProgress*            pProgress,
    double                dProgressStart,
    double                dProgressEnd)
{
    // ── 参数校验 ──────────────────────────────────────────────────────────────
    if (!pImage || !pOutImage)                        return false;
    if (iSize[0] <= 0 || iSize[1] <= 0 || iSize[2] <= 0) return false;
    if (dSpacing[0] <= 0 || dSpacing[1] <= 0 || dSpacing[2] <= 0) return false;
    if (FWHM < 0)                                     return false;

    long long N = static_cast<long long>(iSize[0])
                * static_cast<long long>(iSize[1])
                * static_cast<long long>(iSize[2]);

    // ── FWHM=0：直接复制 ─────────────────────────────────────────────────────
    if (FWHM < 1e-9) {
        for (long long i = 0; i < N; ++i)
            pOutImage[i] = static_cast<double>(pImage[i]);
        if (pProgress) pProgress->SetProgress(dProgressEnd);
        return true;
    }

    // ── step1：unsigned short → double 工作缓冲 ─────────────────────────────
    std::vector<double> buf0(N), buf1(N);
    for (long long i = 0; i < N; ++i)
        buf0[i] = static_cast<double>(pImage[i]);

    if (pProgress) {
        pProgress->SetProgress(dProgressStart + (dProgressEnd - dProgressStart) * 0.1);
        if (pProgress->IsCancelled()) return false;
    }

    // ── step2：计算各轴高斯核 ────────────────────────────────────────────────
    // σ_mm = FWHM / (2√(2ln2))
    // σ_vox[axis] = σ_mm / spacing[axis]   （各轴体素间距可能不同）
    const double sigma_mm = FWHM * FWHM_TO_SIGMA;

    int    radius[3];
    std::vector<double> kernels[3];
    for (int d = 0; d < 3; ++d) {
        double sigma_vox = sigma_mm / dSpacing[d];
        kernels[d] = MakeGaussianKernel1D(sigma_vox, radius[d]);
    }

    // ── step3：三轴可分离卷积  X → Y → Z ────────────────────────────────────
    // 每轴完成后检查取消请求，并更新进度
    const int W = iSize[0], H = iSize[1], D = iSize[2];

    // X 轴：buf0 → buf1
    Convolve1D(buf0.data(), buf1.data(), W, H, D, kernels[0], radius[0], 0);
    if (pProgress) {
        pProgress->SetProgress(dProgressStart + (dProgressEnd - dProgressStart) * 0.4);
        if (pProgress->IsCancelled()) return false;
    }

    // Y 轴：buf1 → buf0
    Convolve1D(buf1.data(), buf0.data(), W, H, D, kernels[1], radius[1], 1);
    if (pProgress) {
        pProgress->SetProgress(dProgressStart + (dProgressEnd - dProgressStart) * 0.7);
        if (pProgress->IsCancelled()) return false;
    }

    // Z 轴：buf0 → pOutImage（直接写入输出，省一次拷贝）
    Convolve1D(buf0.data(), pOutImage, W, H, D, kernels[2], radius[2], 2);

    if (pProgress) {
        pProgress->SetProgress(dProgressEnd);
    }
    return true;
}
