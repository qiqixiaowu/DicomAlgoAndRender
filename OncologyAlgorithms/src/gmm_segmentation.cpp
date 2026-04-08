/**
 * @file gmm_segmentation.cpp
 * @brief GMM EM 肿瘤分割实现
 *
 * 使用 1D 对角协方差（即每个分量只有均值+方差两个参数），
 * 适合单通道灰度图像快速分割。
 */

#include "gmm_segmentation.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Onc {

// ─────────────────────────────────────────────
//  1D 正态分布 PDF
// ─────────────────────────────────────────────
static inline float GaussianPDF(float x, float mu, float sigma) {
    float diff = x - mu;
    float sigma2 = sigma * sigma;
    return static_cast<float>(
        (1.0 / (std::sqrt(2.0 * M_PI) * sigma))
        * std::exp(-0.5 * diff * diff / sigma2));
}

bool GMMSegmentation::segment(
    const short*     data,
    const ImageInfo& info,
    const Point3i&   seedPoint,
    SegmentResult&   result)
{
    if (!data) return false;
    if (!info.inBounds(seedPoint.x, seedPoint.y, seedPoint.z)) return false;

    int m = m_params.roiMargin;
    int x0 = std::max(0, seedPoint.x - m), x1 = std::min(info.dim[0]-1, seedPoint.x + m);
    int y0 = std::max(0, seedPoint.y - m), y1 = std::min(info.dim[1]-1, seedPoint.y + m);
    int z0 = std::max(0, seedPoint.z - m), z1 = std::min(info.dim[2]-1, seedPoint.z + m);

    std::vector<float> samples;
    std::vector<int>   indices;
    samples.reserve((x1-x0+1) * (y1-y0+1) * (z1-z0+1));

    for (int z = z0; z <= z1; ++z)
    for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x) {
        int idx = info.linearIndex(x, y, z);
        samples.push_back(static_cast<float>(data[idx]));
        indices.push_back(idx);
    }

    int N = static_cast<int>(samples.size());
    int K = m_params.numComponents;
    if (N < K) return false;

    // ── 初始化：用均匀分位点初始化均值 ──────────────
    std::vector<float> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    float dataMin = sorted.front(), dataMax = sorted.back();
    float dataRange = dataMax - dataMin;

    std::vector<float> pi(K, 1.f/K);
    std::vector<float> mu(K), sigma(K);
    for (int k = 0; k < K; ++k) {
        mu[k]    = dataMin + (k + 0.5f) * dataRange / K;
        sigma[k] = (dataRange / K) + m_params.minSigma;
    }

    // responsibilities[i*K + k]
    std::vector<float> r(N * K, 0.f);

    float prevLogLik = -std::numeric_limits<float>::max();

    for (int iter = 0; iter < m_params.maxIter; ++iter) {
        // ── E-step ──────────────────────────────
        float logLik = 0.f;
        for (int i = 0; i < N; ++i) {
            float sumR = 0.f;
            for (int k = 0; k < K; ++k) {
                float val = pi[k] * GaussianPDF(samples[i], mu[k], sigma[k]);
                r[i*K+k] = val;
                sumR += val;
            }
            if (sumR < 1e-12f) sumR = 1e-12f;
            logLik += std::log(sumR);
            for (int k = 0; k < K; ++k) r[i*K+k] /= sumR;
        }

        // ── M-step ──────────────────────────────
        for (int k = 0; k < K; ++k) {
            float Nk = 0.f;
            for (int i = 0; i < N; ++i) Nk += r[i*K+k];
            if (Nk < 1e-6f) Nk = 1e-6f;

            float newMu = 0.f;
            for (int i = 0; i < N; ++i) newMu += r[i*K+k] * samples[i];
            newMu /= Nk;

            float newSigma2 = 0.f;
            for (int i = 0; i < N; ++i) {
                float d = samples[i] - newMu;
                newSigma2 += r[i*K+k] * d * d;
            }
            newSigma2 /= Nk;

            pi[k]    = Nk / N;
            mu[k]    = newMu;
            sigma[k] = std::max(m_params.minSigma, std::sqrt(newSigma2));
        }

        // 收敛检测
        if (std::abs(logLik - prevLogLik) < m_params.convergenceThr) break;
        prevLogLik = logLik;
    }

    // ── 确定前景分量 ──────────────────────────────
    // 取种子点灰度值所属 responsibility 最高的分量
    float seedV = static_cast<float>(data[info.linearIndex(seedPoint.x, seedPoint.y, seedPoint.z)]);
    int fgCluster = 0;
    float bestR = -1.f;
    for (int k = 0; k < K; ++k) {
        float pdf = pi[k] * GaussianPDF(seedV, mu[k], sigma[k]);
        if (pdf > bestR) { bestR = pdf; fgCluster = k; }
    }

    // ── 输出前景体素 ──────────────────────────────
    result.clear();
    for (int i = 0; i < N; ++i) {
        // 判断该体素前景分量 responsibility 最高
        int bestK = 0;
        float bestVal = r[i*K];
        for (int k = 1; k < K; ++k) {
            if (r[i*K+k] > bestVal) { bestVal = r[i*K+k]; bestK = k; }
        }
        if (bestK == fgCluster) {
            int idx = indices[i];
            int z = idx / info.sliceSize();
            int rem = idx % info.sliceSize();
            int y = rem / info.dim[0];
            int x = rem % info.dim[0];
            result.push_back({x, y, z});
        }
    }
    return !result.empty();
}

} // namespace Onc
