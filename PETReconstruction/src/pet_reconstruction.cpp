// ============================================================================
// PET 重建 CPU 实现
// ============================================================================

#include "pet_reconstruction.hpp"
#include <cstring>
#include <numeric>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <complex>

// ============================================================================
// PETReconResult
// ============================================================================

void PETReconResult::normalize() {
    if (image.empty()) return;
    float mn = *std::min_element(image.begin(), image.end());
    float mx = *std::max_element(image.begin(), image.end());
    float rg = mx - mn;
    if (rg < 1e-10f) rg = 1.0f;
    for (auto& v : image) v = (v - mn) / rg;
}

std::vector<uint8_t> PETReconResult::toUint8() const {
    std::vector<uint8_t> result(image.size());
    for (size_t i = 0; i < image.size(); ++i) {
        float v = std::clamp(image[i], 0.0f, 1.0f);
        result[i] = static_cast<uint8_t>(v * 255.0f + 0.5f);
    }
    return result;
}

// ============================================================================
// 体模生成
// ============================================================================

std::vector<float> PETReconstructor::generatePhantom(
    int size, float background, const std::vector<PETHotspot>& hotspots)
{
    std::vector<float> image(size * size, 0.0f);
    float halfSize = size / 2.0f;
    float R = 0.85f;

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float px = (x - halfSize + 0.5f) / halfSize;
            float py = (y - halfSize + 0.5f) / halfSize;
            if ((px * px) / (R * R) + (py * py) / ((R * 0.8f) * (R * 0.8f)) <= 1.0f)
                image[y * size + x] = background;
        }
    }
    for (const auto& h : hotspots) {
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                float px = (x - halfSize + 0.5f) / halfSize;
                float py = (y - halfSize + 0.5f) / halfSize;
                float dist = std::sqrt((px - h.cx) * (px - h.cx) + (py - h.cy) * (py - h.cy));
                if (dist <= h.radius) image[y * size + x] = h.activity;
            }
        }
    }
    return image;
}

std::vector<float> PETReconstructor::generateHotColdPhantom(int size) {
    std::vector<PETHotspot> hotspots = {
        {  0.28f,  0.0f,   0.18f, 8.0f },
        { -0.28f,  0.0f,   0.14f, 6.0f },
        {  0.0f,   0.3f,   0.11f, 4.0f },
        {  0.0f,  -0.3f,   0.08f, 4.0f },
        {  0.35f,  0.3f,   0.14f, 0.0f },
        { -0.35f, -0.3f,   0.11f, 0.0f },
    };
    return generatePhantom(size, 1.0f, hotspots);
}

std::vector<float> PETReconstructor::generateDerenzoPhantom(int size) {
    std::vector<float> image(size * size, 0.0f);
    float halfSize = size / 2.0f;
    float R = 0.85f;
    float diameters[] = { 0.14f, 0.11f, 0.09f, 0.07f, 0.05f, 0.04f };
    float sectorAngles[] = { 90.0f, 30.0f, -30.0f, -90.0f, -150.0f, 150.0f };

    for (int sec = 0; sec < 6; ++sec) {
        float d = diameters[sec];
        float spacing = d * 2.0f;
        float secAngle = sectorAngles[sec] * static_cast<float>(M_PI) / 180.0f;
        float secCx = 0.4f * std::cos(secAngle);
        float secCy = 0.4f * std::sin(secAngle);
        int gridN = static_cast<int>(0.35f / spacing) + 1;
        for (int gi = -gridN; gi <= gridN; ++gi) {
            for (int gj = -gridN; gj <= gridN; ++gj) {
                float hx = secCx + gi * spacing;
                float hy = secCy + gj * spacing + (gi % 2) * spacing * 0.5f;
                float distToCenter = std::sqrt((hx - secCx) * (hx - secCx) + (hy - secCy) * (hy - secCy));
                if (distToCenter > 0.3f) continue;
                if (std::sqrt(hx * hx + hy * hy) + d / 2 > R) continue;
                float r = d / 2.0f;
                for (int y = 0; y < size; ++y) {
                    for (int x = 0; x < size; ++x) {
                        float px = (x - halfSize + 0.5f) / halfSize;
                        float py = (y - halfSize + 0.5f) / halfSize;
                        float dist = std::sqrt((px - hx) * (px - hx) + (py - hy) * (py - hy));
                        if (dist <= r) image[y * size + x] = 4.0f;
                    }
                }
            }
        }
    }
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float px = (x - halfSize + 0.5f) / halfSize;
            float py = (y - halfSize + 0.5f) / halfSize;
            if (px * px + py * py <= R * R && image[y * size + x] == 0.0f)
                image[y * size + x] = 1.0f;
        }
    }
    return image;
}

std::vector<float> PETReconstructor::generateUniformCylinder(int size, float activity) {
    std::vector<float> image(size * size, 0.0f);
    float halfSize = size / 2.0f;
    float R = 0.8f;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float px = (x - halfSize + 0.5f) / halfSize;
            float py = (y - halfSize + 0.5f) / halfSize;
            if (px * px + py * py <= R * R) image[y * size + x] = activity;
        }
    }
    return image;
}

// ============================================================================
// 正向投影
// ============================================================================

float PETReconstructor::bilinearSample(
    const std::vector<float>& image, int size, float x, float y)
{
    if (x < 0 || x >= size - 1 || y < 0 || y >= size - 1) return 0.0f;
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = std::min(x0 + 1, size - 1);
    int y1 = std::min(y0 + 1, size - 1);
    float fx = x - x0, fy = y - y0;
    return image[y0 * size + x0] * (1 - fx) * (1 - fy)
         + image[y0 * size + x1] * fx * (1 - fy)
         + image[y1 * size + x0] * (1 - fx) * fy
         + image[y1 * size + x1] * fx * fy;
}

float PETReconstructor::projectLOR(
    const std::vector<float>& image, int size,
    float angle, float radialOffset, float halfSize)
{
    float cosA = std::cos(angle), sinA = std::sin(angle);
    float sum = 0.0f;
    float sMax = halfSize * std::sqrt(2.0f);
    for (float s = -sMax; s <= sMax; s += 1.0f) {
        float x = radialOffset * cosA - s * sinA + halfSize - 0.5f;
        float y = radialOffset * sinA + s * cosA + halfSize - 0.5f;
        sum += bilinearSample(image, size, x, y);
    }
    return sum;
}

void PETReconstructor::backprojectLOR(
    std::vector<float>& image, int size,
    float angle, float radialOffset, float halfSize, float value)
{
    float cosA = std::cos(angle), sinA = std::sin(angle);
    float sMax = halfSize * std::sqrt(2.0f);
    for (float s = -sMax; s <= sMax; s += 1.0f) {
        float fx = radialOffset * cosA - s * sinA + halfSize - 0.5f;
        float fy = radialOffset * sinA + s * cosA + halfSize - 0.5f;
        int ix = static_cast<int>(std::floor(fx));
        int iy = static_cast<int>(std::floor(fy));
        if (ix < 0 || ix >= size || iy < 0 || iy >= size) continue;
        image[iy * size + ix] += value;
    }
}

PETSinogram PETReconstructor::forwardProject(
    const std::vector<float>& image, int imageSize,
    int numAngles, int numRadialBins)
{
    if (numRadialBins <= 0)
        numRadialBins = static_cast<int>(std::ceil(imageSize * std::sqrt(2.0f)));
    PETSinogram sino;
    sino.numAngles = numAngles;
    sino.numRadialBins = numRadialBins;
    sino.data.resize(sino.totalBins(), 0.0f);
    float halfSize = imageSize / 2.0f;
    float halfBins = numRadialBins / 2.0f;
    float binSpacing = (imageSize * std::sqrt(2.0f)) / numRadialBins;
    for (int a = 0; a < numAngles; ++a) {
        float angle = static_cast<float>(M_PI) * a / numAngles;
        for (int r = 0; r < numRadialBins; ++r) {
            float radialOffset = (r - halfBins + 0.5f) * binSpacing;
            sino.at(a, r) = projectLOR(image, imageSize, angle, radialOffset, halfSize);
        }
    }
    return sino;
}

PETSinogram PETReconstructor::forwardProjectAtten(
    const std::vector<float>& activityMap,
    const std::vector<float>& attenuationMap,
    int imageSize, int numAngles, int numRadialBins)
{
    PETSinogram sino = forwardProject(activityMap, imageSize, numAngles, numRadialBins);
    float halfSize = imageSize / 2.0f;
    float halfBins = sino.numRadialBins / 2.0f;
    float binSpacing = (imageSize * std::sqrt(2.0f)) / sino.numRadialBins;
    for (int a = 0; a < sino.numAngles; ++a) {
        float angle = static_cast<float>(M_PI) * a / sino.numAngles;
        for (int r = 0; r < sino.numRadialBins; ++r) {
            float radialOffset = (r - halfBins + 0.5f) * binSpacing;
            float af = computeAttenuationFactor(attenuationMap, imageSize, angle, radialOffset, halfSize);
            sino.at(a, r) *= af;
        }
    }
    return sino;
}

float PETReconstructor::computeAttenuationFactor(
    const std::vector<float>& attenMap, int size,
    float angle, float radialOffset, float halfSize)
{
    float cosA = std::cos(angle), sinA = std::sin(angle);
    float sMax = halfSize * std::sqrt(2.0f);
    float sumMu = 0.0f;
    for (float s = -sMax; s <= sMax; s += 1.0f) {
        float x = radialOffset * cosA - s * sinA + halfSize - 0.5f;
        float y = radialOffset * sinA + s * cosA + halfSize - 0.5f;
        sumMu += bilinearSample(attenMap, size, x, y);
    }
    return std::exp(-sumMu);
}

// ============================================================================
// MLEM 重建
// ============================================================================

PETReconResult PETReconstructor::reconstructMLEM(
    const PETSinogram& sinogram, int outputSize,
    int iterations, const PETCorrections& corrections)
{
    PETReconResult result;
    result.size = outputSize;
    result.image.resize(outputSize * outputSize, 1.0f);

    float halfSize = outputSize / 2.0f;
    float halfBins = sinogram.numRadialBins / 2.0f;
    float binSpacing = (outputSize * std::sqrt(2.0f)) / sinogram.numRadialBins;

    // 灵敏度图
    std::vector<float> sensitivity(outputSize * outputSize, 0.0f);
    for (int a = 0; a < sinogram.numAngles; ++a) {
        float angle = static_cast<float>(M_PI) * a / sinogram.numAngles;
        for (int r = 0; r < sinogram.numRadialBins; ++r) {
            float radialOffset = (r - halfBins + 0.5f) * binSpacing;
            backprojectLOR(sensitivity, outputSize, angle, radialOffset, halfSize, 1.0f);
        }
    }
    for (auto& s : sensitivity) if (s < 1e-10f) s = 1e-10f;

    // 加性校正项
    PETSinogram additive;
    if (corrections.scatterCorrection || corrections.randomsCorrection) {
        additive.numAngles = sinogram.numAngles;
        additive.numRadialBins = sinogram.numRadialBins;
        additive.data.resize(sinogram.totalBins(), 0.0f);
        float meanSino = std::accumulate(sinogram.data.begin(), sinogram.data.end(), 0.0f)
                       / sinogram.totalBins();
        if (corrections.scatterCorrection)
            for (auto& v : additive.data) v += corrections.scatterFraction * meanSino;
        if (corrections.randomsCorrection)
            for (auto& v : additive.data) v += corrections.randomsRate * meanSino;
    }

    for (int iter = 0; iter < iterations; ++iter) {
        // 正向投影
        std::vector<float> estimatedSino(sinogram.totalBins(), 0.0f);
        for (int a = 0; a < sinogram.numAngles; ++a) {
            float angle = static_cast<float>(M_PI) * a / sinogram.numAngles;
            for (int r = 0; r < sinogram.numRadialBins; ++r) {
                float radialOffset = (r - halfBins + 0.5f) * binSpacing;
                float proj = projectLOR(result.image, outputSize, angle, radialOffset, halfSize);
                if (!additive.data.empty()) proj += additive.at(a, r);
                estimatedSino[a * sinogram.numRadialBins + r] = std::max(proj, 1e-10f);
            }
        }

        // 比值
        std::vector<float> ratio(sinogram.totalBins());
        float logLik = 0.0f;
        for (size_t i = 0; i < sinogram.totalBins(); ++i) {
            ratio[i] = sinogram.data[i] / estimatedSino[i];
            if (sinogram.data[i] > 0 && estimatedSino[i] > 0)
                logLik += sinogram.data[i] * std::log(estimatedSino[i]) - estimatedSino[i];
        }

        // 反投影
        std::vector<float> correction(outputSize * outputSize, 0.0f);
        for (int a = 0; a < sinogram.numAngles; ++a) {
            float angle = static_cast<float>(M_PI) * a / sinogram.numAngles;
            for (int r = 0; r < sinogram.numRadialBins; ++r) {
                float radialOffset = (r - halfBins + 0.5f) * binSpacing;
                backprojectLOR(correction, outputSize, angle, radialOffset, halfSize,
                               ratio[a * sinogram.numRadialBins + r]);
            }
        }

        // 更新
        for (int i = 0; i < outputSize * outputSize; ++i) {
            result.image[i] *= correction[i] / sensitivity[i];
            if (result.image[i] < 0.0f) result.image[i] = 0.0f;
        }

        if (corrections.psfModeling) {
            float sigma = corrections.psfFWHM / 2.355f;
            applyGaussianSmooth(result.image, outputSize, outputSize, sigma);
        }

        result.logLikelihood = logLik;
        result.iterationsRun = iter + 1;
        if ((iter + 1) % 5 == 0 || iter == 0)
            std::cout << "  [CPU] MLEM 迭代 " << (iter + 1) << "/" << iterations
                      << "  LogL=" << logLik << std::endl;
    }
    return result;
}

// ============================================================================
// OSEM 重建
// ============================================================================

PETReconResult PETReconstructor::reconstructOSEM(
    const PETSinogram& sinogram, int outputSize,
    int iterations, int numSubsets, const PETCorrections& corrections)
{
    PETReconResult result;
    result.size = outputSize;
    result.image.resize(outputSize * outputSize, 1.0f);

    float halfSize = outputSize / 2.0f;
    float halfBins = sinogram.numRadialBins / 2.0f;
    float binSpacing = (outputSize * std::sqrt(2.0f)) / sinogram.numRadialBins;

    PETSinogram additive;
    if (corrections.scatterCorrection || corrections.randomsCorrection) {
        additive.numAngles = sinogram.numAngles;
        additive.numRadialBins = sinogram.numRadialBins;
        additive.data.resize(sinogram.totalBins(), 0.0f);
        float meanSino = std::accumulate(sinogram.data.begin(), sinogram.data.end(), 0.0f)
                       / sinogram.totalBins();
        if (corrections.scatterCorrection)
            for (auto& v : additive.data) v += corrections.scatterFraction * meanSino;
        if (corrections.randomsCorrection)
            for (auto& v : additive.data) v += corrections.randomsRate * meanSino;
    }

    for (int iter = 0; iter < iterations; ++iter) {
        float logLik = 0.0f;
        for (int subset = 0; subset < numSubsets; ++subset) {
            std::vector<int> subsetAngles;
            for (int a = subset; a < sinogram.numAngles; a += numSubsets)
                subsetAngles.push_back(a);

            // 子集灵敏度
            std::vector<float> subSensitivity(outputSize * outputSize, 0.0f);
            for (int a : subsetAngles) {
                float angle = static_cast<float>(M_PI) * a / sinogram.numAngles;
                for (int r = 0; r < sinogram.numRadialBins; ++r) {
                    float radialOffset = (r - halfBins + 0.5f) * binSpacing;
                    backprojectLOR(subSensitivity, outputSize, angle, radialOffset, halfSize, 1.0f);
                }
            }
            for (auto& s : subSensitivity) if (s < 1e-10f) s = 1e-10f;

            // 正向投影 (子集)
            std::vector<float> ratios;
            std::vector<std::pair<float, float>> lorParams;
            for (int a : subsetAngles) {
                float angle = static_cast<float>(M_PI) * a / sinogram.numAngles;
                for (int r = 0; r < sinogram.numRadialBins; ++r) {
                    float radialOffset = (r - halfBins + 0.5f) * binSpacing;
                    float proj = projectLOR(result.image, outputSize, angle, radialOffset, halfSize);
                    if (!additive.data.empty()) proj += additive.at(a, r);
                    proj = std::max(proj, 1e-10f);
                    float measured = sinogram.at(a, r);
                    ratios.push_back(measured / proj);
                    lorParams.push_back({angle, radialOffset});
                    if (measured > 0 && proj > 0)
                        logLik += measured * std::log(proj) - proj;
                }
            }

            // 反投影
            std::vector<float> correction(outputSize * outputSize, 0.0f);
            for (size_t i = 0; i < ratios.size(); ++i)
                backprojectLOR(correction, outputSize, lorParams[i].first,
                               lorParams[i].second, halfSize, ratios[i]);

            // 更新
            for (int i = 0; i < outputSize * outputSize; ++i) {
                result.image[i] *= correction[i] / subSensitivity[i];
                if (result.image[i] < 0.0f) result.image[i] = 0.0f;
            }
        }
        result.logLikelihood = logLik;
        result.iterationsRun = iter + 1;
        std::cout << "  [CPU] OSEM 迭代 " << (iter + 1) << "/" << iterations
                  << " (" << numSubsets << " subsets)  LogL=" << logLik << std::endl;
    }
    return result;
}

// ============================================================================
// FBP 重建 (PET 版)
// ============================================================================

PETReconResult PETReconstructor::reconstructFBP(
    const PETSinogram& sinogram, int outputSize)
{
    PETReconResult result;
    result.size = outputSize;
    result.image.resize(outputSize * outputSize, 0.0f);

    float halfSize = outputSize / 2.0f;
    float halfBins = sinogram.numRadialBins / 2.0f;
    float binSpacing = (outputSize * std::sqrt(2.0f)) / sinogram.numRadialBins;

    auto nextPow2 = [](int n) { int p = 1; while (p < n) p <<= 1; return p; };
    int N = nextPow2(2 * sinogram.numRadialBins);

    std::vector<std::vector<float>> filteredProj(sinogram.numAngles);
    for (int a = 0; a < sinogram.numAngles; ++a) {
        std::vector<std::complex<float>> freq(N, {0, 0});
        for (int r = 0; r < sinogram.numRadialBins; ++r)
            freq[r] = {sinogram.at(a, r), 0.0f};

        // FFT
        for (int i = 1, j = 0; i < N; ++i) {
            int bit = N >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(freq[i], freq[j]);
        }
        for (int len = 2; len <= N; len <<= 1) {
            float ang = 2.0f * static_cast<float>(M_PI) / len;
            std::complex<float> wn(std::cos(ang), std::sin(ang));
            for (int i = 0; i < N; i += len) {
                std::complex<float> w(1, 0);
                for (int j = 0; j < len / 2; ++j) {
                    auto u = freq[i + j];
                    auto v = freq[i + j + len / 2] * w;
                    freq[i + j] = u + v;
                    freq[i + j + len / 2] = u - v;
                    w *= wn;
                }
            }
        }
        // Ramp 滤波
        for (int i = 0; i < N; ++i) {
            float f = std::abs(i - N / 2.0f) / (N / 2.0f);
            if (f > 1.0f) f = 1.0f;
            freq[i] *= f;
        }
        // IFFT
        for (int i = 1, j = 0; i < N; ++i) {
            int bit = N >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(freq[i], freq[j]);
        }
        for (int len = 2; len <= N; len <<= 1) {
            float ang = -2.0f * static_cast<float>(M_PI) / len;
            std::complex<float> wn(std::cos(ang), std::sin(ang));
            for (int i = 0; i < N; i += len) {
                std::complex<float> w(1, 0);
                for (int j = 0; j < len / 2; ++j) {
                    auto u = freq[i + j];
                    auto v = freq[i + j + len / 2] * w;
                    freq[i + j] = u + v;
                    freq[i + j + len / 2] = u - v;
                    w *= wn;
                }
            }
        }
        filteredProj[a].resize(sinogram.numRadialBins);
        for (int r = 0; r < sinogram.numRadialBins; ++r)
            filteredProj[a][r] = freq[r].real() / N;
    }

    // 反投影
    float angleStep = static_cast<float>(M_PI) / sinogram.numAngles;
    for (int a = 0; a < sinogram.numAngles; ++a) {
        float angle = static_cast<float>(M_PI) * a / sinogram.numAngles;
        float cosA = std::cos(angle), sinA = std::sin(angle);
        for (int y = 0; y < outputSize; ++y) {
            for (int x = 0; x < outputSize; ++x) {
                float px = x - halfSize + 0.5f;
                float py = y - halfSize + 0.5f;
                float t = px * cosA + py * sinA;
                float detIdx = t / binSpacing + halfBins - 0.5f;
                int d0 = static_cast<int>(std::floor(detIdx));
                int d1 = d0 + 1;
                float frac = detIdx - d0;
                float val = 0.0f;
                if (d0 >= 0 && d1 < sinogram.numRadialBins)
                    val = (1 - frac) * filteredProj[a][d0] + frac * filteredProj[a][d1];
                else if (d0 >= 0 && d0 < sinogram.numRadialBins)
                    val = filteredProj[a][d0];
                result.image[y * outputSize + x] += val * angleStep;
            }
        }
    }
    return result;
}

// ============================================================================
// 统一接口
// ============================================================================

PETReconResult PETReconstructor::reconstruct(
    const PETSinogram& sinogram, int outputSize,
    PETReconMethod method, int iterations, int numSubsets,
    const PETCorrections& corrections)
{
    switch (method) {
    case PETReconMethod::MLEM:
        return reconstructMLEM(sinogram, outputSize, iterations, corrections);
    case PETReconMethod::OSEM:
        return reconstructOSEM(sinogram, outputSize, iterations, numSubsets, corrections);
    case PETReconMethod::FBP_PET:
        return reconstructFBP(sinogram, outputSize);
    default:
        return reconstructOSEM(sinogram, outputSize, iterations, numSubsets, corrections);
    }
}

// ============================================================================
// 噪声与工具
// ============================================================================

void PETReconstructor::addPoissonNoise(PETSinogram& sinogram, float scaleFactor) {
    std::mt19937 rng(42);
    for (auto& v : sinogram.data) {
        float lambda = std::max(v * scaleFactor, 0.01f);
        if (lambda < 50.0f) {
            std::poisson_distribution<int> poisson(lambda);
            v = static_cast<float>(poisson(rng)) / scaleFactor;
        } else {
            std::normal_distribution<float> normal(lambda, std::sqrt(lambda));
            v = std::max(normal(rng), 0.0f) / scaleFactor;
        }
    }
}

std::vector<float> PETReconstructor::generateAttenuationMap(int size, float mu) {
    std::vector<float> attenMap(size * size, 0.0f);
    float halfSize = size / 2.0f;
    float R = 0.8f;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float px = (x - halfSize + 0.5f) / halfSize;
            float py = (y - halfSize + 0.5f) / halfSize;
            if (px * px + py * py <= R * R) attenMap[y * size + x] = mu / size;
        }
    }
    return attenMap;
}

void PETReconstructor::applyGaussianSmooth(
    std::vector<float>& data, int width, int height, float sigma)
{
    if (sigma <= 0.0f) return;
    int ksize = static_cast<int>(std::ceil(sigma * 3)) * 2 + 1;
    std::vector<float> kernel(ksize);
    float sum = 0.0f;
    int half = ksize / 2;
    for (int i = 0; i < ksize; ++i) {
        float x = static_cast<float>(i - half);
        kernel[i] = std::exp(-x * x / (2.0f * sigma * sigma));
        sum += kernel[i];
    }
    for (auto& k : kernel) k /= sum;

    std::vector<float> temp(data.size());
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float val = 0.0f;
            for (int k = -half; k <= half; ++k) {
                int sx = std::clamp(x + k, 0, width - 1);
                val += data[y * width + sx] * kernel[k + half];
            }
            temp[y * width + x] = val;
        }
    }
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float val = 0.0f;
            for (int k = -half; k <= half; ++k) {
                int sy = std::clamp(y + k, 0, height - 1);
                val += temp[sy * width + x] * kernel[k + half];
            }
            data[y * width + x] = val;
        }
    }
}

bool PETReconstructor::savePGM(const std::string& filename,
    const std::vector<float>& image, int width, int height)
{
    float minVal = *std::min_element(image.begin(), image.end());
    float maxVal = *std::max_element(image.begin(), image.end());
    float range = maxVal - minVal;
    if (range < 1e-10f) range = 1.0f;
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;
    file << "P5\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        uint8_t v = static_cast<uint8_t>(
            std::clamp((image[i] - minVal) / range * 255.0f, 0.0f, 255.0f));
        file.write(reinterpret_cast<char*>(&v), 1);
    }
    file.close();
    return true;
}
