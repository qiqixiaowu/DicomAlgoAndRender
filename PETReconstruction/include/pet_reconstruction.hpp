// ============================================================================
// PET 重建核心模块 (CPU)
//
// 支持算法: MLEM / OSEM / FBP
// 支持校正: 衰减 / 散射 / 随机 / PSF
// ============================================================================

#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <string>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---- 数据结构 ----

enum class PETReconMethod { MLEM, OSEM, FBP_PET };

struct PETSinogram {
    std::vector<float> data;
    int numAngles = 0;
    int numRadialBins = 0;
    float& at(int angle, int radial) { return data[angle * numRadialBins + radial]; }
    const float& at(int angle, int radial) const { return data[angle * numRadialBins + radial]; }
    size_t totalBins() const { return static_cast<size_t>(numAngles) * numRadialBins; }
};

struct PETReconResult {
    std::vector<float> image;
    int size = 0;
    float logLikelihood = 0.0f;
    int iterationsRun = 0;
    void normalize();
    std::vector<uint8_t> toUint8() const;
};

struct PETHotspot {
    float cx, cy, radius, activity;
};

struct PETCorrections {
    bool attenuationCorrection = false;
    float mu = 0.096f;
    bool scatterCorrection = false;
    float scatterFraction = 0.1f;
    bool randomsCorrection = false;
    float randomsRate = 0.05f;
    bool psfModeling = false;
    float psfFWHM = 4.0f;
};

// ---- PET 重建器 ----

class PETReconstructor {
public:
    // 体模生成
    static std::vector<float> generateHotColdPhantom(int size);
    static std::vector<float> generateDerenzoPhantom(int size);
    static std::vector<float> generateUniformCylinder(int size, float activity = 1.0f);
    static std::vector<float> generatePhantom(int size, float background,
                                               const std::vector<PETHotspot>& hotspots);

    // 正向投影
    static PETSinogram forwardProject(const std::vector<float>& image, int imageSize,
                                      int numAngles, int numRadialBins = 0);
    static PETSinogram forwardProjectAtten(const std::vector<float>& activity,
                                           const std::vector<float>& attenMap,
                                           int imageSize, int numAngles,
                                           int numRadialBins = 0);

    // 重建算法
    static PETReconResult reconstructMLEM(const PETSinogram& sino, int outputSize,
                                          int iterations = 20,
                                          const PETCorrections& corr = {});
    static PETReconResult reconstructOSEM(const PETSinogram& sino, int outputSize,
                                          int iterations = 5, int numSubsets = 12,
                                          const PETCorrections& corr = {});
    static PETReconResult reconstructFBP(const PETSinogram& sino, int outputSize);

    // 统一接口
    static PETReconResult reconstruct(const PETSinogram& sino, int outputSize,
                                      PETReconMethod method,
                                      int iterations = 10, int numSubsets = 12,
                                      const PETCorrections& corr = {});

    // 噪声与工具
    static void addPoissonNoise(PETSinogram& sino, float scaleFactor = 1.0f);
    static std::vector<float> generateAttenuationMap(int size, float mu = 0.096f);
    static bool savePGM(const std::string& path, const std::vector<float>& img,
                        int w, int h);

private:
    static float bilinearSample(const std::vector<float>& img, int size, float x, float y);
    static float projectLOR(const std::vector<float>& img, int size,
                            float angle, float radialOffset, float halfSize);
    static void backprojectLOR(std::vector<float>& img, int size,
                               float angle, float radialOffset, float halfSize, float value);
    static float computeAttenuationFactor(const std::vector<float>& attenMap, int size,
                                          float angle, float radialOffset, float halfSize);
    static void applyGaussianSmooth(std::vector<float>& data, int w, int h, float sigma);
};
