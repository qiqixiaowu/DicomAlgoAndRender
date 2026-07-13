#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <string>
#include <random>
#include <functional>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// PET 重建核心模块
//
// PET (正电子发射断层扫描) 原理:
//   放射性示踪剂在体内发射正电子, 正电子与电子湮灭后产生一对
//   方向相反的 511 keV γ 光子, 探测器环检测这对光子形成符合事件 (LOR).
//   重建目标: 从 LOR 数据恢复放射性分布图像.
//
// 支持算法:
//   1. MLEM  - 最大似然期望最大化 (Maximum Likelihood Expectation Maximization)
//   2. OSEM  - 有序子集期望最大化 (Ordered Subset EM)
//   3. FBP   - 滤波反投影 (作为参考基线)
//
// 支持特性:
//   - 环形探测器几何模拟
//   - 衰减校正 (简化)
//   - 散射校正 (简化常数估计)
//   - 随机校正 (均匀随机符合估计)
//   - 分辨率建模 (高斯 PSF)
//   - Derenzo 体模 / 热球体模 / 自定义点源体模
// ============================================================================

// PET 重建算法
enum class PETReconMethod {
    MLEM,       // Maximum Likelihood Expectation Maximization
    OSEM,       // Ordered Subset EM
    FBP_PET     // 滤波反投影 (PET 版)
};

// PET 正弦图 (2D 环形探测器)
struct PETSinogram {
    std::vector<float> data;    // [numAngles * numRadialBins]
    int numAngles = 0;          // 角度采样数 (半角 [0, π))
    int numRadialBins = 0;      // 径向采样数 (LOR 偏移量)

    float& at(int angle, int radial) {
        return data[angle * numRadialBins + radial];
    }
    const float& at(int angle, int radial) const {
        return data[angle * numRadialBins + radial];
    }

    size_t totalBins() const { return static_cast<size_t>(numAngles) * numRadialBins; }
};

// PET 重建结果
struct PETReconResult {
    std::vector<float> image;   // [size * size] 重建图像
    int size = 0;
    float logLikelihood = 0.0f; // 对数似然 (MLEM/OSEM)
    int iterationsRun = 0;

    void normalize();
    std::vector<uint8_t> toUint8() const;
};

// PET 体模中的热点/冷点区域
struct PETHotspot {
    float cx, cy;       // 中心位置 (归一化 [-1,1])
    float radius;       // 半径 (归一化)
    float activity;     // 放射性活度 (相对值)
};

// PET 3D 体模热点 (球形)
struct PETHotspot3D {
    float cx, cy, cz;   // 3D 中心位置 (归一化 [-1,1])
    float radius;       // 球半径 (归一化)
    float activity;     // 放射性活度
};

// 3D 正弦图 (逐切片, 每切片一个 PETSinogram)
struct PETSinogram3D {
    std::vector<PETSinogram> slices;
    int numSlices     = 0;
    int numAngles     = 0;
    int numRadialBins = 0;
};

// 3D 重建结果
struct PETReconResult3D {
    std::vector<float> volume;  // [numSlices * sizeXY * sizeXY]
    int sizeXY    = 0;
    int numSlices = 0;
    float logLikelihood = 0.0f;
    int iterationsRun   = 0;

    std::vector<float> getSlice(int z) const;
    void normalize();
};

// PET 校正参数
struct PETCorrections {
    bool attenuationCorrection = false;     // 衰减校正
    float mu = 0.096f;                      // 线性衰减系数 (cm^-1, 水 @511keV)
    bool scatterCorrection = false;         // 散射校正
    float scatterFraction = 0.1f;           // 散射比例
    bool randomsCorrection = false;         // 随机符合校正
    float randomsRate = 0.05f;              // 随机符合比例
    bool psfModeling = false;               // 点扩散函数 (PSF) 建模
    float psfFWHM = 4.0f;                   // PSF 半高宽 (mm)
};

// ============================================================================
// PET 重建器
// ============================================================================
class PETReconstructor {
public:
    PETReconstructor() = default;

    // ---- 体模生成 ----

    // Derenzo 体模 (分辨率测试, 6 组不同大小的热点)
    static std::vector<float> generateDerenzoPhantom(int size);

    // 热球体模 (NEMA IEC 标准, 6 个球+背景)
    static std::vector<float> generateHotColdPhantom(int size);

    // 自定义热点体模
    static std::vector<float> generatePhantom(
        int size, float background,
        const std::vector<PETHotspot>& hotspots);

    // 均匀圆柱体模 (用于归一化)
    static std::vector<float> generateUniformCylinder(int size, float activity = 1.0f);

    // ---- 正向投影 (模拟 PET 扫描) ----

    // 2D 环形探测器正向投影
    static PETSinogram forwardProject(
        const std::vector<float>& image, int imageSize,
        int numAngles = 180, int numRadialBins = -1);

    // 带衰减的正向投影
    static PETSinogram forwardProjectAtten(
        const std::vector<float>& activityMap, 
        const std::vector<float>& attenuationMap,
        int imageSize,
        int numAngles = 180, int numRadialBins = -1);

    // ---- 重建算法 ----

    // MLEM 重建
    static PETReconResult reconstructMLEM(
        const PETSinogram& sinogram, int outputSize,
        int iterations = 30,
        const PETCorrections& corrections = {});

    // OSEM 重建
    static PETReconResult reconstructOSEM(
        const PETSinogram& sinogram, int outputSize,
        int iterations = 5, int numSubsets = 12,
        const PETCorrections& corrections = {});

    // FBP 重建 (PET 版)
    static PETReconResult reconstructFBP(
        const PETSinogram& sinogram, int outputSize);

    // 统一接口
    static PETReconResult reconstruct(
        const PETSinogram& sinogram, int outputSize,
        PETReconMethod method = PETReconMethod::OSEM,
        int iterations = 10, int numSubsets = 12,
        const PETCorrections& corrections = {});

    // ---- 噪声与校正 ----

    // 模拟泊松噪声 (PET 固有统计噪声)
    static void addPoissonNoise(PETSinogram& sinogram, float scaleFactor = 1.0f);

    // 生成衰减图 (简化: 均匀椭圆)
    static std::vector<float> generateAttenuationMap(
        int size, float mu = 0.096f);

    // 归一化校正因子 (探测器效率)
    static std::vector<float> computeNormalization(
        int numAngles, int numRadialBins);

    // ---- 3D 体模生成 ----
    static std::vector<float> generate3DHotColdPhantom(int size, int numSlices);
    static std::vector<float> generate3DDerenzoPhantom(int size, int numSlices);

    // ---- 3D 正向投影 (逐切片) ----
    static PETSinogram3D forwardProject3D(
        const std::vector<float>& volume, int imageSize, int numSlices,
        int numAngles = 180, int numRadialBins = -1);

    // ---- 3D 重建 (逐切片) ----
    static PETReconResult3D reconstruct3D(
        const PETSinogram3D& sinogram3D, int outputSize,
        PETReconMethod method = PETReconMethod::OSEM,
        int iterations = 10, int numSubsets = 12,
        const PETCorrections& corrections = {});

    // ---- 3D 泊松噪声 ----
    static void addPoissonNoise3D(PETSinogram3D& sinogram3D, float scaleFactor = 1.0f);

    // ---- 工具 ----

    static bool savePGM(const std::string& filename,
        const std::vector<float>& image, int width, int height);

private:
    // 单条 LOR 的正向投影值
    static float projectLOR(
        const std::vector<float>& image, int size,
        float angle, float radialOffset, float halfSize);

    // 单条 LOR 反投影到图像
    static void backprojectLOR(
        std::vector<float>& image, int size,
        float angle, float radialOffset, float halfSize,
        float value);

    // 双线性插值
    static float bilinearSample(
        const std::vector<float>& image, int size, float x, float y);

    // 衰减线积分
    static float computeAttenuationFactor(
        const std::vector<float>& attenMap, int size,
        float angle, float radialOffset, float halfSize);

    // 1D 高斯平滑 (PSF 建模)
    static void applyGaussianSmooth(
        std::vector<float>& data, int width, int height, float sigma);
};
