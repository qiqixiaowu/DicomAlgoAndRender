#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <complex>
#include <algorithm>
#include <functional>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// CT 重建核心模块
// 
// 支持算法:
//   1. 正向投影 (Radon 变换) - 模拟 CT 扫描生成正弦图
//   2. 滤波反投影 (FBP)      - 经典解析重建
//   3. 迭代重建 (SIRT/ART)   - 代数迭代重建
//
// 支持滤波器:
//   Ram-Lak, Shepp-Logan, Cosine, Hamming, Hann
// ============================================================================

// 滤波器类型
enum class FilterType {
    RAM_LAK,        // Ramp filter (Ram-Lak)
    SHEPP_LOGAN,    // Shepp-Logan filter
    COSINE,         // Cosine filter
    HAMMING,        // Hamming window
    HANN            // Hann window
};

// 重建算法类型
enum class ReconMethod {
    FBP,            // 滤波反投影 (Filtered Back Projection)
    SIRT,           // 同时迭代重建 (Simultaneous Iterative Reconstruction)
    ART             // 代数重建 (Algebraic Reconstruction Technique)
};

// 正弦图数据 (Sinogram)
struct Sinogram {
    std::vector<float> data;    // [numAngles * numDetectors]
    int numAngles = 0;          // 投影角度数
    int numDetectors = 0;       // 探测器通道数
    float angleStart = 0.0f;    // 起始角度 (弧度)
    float angleEnd = 0.0f;      // 终止角度 (弧度)

    float& at(int angle, int detector) {
        return data[angle * numDetectors + detector];
    }
    const float& at(int angle, int detector) const {
        return data[angle * numDetectors + detector];
    }
};

// 重建结果
struct ReconResult {
    std::vector<float> image;   // 重建图像 [size * size], 浮点
    int size = 0;               // 图像边长 (正方形)
    float convergenceError = 0; // 迭代收敛误差 (仅迭代法有效)

    // 归一化到 [0, 1]
    void normalize();
    // 转换为 8-bit 灰度
    std::vector<uint8_t> toUint8() const;
};

// Shepp-Logan 体模椭圆参数
struct PhantomEllipse {
    float intensity;    // 灰度值
    float cx, cy;       // 椭圆中心 (归一化坐标 [-1,1])
    float a, b;         // 半轴长度
    float theta;        // 旋转角度 (度)
};

// ============================================================================
// CT 重建器
// ============================================================================
class CTReconstructor {
public:
    CTReconstructor() = default;

    // ---- 体模生成 ----
    // 生成 Shepp-Logan 体模 (经典 CT 测试图像)
    static std::vector<float> generateSheppLoganPhantom(int size);

    // 自定义椭圆体模
    static std::vector<float> generatePhantom(
        int size, const std::vector<PhantomEllipse>& ellipses);

    // ---- 正向投影 (模拟 CT 扫描) ----
    // 平行束正向投影 (Radon 变换)
    static Sinogram forwardProject(
        const std::vector<float>& image, int imageSize,
        int numAngles = 180, int numDetectors = -1);

    // ---- 重建算法 ----
    // 滤波反投影 (FBP)
    static ReconResult reconstructFBP(
        const Sinogram& sinogram, int outputSize,
        FilterType filter = FilterType::RAM_LAK);

    // SIRT 迭代重建
    static ReconResult reconstructSIRT(
        const Sinogram& sinogram, int outputSize,
        int iterations = 50, float relaxation = 0.1f);

    // ART 迭代重建
    static ReconResult reconstructART(
        const Sinogram& sinogram, int outputSize,
        int iterations = 10, float relaxation = 0.25f);

    // 统一接口
    static ReconResult reconstruct(
        const Sinogram& sinogram, int outputSize,
        ReconMethod method = ReconMethod::FBP,
        FilterType filter = FilterType::RAM_LAK,
        int iterations = 50, float relaxation = 0.1f);

    // ---- 滤波器 ----
    // 生成频域滤波器
    static std::vector<float> createFilter(int size, FilterType type);

    // 对单条投影线进行1D滤波
    static std::vector<float> applyFilter(
        const std::vector<float>& projection, FilterType type);

    // ---- 工具函数 ----
    // 添加泊松噪声 (模拟光子噪声)
    static void addPoissonNoise(Sinogram& sinogram, float photonCount = 1e5f);

    // 保存为 PGM 灰度图 (方便调试, 无需额外库)
    static bool savePGM(const std::string& filename,
        const std::vector<float>& image, int width, int height);
    static bool savePGM(const std::string& filename,
        const std::vector<uint8_t>& image, int width, int height);

private:
    // 1D FFT (Cooley-Tukey, 就地计算)
    static void fft1D(std::vector<std::complex<float>>& data, bool inverse = false);
    // 下一个 2 的幂
    static int nextPowerOf2(int n);
    // 双线性插值取值
    static float bilinearSample(const std::vector<float>& image, int size,
        float x, float y);
};
