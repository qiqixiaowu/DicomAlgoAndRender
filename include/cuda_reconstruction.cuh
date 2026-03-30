#pragma once

// ============================================================================
// CUDA 加速重建模块
//
// 提供 CT 和 PET 重建的 GPU 加速版本:
//   - CT:  FBP (GPU 反投影)
//   - CT:  SIRT (GPU 正投影 + 反投影)
//   - PET: MLEM (GPU EM 迭代)
//   - PET: OSEM (GPU 子集 EM)
//
// 部分函数提供 C++ 包装, 内部调用 CUDA kernel
// 需要: CUDA Toolkit >= 11.0, 单精度浮点
// ============================================================================

#include <vector>
#include <cstdint>
#include <string>

// 前向声明, 避免在非 CUDA 编译单元引入 cuda_runtime.h
struct CudaReconContext;

// GPU 设备信息
struct CudaDeviceInfo {
    std::string name;
    int major = 0, minor = 0;          // 计算能力
    size_t totalMemMB = 0;
    int smCount = 0;                   // SM 数
    int maxThreadsPerBlock = 0;
    bool available = false;
};

// CT 重建参数 (GPU)
struct CudaCTParams {
    int numAngles = 360;
    int numDetectors = 0;       // 0 = 自动
    int outputSize = 256;
    float angleStart = 0.0f;
    float angleEnd = 3.14159265f;
    int filterType = 0;         // 0=RamLak 1=SheppLogan 2=Cosine 3=Hamming 4=Hann
    // 迭代参数 (SIRT)
    int iterations = 50;
    float relaxation = 0.1f;
};

// PET 重建参数 (GPU)
struct CudaPETParams {
    int numAngles = 180;
    int numRadialBins = 0;      // 0 = 自动
    int outputSize = 128;
    int iterations = 10;
    int numSubsets = 12;
    // 校正
    bool attenuationCorrection = false;
    bool scatterCorrection = false;
    float scatterFraction = 0.1f;
};

// GPU 重建结果
struct CudaReconResult {
    std::vector<float> image;
    int size = 0;
    float elapsedMs = 0.0f;    // GPU 计时
    float convergenceError = 0;
};

// ============================================================================
// CUDA 重建器 - C++ 接口
// ============================================================================
class CudaReconstructor {
public:
    // ---- 设备管理 ----
    static CudaDeviceInfo queryDevice(int deviceId = 0);
    static bool isAvailable();

    // ---- CT 重建 (GPU) ----

    // GPU FBP: 滤波在 CPU (或 cuFFT), 反投影在 GPU
    static CudaReconResult ctFBP(
        const float* sinogramData, const CudaCTParams& params);

    // GPU SIRT
    static CudaReconResult ctSIRT(
        const float* sinogramData, const CudaCTParams& params);

    // ---- PET 重建 (GPU) ----

    // GPU MLEM
    static CudaReconResult petMLEM(
        const float* sinogramData, const CudaPETParams& params);

    // GPU OSEM
    static CudaReconResult petOSEM(
        const float* sinogramData, const CudaPETParams& params);

    // ---- 工具 ----

    // 正向投影 (GPU)
    static void gpuForwardProject(
        const float* image, int imageSize,
        float* sinogram, int numAngles, int numDetectors,
        float angleStart, float angleEnd);

    // 反投影 (GPU)
    static void gpuBackProject(
        const float* sinogram, int numAngles, int numDetectors,
        float* image, int imageSize,
        float angleStart, float angleEnd);

    // GPU 1D 滤波 (批量, 对 sinogram 每行)
    static void gpuApplyRampFilter(
        float* sinogram, int numAngles, int numDetectors,
        int filterType);
};
