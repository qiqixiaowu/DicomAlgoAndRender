// ============================================================================
// CUDA 加速 PET 重建模块
//
// 提供 MLEM / OSEM 的 GPU 加速版本
// ============================================================================

#pragma once

#include <vector>
#include <cstdint>
#include <string>

// ---- GPU 设备信息 ----
struct CudaDeviceInfo {
    std::string name;
    int major = 0, minor = 0;
    size_t totalMemMB = 0;
    int smCount = 0;
    int maxThreadsPerBlock = 0;
    bool available = false;
};

// ---- PET 重建参数 (GPU) ----
struct CudaPETParams {
    int numAngles = 180;
    int numRadialBins = 0;          // 0 = 自动
    int outputSize = 128;
    int iterations = 5;
    int numSubsets = 12;
    // 校正
    bool scatterCorrection = false;
    float scatterFraction = 0.1f;
};

// ---- GPU 重建结果 ----
struct CudaReconResult {
    std::vector<float> image;
    int size = 0;
    float elapsedMs = 0.0f;
    float convergenceError = 0;
    int iterationsRun = 0;
};

// ---- CUDA 重建器 ----
class CudaReconstructor {
public:
    static CudaDeviceInfo queryDevice(int deviceId = 0);
    static bool isAvailable();

    static CudaReconResult petMLEM(const float* sinogramData,
                                   const CudaPETParams& params);
    static CudaReconResult petOSEM(const float* sinogramData,
                                   const CudaPETParams& params);
};
