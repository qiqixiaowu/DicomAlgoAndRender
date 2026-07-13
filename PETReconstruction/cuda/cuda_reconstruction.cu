// ============================================================================
// CUDA 加速 PET 重建 - 核心实现
//
// Kernels:
//   kernelPETForwardProject       - 全角度正投影 (MLEM)
//   kernelPETForwardProjectSubset - 子集正投影 (OSEM)
//   kernelPETBackProject          - 反投影 (支持子集角度索引)
//   kernelElementDiv              - 逐元素除法 (ratio = y / ȳ)
//   kernelElementClamp            - 逐元素钳位 (防除零)
//   kernelMLEMUpdate              - 乘法更新 λ *= corr / S
// ============================================================================

#include "cuda_reconstruction.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            return {};                                                         \
        }                                                                      \
    } while (0)

// ============================================================================
// CUDA Kernels
// ============================================================================

// ---- PET 正向投影 kernel (全角度, MLEM 用) ----
__global__ void kernelPETForwardProject(
    const float* __restrict__ image, int imageSize,
    float* __restrict__ sinogram, int numAngles, int numRadialBins,
    float binSpacing)
{
    int a = blockIdx.y * blockDim.y + threadIdx.y;
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= numAngles || r >= numRadialBins) return;

    float halfImg = imageSize / 2.0f;
    float halfBins = numRadialBins / 2.0f;
    float angle = M_PI * a / numAngles;
    float cosA = cosf(angle), sinA = sinf(angle);
    float radialOffset = (r - halfBins + 0.5f) * binSpacing;
    float sMax = halfImg * sqrtf(2.0f);

    float sum = 0.0f;
    for (float s = -sMax; s <= sMax; s += 1.0f) {
        float fx = radialOffset * cosA - s * sinA + halfImg - 0.5f;
        float fy = radialOffset * sinA + s * cosA + halfImg - 0.5f;
        int x0 = (int)floorf(fx);
        int y0 = (int)floorf(fy);
        if (x0 < 0 || x0 >= imageSize - 1 || y0 < 0 || y0 >= imageSize - 1) continue;
        float dx = fx - x0, dy = fy - y0;
        sum += image[y0 * imageSize + x0] * (1 - dx) * (1 - dy)
             + image[y0 * imageSize + x0 + 1] * dx * (1 - dy)
             + image[(y0 + 1) * imageSize + x0] * (1 - dx) * dy
             + image[(y0 + 1) * imageSize + x0 + 1] * dx * dy;
    }
    sinogram[a * numRadialBins + r] = sum;
}

// ---- PET 正向投影 kernel (子集版, OSEM 用) ----
__global__ void kernelPETForwardProjectSubset(
    const float* __restrict__ image, int imageSize,
    float* __restrict__ sinogram, int numAngles, int numRadialBins,
    float binSpacing,
    const int* __restrict__ angleIndices, int numAngleIndices)
{
    int ai = blockIdx.y * blockDim.y + threadIdx.y;
    int r  = blockIdx.x * blockDim.x + threadIdx.x;
    if (ai >= numAngleIndices || r >= numRadialBins) return;

    int a = angleIndices[ai];
    float halfImg  = imageSize / 2.0f;
    float halfBins = numRadialBins / 2.0f;
    float angle = M_PI * a / numAngles;
    float cosA = cosf(angle), sinA = sinf(angle);
    float radialOffset = (r - halfBins + 0.5f) * binSpacing;
    float sMax = halfImg * sqrtf(2.0f);

    float sum = 0.0f;
    for (float s = -sMax; s <= sMax; s += 1.0f) {
        float fx = radialOffset * cosA - s * sinA + halfImg - 0.5f;
        float fy = radialOffset * sinA + s * cosA + halfImg - 0.5f;
        int x0 = (int)floorf(fx);
        int y0 = (int)floorf(fy);
        if (x0 < 0 || x0 >= imageSize - 1 || y0 < 0 || y0 >= imageSize - 1) continue;
        float dx = fx - x0, dy = fy - y0;
        sum += image[y0 * imageSize + x0]       * (1 - dx) * (1 - dy)
             + image[y0 * imageSize + x0 + 1]   * dx       * (1 - dy)
             + image[(y0 + 1) * imageSize + x0] * (1 - dx) * dy
             + image[(y0 + 1) * imageSize + x0 + 1] * dx   * dy;
    }
    sinogram[ai * numRadialBins + r] = sum;
}

// ---- PET 反投影 kernel (支持子集角度索引) ----
__global__ void kernelPETBackProject(
    const float* __restrict__ sinogram, int numAngles, int numRadialBins,
    float* __restrict__ image, int imageSize,
    float binSpacing,
    const int* __restrict__ angleIndices, int numAngleIndices)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= imageSize || y >= imageSize) return;

    float halfImg = imageSize / 2.0f;
    float halfBins = numRadialBins / 2.0f;
    float px = x - halfImg + 0.5f;
    float py = y - halfImg + 0.5f;

    float sum = 0.0f;
    int nAngles = (angleIndices != nullptr) ? numAngleIndices : numAngles;

    for (int ai = 0; ai < nAngles; ++ai) {
        int a = (angleIndices != nullptr) ? angleIndices[ai] : ai;
        float angle = M_PI * a / numAngles;
        float cosA = cosf(angle), sinA = sinf(angle);
        float t = px * cosA + py * sinA;
        float binIdx = t / binSpacing + halfBins - 0.5f;

        int b0 = (int)floorf(binIdx);
        int b1 = b0 + 1;
        float frac = binIdx - b0;

        float val = 0.0f;
        if (b0 >= 0 && b1 < numRadialBins) {
            val = (1.0f - frac) * sinogram[a * numRadialBins + b0]
                + frac * sinogram[a * numRadialBins + b1];
        } else if (b0 >= 0 && b0 < numRadialBins) {
            val = sinogram[a * numRadialBins + b0];
        }
        sum += val;
    }
    image[y * imageSize + x] = sum;
}

// ---- 逐元素除法 (ratio = measured / estimated) ----
__global__ void kernelElementDiv(
    const float* __restrict__ measured,
    const float* __restrict__ estimated,
    float* __restrict__ ratio, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float est = estimated[i];
    ratio[i] = (est > 1e-10f) ? (measured[i] / est) : 0.0f;
}

// ---- 逐元素钳位 (val = max(val, epsilon)) ----
__global__ void kernelElementClamp(
    float* __restrict__ data, float eps, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (data[i] < eps) data[i] = eps;
}

// ---- MLEM 乘法更新: λ_j *= correction_j / sensitivity_j ----
__global__ void kernelMLEMUpdate(
    float* __restrict__ image, int imageSize,
    const float* __restrict__ correction,
    const float* __restrict__ sensitivity)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = imageSize * imageSize;
    if (idx >= total) return;

    float s = sensitivity[idx];
    if (s < 1e-10f) s = 1e-10f;
    image[idx] *= correction[idx] / s;
    if (image[idx] < 0.0f) image[idx] = 0.0f;  // 非负约束
}

// ============================================================================
// Host 端实现
// ============================================================================

CudaDeviceInfo CudaReconstructor::queryDevice(int deviceId) {
    CudaDeviceInfo info;
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0 || deviceId >= count) {
        info.available = false;
        return info;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    info.name = prop.name;
    info.major = prop.major;
    info.minor = prop.minor;
    info.totalMemMB = prop.totalGlobalMem / (1024 * 1024);
    info.smCount = prop.multiProcessorCount;
    info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
    info.available = true;
    return info;
}

bool CudaReconstructor::isAvailable() {
    int count = 0;
    return (cudaGetDeviceCount(&count) == cudaSuccess && count > 0);
}

// ============================================================================
// PET MLEM (GPU)
// ============================================================================

CudaReconResult CudaReconstructor::petMLEM(
    const float* sinogramData, const CudaPETParams& params)
{
    CudaReconResult result;
    int numAng = params.numAngles;
    int numBins = params.numRadialBins > 0 ? params.numRadialBins
        : static_cast<int>(std::ceil(params.outputSize * std::sqrt(2.0f)));
    int outSize = params.outputSize;
    float binSpacing = (outSize * std::sqrt(2.0f)) / numBins;
    size_t sinoBytes = numAng * numBins * sizeof(float);
    size_t imgBytes = outSize * outSize * sizeof(float);
    int sinoTotal = numAng * numBins;
    int imgTotal = outSize * outSize;

    float *d_measured, *d_image, *d_estimated, *d_ratio, *d_correction, *d_sensitivity;
    CUDA_CHECK(cudaMalloc(&d_measured, sinoBytes));
    CUDA_CHECK(cudaMalloc(&d_image, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_estimated, sinoBytes));
    CUDA_CHECK(cudaMalloc(&d_ratio, sinoBytes));
    CUDA_CHECK(cudaMalloc(&d_correction, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_sensitivity, imgBytes));

    CUDA_CHECK(cudaMemcpy(d_measured, sinogramData, sinoBytes, cudaMemcpyHostToDevice));
    std::vector<float> initImg(imgTotal, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_image, initImg.data(), imgBytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    dim3 sinoBlock(16, 16);
    dim3 sinoGrid((numBins + 15) / 16, (numAng + 15) / 16);
    dim3 imgBlock(16, 16);
    dim3 imgGrid((outSize + 15) / 16, (outSize + 15) / 16);
    int linBlock = 256;

    // 预计算灵敏度图
    {
        std::vector<float> ones(sinoTotal, 1.0f);
        float* d_ones;
        CUDA_CHECK(cudaMalloc(&d_ones, sinoBytes));
        CUDA_CHECK(cudaMemcpy(d_ones, ones.data(), sinoBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_sensitivity, 0, imgBytes));
        kernelPETBackProject<<<imgGrid, imgBlock>>>(
            d_ones, numAng, numBins, d_sensitivity, outSize, binSpacing, nullptr, 0);
        CUDA_CHECK(cudaGetLastError());
        cudaFree(d_ones);
    }

    for (int iter = 0; iter < params.iterations; ++iter) {
        // 1. 正向投影
        CUDA_CHECK(cudaMemset(d_estimated, 0, sinoBytes));
        kernelPETForwardProject<<<sinoGrid, sinoBlock>>>(
            d_image, outSize, d_estimated, numAng, numBins, binSpacing);
        CUDA_CHECK(cudaGetLastError());

        // 2. ratio = measured / estimated
        {
            int blocks = (sinoTotal + linBlock - 1) / linBlock;
            kernelElementDiv<<<blocks, linBlock>>>(d_measured, d_estimated, d_ratio, sinoTotal);
            CUDA_CHECK(cudaGetLastError());
        }

        // 3. 反投影 ratio
        CUDA_CHECK(cudaMemset(d_correction, 0, imgBytes));
        kernelPETBackProject<<<imgGrid, imgBlock>>>(
            d_ratio, numAng, numBins, d_correction, outSize, binSpacing, nullptr, 0);
        CUDA_CHECK(cudaGetLastError());

        // 4. MLEM 更新
        {
            int blocks = (imgTotal + linBlock - 1) / linBlock;
            kernelMLEMUpdate<<<blocks, linBlock>>>(d_image, outSize, d_correction, d_sensitivity);
            CUDA_CHECK(cudaGetLastError());
        }

        if ((iter + 1) % 5 == 0 || iter == 0)
            std::cout << "  [GPU] MLEM 迭代 " << (iter + 1) << "/" << params.iterations << std::endl;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result.elapsedMs, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    result.size = outSize;
    result.image.resize(imgTotal);
    CUDA_CHECK(cudaMemcpy(result.image.data(), d_image, imgBytes, cudaMemcpyDeviceToHost));
    result.iterationsRun = params.iterations;

    cudaFree(d_measured); cudaFree(d_image); cudaFree(d_estimated);
    cudaFree(d_ratio); cudaFree(d_correction); cudaFree(d_sensitivity);

    std::cout << "  [GPU] PET MLEM 完成: " << result.elapsedMs << " ms" << std::endl;
    return result;
}

// ============================================================================
// PET OSEM (GPU)
// ============================================================================

CudaReconResult CudaReconstructor::petOSEM(
    const float* sinogramData, const CudaPETParams& params)
{
    CudaReconResult result;
    int numAng   = params.numAngles;
    int numBins  = params.numRadialBins > 0 ? params.numRadialBins
        : static_cast<int>(std::ceil(params.outputSize * std::sqrt(2.0f)));
    int outSize  = params.outputSize;
    float binSpacing = (outSize * std::sqrt(2.0f)) / numBins;
    int numSubsets   = params.numSubsets;
    int imgTotal     = outSize * outSize;

    int nSubAng = (numAng + numSubsets - 1) / numSubsets;
    int subSinoTotal = nSubAng * numBins;

    size_t sinoBytes     = numAng * numBins * sizeof(float);
    size_t imgBytes      = imgTotal * sizeof(float);
    size_t subSinoBytes  = subSinoTotal * sizeof(float);
    size_t subAngBytes   = nSubAng * sizeof(int);

    float *d_measured, *d_image, *d_estimated, *d_ratio, *d_correction, *d_sensitivity;
    float *d_subOnes;
    int   *d_angleIndices;
    CUDA_CHECK(cudaMalloc(&d_measured,     sinoBytes));
    CUDA_CHECK(cudaMalloc(&d_image,        imgBytes));
    CUDA_CHECK(cudaMalloc(&d_estimated,    subSinoBytes));
    CUDA_CHECK(cudaMalloc(&d_ratio,        subSinoBytes));
    CUDA_CHECK(cudaMalloc(&d_correction,   imgBytes));
    CUDA_CHECK(cudaMalloc(&d_sensitivity,  imgBytes));
    CUDA_CHECK(cudaMalloc(&d_subOnes,      subSinoBytes));
    CUDA_CHECK(cudaMalloc(&d_angleIndices, subAngBytes));

    CUDA_CHECK(cudaMemcpy(d_measured, sinogramData, sinoBytes, cudaMemcpyHostToDevice));
    std::vector<float> initImg(imgTotal, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_image, initImg.data(), imgBytes, cudaMemcpyHostToDevice));
    std::vector<float> subOnesHost(subSinoTotal, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_subOnes, subOnesHost.data(), subSinoBytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    dim3 sinoBlock(16, 16);
    dim3 imgBlock(16, 16);
    dim3 imgGrid((outSize + 15) / 16, (outSize + 15) / 16);
    int  linBlock = 256;

    // 预计算子集角度索引
    std::vector<std::vector<int>> allSubAngles(numSubsets);
    for (int sub = 0; sub < numSubsets; ++sub)
        for (int a = sub; a < numAng; a += numSubsets)
            allSubAngles[sub].push_back(a);

    for (int iter = 0; iter < params.iterations; ++iter) {
        for (int sub = 0; sub < numSubsets; ++sub) {
            int curNSub = static_cast<int>(allSubAngles[sub].size());
            int curSubSinoTotal = curNSub * numBins;

            CUDA_CHECK(cudaMemcpy(d_angleIndices, allSubAngles[sub].data(),
                curNSub * sizeof(int), cudaMemcpyHostToDevice));

            // ① 子集灵敏度
            CUDA_CHECK(cudaMemset(d_sensitivity, 0, imgBytes));
            kernelPETBackProject<<<imgGrid, imgBlock>>>(
                d_subOnes, numAng, numBins, d_sensitivity, outSize,
                binSpacing, d_angleIndices, curNSub);
            CUDA_CHECK(cudaGetLastError());

            // ② 正投影 (子集)
            CUDA_CHECK(cudaMemset(d_estimated, 0, subSinoBytes));
            {
                dim3 grid((numBins + 15) / 16, (curNSub + 15) / 16);
                kernelPETForwardProjectSubset<<<grid, sinoBlock>>>(
                    d_image, outSize, d_estimated, numAng, numBins,
                    binSpacing, d_angleIndices, curNSub);
                CUDA_CHECK(cudaGetLastError());
            }

            // 钳位
            {
                int blocks = (curSubSinoTotal + linBlock - 1) / linBlock;
                kernelElementClamp<<<blocks, linBlock>>>(d_estimated, 1e-10f, curSubSinoTotal);
                CUDA_CHECK(cudaGetLastError());
            }

            // ③ ratio = y / ȳ (gather 子集测量值)
            {
                std::vector<float> subMeasured(curSubSinoTotal);
                for (int ai = 0; ai < curNSub; ++ai) {
                    int a = allSubAngles[sub][ai];
                    CUDA_CHECK(cudaMemcpy(subMeasured.data() + ai * numBins,
                        d_measured + a * numBins, numBins * sizeof(float),
                        cudaMemcpyDeviceToHost));
                }
                float* d_subMeasured;
                CUDA_CHECK(cudaMalloc(&d_subMeasured, curSubSinoTotal * sizeof(float)));
                CUDA_CHECK(cudaMemcpy(d_subMeasured, subMeasured.data(),
                    curSubSinoTotal * sizeof(float), cudaMemcpyHostToDevice));
                int blocks = (curSubSinoTotal + linBlock - 1) / linBlock;
                kernelElementDiv<<<blocks, linBlock>>>(
                    d_subMeasured, d_estimated, d_ratio, curSubSinoTotal);
                CUDA_CHECK(cudaGetLastError());
                cudaFree(d_subMeasured);
            }

            // ④ 反投影 ratio (子集)
            CUDA_CHECK(cudaMemset(d_correction, 0, imgBytes));
            kernelPETBackProject<<<imgGrid, imgBlock>>>(
                d_ratio, numAng, numBins, d_correction, outSize,
                binSpacing, d_angleIndices, curNSub);
            CUDA_CHECK(cudaGetLastError());

            // ⑤ 乘法更新
            {
                int blocks = (imgTotal + linBlock - 1) / linBlock;
                kernelMLEMUpdate<<<blocks, linBlock>>>(
                    d_image, outSize, d_correction, d_sensitivity);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        if ((iter + 1) % 5 == 0 || iter == 0)
            std::cout << "  [GPU] OSEM 迭代 " << (iter + 1) << "/" << params.iterations
                      << " (" << numSubsets << " subsets)" << std::endl;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result.elapsedMs, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    result.size = outSize;
    result.image.resize(imgTotal);
    CUDA_CHECK(cudaMemcpy(result.image.data(), d_image, imgBytes, cudaMemcpyDeviceToHost));
    result.iterationsRun = params.iterations;

    cudaFree(d_measured);    cudaFree(d_image);       cudaFree(d_estimated);
    cudaFree(d_ratio);       cudaFree(d_correction);  cudaFree(d_sensitivity);
    cudaFree(d_subOnes);     cudaFree(d_angleIndices);

    std::cout << "  [GPU] PET OSEM 完成: " << result.elapsedMs << " ms ("
              << params.iterations << " iters × " << numSubsets << " subsets)" << std::endl;
    return result;
}
