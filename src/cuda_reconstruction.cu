// ============================================================================
// CUDA 加速重建 - 核心实现
//
// 包含所有 CUDA kernel 和 host 端管理代码
// ============================================================================

#include "cuda_reconstruction.cuh"

#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <iostream>
#include <algorithm>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ============================================================================
// 错误检查宏
// ============================================================================
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            return {};                                                         \
        }                                                                      \
    } while (0)

#define CUFFT_CHECK(call)                                                      \
    do {                                                                       \
        cufftResult err = (call);                                              \
        if (err != CUFFT_SUCCESS) {                                            \
            std::cerr << "cuFFT Error: " << err                                \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            return {};                                                         \
        }                                                                      \
    } while (0)

// ============================================================================
// CUDA Kernels
// ============================================================================

// ---- CT 正向投影 kernel ----
// 每个线程计算一个 (angle, detector) 的投影值
__global__ void kernelForwardProject(
    const float* __restrict__ image, int imageSize,
    float* __restrict__ sinogram, int numAngles, int numDetectors,
    float angleStart, float angleStep, float detectorSpacing)
{
    int a = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= numAngles || d >= numDetectors) return;

    float halfImg = imageSize / 2.0f;
    float halfDet = numDetectors / 2.0f;
    float theta = angleStart + a * angleStep;
    float cosT = cosf(theta);
    float sinT = sinf(theta);
    float t = (d - halfDet + 0.5f) * detectorSpacing;
    float sMax = halfImg * sqrtf(2.0f);

    float sum = 0.0f;
    for (float s = -sMax; s <= sMax; s += 1.0f) {
        float fx = t * cosT - s * sinT + halfImg - 0.5f;
        float fy = t * sinT + s * cosT + halfImg - 0.5f;

        // 双线性插值
        int x0 = (int)floorf(fx);
        int y0 = (int)floorf(fy);
        if (x0 < 0 || x0 >= imageSize - 1 || y0 < 0 || y0 >= imageSize - 1) continue;
        float dx = fx - x0, dy = fy - y0;
        sum += image[y0 * imageSize + x0] * (1 - dx) * (1 - dy)
             + image[y0 * imageSize + x0 + 1] * dx * (1 - dy)
             + image[(y0 + 1) * imageSize + x0] * (1 - dx) * dy
             + image[(y0 + 1) * imageSize + x0 + 1] * dx * dy;
    }
    sinogram[a * numDetectors + d] = sum;
}

// ---- CT 反投影 kernel ----
// 每个线程计算一个像素的反投影累加
__global__ void kernelBackProject(
    const float* __restrict__ sinogram, int numAngles, int numDetectors,
    float* __restrict__ image, int outputSize,
    float angleStart, float angleStep, float detectorSpacing)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= outputSize || y >= outputSize) return;

    float halfOut = outputSize / 2.0f;
    float halfDet = numDetectors / 2.0f;
    float px = x - halfOut + 0.5f;
    float py = y - halfOut + 0.5f;

    float sum = 0.0f;
    for (int a = 0; a < numAngles; ++a) {
        float theta = angleStart + a * angleStep;
        float t = px * cosf(theta) + py * sinf(theta);
        float detIdx = t / detectorSpacing + halfDet - 0.5f;

        int d0 = (int)floorf(detIdx);
        int d1 = d0 + 1;
        float frac = detIdx - d0;

        float val = 0.0f;
        if (d0 >= 0 && d1 < numDetectors) {
            val = (1.0f - frac) * sinogram[a * numDetectors + d0]
                + frac * sinogram[a * numDetectors + d1];
        } else if (d0 >= 0 && d0 < numDetectors) {
            val = sinogram[a * numDetectors + d0];
        }
        sum += val * angleStep;
    }
    image[y * outputSize + x] = sum;
}

// ---- Ramp 滤波 kernel (频域) ----
// 对 cuFFT 输出的频域数据乘以滤波器
__global__ void kernelApplyFilter(
    cufftComplex* __restrict__ data, int N, int numRows, int filterType)
{
    int row = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows || i >= N) return;

    float halfN = N / 2.0f;
    float freq = fabsf(i - halfN) / halfN;
    if (freq > 1.0f) freq = 1.0f;

    float ramp = freq;
    float filterVal = ramp;

    switch (filterType) {
    case 1: // Shepp-Logan
        filterVal = (freq < 1e-6f) ? 0.0f : ramp * sinf(M_PI * freq / 2.0f) / (M_PI * freq / 2.0f);
        break;
    case 2: // Cosine
        filterVal = ramp * cosf(M_PI * freq / 2.0f);
        break;
    case 3: // Hamming
        filterVal = ramp * (0.54f + 0.46f * cosf(M_PI * freq));
        break;
    case 4: // Hann
        filterVal = ramp * 0.5f * (1.0f + cosf(M_PI * freq));
        break;
    default: // Ram-Lak
        break;
    }

    int idx = row * N + i;
    data[idx].x *= filterVal;
    data[idx].y *= filterVal;
}

// ---- PET MLEM 更新 kernel ----
// x_new = x_old * (H^T * (y / (H*x_old))) / sensitivity
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
    if (image[idx] < 0.0f) image[idx] = 0.0f;
}

// ---- PET 正向投影 kernel (环形) ----
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

// ---- PET 反投影 kernel ----
// 每个线程处理一个像素, 累加所有 LOR 贡献
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

// ---- 逐元素除法 kernel (ratio = measured / estimated) ----
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
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

// ============================================================================
// CT FBP (GPU)
// ============================================================================

CudaReconResult CudaReconstructor::ctFBP(
    const float* sinogramData, const CudaCTParams& params)
{
    CudaReconResult result;
    int numDet = params.numDetectors > 0 ? params.numDetectors
        : static_cast<int>(std::ceil(params.outputSize * std::sqrt(2.0f)));
    int numAng = params.numAngles;
    int outSize = params.outputSize;
    float angleStep = (params.angleEnd - params.angleStart) / numAng;
    float detSpacing = (outSize * std::sqrt(2.0f)) / numDet;

    auto t0 = std::chrono::high_resolution_clock::now();

    // 分配 GPU 内存
    float* d_sino = nullptr;
    float* d_image = nullptr;
    size_t sinoBytes = numAng * numDet * sizeof(float);
    size_t imgBytes = outSize * outSize * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_sino, sinoBytes));
    CUDA_CHECK(cudaMalloc(&d_image, imgBytes));
    CUDA_CHECK(cudaMemcpy(d_sino, sinogramData, sinoBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_image, 0, imgBytes));

    // --- 滤波 (使用 cuFFT) ---
    int N = 1;
    while (N < 2 * numDet) N <<= 1;

    // 准备零填充数据
    std::vector<float> paddedSino(numAng * N, 0.0f);
    for (int a = 0; a < numAng; ++a) {
        for (int d = 0; d < numDet; ++d) {
            paddedSino[a * N + d] = sinogramData[a * numDet + d];
        }
    }

    cufftComplex* d_freq = nullptr;
    float* d_padded = nullptr;
    CUDA_CHECK(cudaMalloc(&d_padded, numAng * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_freq, numAng * N * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMemcpy(d_padded, paddedSino.data(),
        numAng * N * sizeof(float), cudaMemcpyHostToDevice));

    // 批量 1D FFT (每行一个)
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_R2C, numAng));

    // R2C 输出大小 = N/2+1
    int freqSize = N / 2 + 1;
    cufftComplex* d_freqR2C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_freqR2C, numAng * freqSize * sizeof(cufftComplex)));

    CUFFT_CHECK(cufftExecR2C(plan, d_padded, d_freqR2C));

    // 应用滤波器
    {
        dim3 block(256);
        dim3 grid((freqSize + 255) / 256, numAng);
        kernelApplyFilter<<<grid, block>>>(d_freqR2C, freqSize, numAng, params.filterType);
        CUDA_CHECK(cudaGetLastError());
    }

    // IFFT (C2R)
    cufftHandle planInv;
    CUFFT_CHECK(cufftPlan1d(&planInv, N, CUFFT_C2R, numAng));

    float* d_filtered = nullptr;
    CUDA_CHECK(cudaMalloc(&d_filtered, numAng * N * sizeof(float)));
    CUFFT_CHECK(cufftExecC2R(planInv, d_freqR2C, d_filtered));

    // 需要除以 N (cuFFT 不做归一化)
    // 拷贝回 d_sino (只取前 numDet 列)
    {
        std::vector<float> filteredHost(numAng * N);
        CUDA_CHECK(cudaMemcpy(filteredHost.data(), d_filtered,
            numAng * N * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<float> filteredSino(numAng * numDet);
        for (int a = 0; a < numAng; ++a) {
            for (int d = 0; d < numDet; ++d) {
                filteredSino[a * numDet + d] = filteredHost[a * N + d] / N;
            }
        }
        CUDA_CHECK(cudaMemcpy(d_sino, filteredSino.data(), sinoBytes,
            cudaMemcpyHostToDevice));
    }

    cufftDestroy(plan);
    cufftDestroy(planInv);
    cudaFree(d_padded);
    cudaFree(d_freq);
    cudaFree(d_freqR2C);
    cudaFree(d_filtered);

    // --- 反投影 (GPU) ---
    {
        dim3 block(16, 16);
        dim3 grid((outSize + 15) / 16, (outSize + 15) / 16);
        kernelBackProject<<<grid, block>>>(
            d_sino, numAng, numDet, d_image, outSize,
            params.angleStart, angleStep, detSpacing);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 拷贝结果
    result.size = outSize;
    result.image.resize(outSize * outSize);
    CUDA_CHECK(cudaMemcpy(result.image.data(), d_image, imgBytes,
        cudaMemcpyDeviceToHost));

    cudaFree(d_sino);
    cudaFree(d_image);

    auto t1 = std::chrono::high_resolution_clock::now();
    result.elapsedMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
    std::cout << "  [GPU] CT FBP 完成: " << result.elapsedMs << " ms" << std::endl;
    return result;
}

// ============================================================================
// CT SIRT (GPU)
// ============================================================================

CudaReconResult CudaReconstructor::ctSIRT(
    const float* sinogramData, const CudaCTParams& params)
{
    CudaReconResult result;
    int numDet = params.numDetectors > 0 ? params.numDetectors
        : static_cast<int>(std::ceil(params.outputSize * std::sqrt(2.0f)));
    int numAng = params.numAngles;
    int outSize = params.outputSize;
    float angleStep = (params.angleEnd - params.angleStart) / numAng;
    float detSpacing = (outSize * std::sqrt(2.0f)) / numDet;

    auto t0 = std::chrono::high_resolution_clock::now();

    size_t sinoBytes = numAng * numDet * sizeof(float);
    size_t imgBytes = outSize * outSize * sizeof(float);

    float *d_measured, *d_image, *d_simSino, *d_residual, *d_correction, *d_weight;
    CUDA_CHECK(cudaMalloc(&d_measured, sinoBytes));
    CUDA_CHECK(cudaMalloc(&d_image, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_simSino, sinoBytes));
    CUDA_CHECK(cudaMalloc(&d_residual, sinoBytes));
    CUDA_CHECK(cudaMalloc(&d_correction, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_weight, imgBytes));

    CUDA_CHECK(cudaMemcpy(d_measured, sinogramData, sinoBytes, cudaMemcpyHostToDevice));

    // 初始化图像为 1.0
    std::vector<float> initImg(outSize * outSize, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_image, initImg.data(), imgBytes, cudaMemcpyHostToDevice));

    dim3 sinoBlock(16, 16);
    dim3 sinoGrid((numDet + 15) / 16, (numAng + 15) / 16);
    dim3 imgBlock(16, 16);
    dim3 imgGrid((outSize + 15) / 16, (outSize + 15) / 16);

    for (int iter = 0; iter < params.iterations; ++iter) {
        // 1. 正向投影
        CUDA_CHECK(cudaMemset(d_simSino, 0, sinoBytes));
        kernelForwardProject<<<sinoGrid, sinoBlock>>>(
            d_image, outSize, d_simSino, numAng, numDet,
            params.angleStart, angleStep, detSpacing);
        CUDA_CHECK(cudaGetLastError());

        // 2. 残差 = measured - simulated (在 GPU 上用简单 kernel)
        {
            int N = numAng * numDet;
            int threads = 256;
            int blocks = (N + threads - 1) / threads;
            // 用 elementDiv kernel 的变体来做减法?
            // 简化: 拷贝回 host 做
            std::vector<float> sim(N), res(N);
            CUDA_CHECK(cudaMemcpy(sim.data(), d_simSino, sinoBytes, cudaMemcpyDeviceToHost));
            const float* meas = sinogramData;
            float totalErr = 0;
            for (int i = 0; i < N; ++i) {
                res[i] = meas[i] - sim[i];
                totalErr += res[i] * res[i];
            }
            CUDA_CHECK(cudaMemcpy(d_residual, res.data(), sinoBytes, cudaMemcpyHostToDevice));
            result.convergenceError = totalErr / N;
        }

        // 3. 反投影残差
        CUDA_CHECK(cudaMemset(d_correction, 0, imgBytes));
        kernelBackProject<<<imgGrid, imgBlock>>>(
            d_residual, numAng, numDet, d_correction, outSize,
            params.angleStart, angleStep, detSpacing);
        CUDA_CHECK(cudaGetLastError());

        // 4. 更新 (host 端做简单的加权更新)
        {
            std::vector<float> img(outSize * outSize), corr(outSize * outSize);
            CUDA_CHECK(cudaMemcpy(img.data(), d_image, imgBytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(corr.data(), d_correction, imgBytes, cudaMemcpyDeviceToHost));
            for (int i = 0; i < outSize * outSize; ++i) {
                img[i] += params.relaxation * corr[i] / (numAng * numDet);
                if (img[i] < 0) img[i] = 0;
            }
            CUDA_CHECK(cudaMemcpy(d_image, img.data(), imgBytes, cudaMemcpyHostToDevice));
        }

        if ((iter + 1) % 10 == 0 || iter == 0) {
            std::cout << "  [GPU] SIRT 迭代 " << (iter + 1)
                      << "/" << params.iterations
                      << "  MSE=" << result.convergenceError << std::endl;
        }
    }

    result.size = outSize;
    result.image.resize(outSize * outSize);
    CUDA_CHECK(cudaMemcpy(result.image.data(), d_image, imgBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_measured); cudaFree(d_image); cudaFree(d_simSino);
    cudaFree(d_residual); cudaFree(d_correction); cudaFree(d_weight);

    auto t1 = std::chrono::high_resolution_clock::now();
    result.elapsedMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
    std::cout << "  [GPU] CT SIRT 完成: " << result.elapsedMs << " ms" << std::endl;
    return result;
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

    auto t0 = std::chrono::high_resolution_clock::now();

    float *d_measured, *d_image, *d_estimated, *d_ratio, *d_correction, *d_sensitivity;
    CUDA_CHECK(cudaMalloc(&d_measured, sinoBytes));
    CUDA_CHECK(cudaMalloc(&d_image, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_estimated, sinoBytes));
    CUDA_CHECK(cudaMalloc(&d_ratio, sinoBytes));
    CUDA_CHECK(cudaMalloc(&d_correction, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_sensitivity, imgBytes));

    CUDA_CHECK(cudaMemcpy(d_measured, sinogramData, sinoBytes, cudaMemcpyHostToDevice));

    // 初始图像 = 1.0
    std::vector<float> initImg(imgTotal, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_image, initImg.data(), imgBytes, cudaMemcpyHostToDevice));

    dim3 sinoBlock(16, 16);
    dim3 sinoGrid((numBins + 15) / 16, (numAng + 15) / 16);
    dim3 imgBlock(16, 16);
    dim3 imgGrid((outSize + 15) / 16, (outSize + 15) / 16);
    int linBlock = 256;

    // 预计算灵敏度图: 反投影全 1 正弦图
    {
        std::vector<float> ones(sinoTotal, 1.0f);
        float* d_ones;
        CUDA_CHECK(cudaMalloc(&d_ones, sinoBytes));
        CUDA_CHECK(cudaMemcpy(d_ones, ones.data(), sinoBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_sensitivity, 0, imgBytes));

        kernelPETBackProject<<<imgGrid, imgBlock>>>(
            d_ones, numAng, numBins, d_sensitivity, outSize,
            binSpacing, nullptr, 0);
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
            kernelElementDiv<<<blocks, linBlock>>>(
                d_measured, d_estimated, d_ratio, sinoTotal);
            CUDA_CHECK(cudaGetLastError());
        }

        // 3. 反投影 ratio
        CUDA_CHECK(cudaMemset(d_correction, 0, imgBytes));
        kernelPETBackProject<<<imgGrid, imgBlock>>>(
            d_ratio, numAng, numBins, d_correction, outSize,
            binSpacing, nullptr, 0);
        CUDA_CHECK(cudaGetLastError());

        // 4. MLEM 更新: image *= correction / sensitivity
        {
            int blocks = (imgTotal + linBlock - 1) / linBlock;
            kernelMLEMUpdate<<<blocks, linBlock>>>(
                d_image, outSize, d_correction, d_sensitivity);
            CUDA_CHECK(cudaGetLastError());
        }

        if ((iter + 1) % 5 == 0 || iter == 0) {
            std::cout << "  [GPU] PET MLEM 迭代 " << (iter + 1)
                      << "/" << params.iterations << std::endl;
        }
    }

    result.size = outSize;
    result.image.resize(imgTotal);
    CUDA_CHECK(cudaMemcpy(result.image.data(), d_image, imgBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_measured); cudaFree(d_image); cudaFree(d_estimated);
    cudaFree(d_ratio); cudaFree(d_correction); cudaFree(d_sensitivity);

    auto t1 = std::chrono::high_resolution_clock::now();
    result.elapsedMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
    result.iterationsRun = params.iterations;
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
    int numAng = params.numAngles;
    int numBins = params.numRadialBins > 0 ? params.numRadialBins
        : static_cast<int>(std::ceil(params.outputSize * std::sqrt(2.0f)));
    int outSize = params.outputSize;
    float binSpacing = (outSize * std::sqrt(2.0f)) / numBins;
    size_t sinoBytes = numAng * numBins * sizeof(float);
    size_t imgBytes = outSize * outSize * sizeof(float);
    int imgTotal = outSize * outSize;

    auto t0 = std::chrono::high_resolution_clock::now();

    float *d_measured, *d_image, *d_estimated, *d_ratio, *d_correction, *d_sensitivity;
    int* d_angleIndices;
    CUDA_CHECK(cudaMalloc(&d_measured, sinoBytes));
    CUDA_CHECK(cudaMalloc(&d_image, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_estimated, sinoBytes));
    CUDA_CHECK(cudaMalloc(&d_ratio, sinoBytes));
    CUDA_CHECK(cudaMalloc(&d_correction, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_sensitivity, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_angleIndices, numAng * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_measured, sinogramData, sinoBytes, cudaMemcpyHostToDevice));

    std::vector<float> initImg(imgTotal, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_image, initImg.data(), imgBytes, cudaMemcpyHostToDevice));

    dim3 sinoBlock(16, 16);
    dim3 imgBlock(16, 16);
    dim3 imgGrid((outSize + 15) / 16, (outSize + 15) / 16);
    int linBlock = 256;

    int numSubsets = params.numSubsets;

    for (int iter = 0; iter < params.iterations; ++iter) {
        for (int sub = 0; sub < numSubsets; ++sub) {
            // 子集角度
            std::vector<int> subAngles;
            for (int a = sub; a < numAng; a += numSubsets) subAngles.push_back(a);
            int nSubAng = static_cast<int>(subAngles.size());
            CUDA_CHECK(cudaMemcpy(d_angleIndices, subAngles.data(),
                nSubAng * sizeof(int), cudaMemcpyHostToDevice));

            // 子集灵敏度
            {
                std::vector<float> ones(numAng * numBins, 0.0f);
                for (int a : subAngles)
                    for (int r = 0; r < numBins; ++r)
                        ones[a * numBins + r] = 1.0f;
                float* d_ones;
                CUDA_CHECK(cudaMalloc(&d_ones, sinoBytes));
                CUDA_CHECK(cudaMemcpy(d_ones, ones.data(), sinoBytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(d_sensitivity, 0, imgBytes));
                kernelPETBackProject<<<imgGrid, imgBlock>>>(
                    d_ones, numAng, numBins, d_sensitivity, outSize,
                    binSpacing, d_angleIndices, nSubAng);
                CUDA_CHECK(cudaGetLastError());
                cudaFree(d_ones);
            }

            // 正向投影 (完整, 但只用子集角度做比较)
            CUDA_CHECK(cudaMemset(d_estimated, 0, sinoBytes));
            {
                dim3 grid((numBins + 15) / 16, (numAng + 15) / 16);
                kernelPETForwardProject<<<grid, sinoBlock>>>(
                    d_image, outSize, d_estimated, numAng, numBins, binSpacing);
                CUDA_CHECK(cudaGetLastError());
            }

            // ratio
            {
                int N = numAng * numBins;
                int blocks = (N + linBlock - 1) / linBlock;
                kernelElementDiv<<<blocks, linBlock>>>(
                    d_measured, d_estimated, d_ratio, N);
                CUDA_CHECK(cudaGetLastError());
            }

            // 反投影 ratio (仅子集角度)
            CUDA_CHECK(cudaMemset(d_correction, 0, imgBytes));
            kernelPETBackProject<<<imgGrid, imgBlock>>>(
                d_ratio, numAng, numBins, d_correction, outSize,
                binSpacing, d_angleIndices, nSubAng);
            CUDA_CHECK(cudaGetLastError());

            // MLEM 更新
            {
                int blocks = (imgTotal + linBlock - 1) / linBlock;
                kernelMLEMUpdate<<<blocks, linBlock>>>(
                    d_image, outSize, d_correction, d_sensitivity);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        std::cout << "  [GPU] PET OSEM 迭代 " << (iter + 1)
                  << "/" << params.iterations
                  << " (" << numSubsets << " subsets)" << std::endl;
    }

    result.size = outSize;
    result.image.resize(imgTotal);
    CUDA_CHECK(cudaMemcpy(result.image.data(), d_image, imgBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_measured); cudaFree(d_image); cudaFree(d_estimated);
    cudaFree(d_ratio); cudaFree(d_correction); cudaFree(d_sensitivity);
    cudaFree(d_angleIndices);

    auto t1 = std::chrono::high_resolution_clock::now();
    result.elapsedMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
    result.iterationsRun = params.iterations;
    std::cout << "  [GPU] PET OSEM 完成: " << result.elapsedMs << " ms" << std::endl;
    return result;
}

// ============================================================================
// 工具函数包装
// ============================================================================

void CudaReconstructor::gpuForwardProject(
    const float* image, int imageSize,
    float* sinogram, int numAngles, int numDetectors,
    float angleStart, float angleEnd)
{
    float angleStep = (angleEnd - angleStart) / numAngles;
    float detSpacing = (imageSize * std::sqrt(2.0f)) / numDetectors;
    size_t imgBytes = imageSize * imageSize * sizeof(float);
    size_t sinoBytes = numAngles * numDetectors * sizeof(float);

    float *d_img, *d_sino;
    cudaMalloc(&d_img, imgBytes);
    cudaMalloc(&d_sino, sinoBytes);
    cudaMemcpy(d_img, image, imgBytes, cudaMemcpyHostToDevice);
    cudaMemset(d_sino, 0, sinoBytes);

    dim3 block(16, 16);
    dim3 grid((numDetectors + 15) / 16, (numAngles + 15) / 16);
    kernelForwardProject<<<grid, block>>>(
        d_img, imageSize, d_sino, numAngles, numDetectors,
        angleStart, angleStep, detSpacing);
    cudaDeviceSynchronize();

    cudaMemcpy(sinogram, d_sino, sinoBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_img); cudaFree(d_sino);
}

void CudaReconstructor::gpuBackProject(
    const float* sinogram, int numAngles, int numDetectors,
    float* image, int imageSize,
    float angleStart, float angleEnd)
{
    float angleStep = (angleEnd - angleStart) / numAngles;
    float detSpacing = (imageSize * std::sqrt(2.0f)) / numDetectors;
    size_t imgBytes = imageSize * imageSize * sizeof(float);
    size_t sinoBytes = numAngles * numDetectors * sizeof(float);

    float *d_sino, *d_img;
    cudaMalloc(&d_sino, sinoBytes);
    cudaMalloc(&d_img, imgBytes);
    cudaMemcpy(d_sino, sinogram, sinoBytes, cudaMemcpyHostToDevice);
    cudaMemset(d_img, 0, imgBytes);

    dim3 block(16, 16);
    dim3 grid((imageSize + 15) / 16, (imageSize + 15) / 16);
    kernelBackProject<<<grid, block>>>(
        d_sino, numAngles, numDetectors, d_img, imageSize,
        angleStart, angleStep, detSpacing);
    cudaDeviceSynchronize();

    cudaMemcpy(image, d_img, imgBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_sino); cudaFree(d_img);
}

void CudaReconstructor::gpuApplyRampFilter(
    float* sinogram, int numAngles, int numDetectors, int filterType)
{
    // 简化: host 端 cuFFT 流程
    CudaCTParams p;
    p.numAngles = numAngles;
    p.numDetectors = numDetectors;
    p.filterType = filterType;
    // 使用 ctFBP 内部的滤波逻辑即可, 此处仅作占位提示
    std::cout << "[GPU] gpuApplyRampFilter: 请直接使用 ctFBP 内置滤波" << std::endl;
}
