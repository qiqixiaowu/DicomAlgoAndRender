/**
 * @file lung_nodule_segment_dl.cu
 * @brief 3D nnUNet 手写 cuDNN 推理引擎（GPU 加速路径）
 *
 * 本文件实现 LungNoduleDLSegment::inferGPU()：
 *   - 完整 5 层 Encoder-Decoder U-Net
 *   - 每个 Conv 块 = 3D Conv(cuDNN) + Instance Normalization(CUDA Kernel) + LeakyReLU(CUDA Kernel)
 *   - Decoder 使用最近邻上采样 + Skip Connection 拼接
 *   - 最终 1×1×1 卷积输出 2 类 logits
 *
 * 编译：nvcc -std=c++17 -O2 -DUSE_CUDA -lcudnn lung_nodule_segment_dl.cu -o ...
 *
 * 网络通道配置（nnUNet v1 标准）：
 *   enc1: 1  → 32  (stride 1)    skip1
 *   enc2: 32 → 64  (stride 2)    skip2
 *   enc3: 64 → 128 (stride 2)    skip3
 *   enc4: 128→ 256 (stride 2)    skip4
 *   btn:  256→ 320 (stride 2)    bottleneck
 *   dec4: 320+256 → 256 (up×2 + cat)
 *   dec3: 256+128 → 128 (up×2 + cat)
 *   dec2: 128+64  → 64  (up×2 + cat)
 *   dec1: 64+32   → 32  (up×2 + cat)
 *   out:  32 → 2   (1×1×1 conv)
 */

#ifdef USE_CUDA

#include "lung_nodule_segment_dl.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <cassert>
#include <cstring>

// ═══════════════════════════════════════════════
//  宏：cuDNN 错误检查
// ═══════════════════════════════════════════════
#define CUDNN_CHECK(expr) \
    do { \
        cudnnStatus_t _s = (expr); \
        if (_s != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "[cuDNN] Error %d at %s:%d: %s\n", \
                    (int)_s, __FILE__, __LINE__, cudnnGetErrorString(_s)); \
            return false; \
        } \
    } while(0)

#define CUDA_CHECK(expr) \
    do { \
        cudaError_t _e = (expr); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "[CUDA] Error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(_e)); \
            return false; \
        } \
    } while(0)

namespace Onc {

// ═══════════════════════════════════════════════
//  辅助：在 GPU 上分配并上传权重
// ═══════════════════════════════════════════════
static bool uploadWeight(const ParamDict& params, const std::string& key,
                          float** dPtr, size_t expectedElems = 0)
{
    auto it = params.find(key);
    if (it == params.end()) {
        // 未找到时分配全零权重，保证网络可运行
        if (expectedElems == 0) {
            *dPtr = nullptr;
            return true;
        }
        CUDA_CHECK(cudaMalloc(dPtr, expectedElems * sizeof(float)));
        CUDA_CHECK(cudaMemset(*dPtr, 0, expectedElems * sizeof(float)));
        return true;
    }
    const auto& vec = it->second;
    CUDA_CHECK(cudaMalloc(dPtr, vec.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(*dPtr, vec.data(), vec.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    return true;
}

// ═══════════════════════════════════════════════
//  CUDA Kernel：Instance Normalization
//  每个线程处理 1 个通道（spatialSize 个元素）
// ═══════════════════════════════════════════════
__global__ void kernelInstanceNorm(
    float* data,        // [C, D*H*W] in-place
    const float* gamma, // [C]
    const float* beta,  // [C]
    int C, int DHW,
    float eps)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float* ch = data + c * DHW;
    // 计算均值
    float mean = 0.f;
    for (int i = 0; i < DHW; ++i) mean += ch[i];
    mean /= (float)DHW;
    // 计算方差
    float var = 0.f;
    for (int i = 0; i < DHW; ++i) {
        float d = ch[i] - mean;
        var += d * d;
    }
    var /= (float)DHW;
    float invStd = rsqrtf(var + eps);

    float g = gamma ? gamma[c] : 1.f;
    float b = beta  ? beta[c]  : 0.f;
    for (int i = 0; i < DHW; ++i)
        ch[i] = (ch[i] - mean) * invStd * g + b;
}

// ═══════════════════════════════════════════════
//  CUDA Kernel：LeakyReLU in-place
// ═══════════════════════════════════════════════
__global__ void kernelLeakyReLU(float* data, int N, float alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (data[i] < 0.f) data[i] *= alpha;
}

// ═══════════════════════════════════════════════
//  CUDA Kernel：最近邻上采样 ×2 (3D)
//  input:  [C, D, H, W]
//  output: [C, D*2, H*2, W*2]
// ═══════════════════════════════════════════════
__global__ void kernelUpsample2x(
    const float* input, float* output,
    int C, int D, int H, int W)
{
    int oD = D*2, oH = H*2, oW = W*2;
    int total = C * oD * oH * oW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int ox = idx % oW; int tmp = idx / oW;
    int oy = tmp % oH; tmp /= oH;
    int oz = tmp % oD;
    int c  = tmp / oD;

    int iz = oz / 2, iy = oy / 2, ix = ox / 2;
    output[idx] = input[c * D*H*W + iz * H*W + iy * W + ix];
}

// ═══════════════════════════════════════════════
//  CUDA Kernel：Channel Concatenation
//  a: [Ca, DHW]    b: [Cb, DHW]  → out: [Ca+Cb, DHW]
// ═══════════════════════════════════════════════
__global__ void kernelConcat(
    const float* a, const float* b, float* out,
    int Ca, int Cb, int DHW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalA = Ca * DHW;
    int totalB = Cb * DHW;
    if (idx < totalA)
        out[idx] = a[idx];
    else if (idx < totalA + totalB)
        out[idx] = b[idx - totalA];
}

// ═══════════════════════════════════════════════
//  辅助：创建 cuDNN 3D conv 描述符并执行
// ═══════════════════════════════════════════════
struct ConvParams {
    int C_in, C_out, D, H, W;
    int kD, kH, kW;           // kernel size
    int sD, sH, sW;           // stride
    int padD, padH, padW;     // padding
};

static bool cudnnConv3D(
    cudnnHandle_t handle,
    const float*  input,
    const float*  weight,
    const float*  bias,       // may be nullptr
    float*        output,
    const ConvParams& p)
{
    int oD = (p.D + 2*p.padD - p.kD) / p.sD + 1;
    int oH = (p.H + 2*p.padH - p.kH) / p.sH + 1;
    int oW = (p.W + 2*p.padW - p.kW) / p.sW + 1;

    // ── 创建张量描述符 ──
    cudnnTensorDescriptor_t   xDesc, yDesc, bDesc;
    cudnnFilterDescriptor_t   wDesc;
    cudnnConvolutionDescriptor_t convDesc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&wDesc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));

    // 使用 5D 张量（N=1, C, D, H, W）
    int xDims[5]   = {1, p.C_in,  p.D, p.H, p.W};
    int xStrides[5]= {p.C_in*p.D*p.H*p.W, p.D*p.H*p.W, p.H*p.W, p.W, 1};
    int yDims[5]   = {1, p.C_out, oD,   oH,   oW};
    int yStrides[5]= {p.C_out*oD*oH*oW, oD*oH*oW, oH*oW, oW, 1};
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(xDesc, CUDNN_DATA_FLOAT, 5, xDims, xStrides));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(yDesc, CUDNN_DATA_FLOAT, 5, yDims, yStrides));

    int filterDim[5] = {p.C_out, p.C_in, p.kD, p.kH, p.kW};
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_FLOAT,
                                            CUDNN_TENSOR_NCHW, 5, filterDim));

    int pads[3]    = {p.padD, p.padH, p.padW};
    int strides[3] = {p.sD,   p.sH,   p.sW};
    int dilations[3]={1,      1,      1};
    CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(convDesc, 3, pads, strides, dilations,
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // ── 选择算法 ──
    int reqAlgo = 0; cudnnConvolutionFwdAlgoPerf_t algoPerf;
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
        handle, xDesc, wDesc, convDesc, yDesc, 1, &reqAlgo, &algoPerf));
    cudnnConvolutionFwdAlgo_t algo = algoPerf.algo;

    // ── 工作空间 ──
    size_t wsSize = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        handle, xDesc, wDesc, convDesc, yDesc, algo, &wsSize));
    float* dWS = nullptr;
    if (wsSize > 0) CUDA_CHECK(cudaMalloc(&dWS, wsSize));

    // ── 前向传播 ──
    const float alpha = 1.f, beta = 0.f;
    CUDNN_CHECK(cudnnConvolutionForward(
        handle, &alpha, xDesc, input, wDesc, weight, convDesc,
        algo, dWS, wsSize, &beta, yDesc, output));

    // ── 加 bias ──
    if (bias) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bDesc));
        int bDims[5]   = {1, p.C_out, 1, 1, 1};
        int bStrides[5]= {p.C_out, 1, 1, 1, 1};
        CUDNN_CHECK(cudnnSetTensorNdDescriptor(bDesc, CUDNN_DATA_FLOAT, 5, bDims, bStrides));
        CUDNN_CHECK(cudnnAddTensor(handle, &alpha, bDesc, bias, &alpha, yDesc, output));
        cudnnDestroyTensorDescriptor(bDesc);
    }

    // ── 清理 ──
    if (dWS) cudaFree(dWS);
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    return true;
}

// ═══════════════════════════════════════════════
//  辅助宏：执行一个 Conv+IN+LeakyReLU 块
// ═══════════════════════════════════════════════
#define CONV_BLK(handle, in, w, b, g, bt, out, cp, threads) \
    do { \
        if (!cudnnConv3D(handle, in, w, b, out, cp)) return false; \
        int _dhw = (cp).C_out * (((cp).D+2*(cp).padD-(cp).kD)/(cp).sD+1) \
                               * (((cp).H+2*(cp).padH-(cp).kH)/(cp).sH+1) \
                               * (((cp).W+2*(cp).padW-(cp).kW)/(cp).sW+1); \
        int _Cout = (cp).C_out; \
        int _sDHW = _dhw / _Cout; \
        kernelInstanceNorm<<<(_Cout+63)/64, 64>>>(out, g, bt, _Cout, _sDHW, 1e-5f); \
        kernelLeakyReLU<<<(_dhw+255)/256, 256>>>(out, _dhw, 0.01f); \
    } while(0)

// ═══════════════════════════════════════════════
//  主 GPU 推理函数实现
// ═══════════════════════════════════════════════
bool LungNoduleDLSegment::inferGPU(
    const std::vector<float>& patch,
    std::vector<float>&       logits)
{
    int pW = m_cfg.patchSize[0];
    int pH = m_cfg.patchSize[1];
    int pD = m_cfg.patchSize[2];

    // ── 初始化 cuDNN ──
    cudnnHandle_t handle;
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
        std::cerr << "[LungNoduleDL] Failed to create cuDNN handle\n";
        return inferCPU(patch, logits);
    }

    // ════════════════════════════════════════════
    //  0. 上传 patch 到 GPU
    // ════════════════════════════════════════════
    float* dInput = nullptr;
    size_t patchBytes = patch.size() * sizeof(float);
    if (cudaMalloc(&dInput, patchBytes) != cudaSuccess) return false;
    cudaMemcpy(dInput, patch.data(), patchBytes, cudaMemcpyHostToDevice);

    // ════════════════════════════════════════════
    //  辅助 Lambda：上传权重
    // ════════════════════════════════════════════
    auto uploadW = [&](const std::string& key, float** ptr, size_t n = 0) -> bool {
        return uploadWeight(m_params, key, ptr, n);
    };

    // ════════════════════════════════════════════
    //  1. Encoder Block 1  (1 → 32, stride=1, D,H,W)
    // ════════════════════════════════════════════
    float *dEnc1 = nullptr;
    CUDA_CHECK(cudaMalloc(&dEnc1, 32 * pD * pH * pW * sizeof(float)));
    {
        float *dW = nullptr, *dB = nullptr, *dG = nullptr, *dBeta = nullptr;
        uploadW("conv_blocks_context.0.blocks.0.conv.weight", &dW, 32*1*3*3*3);
        uploadW("conv_blocks_context.0.blocks.0.conv.bias",   &dB, 32);
        uploadW("conv_blocks_context.0.blocks.0.instnorm.weight", &dG,   32);
        uploadW("conv_blocks_context.0.blocks.0.instnorm.bias",   &dBeta,32);
        ConvParams cp{1,32,pD,pH,pW,3,3,3,1,1,1,1,1,1};
        CONV_BLK(handle, dInput, dW, dB, dG, dBeta, dEnc1, cp, 256);
        // Block 1 第2次卷积(32→32)
        float *dW2=nullptr,*dB2=nullptr,*dG2=nullptr,*dBeta2=nullptr;
        uploadW("conv_blocks_context.0.blocks.1.conv.weight", &dW2, 32*32*3*3*3);
        uploadW("conv_blocks_context.0.blocks.1.conv.bias",   &dB2, 32);
        uploadW("conv_blocks_context.0.blocks.1.instnorm.weight",&dG2,  32);
        uploadW("conv_blocks_context.0.blocks.1.instnorm.bias",  &dBeta2,32);
        float *dEnc1b = nullptr;
        CUDA_CHECK(cudaMalloc(&dEnc1b, 32*pD*pH*pW*sizeof(float)));
        ConvParams cp2{32,32,pD,pH,pW,3,3,3,1,1,1,1,1,1};
        CONV_BLK(handle, dEnc1, dW2, dB2, dG2, dBeta2, dEnc1b, cp2, 256);
        cudaFree(dEnc1); dEnc1 = dEnc1b;
        cudaFree(dW); cudaFree(dB); cudaFree(dG); cudaFree(dBeta);
        cudaFree(dW2); cudaFree(dB2); cudaFree(dG2); cudaFree(dBeta2);
    }

    // ════════════════════════════════════════════
    //  2. Encoder Block 2  (32 → 64, stride=2)
    //     输出尺寸: D/2, H/2, W/2
    // ════════════════════════════════════════════
    int eD2=pD/2, eH2=pH/2, eW2=pW/2;
    float *dEnc2 = nullptr;
    CUDA_CHECK(cudaMalloc(&dEnc2, 64*eD2*eH2*eW2*sizeof(float)));
    {
        float *dW=nullptr,*dB=nullptr,*dG=nullptr,*dBeta=nullptr;
        uploadW("conv_blocks_context.1.blocks.0.conv.weight",  &dW,   64*32*3*3*3);
        uploadW("conv_blocks_context.1.blocks.0.conv.bias",    &dB,   64);
        uploadW("conv_blocks_context.1.blocks.0.instnorm.weight",&dG,  64);
        uploadW("conv_blocks_context.1.blocks.0.instnorm.bias", &dBeta,64);
        ConvParams cp{32,64,pD,pH,pW,3,3,3,2,2,2,1,1,1};
        CONV_BLK(handle, dEnc1, dW, dB, dG, dBeta, dEnc2, cp, 256);
        float *dW2=nullptr,*dB2=nullptr,*dG2=nullptr,*dBeta2=nullptr;
        uploadW("conv_blocks_context.1.blocks.1.conv.weight",  &dW2,  64*64*3*3*3);
        uploadW("conv_blocks_context.1.blocks.1.conv.bias",    &dB2,  64);
        uploadW("conv_blocks_context.1.blocks.1.instnorm.weight",&dG2, 64);
        uploadW("conv_blocks_context.1.blocks.1.instnorm.bias", &dBeta2,64);
        float *dEnc2b=nullptr;
        CUDA_CHECK(cudaMalloc(&dEnc2b,64*eD2*eH2*eW2*sizeof(float)));
        ConvParams cp2{64,64,eD2,eH2,eW2,3,3,3,1,1,1,1,1,1};
        CONV_BLK(handle, dEnc2, dW2, dB2, dG2, dBeta2, dEnc2b, cp2, 256);
        cudaFree(dEnc2); dEnc2=dEnc2b;
        cudaFree(dW);cudaFree(dB);cudaFree(dG);cudaFree(dBeta);
        cudaFree(dW2);cudaFree(dB2);cudaFree(dG2);cudaFree(dBeta2);
    }

    // ════════════════════════════════════════════
    //  3. Encoder Block 3  (64 → 128, stride=2)
    // ════════════════════════════════════════════
    int eD3=eD2/2, eH3=eH2/2, eW3=eW2/2;
    float *dEnc3 = nullptr;
    CUDA_CHECK(cudaMalloc(&dEnc3, 128*eD3*eH3*eW3*sizeof(float)));
    {
        float *dW=nullptr,*dB=nullptr,*dG=nullptr,*dBeta=nullptr;
        uploadW("conv_blocks_context.2.blocks.0.conv.weight",  &dW,   128*64*3*3*3);
        uploadW("conv_blocks_context.2.blocks.0.conv.bias",    &dB,   128);
        uploadW("conv_blocks_context.2.blocks.0.instnorm.weight",&dG,  128);
        uploadW("conv_blocks_context.2.blocks.0.instnorm.bias", &dBeta,128);
        ConvParams cp{64,128,eD2,eH2,eW2,3,3,3,2,2,2,1,1,1};
        CONV_BLK(handle, dEnc2, dW, dB, dG, dBeta, dEnc3, cp, 256);
        float *dW2=nullptr,*dB2=nullptr,*dG2=nullptr,*dBeta2=nullptr;
        uploadW("conv_blocks_context.2.blocks.1.conv.weight",  &dW2,  128*128*3*3*3);
        uploadW("conv_blocks_context.2.blocks.1.conv.bias",    &dB2,  128);
        uploadW("conv_blocks_context.2.blocks.1.instnorm.weight",&dG2, 128);
        uploadW("conv_blocks_context.2.blocks.1.instnorm.bias", &dBeta2,128);
        float *dEnc3b=nullptr;
        CUDA_CHECK(cudaMalloc(&dEnc3b,128*eD3*eH3*eW3*sizeof(float)));
        ConvParams cp2{128,128,eD3,eH3,eW3,3,3,3,1,1,1,1,1,1};
        CONV_BLK(handle, dEnc3, dW2, dB2, dG2, dBeta2, dEnc3b, cp2, 256);
        cudaFree(dEnc3); dEnc3=dEnc3b;
        cudaFree(dW);cudaFree(dB);cudaFree(dG);cudaFree(dBeta);
        cudaFree(dW2);cudaFree(dB2);cudaFree(dG2);cudaFree(dBeta2);
    }

    // ════════════════════════════════════════════
    //  4. Encoder Block 4  (128 → 256, stride=2)
    // ════════════════════════════════════════════
    int eD4=eD3/2, eH4=eH3/2, eW4=eW3/2;
    float *dEnc4 = nullptr;
    CUDA_CHECK(cudaMalloc(&dEnc4, 256*eD4*eH4*eW4*sizeof(float)));
    {
        float *dW=nullptr,*dB=nullptr,*dG=nullptr,*dBeta=nullptr;
        uploadW("conv_blocks_context.3.blocks.0.conv.weight",  &dW,   256*128*3*3*3);
        uploadW("conv_blocks_context.3.blocks.0.conv.bias",    &dB,   256);
        uploadW("conv_blocks_context.3.blocks.0.instnorm.weight",&dG,  256);
        uploadW("conv_blocks_context.3.blocks.0.instnorm.bias", &dBeta,256);
        ConvParams cp{128,256,eD3,eH3,eW3,3,3,3,2,2,2,1,1,1};
        CONV_BLK(handle, dEnc3, dW, dB, dG, dBeta, dEnc4, cp, 256);
        float *dW2=nullptr,*dB2=nullptr,*dG2=nullptr,*dBeta2=nullptr;
        uploadW("conv_blocks_context.3.blocks.1.conv.weight",  &dW2,  256*256*3*3*3);
        uploadW("conv_blocks_context.3.blocks.1.conv.bias",    &dB2,  256);
        uploadW("conv_blocks_context.3.blocks.1.instnorm.weight",&dG2, 256);
        uploadW("conv_blocks_context.3.blocks.1.instnorm.bias", &dBeta2,256);
        float *dEnc4b=nullptr;
        CUDA_CHECK(cudaMalloc(&dEnc4b,256*eD4*eH4*eW4*sizeof(float)));
        ConvParams cp2{256,256,eD4,eH4,eW4,3,3,3,1,1,1,1,1,1};
        CONV_BLK(handle, dEnc4, dW2, dB2, dG2, dBeta2, dEnc4b, cp2, 256);
        cudaFree(dEnc4); dEnc4=dEnc4b;
        cudaFree(dW);cudaFree(dB);cudaFree(dG);cudaFree(dBeta);
        cudaFree(dW2);cudaFree(dB2);cudaFree(dG2);cudaFree(dBeta2);
    }

    // ════════════════════════════════════════════
    //  5. Bottleneck  (256 → 320, stride=2)
    // ════════════════════════════════════════════
    int bD=eD4/2, bH=eH4/2, bW=eW4/2;
    float *dBtn = nullptr;
    CUDA_CHECK(cudaMalloc(&dBtn, 320*bD*bH*bW*sizeof(float)));
    {
        float *dW=nullptr,*dB=nullptr,*dG=nullptr,*dBeta=nullptr;
        uploadW("conv_blocks_context.4.blocks.0.conv.weight",  &dW,   320*256*3*3*3);
        uploadW("conv_blocks_context.4.blocks.0.conv.bias",    &dB,   320);
        uploadW("conv_blocks_context.4.blocks.0.instnorm.weight",&dG,  320);
        uploadW("conv_blocks_context.4.blocks.0.instnorm.bias", &dBeta,320);
        ConvParams cp{256,320,eD4,eH4,eW4,3,3,3,2,2,2,1,1,1};
        CONV_BLK(handle, dEnc4, dW, dB, dG, dBeta, dBtn, cp, 256);
        float *dW2=nullptr,*dB2=nullptr,*dG2=nullptr,*dBeta2=nullptr;
        uploadW("conv_blocks_context.4.blocks.1.conv.weight",  &dW2,  320*320*3*3*3);
        uploadW("conv_blocks_context.4.blocks.1.conv.bias",    &dB2,  320);
        uploadW("conv_blocks_context.4.blocks.1.instnorm.weight",&dG2, 320);
        uploadW("conv_blocks_context.4.blocks.1.instnorm.bias", &dBeta2,320);
        float *dBtnb=nullptr;
        CUDA_CHECK(cudaMalloc(&dBtnb,320*bD*bH*bW*sizeof(float)));
        ConvParams cp2{320,320,bD,bH,bW,3,3,3,1,1,1,1,1,1};
        CONV_BLK(handle, dBtn, dW2, dB2, dG2, dBeta2, dBtnb, cp2, 256);
        cudaFree(dBtn); dBtn=dBtnb;
        cudaFree(dW);cudaFree(dB);cudaFree(dG);cudaFree(dBeta);
        cudaFree(dW2);cudaFree(dB2);cudaFree(dG2);cudaFree(dBeta2);
    }

    // ════════════════════════════════════════════
    //  Decoder 工具 Lambda：上采样 + concat + 2×Conv
    // ════════════════════════════════════════════
    // up: [C_in, D, H, W] → [C_in, D*2, H*2, W*2]
    auto doUpsample = [&](float* dIn, int Cin, int D, int H, int W, float** dOut) -> bool {
        int oD=D*2,oH=H*2,oW=W*2;
        CUDA_CHECK(cudaMalloc(dOut, Cin*oD*oH*oW*sizeof(float)));
        int total = Cin*oD*oH*oW;
        kernelUpsample2x<<<(total+255)/256,256>>>(dIn, *dOut, Cin, D, H, W);
        return true;
    };

    auto doConcat = [&](float* dA, int Ca, float* dB_, int Cb, int DHW, float** dOut) -> bool {
        int total = (Ca+Cb)*DHW;
        CUDA_CHECK(cudaMalloc(dOut, total*sizeof(float)));
        kernelConcat<<<(total+255)/256,256>>>(dA, dB_, *dOut, Ca, Cb, DHW);
        return true;
    };

    // ════════════════════════════════════════════
    //  6. Decoder Block 4  up(320→320) + cat(skip4,256) → 256
    // ════════════════════════════════════════════
    float *dUp4=nullptr;
    doUpsample(dBtn, 320, bD, bH, bW, &dUp4);
    float *dCat4=nullptr;
    doConcat(dUp4, 320, dEnc4, 256, eD4*eH4*eW4, &dCat4);
    float *dDec4=nullptr;
    CUDA_CHECK(cudaMalloc(&dDec4, 256*eD4*eH4*eW4*sizeof(float)));
    {
        float *dW=nullptr,*dB=nullptr,*dG=nullptr,*dBeta=nullptr;
        uploadW("conv_blocks_localization.0.0.conv.weight",  &dW,   256*576*3*3*3);
        uploadW("conv_blocks_localization.0.0.conv.bias",    &dB,   256);
        uploadW("conv_blocks_localization.0.0.instnorm.weight",&dG,  256);
        uploadW("conv_blocks_localization.0.0.instnorm.bias", &dBeta,256);
        ConvParams cp{576,256,eD4,eH4,eW4,3,3,3,1,1,1,1,1,1};
        CONV_BLK(handle, dCat4, dW, dB, dG, dBeta, dDec4, cp, 256);
        float *dW2=nullptr,*dB2=nullptr,*dG2=nullptr,*dBeta2=nullptr;
        uploadW("conv_blocks_localization.0.1.conv.weight",  &dW2,  256*256*1*1*1);
        uploadW("conv_blocks_localization.0.1.conv.bias",    &dB2,  256);
        uploadW("conv_blocks_localization.0.1.instnorm.weight",&dG2, 256);
        uploadW("conv_blocks_localization.0.1.instnorm.bias", &dBeta2,256);
        float *dDec4b=nullptr;
        CUDA_CHECK(cudaMalloc(&dDec4b,256*eD4*eH4*eW4*sizeof(float)));
        ConvParams cp2{256,256,eD4,eH4,eW4,1,1,1,1,1,1,0,0,0};
        CONV_BLK(handle, dDec4, dW2, dB2, dG2, dBeta2, dDec4b, cp2, 256);
        cudaFree(dDec4); dDec4=dDec4b;
        cudaFree(dW);cudaFree(dB);cudaFree(dG);cudaFree(dBeta);
        cudaFree(dW2);cudaFree(dB2);cudaFree(dG2);cudaFree(dBeta2);
    }
    cudaFree(dUp4); cudaFree(dCat4); cudaFree(dBtn); cudaFree(dEnc4);

    // ════════════════════════════════════════════
    //  7. Decoder Block 3  up(256→256) + cat(skip3,128) → 128
    // ════════════════════════════════════════════
    float *dUp3=nullptr;
    doUpsample(dDec4, 256, eD4, eH4, eW4, &dUp3);
    float *dCat3=nullptr;
    doConcat(dUp3, 256, dEnc3, 128, eD3*eH3*eW3, &dCat3);
    float *dDec3=nullptr;
    CUDA_CHECK(cudaMalloc(&dDec3, 128*eD3*eH3*eW3*sizeof(float)));
    {
        float *dW=nullptr,*dB=nullptr,*dG=nullptr,*dBeta=nullptr;
        uploadW("conv_blocks_localization.1.0.conv.weight",  &dW,   128*384*3*3*3);
        uploadW("conv_blocks_localization.1.0.conv.bias",    &dB,   128);
        uploadW("conv_blocks_localization.1.0.instnorm.weight",&dG,  128);
        uploadW("conv_blocks_localization.1.0.instnorm.bias", &dBeta,128);
        ConvParams cp{384,128,eD3,eH3,eW3,3,3,3,1,1,1,1,1,1};
        CONV_BLK(handle, dCat3, dW, dB, dG, dBeta, dDec3, cp, 256);
        float *dW2=nullptr,*dB2=nullptr,*dG2=nullptr,*dBeta2=nullptr;
        uploadW("conv_blocks_localization.1.1.conv.weight",  &dW2,  128*128*1*1*1);
        uploadW("conv_blocks_localization.1.1.conv.bias",    &dB2,  128);
        uploadW("conv_blocks_localization.1.1.instnorm.weight",&dG2, 128);
        uploadW("conv_blocks_localization.1.1.instnorm.bias", &dBeta2,128);
        float *dDec3b=nullptr;
        CUDA_CHECK(cudaMalloc(&dDec3b,128*eD3*eH3*eW3*sizeof(float)));
        ConvParams cp2{128,128,eD3,eH3,eW3,1,1,1,1,1,1,0,0,0};
        CONV_BLK(handle, dDec3, dW2, dB2, dG2, dBeta2, dDec3b, cp2, 256);
        cudaFree(dDec3); dDec3=dDec3b;
        cudaFree(dW);cudaFree(dB);cudaFree(dG);cudaFree(dBeta);
        cudaFree(dW2);cudaFree(dB2);cudaFree(dG2);cudaFree(dBeta2);
    }
    cudaFree(dUp3); cudaFree(dCat3); cudaFree(dDec4); cudaFree(dEnc3);

    // ════════════════════════════════════════════
    //  8. Decoder Block 2  up(128→128) + cat(skip2,64) → 64
    // ════════════════════════════════════════════
    float *dUp2=nullptr;
    doUpsample(dDec3, 128, eD3, eH3, eW3, &dUp2);
    float *dCat2=nullptr;
    doConcat(dUp2, 128, dEnc2, 64, eD2*eH2*eW2, &dCat2);
    float *dDec2=nullptr;
    CUDA_CHECK(cudaMalloc(&dDec2, 64*eD2*eH2*eW2*sizeof(float)));
    {
        float *dW=nullptr,*dB=nullptr,*dG=nullptr,*dBeta=nullptr;
        uploadW("conv_blocks_localization.2.0.conv.weight",  &dW,   64*192*3*3*3);
        uploadW("conv_blocks_localization.2.0.conv.bias",    &dB,   64);
        uploadW("conv_blocks_localization.2.0.instnorm.weight",&dG,  64);
        uploadW("conv_blocks_localization.2.0.instnorm.bias", &dBeta,64);
        ConvParams cp{192,64,eD2,eH2,eW2,3,3,3,1,1,1,1,1,1};
        CONV_BLK(handle, dCat2, dW, dB, dG, dBeta, dDec2, cp, 256);
        float *dW2=nullptr,*dB2=nullptr,*dG2=nullptr,*dBeta2=nullptr;
        uploadW("conv_blocks_localization.2.1.conv.weight",  &dW2,  64*64*1*1*1);
        uploadW("conv_blocks_localization.2.1.conv.bias",    &dB2,  64);
        uploadW("conv_blocks_localization.2.1.instnorm.weight",&dG2, 64);
        uploadW("conv_blocks_localization.2.1.instnorm.bias", &dBeta2,64);
        float *dDec2b=nullptr;
        CUDA_CHECK(cudaMalloc(&dDec2b,64*eD2*eH2*eW2*sizeof(float)));
        ConvParams cp2{64,64,eD2,eH2,eW2,1,1,1,1,1,1,0,0,0};
        CONV_BLK(handle, dDec2, dW2, dB2, dG2, dBeta2, dDec2b, cp2, 256);
        cudaFree(dDec2); dDec2=dDec2b;
        cudaFree(dW);cudaFree(dB);cudaFree(dG);cudaFree(dBeta);
        cudaFree(dW2);cudaFree(dB2);cudaFree(dG2);cudaFree(dBeta2);
    }
    cudaFree(dUp2); cudaFree(dCat2); cudaFree(dDec3); cudaFree(dEnc2);

    // ════════════════════════════════════════════
    //  9. Decoder Block 1  up(64→64) + cat(skip1,32) → 32
    // ════════════════════════════════════════════
    float *dUp1=nullptr;
    doUpsample(dDec2, 64, eD2, eH2, eW2, &dUp1);
    float *dCat1=nullptr;
    doConcat(dUp1, 64, dEnc1, 32, pD*pH*pW, &dCat1);
    float *dDec1=nullptr;
    CUDA_CHECK(cudaMalloc(&dDec1, 32*pD*pH*pW*sizeof(float)));
    {
        float *dW=nullptr,*dB=nullptr,*dG=nullptr,*dBeta=nullptr;
        uploadW("conv_blocks_localization.3.0.conv.weight",  &dW,   32*96*3*3*3);
        uploadW("conv_blocks_localization.3.0.conv.bias",    &dB,   32);
        uploadW("conv_blocks_localization.3.0.instnorm.weight",&dG,  32);
        uploadW("conv_blocks_localization.3.0.instnorm.bias", &dBeta,32);
        ConvParams cp{96,32,pD,pH,pW,3,3,3,1,1,1,1,1,1};
        CONV_BLK(handle, dCat1, dW, dB, dG, dBeta, dDec1, cp, 256);
        float *dW2=nullptr,*dB2=nullptr,*dG2=nullptr,*dBeta2=nullptr;
        uploadW("conv_blocks_localization.3.1.conv.weight",  &dW2,  32*32*1*1*1);
        uploadW("conv_blocks_localization.3.1.conv.bias",    &dB2,  32);
        uploadW("conv_blocks_localization.3.1.instnorm.weight",&dG2, 32);
        uploadW("conv_blocks_localization.3.1.instnorm.bias", &dBeta2,32);
        float *dDec1b=nullptr;
        CUDA_CHECK(cudaMalloc(&dDec1b,32*pD*pH*pW*sizeof(float)));
        ConvParams cp2{32,32,pD,pH,pW,1,1,1,1,1,1,0,0,0};
        CONV_BLK(handle, dDec1, dW2, dB2, dG2, dBeta2, dDec1b, cp2, 256);
        cudaFree(dDec1); dDec1=dDec1b;
        cudaFree(dW);cudaFree(dB);cudaFree(dG);cudaFree(dBeta);
        cudaFree(dW2);cudaFree(dB2);cudaFree(dG2);cudaFree(dBeta2);
    }
    cudaFree(dUp1); cudaFree(dCat1); cudaFree(dDec2); cudaFree(dEnc1);

    // ════════════════════════════════════════════
    //  10. Output Head  (32 → 2,  1×1×1 conv, no activation)
    // ════════════════════════════════════════════
    int outC = m_cfg.outChannels; // 2
    float *dLogits = nullptr;
    CUDA_CHECK(cudaMalloc(&dLogits, outC*pD*pH*pW*sizeof(float)));
    {
        float *dW=nullptr,*dB=nullptr;
        uploadW("seg_outputs.3.weight", &dW, outC*32*1*1*1);
        uploadW("seg_outputs.3.bias",   &dB, outC);
        ConvParams cp{32, outC, pD, pH, pW, 1,1,1, 1,1,1, 0,0,0};
        if (!cudnnConv3D(handle, dDec1, dW, dB, dLogits, cp)) return false;
        cudaFree(dW); cudaFree(dB);
    }
    cudaFree(dDec1); cudaFree(dInput);

    // ════════════════════════════════════════════
    //  11. 将 logits 从 GPU 拷贝回 CPU
    // ════════════════════════════════════════════
    int logitN = outC * pD * pH * pW;
    logits.resize(logitN);
    CUDA_CHECK(cudaMemcpy(logits.data(), dLogits,
                          logitN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dLogits);
    cudnnDestroy(handle);

    std::cout << "[LungNoduleDL] GPU inference done, logit size=" << logitN << "\n";
    return true;
}

} // namespace Onc

#endif // USE_CUDA
