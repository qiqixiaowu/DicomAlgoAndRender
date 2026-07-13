/**
 * ============================================================
 *  CUDA 第四课: 并行归约 (Parallel Reduction)
 * ============================================================
 *
 * 学习目标：
 *   1. 理解并行归约的核心思想 —— 树形计算
 *   2. 掌握从 Naive → 无 Bank Conflict → Warp Shuffle 的优化路径
 *   3. 理解 Warp Divergence 的危害和消除方法
 *   4. 学习 Warp-level 原语 (__shfl_down_sync)
 *   5. 理解为什么归约是面试高频考点
 *
 * 面试必考：
 *   - 归约的时间复杂度: O(N/P + log P)
 *   - 三种优化阶段的 Bank Conflict 和 Warp Divergence 分析
 *   - __shfl_down_sync 的含义和使用
 *   - 如何处理超大数组（多 Block 归约）
 *
 * 编译:
 *   nvcc 04_reduction.cu -o 04_reduction.exe
 * ============================================================
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define BLOCK_SIZE 256
#define N (1 << 22)  // 4M 个元素


// ═══════════════════════════════════════════════════════════════
//  什么是归约 (Reduction)？
// ═══════════════════════════════════════════════════════════════
//
//  把一个数组 "缩小" 成一个值：
//    sum([1, 2, 3, 4, 5, 6, 7, 8]) → 36
//    max([1, 2, 3, 4, 5, 6, 7, 8]) → 8
//
//  CPU 做法：顺序累加，O(N) 时间
//  GPU 做法：树形归约，O(log N) 步骤 × O(N/P) 每步
//
//  树形归约示意 (8个元素, 4个线程):
//
//  Step 0:  [1] [2] [3] [4] [5] [6] [7] [8]
//            ↘↙      ↘↙      ↘↙      ↘↙
//  Step 1:  [3]     [7]     [11]    [15]
//            ↘      ↙          ↘     ↙
//  Step 2:  [10]              [26]
//                 ↘       ↙
//  Step 3:       [36]                         ← 最终结果
//


// ═══════════════════════════════════════════════════════════════
//  版本1: Naive 归约 (有 Warp Divergence + Bank Conflict)
// ═══════════════════════════════════════════════════════════════
//
//  让线程 0 和线程 1 做一对，线程 2 和线程 3 做一对...
//
//  stride = 1:  T0 += T1,  T2 += T3,  T4 += T5, ...
//  stride = 2:  T0 += T2,  T4 += T6, ...
//  stride = 4:  T0 += T4, ...
//
//  问题1 - Warp Divergence:
//    stride=1 时，Warp 中只有偶数线程工作 → 50% 浪费
//    stride=2 时，只有 1/4 线程工作 → 75% 浪费
//    每一步活跃线程减半！
//
//  问题2 - Bank Conflict:
//    stride=1 时，T0 读 sdata[0] 和 sdata[1] → Bank 0, Bank 1 ✓
//    stride=2 时，T0 读 sdata[0] 和 sdata[2] → Bank 0, Bank 2 ✓
//    但 stride=16 时，T0 读 sdata[0] 和 sdata[16] → Bank 0, Bank 0 ← Conflict!
//

__global__ void reduceNaive(const float* input, float* output, int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载到 Shared Memory
    sdata[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // 树形归约
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {  // ← 问题所在！
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // 线程0写出结果
    if (tid == 0) output[blockIdx.x] = sdata[0];
}


// ═══════════════════════════════════════════════════════════════
//  版本2: 连续线程工作 (消除 Warp Divergence)
// ═══════════════════════════════════════════════════════════════
//
//  思路：让连续的线程做工作，不是交错的。
//
//  stride = blockDim/2:  T0 += T[blockDim/2],  T1 += T[blockDim/2+1], ...
//  stride = blockDim/4:  T0 += T[blockDim/4],  T1 += T[blockDim/4+1], ...
//
//  好处：
//    stride = blockDim/2 时，前半 Warp 全部工作 → 无 Divergence!
//    stride = blockDim/4 时，前 1/4 的 Warp 全部活跃
//    → 在每一步中，活跃线程是连续的
//
//  还解决了 Bank Conflict！
//    stride 是 2 的幂，访问 sdata[tid] 和 sdata[tid + stride]
//    → 不同线程访问不同 Bank
//

__global__ void reduceSequential(const float* input, float* output, int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // 反向归约 —— 连续线程工作
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {  // ← 连续线程！
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}


// ═══════════════════════════════════════════════════════════════
//  版本3: First-Add During Load (减少一半线程闲置)
// ═══════════════════════════════════════════════════════════════
//
//  观察：版本2 的第一步，stride = blockDim/2，
//  前半线程把后半的值加过来。后半线程只是加载了数据然后就没用了。
//  浪费！
//
//  优化：每个 Block "负责" 2 * blockDim 个元素，
//  加载时就做第一次加法：
//
//  sdata[tid] = input[gid] + input[gid + blockDim * gridDim]
//
//  这样同样的 Block 数就能处理 2 倍的数据。
//

__global__ void reduceFirstAdd(const float* input, float* output, int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // 加载时做第一次加法！
    float mySum = 0.0f;
    if (gid < n)                    mySum  = input[gid];
    if (gid + blockDim.x < n)      mySum += input[gid + blockDim.x];
    sdata[tid] = mySum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}


// ═══════════════════════════════════════════════════════════════
//  版本4: Warp Shuffle (终极优化)
// ═══════════════════════════════════════════════════════════════
//
//  当 stride < 32（一个 Warp 内）时，不需要 Shared Memory！
//  因为 Warp 内的线程是锁步执行的 (SIMT)，可以直接交换寄存器值。
//
//  __shfl_down_sync(mask, val, delta):
//    线程 T 从线程 T+delta 获取 val 的值
//    mask = 0xFFFFFFFF 表示 Warp 中所有 32 个线程都参与
//
//  Warp 内归约过程 (假设 Warp 从 T0-T31):
//    delta=16: T0 += T16, T1 += T17, ..., T15 += T31
//    delta=8:  T0 += T8,  T1 += T9,  ..., T7  += T15
//    delta=4:  T0 += T4,  T1 += T5,  T2  += T6,  T3 += T7
//    delta=2:  T0 += T2,  T1 += T3
//    delta=1:  T0 += T1
//    → T0 拥有整个 Warp 的总和！
//
//  优势：
//    1. 不需要 Shared Memory，直接用寄存器 → 更快
//    2. 不需要 __syncthreads() → 减少同步开销
//    3. 延迟更低
//

__device__ float warpReduce(float val)
{
    // Warp 内归约：5 步搞定 32 个线程
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;  // 只有 lane 0 的结果是正确的完整总和
}

__global__ void reduceWarpShuffle(const float* input, float* output, int n)
{
    __shared__ float warpSums[BLOCK_SIZE / 32];  // 每个 Warp 一个位置

    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int laneId = tid % 32;       // 在 Warp 内的位置 (0-31)
    int warpId = tid / 32;       // 哪个 Warp (0-7, 当 blockSize=256)

    // First-add during load
    float mySum = 0.0f;
    if (gid < n)               mySum  = input[gid];
    if (gid + blockDim.x < n)  mySum += input[gid + blockDim.x];

    // Step 1: Warp 内归约（用 Shuffle，不用 Shared Memory）
    mySum = warpReduce(mySum);

    // Step 2: 每个 Warp 的 lane 0 写到 Shared Memory
    if (laneId == 0) warpSums[warpId] = mySum;
    __syncthreads();

    // Step 3: 第一个 Warp 做最终归约
    // (因为 BLOCK_SIZE/32 = 8 ≤ 32，一个 Warp 就够了)
    if (warpId == 0) {
        mySum = (tid < blockDim.x / 32) ? warpSums[tid] : 0.0f;
        mySum = warpReduce(mySum);
    }

    if (tid == 0) output[blockIdx.x] = mySum;
}


// ═══════════════════════════════════════════════════════════════
//  CPU 参考实现
// ═══════════════════════════════════════════════════════════════

float reduceCPU(const float* data, int n)
{
    double sum = 0;  // 用 double 减少浮点误差
    for (int i = 0; i < n; ++i) sum += data[i];
    return (float)sum;
}


// ═══════════════════════════════════════════════════════════════
//  辅助：两阶段归约（处理多 Block 的情况）
// ═══════════════════════════════════════════════════════════════
//
//  问题：每个 Block 产生一个部分和，但 Block 之间无法同步！
//  方案：分两步:
//    Phase 1: 每个 Block 归约出一个值 → 存到 d_partial[blockIdx.x]
//    Phase 2: 启动一个新 Kernel，归约 d_partial 数组
//
//    数据量: N=4M → 4M/256/2 = 8192 个部分和
//    Phase 2: 8192 个数 → 8192/256/2 = 16 个部分和
//    Phase 3: 16 个数 → 1 个结果 (一个 Block 搞定)
//

typedef void (*ReduceKernel)(const float*, float*, int);

float gpuReduce(const float* d_input, int n, ReduceKernel kernel,
                bool firstAddDuringLoad)
{
    int elemPerBlock = firstAddDuringLoad ? (BLOCK_SIZE * 2) : BLOCK_SIZE;
    int numBlocks = (n + elemPerBlock - 1) / elemPerBlock;

    float* d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, numBlocks * sizeof(float)));

    // Phase 1
    kernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_partial, n);
    CUDA_CHECK(cudaGetLastError());

    // 递归调用直到数据量足够小
    float result;
    if (numBlocks > 1) {
        result = gpuReduce(d_partial, numBlocks, kernel, firstAddDuringLoad);
    } else {
        CUDA_CHECK(cudaMemcpy(&result, d_partial, sizeof(float),
                               cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_partial));
    return result;
}


// ═══════════════════════════════════════════════════════════════
//  Main: 性能对比
// ═══════════════════════════════════════════════════════════════

int main()
{
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  CUDA 第四课: 并行归约 (Reduction)       ║\n");
    printf("╚══════════════════════════════════════════╝\n");
    printf("数据量: %d 个浮点数 (%.1f MB)\n\n", N, N * sizeof(float) / 1e6);

    // 分配和初始化
    float* h_data = (float*)malloc(N * sizeof(float));
    srand(42);
    for (int i = 0; i < N; ++i) {
        h_data[i] = (float)(rand() % 100) / 100.0f;  // 0.00 ~ 0.99
    }

    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float),
                           cudaMemcpyHostToDevice));

    // CPU 参考
    float cpuSum = reduceCPU(h_data, N);
    printf("CPU 结果: %.2f\n\n", cpuSum);

    // 各版本测试
    struct {
        const char* name;
        ReduceKernel kernel;
        bool firstAdd;
    } tests[] = {
        { "Naive (Warp Divergence)",       reduceNaive,        false },
        { "Sequential (连续线程)",          reduceSequential,   false },
        { "First-Add During Load",          reduceFirstAdd,     true  },
        { "Warp Shuffle (终极版)",          reduceWarpShuffle,  true  },
    };

    for (auto& test : tests) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // 预热
        gpuReduce(d_data, N, test.kernel, test.firstAdd);

        // 测量 (跑 10 次取平均)
        CUDA_CHECK(cudaEventRecord(start));
        float gpuSum = 0;
        for (int r = 0; r < 10; ++r)
            gpuSum = gpuReduce(d_data, N, test.kernel, test.firstAdd);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 10.0f;

        float bandwidth = N * sizeof(float) / ms / 1e6;  // GB/s
        float error = fabsf(gpuSum - cpuSum) / fabsf(cpuSum) * 100.0f;

        printf("[%s]\n", test.name);
        printf("  耗时: %.3f ms | 带宽: %.1f GB/s | 误差: %.4f%%\n\n",
               ms, bandwidth, error);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    // 清理
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);

    printf("课后思考：\n");
    printf("  1. 为什么 Naive 版本比 Sequential 慢？画出 Warp 内线程的活跃情况。\n");
    printf("  2. First-Add 实际上减少了多少个 Block？\n");
    printf("  3. __shfl_down_sync 的 mask 参数有什么作用？全 1 意味着什么？\n");
    printf("  4. 如果数据量不是 2 的幂，哪些版本可能出错？怎么修复？\n");
    printf("  5. 用实际带宽除以理论显存带宽，利用率是多少？\n");

    return 0;
}
