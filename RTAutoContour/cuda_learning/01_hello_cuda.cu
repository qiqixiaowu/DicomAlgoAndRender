/**
 * ============================================================
 *  CUDA 第一课: Hello CUDA — 设备查询 & 第一个 Kernel
 * ============================================================
 *
 * 学习目标：
 *   1. 了解 GPU 硬件信息查询
 *   2. 理解 __global__ / __device__ / __host__ 修饰符
 *   3. 理解 Grid → Block → Thread 的层次结构
 *   4. 掌握 threadIdx / blockIdx / blockDim 的含义
 *   5. 学会 CUDA 错误检查
 *
 * 编译运行:
 *   nvcc 01_hello_cuda.cu -o 01_hello_cuda.exe
 *   ./01_hello_cuda.exe
 * ============================================================
 */

#include <cstdio>
#include <cuda_runtime.h>

// ═══════════════════════════════════════════════════════════════
//  Part 0: CUDA 错误检查宏（务必养成习惯！）
// ═══════════════════════════════════════════════════════════════
//
//  CUDA API 返回 cudaError_t，不检查的话出了错一脸迷茫。
//  这个宏在出错时打印文件名、行号和错误信息。
//
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)


// ═══════════════════════════════════════════════════════════════
//  Part 1: 设备信息查询
// ═══════════════════════════════════════════════════════════════
//
//  面试考点：SM 数量、Warp 大小、Shared Memory 大小、
//           最大线程数、计算能力(Compute Capability)
//
void queryDeviceInfo()
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    printf("=== GPU 设备信息 ===\n");
    printf("检测到 %d 个 CUDA 设备\n\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        printf("设备 %d: %s\n", i, prop.name);
        printf("  计算能力:            %d.%d\n", prop.major, prop.minor);
        printf("  SM 数量:             %d\n", prop.multiProcessorCount);
        printf("  每个 SM 最大线程数:   %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  每个 Block 最大线程:  %d\n", prop.maxThreadsPerBlock);
        printf("  Warp 大小:           %d\n", prop.warpSize);
        printf("  全局显存:            %.1f GB\n", prop.totalGlobalMem / 1e9);
        printf("  每个 Block Shared:   %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  每个 SM Shared:      %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
        printf("  L2 缓存大小:         %d KB\n", prop.l2CacheSize / 1024);
        printf("  显存带宽:            %.0f GB/s\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
        printf("  最大 Grid 维度:      (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  最大 Block 维度:     (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("\n");
    }
}


// ═══════════════════════════════════════════════════════════════
//  Part 2: 修饰符详解
// ═══════════════════════════════════════════════════════════════
//
//  __global__  — "入口函数"，CPU 调用，GPU 执行
//                必须返回 void，通过 <<<grid, block>>> 启动
//
//  __device__  — GPU 内部函数，只能被 __global__ 或其他 __device__ 调用
//                不能被 CPU 代码直接调用
//
//  __host__    — 普通 CPU 函数（默认就是 __host__，可省略）
//
//  __host__ __device__ — 两边都能用，编译器生成 CPU 和 GPU 两个版本
//

// __device__ 函数：只能在 GPU 上调用
__device__ int gpu_square(int x)
{
    return x * x;
}

// __host__ __device__ 函数：CPU 和 GPU 都能调用
__host__ __device__ int both_add(int a, int b)
{
    return a + b;
}


// ═══════════════════════════════════════════════════════════════
//  Part 3: 线程层次结构
// ═══════════════════════════════════════════════════════════════
//
//  Grid (网格)
//  ├── Block (0,0)    Block (1,0)    Block (2,0)    ...
//  │   └── Thread 0,1,2,...,N-1   ← 共享 Shared Memory
//  ├── Block (0,1)    Block (1,1)    ...
//  └── ...
//
//  每个线程有唯一标识：
//    线程在 Block 内的位置:  threadIdx.x, threadIdx.y, threadIdx.z
//    Block 在 Grid 内的位置:  blockIdx.x,  blockIdx.y,  blockIdx.z
//    Block 的维度:           blockDim.x,  blockDim.y,  blockDim.z
//    Grid 的维度:            gridDim.x,   gridDim.y,   gridDim.z
//
//  计算全局唯一线程 ID（1D 情况）：
//    globalId = blockIdx.x * blockDim.x + threadIdx.x
//
//  计算全局唯一线程 ID（2D 情况，你的 Marching Squares 用的）：
//    globalX = blockIdx.x * blockDim.x + threadIdx.x
//    globalY = blockIdx.y * blockDim.y + threadIdx.y
//

// 第一个 Kernel: 打印每个线程的坐标
__global__ void helloKernel()
{
    // printf 在 CUDA kernel 里可以用（Compute Capability >= 2.0）
    // 但是要注意：所有线程同时 printf，输出顺序是不确定的！
    printf("  Thread (%d,%d) in Block (%d,%d) | globalId=(%d,%d)\n",
           threadIdx.x, threadIdx.y,
           blockIdx.x, blockIdx.y,
           blockIdx.x * blockDim.x + threadIdx.x,
           blockIdx.y * blockDim.y + threadIdx.y);
}


// ═══════════════════════════════════════════════════════════════
//  Part 4: Kernel 启动配置
// ═══════════════════════════════════════════════════════════════
//
//  kernel<<<gridDim, blockDim>>>(args...);
//
//  gridDim  — Grid 中有多少个 Block（dim3 类型）
//  blockDim — 每个 Block 中有多少个 Thread（dim3 类型）
//
//  dim3 是 CUDA 的三维整数类型：
//    dim3(4)      → (4, 1, 1)
//    dim3(4, 2)   → (4, 2, 1)
//    dim3(4, 2, 3)
//
//  总线程数 = gridDim.x * gridDim.y * gridDim.z
//           * blockDim.x * blockDim.y * blockDim.z
//
//  常用配置原则：
//    1. blockDim 应为 32 的倍数（Warp 大小）
//    2. 通常 128 或 256 效果好
//    3. gridDim = ceil(数据量 / blockDim)
//

void demo_thread_hierarchy()
{
    printf("\n=== 线程层次结构演示 ===\n");
    printf("启动 Grid(2,2), Block(2,2) → 共 16 个线程\n\n");

    dim3 grid(2, 2);    // 2×2 = 4 个 Block
    dim3 block(2, 2);   // 每个 Block 2×2 = 4 个线程

    helloKernel<<<grid, block>>>();

    // 重要！Kernel 调用是异步的！
    // cudaDeviceSynchronize() 等待 GPU 上所有工作完成
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n注意观察：输出顺序是不确定的（GPU 并行执行）\n");
}


// ═══════════════════════════════════════════════════════════════
//  Part 5: 实战 — 用 GPU 计算数组每个元素的平方
// ═══════════════════════════════════════════════════════════════

__global__ void squareKernel(const int* input, int* output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查！因为线程数通常不恰好等于数据量
    // 比如 N=1000, blockDim=256, 则需要 4 个 Block = 1024 个线程
    // 最后 24 个线程是"多余的"，必须跳过
    if (idx >= N) return;

    output[idx] = gpu_square(input[idx]);  // 调用 __device__ 函数
}

void demo_square_array()
{
    printf("\n=== GPU 数组平方演示 ===\n");

    const int N = 10;
    const int bytes = N * sizeof(int);

    // 1. 在 CPU 端准备数据
    int h_input[N]  = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int h_output[N] = {0};

    // 2. 在 GPU 端分配内存
    int *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    // 3. 把数据从 CPU 拷贝到 GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // 4. 启动 Kernel
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;  // 向上取整
    printf("  Grid=%d, Block=%d, 总线程=%d (数据量=%d)\n",
           gridSize, blockSize, gridSize * blockSize, N);

    squareKernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. 把结果从 GPU 拷贝回 CPU
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    // 6. 验证
    printf("  输入:  ");
    for (int i = 0; i < N; ++i) printf("%d ", h_input[i]);
    printf("\n  输出:  ");
    for (int i = 0; i < N; ++i) printf("%d ", h_output[i]);
    printf("\n");

    // 7. 释放 GPU 内存（别忘了！）
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}


// ═══════════════════════════════════════════════════════════════
//  主函数
// ═══════════════════════════════════════════════════════════════
int main()
{
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  CUDA 第一课: Hello CUDA                ║\n");
    printf("╚══════════════════════════════════════════╝\n");

    queryDeviceInfo();
    demo_thread_hierarchy();
    demo_square_array();

    printf("\n课后思考：\n");
    printf("  1. 如果 N = 10, Block = 256，有多少线程在空跑？\n");
    printf("  2. 如果把 squareKernel 中的 if(idx>=N) return; 删掉，会怎样？\n");
    printf("  3. helloKernel 的输出为什么每次运行顺序不同？\n");

    return 0;
}
