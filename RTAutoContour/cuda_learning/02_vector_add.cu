/**
 * ============================================================
 *  CUDA 第二课: 向量加法 — 内存管理 & 性能测量
 * ============================================================
 *
 * 学习目标：
 *   1. 完整的 GPU 内存管理流程 (malloc → memcpy → kernel → memcpy → free)
 *   2. CUDA Event 计时（精确到微秒级的 GPU 时间测量）
 *   3. 带宽计算与理论峰值对比
 *   4. 错误处理的完整流程
 *   5. Unified Memory (cudaMallocManaged) 简化版
 *
 * 面试考点：
 *   - cudaMemcpy 的方向参数
 *   - 同步 vs 异步内存拷贝
 *   - Pinned Memory (cudaMallocHost) vs 普通 malloc
 *   - cudaEvent 计时为什么比 CPU 计时更准确
 *
 * 编译运行:
 *   nvcc 02_vector_add.cu -o 02_vector_add.exe
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


// ═══════════════════════════════════════════════════════════════
//  Part 1: 基础向量加法 Kernel
// ═══════════════════════════════════════════════════════════════
//
//  最简单的并行模式：每个线程处理一个元素
//  时间复杂度: O(1) per thread, O(N) total (N个线程)
//

__global__ void vectorAdd(const float* a, const float* b, float* c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}


// ═══════════════════════════════════════════════════════════════
//  Part 2: Grid-Stride Loop — 处理任意大小数据
// ═══════════════════════════════════════════════════════════════
//
//  问题：如果 N = 1亿，我们不可能启动 1亿个线程
//  解决：每个线程处理多个元素，用循环+步长跳过
//
//  步长 = gridDim.x * blockDim.x（总线程数）
//
//  这是 CUDA 最重要的编程模式之一！
//  好处：
//    1. 一份代码适配任意数据大小
//    2. 可以固定 Grid/Block 大小，方便调优
//    3. 线程复用，减少 Kernel 启动开销
//

__global__ void vectorAddStride(const float* a, const float* b, float* c, int N)
{
    //     ↓ 起始位置      ↓ 步长 = 总线程数
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N;
         idx += gridDim.x * blockDim.x)  // ← Grid-Stride!
    {
        c[idx] = a[idx] + b[idx];
    }
}

// 图解 Grid-Stride (4线程处理10个元素):
//
//  线程 0: 处理 [0] [4] [8]
//  线程 1: 处理 [1] [5] [9]
//  线程 2: 处理 [2] [6]
//  线程 3: 处理 [3] [7]
//
//  ↓ 相当于把数组切成若干"条纹"，每个线程负责一条


// ═══════════════════════════════════════════════════════════════
//  Part 3: 手动内存管理方式
// ═══════════════════════════════════════════════════════════════
//
//  流程图：
//
//  CPU (Host)                    GPU (Device)
//  ──────────                    ────────────
//  malloc h_a, h_b, h_c          cudaMalloc d_a, d_b, d_c
//  初始化 h_a, h_b               
//  ─── cudaMemcpy H→D ──────→   d_a, d_b 有数据了
//                                kernel<<<>>>() 执行
//  ←── cudaMemcpy D→H ──────    h_c 有结果了
//  验证/使用 h_c                 cudaFree d_a, d_b, d_c
//  free h_a, h_b, h_c
//

void demo_manual_memory(int N)
{
    printf("\n=== 方式1: 手动内存管理 (N=%d) ===\n", N);

    size_t bytes = N * sizeof(float);

    // --- Host 端 ---
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);

    for (int i = 0; i < N; ++i) {
        h_a[i] = sinf(i * 0.01f);
        h_b[i] = cosf(i * 0.01f);
    }

    // --- Device 端 ---
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // --- CUDA Event 计时 ---
    // 为什么用 cudaEvent 而不是 CPU 的 clock()?
    // 因为 Kernel 是异步执行的！CPU 时间测不准。
    // cudaEvent 打的是 GPU 时间戳，精确可靠。
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- 拷贝 + 计算 + 拷贝 ---
    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // 带宽计算: 读 a + 读 b + 写 c = 3N 个 float
    double gb = 3.0 * N * sizeof(float) / 1e9;
    printf("  耗时: %.3f ms\n", ms);
    printf("  有效带宽: %.1f GB/s\n", gb / (ms / 1000.0));

    // --- 验证 ---
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabsf(h_c[i] - expected) > 1e-5f) errors++;
    }
    printf("  验证: %s (%d errors)\n", errors ? "FAIL" : "PASS", errors);

    // --- 清理 ---
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a); free(h_b); free(h_c);
}


// ═══════════════════════════════════════════════════════════════
//  Part 4: Pinned Memory（页锁定内存）
// ═══════════════════════════════════════════════════════════════
//
//  普通 malloc 分配的内存是"可分页"(pageable)的，
//  OS 可能把它换出到磁盘（虚拟内存）。
//
//  cudaMemcpy 内部实际做了两步：
//    pageable → 临时 pinned buffer → GPU
//  多了一次拷贝！
//
//  cudaMallocHost 分配的是"页锁定内存"(pinned memory):
//    pinned → GPU  (DMA 直传，省一次拷贝)
//
//  好处：
//    1. H→D / D→H 传输带宽提高 ~2x
//    2. 可以用 cudaMemcpyAsync（异步传输的前提条件！）
//
//  坏处：
//    1. 不能被 OS 换页，占实际物理内存
//    2. 分配成本比 malloc 高
//    3. 不要分配太多（会挤压 OS 和其他进程的内存）
//

void demo_pinned_memory(int N)
{
    printf("\n=== 方式2: Pinned Memory (N=%d) ===\n", N);

    size_t bytes = N * sizeof(float);

    // 用 cudaMallocHost 代替 malloc
    float *h_a, *h_b, *h_c;
    CUDA_CHECK(cudaMallocHost(&h_a, bytes));  // pinned!
    CUDA_CHECK(cudaMallocHost(&h_b, bytes));
    CUDA_CHECK(cudaMallocHost(&h_c, bytes));

    for (int i = 0; i < N; ++i) {
        h_a[i] = sinf(i * 0.01f);
        h_b[i] = cosf(i * 0.01f);
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double gb = 3.0 * N * sizeof(float) / 1e9;
    printf("  耗时: %.3f ms\n", ms);
    printf("  有效带宽: %.1f GB/s\n", gb / (ms / 1000.0));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFreeHost(h_a));  // 注意用 cudaFreeHost!
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));
}


// ═══════════════════════════════════════════════════════════════
//  Part 5: Unified Memory (统一内存)
// ═══════════════════════════════════════════════════════════════
//
//  cudaMallocManaged 分配的内存 CPU 和 GPU 都能直接访问。
//  不需要手动 cudaMemcpy！驱动自动在两端之间迁移页面。
//
//  好处：代码简洁，适合原型开发
//  坏处：
//    1. 隐式迁移有开销，性能不如手动管理
//    2. 第一次访问时触发"页面错误"迁移，有延迟尖峰
//    3. 调试困难（不清楚数据在哪端）
//
//  面试答：常用于开发阶段快速验证，产品代码建议显式管理
//

void demo_unified_memory(int N)
{
    printf("\n=== 方式3: Unified Memory (N=%d) ===\n", N);

    size_t bytes = N * sizeof(float);

    // 一次分配，两端可用
    float *a, *b, *c;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    CUDA_CHECK(cudaMallocManaged(&c, bytes));

    // 在 CPU 端初始化（像普通指针一样用）
    for (int i = 0; i < N; ++i) {
        a[i] = sinf(i * 0.01f);
        b[i] = cosf(i * 0.01f);
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 提示：cudaMemPrefetchAsync 可以提前把数据迁移到 GPU
    // 避免 kernel 执行时的页面错误延迟
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaMemPrefetchAsync(a, bytes, device));
    CUDA_CHECK(cudaMemPrefetchAsync(b, bytes, device));

    CUDA_CHECK(cudaEventRecord(start));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vectorAddStride<<<gridSize, blockSize>>>(a, b, c, N);  // 用 stride 版

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double gb = 3.0 * N * sizeof(float) / 1e9;
    printf("  Kernel 耗时: %.3f ms (不含传输)\n", ms);
    printf("  有效带宽: %.1f GB/s\n", gb / (ms / 1000.0));

    // 在 CPU 端直接读 c（驱动自动迁移回 CPU）
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (fabsf(c[i] - (a[i] + b[i])) > 1e-5f) errors++;
    }
    printf("  验证: %s\n", errors ? "FAIL" : "PASS");

    // 统一用 cudaFree
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}


// ═══════════════════════════════════════════════════════════════
//  主函数
// ═══════════════════════════════════════════════════════════════
int main()
{
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  CUDA 第二课: 向量加法 & 内存管理       ║\n");
    printf("╚══════════════════════════════════════════╝\n");

    const int N = 1 << 24;  // 16M 个元素
    printf("数据量: %d 个 float (%.1f MB)\n", N, N * sizeof(float) / 1e6);

    demo_manual_memory(N);
    demo_pinned_memory(N);
    demo_unified_memory(N);

    printf("\n课后思考：\n");
    printf("  1. Pinned 比 Pageable 快多少？为什么？\n");
    printf("  2. 向量加法是计算密集型还是访存密集型？为什么？\n");
    printf("  3. Grid-Stride Loop 启动 256 个线程处理 16M 数据，为什么不慢？\n");
    printf("  4. 算的有效带宽占 GPU 理论峰值带宽的百分比是多少？\n");
    printf("     (提示: 用第一课查到的显存带宽)\n");

    return 0;
}
