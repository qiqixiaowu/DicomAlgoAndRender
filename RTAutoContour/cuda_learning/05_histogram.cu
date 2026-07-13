/**
 * ============================================================
 *  CUDA 第五课: 直方图 & Atomic 操作
 * ============================================================
 *
 * 学习目标：
 *   1. 理解 Atomic 操作的含义和使用场景
 *   2. 掌握 Global Atomic vs Shared Memory Privatization 模式
 *   3. 理解为什么 Atomic 会 "序列化" 导致性能下降
 *   4. 学习 Privatization 优化 —— 先私有化到 Shared，最后合并到 Global
 *   5. 了解实际应用：医学图像灰度直方图
 *
 * 面试必考：
 *   - atomicAdd 是如何保证原子性的（CAS 循环、总线锁）
 *   - 为什么大量 Atomic 会成为瓶颈
 *   - Privatization 模式：面试中常问的优化技巧
 *   - Shared Memory 大小限制对 bin 数的影响
 *
 * 编译:
 *   nvcc 05_histogram.cu -o 05_histogram.exe
 * ============================================================
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
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

#define NUM_BINS 256       // 256 个灰度级
#define BLOCK_SIZE 256
#define N (1 << 24)        // 16M 个像素


// ═══════════════════════════════════════════════════════════════
//  什么是 Atomic 操作？
// ═══════════════════════════════════════════════════════════════
//
//  问题场景：直方图 —— 多个线程可能同时增加同一个 bin
//
//  没有 Atomic:
//    线程 A: 读 bin[5] = 10
//    线程 B: 读 bin[5] = 10        ← 同时读到旧值！
//    线程 A: 写 bin[5] = 11
//    线程 B: 写 bin[5] = 11        ← 丢失了一次更新！正确应该是 12
//
//  有 Atomic:
//    atomicAdd(&bin[5], 1)
//    → 硬件保证 "读-改-写" 是一个不可分割的整体操作
//    → 不会丢失更新
//
//  代价：
//    多个线程竞争同一地址时会 "排队" → 序列化 → 性能下降
//    竞争越多，性能越差
//


// ═══════════════════════════════════════════════════════════════
//  版本1: Global Atomic (最简单，最慢)
// ═══════════════════════════════════════════════════════════════
//
//  每个线程直接对 Global Memory 的直方图 bin 做 atomicAdd
//
//  问题：
//    256 个 bin，成千上万的线程 → 大量竞争
//    尤其如果数据分布不均匀（比如图像大部分是暗的）
//    → 某些 bin 的竞争极其严重
//

__global__ void histogramGlobal(const unsigned char* data, int* hist, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-Stride Loop: 一个 Grid 处理整个数组
    for (int i = gid; i < n; i += blockDim.x * gridDim.x) {
        atomicAdd(&hist[data[i]], 1);
        //         ↑ 对 Global Memory 做原子操作 → 慢！
    }
}


// ═══════════════════════════════════════════════════════════════
//  版本2: Shared Memory Privatization (经典优化)
// ═══════════════════════════════════════════════════════════════
//
//  核心思想：
//  1. 每个 Block 在 Shared Memory 中维护一份 "私有直方图"
//  2. Block 内的线程只对 Shared Memory 做 atomicAdd → 竞争少得多
//  3. Block 结束后，把私有直方图合并到 Global Memory
//
//  ┌──────────┐   ┌──────────┐   ┌──────────┐
//  │ Block 0  │   │ Block 1  │   │ Block 2  │
//  │ ┌──────┐ │   │ ┌──────┐ │   │ ┌──────┐ │
//  │ │Shared│ │   │ │Shared│ │   │ │Shared│ │  ← 私有直方图
//  │ │ Hist │ │   │ │ Hist │ │   │ │ Hist │ │
//  │ └──┬───┘ │   │ └──┬───┘ │   │ └──┬───┘ │
//  └────┼─────┘   └────┼─────┘   └────┼─────┘
//       └───────────────┼───────────────┘
//                       ↓
//              ┌──────────────┐
//              │ Global Hist  │  ← 最终合并
//              └──────────────┘
//
//  为什么更快：
//    一个 Block 有 256 个线程，只在 Block 内竞争
//    比上万线程同时竞争 Global Memory 快得多
//    Shared Memory 带宽远高于 Global Memory
//
//  内存开销：256 bins × 4 bytes = 1KB → 很小，每个 Block 轻松放得下
//

__global__ void histogramShared(const unsigned char* data, int* hist, int n)
{
    // Step 1: 初始化 Shared Memory 直方图
    __shared__ int sharedHist[NUM_BINS];

    // 让线程协作初始化（比单线程循环快）
    // 当 BLOCK_SIZE >= NUM_BINS 时，每个线程清零一个 bin
    if (threadIdx.x < NUM_BINS) {
        sharedHist[threadIdx.x] = 0;
    }
    __syncthreads();

    // Step 2: 私有直方图累计
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = gid; i < n; i += blockDim.x * gridDim.x) {
        atomicAdd(&sharedHist[data[i]], 1);
        //         ↑ 对 Shared Memory 做原子 → 比 Global 快很多
    }
    __syncthreads();  // 确保 Block 内所有线程都写完了

    // Step 3: 合并到 Global Memory
    if (threadIdx.x < NUM_BINS) {
        atomicAdd(&hist[threadIdx.x], sharedHist[threadIdx.x]);
        //         ↑ 合并时才访问 Global，且每个 bin 只有 gridDim.x 次竞争
    }
}


// ═══════════════════════════════════════════════════════════════
//  版本3: 线程内局部累加 (进一步减少 Atomic)
// ═══════════════════════════════════════════════════════════════
//
//  观察：如果连续像素值相同（常见于暗背景图像），
//  可以在寄存器中先累加，再一次性 atomicAdd
//
//  这是一种实用的微优化，但代码更复杂

__global__ void histogramLocalAccum(const unsigned char* data, int* hist, int n)
{
    __shared__ int sharedHist[NUM_BINS];

    if (threadIdx.x < NUM_BINS) sharedHist[threadIdx.x] = 0;
    __syncthreads();

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 每个线程用局部变量记录上一个值和连续计数
    unsigned char prevVal = 0;
    int localCount = 0;

    for (int i = gid; i < n; i += stride) {
        unsigned char val = data[i];
        if (val == prevVal) {
            localCount++;  // 相同值，继续累加
        } else {
            if (localCount > 0)
                atomicAdd(&sharedHist[prevVal], localCount);
            prevVal = val;
            localCount = 1;
        }
    }
    // 别忘了最后一批
    if (localCount > 0)
        atomicAdd(&sharedHist[prevVal], localCount);

    __syncthreads();

    if (threadIdx.x < NUM_BINS)
        atomicAdd(&hist[threadIdx.x], sharedHist[threadIdx.x]);
}


// ═══════════════════════════════════════════════════════════════
//  CPU 参考
// ═══════════════════════════════════════════════════════════════

void histogramCPU(const unsigned char* data, int* hist, int n)
{
    memset(hist, 0, NUM_BINS * sizeof(int));
    for (int i = 0; i < n; ++i) {
        hist[data[i]]++;
    }
}


// ═══════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════

int main()
{
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  CUDA 第五课: 直方图 & Atomic 操作       ║\n");
    printf("╚══════════════════════════════════════════╝\n");
    printf("数据量: %d 像素 (%.1f MB)\n\n", N, N / 1e6);

    // --- 生成模拟医学图像数据 ---
    // 多数像素偏暗 (模拟 CT 背景区域)，制造不均匀分布
    unsigned char* h_data = (unsigned char*)malloc(N);
    srand(42);
    for (int i = 0; i < N; ++i) {
        float r = (float)rand() / RAND_MAX;
        if (r < 0.6f)       h_data[i] = rand() % 30;                   // 60% 暗区域
        else if (r < 0.9f)  h_data[i] = 100 + rand() % 80;            // 30% 中间
        else                 h_data[i] = 200 + rand() % 56;            // 10% 高亮
    }

    // CPU 参考
    int h_histCPU[NUM_BINS];
    histogramCPU(h_data, h_histCPU, N);

    // GPU 数据
    unsigned char* d_data;
    int* d_hist;
    CUDA_CHECK(cudaMalloc(&d_data, N));
    CUDA_CHECK(cudaMalloc(&d_hist, NUM_BINS * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice));

    int numBlocks = 256;  // 用固定 Block 数 + Grid-Stride Loop

    struct {
        const char* name;
        void (*kernel)(const unsigned char*, int*, int);
    } tests[] = {
        { "Global Atomic (朴素版)",       histogramGlobal     },
        { "Shared Privatization (经典)",   histogramShared     },
        { "Local Accumulate (进阶)",       histogramLocalAccum },
    };

    for (auto& test : tests) {
        // 清零直方图
        CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(int)));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // 预热
        test.kernel<<<numBlocks, BLOCK_SIZE>>>(d_data, d_hist, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(int)));

        // 测量
        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < 10; ++r) {
            CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(int)));
            test.kernel<<<numBlocks, BLOCK_SIZE>>>(d_data, d_hist, N);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 10.0f;

        // 验证
        int h_histGPU[NUM_BINS];
        CUDA_CHECK(cudaMemcpy(h_histGPU, d_hist, NUM_BINS * sizeof(int),
                               cudaMemcpyDeviceToHost));

        int errors = 0;
        for (int i = 0; i < NUM_BINS; ++i) {
            if (h_histCPU[i] != h_histGPU[i]) {
                errors++;
                if (errors <= 3)
                    printf("  bin[%d]: CPU=%d GPU=%d\n",
                           i, h_histCPU[i], h_histGPU[i]);
            }
        }

        float bandwidth = N / ms / 1e6;  // GB/s (1 byte per pixel)
        printf("[%s]\n", test.name);
        printf("  耗时: %.3f ms | 带宽: %.1f GB/s | %s\n\n",
               ms, bandwidth, errors == 0 ? "PASS" : "FAIL");

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    // --- 打印直方图分布信息 ---
    printf("=== 数据分布 (前30 bins 和后 56 bins) ===\n");
    printf("暗区域 (0-29):   ");
    long darkTotal = 0;
    for (int i = 0; i < 30; ++i) darkTotal += h_histCPU[i];
    printf("%ld 像素 (%.1f%%)\n", darkTotal, darkTotal * 100.0 / N);

    printf("中间区域 (100-179): ");
    long midTotal = 0;
    for (int i = 100; i < 180; ++i) midTotal += h_histCPU[i];
    printf("%ld 像素 (%.1f%%)\n", midTotal, midTotal * 100.0 / N);

    printf("高亮区域 (200-255): ");
    long brightTotal = 0;
    for (int i = 200; i < 256; ++i) brightTotal += h_histCPU[i];
    printf("%ld 像素 (%.1f%%)\n", brightTotal, brightTotal * 100.0 / N);

    // 清理
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_hist));
    free(h_data);

    printf("\n课后思考：\n");
    printf("  1. 为什么不均匀分布下 Global Atomic 更慢？\n");
    printf("     提示：暗区域 60%% 的像素都挤在 0-29 的 30 个 bin 里\n");
    printf("  2. Shared Memory Privatization 为什么能显著加速？\n");
    printf("     提示：Block 内只有 256 个线程竞争 vs 65536 个线程\n");
    printf("  3. 如果 bin 数从 256 变成 65536，Shared 版还能用吗？\n");
    printf("     提示：65536 × 4 = 256KB > Shared Memory 容量\n");
    printf("  4. atomicAdd 在 float 上和 int 上有什么区别？\n");
    printf("     提示：float 的 atomicAdd 用 CAS 循环实现，更慢\n");

    return 0;
}
