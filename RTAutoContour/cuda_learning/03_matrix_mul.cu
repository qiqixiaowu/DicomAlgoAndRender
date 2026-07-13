/**
 * ============================================================
 *  CUDA 第三课: 矩阵乘法 — Shared Memory & Tiling
 * ============================================================
 *
 * 学习目标：
 *   1. 理解 Shared Memory 的作用和使用方式
 *   2. 掌握 Tiling（分块）技术 —— CUDA 最核心的优化手段
 *   3. 理解 __syncthreads() 的必要性
 *   4. 体会 Naive vs Tiled 的巨大性能差异
 *   5. 了解 Bank Conflict 的概念
 *
 * 面试必考：
 *   - Shared Memory vs Global Memory 的延迟差异
 *   - 为什么 Tiling 能减少 Global Memory 访问次数
 *   - __syncthreads() 放错位置会怎样
 *   - Bank Conflict 是什么，怎么避免
 *
 * 编译:
 *   nvcc 03_matrix_mul.cu -o 03_matrix_mul.exe
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

// 矩阵大小 (方阵简化问题)
#define M_SIZE 1024
#define TILE_SIZE 16  // 也是 Block 大小


// ═══════════════════════════════════════════════════════════════
//  Part 1: Naive 矩阵乘法 (不用 Shared Memory)
// ═══════════════════════════════════════════════════════════════
//
//  C[row][col] = sum_k( A[row][k] * B[k][col] )
//
//  每个线程计算 C 的一个元素，需要读取 A 的一整行和 B 的一整列。
//
//  ┌─────┐     ┌─────┐     ┌─────┐
//  │  A  │  ×  │  B  │  =  │  C  │
//  │ M×K │     │ K×N │     │ M×N │
//  └─────┘     └─────┘     └─────┘
//
//  Global Memory 访问次数分析（对于 C 的一个元素）：
//    读 A 的一行: K 次
//    读 B 的一列: K 次
//    写 C 的一个元素: 1 次
//    总计: 2K + 1 次 Global Memory 访问
//
//  对于整个矩阵：M*N 个元素 × 2K 次 = 2MNK 次
//  当 M=N=K=1024 时：2 × 1024³ ≈ 21 亿次！慢得要死。
//

__global__ void matMulNaive(const float* A, const float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
            //      ↑ 读 A 的第 row 行   ↑ 读 B 的第 col 列
        }
        C[row * N + col] = sum;
    }
}

// 问题在哪？看 B[k * N + col]:
//   线程 (row, 0) 读 B[k*N + 0]
//   线程 (row, 1) 读 B[k*N + 1]
//   ...
//   → 同一 Warp 中的线程读 B 的同一行的连续元素 ✓ (合并访存)
//     但读 A 时，它们都读 A[row*N + k]，同一个地址 → 广播 ✓
//
// 真正的问题是：同一个 B[k][col] 被不同 row 的线程重复读了 M 次！
// A[row][k] 被不同 col 的线程重复读了 N 次！
// → 数据复用率极低


// ═══════════════════════════════════════════════════════════════
//  Part 2: Tiled 矩阵乘法 (用 Shared Memory)
// ═══════════════════════════════════════════════════════════════
//
//  核心思想：把矩阵分成 TILE×TILE 的小块，一次加载一块到 Shared Memory，
//  Block 内所有线程共享这块数据，反复使用。
//
//  ┌──┬──┬──┐     ┌──┬──┬──┐
//  │T0│T1│T2│     │T0│T1│T2│     每个 Tile: TILE_SIZE × TILE_SIZE
//  ├──┼──┼──┤  ×  ├──┼──┼──┤
//  │T3│T4│T5│     │T3│T4│T5│     Block 负责计算 C 的一个 Tile
//  ├──┼──┼──┤     ├──┼──┼──┤     需要 A 的一行 Tiles × B 的一列 Tiles
//  │T6│T7│T8│     │T6│T7│T8│
//  └──┴──┴──┘     └──┴──┴──┘
//
//  步骤：
//  for each tile_k in [0, N/TILE):
//    1. 每个线程加载 A 和 B 各一个元素到 Shared Memory
//    2. __syncthreads()  ← 确保 Tile 加载完毕
//    3. 每个线程用 Shared Memory 中的数据计算部分和
//    4. __syncthreads()  ← 确保计算完毕再加载下一个 Tile
//
//  Global Memory 访问次数：
//    每个 Tile 加载: Block 中每个线程读 A 和 B 各 1 次
//    共 N/TILE 个 Tile: 2 × N/TILE 次 Global Memory 读取
//    当 TILE=16, N=1024: 2 × 64 = 128 次
//    对比 Naive: 2 × 1024 = 2048 次
//    节省了 TILE_SIZE 倍！！
//

__global__ void matMulTiled(const float* A, const float* B, float* C, int N)
{
    // Shared Memory 声明（只在 Block 内可见）
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // 遍历所有 Tile
    for (int t = 0; t < N / TILE_SIZE; ++t) {

        // Step 1: 协作加载 —— 每个线程加载一个元素到 Shared Memory
        tileA[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE_SIZE + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];

        // Step 2: 同步 —— 等所有线程都加载完毕
        // 如果不同步：有些线程可能已经开始用还没被其他线程加载的数据！
        __syncthreads();

        // Step 3: 计算部分和 —— 完全在 Shared Memory 上操作（快！）
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Step 4: 再次同步 —— 确保所有线程用完了当前 Tile 数据
        // 如果不同步：有些线程可能开始加载下一个 Tile，覆盖了还在用的数据！
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}


// ═══════════════════════════════════════════════════════════════
//  Part 3: Bank Conflict 讲解
// ═══════════════════════════════════════════════════════════════
//
//  Shared Memory 被划分为 32 个 Bank（跟 Warp 大小一样）
//  地址 i 属于 Bank (i / 4) % 32
//
//  float tileA[16][16]:
//    tileA[row][0] → Bank 0
//    tileA[row][1] → Bank 1
//    tileA[row][2] → Bank 2
//    ...
//    tileA[row][15] → Bank 15
//
//  当同一 Warp 的 32 个线程同时访问:
//    tileA[threadIdx.y][k]  (k 相同, threadIdx.y 不同)
//    → 每个线程访问不同 row 的同一列
//    → 地址 = (threadIdx.y * 16 + k) * 4
//    → Bank = (threadIdx.y * 16 + k) % 32
//    → 因为 16 是偶数，可能产生 2-way bank conflict
//
//  解决方案: Padding！
//    float tileA[TILE_SIZE][TILE_SIZE + 1];  // 加一列！
//    → 每行 17 个 float，错开了 Bank 对齐
//    → 消除 conflict
//
// （本例 TILE=16 影响不大，当 TILE=32 时 bank conflict 很严重）


// ═══════════════════════════════════════════════════════════════
//  CPU 参考实现
// ═══════════════════════════════════════════════════════════════

void matMulCPU(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
        float sum = 0;
        for (int k = 0; k < N; ++k)
            sum += A[i * N + k] * B[k * N + j];
        C[i * N + j] = sum;
    }
}


// ═══════════════════════════════════════════════════════════════
//  性能对比
// ═══════════════════════════════════════════════════════════════

int main()
{
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  CUDA 第三课: 矩阵乘法 & Shared Memory  ║\n");
    printf("╚══════════════════════════════════════════╝\n");

    const int N = M_SIZE;
    size_t bytes = N * N * sizeof(float);
    printf("矩阵大小: %dx%d (%.1f MB)\n\n", N, N, bytes / 1e6);

    // 分配
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C_naive = (float*)malloc(bytes);
    float* h_C_tiled = (float*)malloc(bytes);

    srand(42);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
        h_B[i] = (float)(rand() % 100) / 100.0f;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms;

    // --- Naive ---
    CUDA_CHECK(cudaEventRecord(start));
    matMulNaive<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, bytes, cudaMemcpyDeviceToHost));

    // GFLOPS = 2*N^3 / time / 1e9
    double gflops_naive = 2.0 * N * N * N / (ms / 1000.0) / 1e9;
    printf("[Naive]  耗时: %7.2f ms  |  %.1f GFLOPS\n", ms, gflops_naive);

    // --- Tiled ---
    CUDA_CHECK(cudaEventRecord(start));
    matMulTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, bytes, cudaMemcpyDeviceToHost));

    double gflops_tiled = 2.0 * N * N * N / (ms / 1000.0) / 1e9;
    printf("[Tiled]  耗时: %7.2f ms  |  %.1f GFLOPS\n", ms, gflops_tiled);
    printf("[加速比]  %.1fx\n\n", gflops_tiled / gflops_naive);

    // --- 验证正确性 ---
    int errors = 0;
    for (int i = 0; i < N * N; ++i) {
        if (fabsf(h_C_naive[i] - h_C_tiled[i]) > 0.01f) {
            errors++;
            if (errors <= 3)
                printf("  差异 [%d]: naive=%.4f tiled=%.4f\n",
                       i, h_C_naive[i], h_C_tiled[i]);
        }
    }
    printf("Naive vs Tiled 一致性: %s (%d 差异)\n",
           errors == 0 ? "PASS" : "FAIL", errors);

    // 清理
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C_naive); free(h_C_tiled);

    printf("\n课后思考：\n");
    printf("  1. Tiled 版快了多少倍？理论上限是 TILE_SIZE=%d 倍，实际接近吗？\n", TILE_SIZE);
    printf("  2. 如果把两个 __syncthreads() 删掉，结果会怎样？\n");
    printf("  3. 把 TILE_SIZE 改成 32 试试，性能有变化吗？\n");
    printf("  4. 声明 tileA[TILE_SIZE][TILE_SIZE+1] (padding) 对性能有影响吗？\n");

    return 0;
}
