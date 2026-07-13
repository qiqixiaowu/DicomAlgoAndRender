/**
 * ============================================================
 *  CUDA 第六课: CUDA Streams & 异步操作
 * ============================================================
 *
 * 学习目标：
 *   1. 理解 CUDA Stream 的概念 —— GPU 任务队列
 *   2. 掌握异步 memcpy (cudaMemcpyAsync) 的使用
 *   3. 理解如何用多 Stream 实现 "计算-传输" 重叠
 *   4. 学习 CUDA Event 在跨 Stream 同步中的作用
 *   5. 了解 Pipeline 模式 —— 最大化 GPU 利用率
 *
 * 面试必考：
 *   - Default Stream 和 Non-default Stream 的区别
 *   - 为什么异步 memcpy 需要 Pinned Memory
 *   - 如何判断计算和传输能否重叠
 *   - cudaStreamSynchronize vs cudaDeviceSynchronize
 *
 * 编译:
 *   nvcc 06_streams.cu -o 06_streams.exe
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
//  什么是 CUDA Stream？
// ═══════════════════════════════════════════════════════════════
//
//  Stream = GPU 上的一个任务队列。
//  同一个 Stream 中的操作按顺序执行（FIFO）。
//  不同 Stream 中的操作可以并行执行！
//
//  没有 Stream 时 (Default Stream):
//    ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
//    │ H→D  ├→→┤Kernel├→→┤ D→H  ├→→┤Kernel├→→┤ D→H  │
//    └──────┘  └──────┘  └──────┘  └──────┘  └──────┘
//    ─────────────────────────────────────────────────→ 时间
//              全部串行，GPU 很多时候在等数据传输
//
//  有多个 Stream 时:
//    Stream 0: ┌──────┐     ┌──────┐     ┌──────┐
//              │ H→D  ├→→→→→┤Kernel├→→→→→┤ D→H  │
//              └──────┘     └──────┘     └──────┘
//    Stream 1:     ┌──────┐     ┌──────┐     ┌──────┐
//                  │ H→D  ├→→→→→┤Kernel├→→→→→┤ D→H  │
//                  └──────┘     └──────┘     └──────┘
//    Stream 2:         ┌──────┐     ┌──────┐     ┌──────┐
//                      │ H→D  ├→→→→→┤Kernel├→→→→→┤ D→H  │
//                      └──────┘     └──────┘     └──────┘
//    ─────────────────────────────────────────────────→ 时间
//              传输和计算重叠！总时间大幅缩短！
//
//  GPU 硬件支持:
//    - 1 个 Copy Engine (H→D)
//    - 1 个 Copy Engine (D→H)
//    - N 个 SM (计算)
//    → 可以同时做: 一个 H→D 传输 + 一个计算 + 一个 D→H 传输
//


// ═══════════════════════════════════════════════════════════════
//  重要前提：Pinned Memory (Page-locked Memory)
// ═══════════════════════════════════════════════════════════════
//
//  普通 malloc 分配的内存是 "Pageable" 的（可被操作系统换出到磁盘）。
//  CUDA 的 DMA 引擎不能直接访问 Pageable 内存。
//
//  所以 cudaMemcpy (同步版) 会：
//    1. 先把数据拷贝到一个临时 Pinned Buffer
//    2. 再从 Pinned Buffer DMA 到 GPU
//    → 两次拷贝！
//
//  cudaMallocHost 分配的内存是 "Pinned" 的（锁定在物理内存中）。
//  DMA 引擎可以直接访问，只需一次拷贝。
//
//  而且：cudaMemcpyAsync 必须使用 Pinned Memory 才能真正异步！
//  如果用 Pageable Memory + Async，实际上会退化为同步操作。
//


// ═══════════════════════════════════════════════════════════════
//  模拟计算任务：对向量每个元素做多次 sin/cos（人为增加计算量）
// ═══════════════════════════════════════════════════════════════

__global__ void heavyKernel(float* data, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float val = data[gid];
        // 做 100 次 sin/cos 模拟大量计算
        for (int j = 0; j < 100; ++j) {
            val = sinf(val) * cosf(val) + 0.001f;
        }
        data[gid] = val;
    }
}


// ═══════════════════════════════════════════════════════════════
//  Demo 1: 无 Stream (Default Stream，全部串行)
// ═══════════════════════════════════════════════════════════════

float demo_no_stream(float* h_data, int totalN)
{
    size_t bytes = totalN * sizeof(float);

    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // 全量传输 → 全量计算 → 全量回传
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (totalN + threads - 1) / threads;
    heavyKernel<<<blocks, threads>>>(d_data, totalN);

    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));

    return ms;
}


// ═══════════════════════════════════════════════════════════════
//  Demo 2: 多 Stream Pipeline (传输-计算重叠)
// ═══════════════════════════════════════════════════════════════

float demo_multi_stream(float* h_data, int totalN, int numStreams)
{
    size_t bytes = totalN * sizeof(float);
    int chunkN = totalN / numStreams;
    size_t chunkBytes = chunkN * sizeof(float);

    // 分配 GPU 内存
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    // 创建多个 Stream
    cudaStream_t* streams = new cudaStream_t[numStreams];
    for (int i = 0; i < numStreams; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    int threads = 256;

    for (int i = 0; i < numStreams; ++i) {
        int offset = i * chunkN;
        size_t offsetBytes = offset * sizeof(float);

        // 每个 Stream: 传入 → 计算 → 传出
        // 不同 Stream 之间可以重叠！

        // Step 1: 异步传入 (H → D)
        CUDA_CHECK(cudaMemcpyAsync(d_data + offset, h_data + offset,
                                    chunkBytes, cudaMemcpyHostToDevice,
                                    streams[i]));

        // Step 2: 在同一 Stream 中启动 Kernel (会等 Step 1 完成)
        int blocks = (chunkN + threads - 1) / threads;
        heavyKernel<<<blocks, threads, 0, streams[i]>>>(
            d_data + offset, chunkN);

        // Step 3: 异步传出 (D → H) (会等 Step 2 完成)
        CUDA_CHECK(cudaMemcpyAsync(h_data + offset, d_data + offset,
                                    chunkBytes, cudaMemcpyDeviceToHost,
                                    streams[i]));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // 清理
    for (int i = 0; i < numStreams; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    delete[] streams;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));

    return ms;
}


// ═══════════════════════════════════════════════════════════════
//  Demo 3: 展示 Event 跨 Stream 同步
// ═══════════════════════════════════════════════════════════════
//
//  场景：Stream A 做计算，Stream B 需要用 Stream A 的结果
//
//  cudaEventRecord(event, streamA)  → 在 Stream A 中记录事件
//  cudaStreamWaitEvent(streamB, event) → Stream B 等待该事件完成
//
//  这比 cudaDeviceSynchronize() 更细粒度 —— 只阻塞相关的 Stream
//

float demo_event_sync(float* h_data, int totalN)
{
    int halfN = totalN / 2;
    size_t halfBytes = halfN * sizeof(float);

    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, halfBytes));
    CUDA_CHECK(cudaMalloc(&d_B, halfBytes));

    cudaStream_t streamA, streamB;
    CUDA_CHECK(cudaStreamCreate(&streamA));
    CUDA_CHECK(cudaStreamCreate(&streamB));

    cudaEvent_t eventA, startTimer, stopTimer;
    CUDA_CHECK(cudaEventCreate(&eventA));
    CUDA_CHECK(cudaEventCreate(&startTimer));
    CUDA_CHECK(cudaEventCreate(&stopTimer));

    CUDA_CHECK(cudaEventRecord(startTimer));

    int threads = 256;
    int blocks = (halfN + threads - 1) / threads;

    // Stream A: 传入 + 计算
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_data, halfBytes,
                                cudaMemcpyHostToDevice, streamA));
    heavyKernel<<<blocks, threads, 0, streamA>>>(d_A, halfN);
    CUDA_CHECK(cudaEventRecord(eventA, streamA));  // ← 记录完成事件

    // Stream B: 传入自己的数据（与 Stream A 并行）
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_data + halfN, halfBytes,
                                cudaMemcpyHostToDevice, streamB));

    // Stream B: 等待 Stream A 完成（跨 Stream 依赖）
    CUDA_CHECK(cudaStreamWaitEvent(streamB, eventA, 0));

    // Stream B: 继续计算
    heavyKernel<<<blocks, threads, 0, streamB>>>(d_B, halfN);

    // 回传
    CUDA_CHECK(cudaMemcpyAsync(h_data, d_A, halfBytes,
                                cudaMemcpyDeviceToHost, streamA));
    CUDA_CHECK(cudaMemcpyAsync(h_data + halfN, d_B, halfBytes,
                                cudaMemcpyDeviceToHost, streamB));

    CUDA_CHECK(cudaEventRecord(stopTimer));
    CUDA_CHECK(cudaEventSynchronize(stopTimer));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, startTimer, stopTimer));

    CUDA_CHECK(cudaStreamDestroy(streamA));
    CUDA_CHECK(cudaStreamDestroy(streamB));
    CUDA_CHECK(cudaEventDestroy(eventA));
    CUDA_CHECK(cudaEventDestroy(startTimer));
    CUDA_CHECK(cudaEventDestroy(stopTimer));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    return ms;
}


// ═══════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════

int main()
{
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  CUDA 第六课: Streams & 异步操作         ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    // 查询设备能力
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("Async Engine Count: %d (支持 %d 个方向的并行DMA传输)\n",
           prop.asyncEngineCount, prop.asyncEngineCount);
    printf("Concurrent Kernels: %s\n\n",
           prop.concurrentKernels ? "Yes" : "No");

    const int totalN = 1 << 22;  // 4M 元素
    size_t bytes = totalN * sizeof(float);
    printf("数据量: %d 元素 (%.1f MB)\n\n", totalN, bytes / 1e6);

    // 使用 Pinned Memory（关键！异步传输的前提）
    float* h_data;
    CUDA_CHECK(cudaMallocHost(&h_data, bytes));  // ← Pinned Memory

    srand(42);
    for (int i = 0; i < totalN; ++i)
        h_data[i] = (float)(rand() % 1000) / 1000.0f;

    // ---- 测试 1: 无 Stream ----
    printf("=== Demo 1: Default Stream (全部串行) ===\n");
    float ms_none = demo_no_stream(h_data, totalN);
    printf("  耗时: %.2f ms\n\n", ms_none);

    // 重新初始化
    for (int i = 0; i < totalN; ++i)
        h_data[i] = (float)(rand() % 1000) / 1000.0f;

    // ---- 测试 2: 多 Stream ----
    int streamCounts[] = {2, 4, 8, 16};
    printf("=== Demo 2: Multi-Stream Pipeline ===\n");
    for (int ns : streamCounts) {
        // 重新初始化
        for (int i = 0; i < totalN; ++i)
            h_data[i] = (float)(rand() % 1000) / 1000.0f;

        float ms = demo_multi_stream(h_data, totalN, ns);
        printf("  %2d Streams: %.2f ms (加速 %.2fx)\n",
               ns, ms, ms_none / ms);
    }
    printf("\n");

    // ---- 测试 3: Event 跨 Stream 同步 ----
    for (int i = 0; i < totalN; ++i)
        h_data[i] = (float)(rand() % 1000) / 1000.0f;

    printf("=== Demo 3: Event 跨 Stream 同步 ===\n");
    float ms_event = demo_event_sync(h_data, totalN);
    printf("  耗时: %.2f ms\n\n", ms_event);

    // ---- 测试 4: Pinned vs Pageable Memory 带宽对比 ----
    printf("=== Demo 4: Pinned vs Pageable Memory 传输带宽 ===\n");
    {
        float* h_page = (float*)malloc(bytes);    // Pageable
        float* h_pin;                              // Pinned (已有 h_data)
        CUDA_CHECK(cudaMallocHost(&h_pin, bytes));
        float* d_buf;
        CUDA_CHECK(cudaMalloc(&d_buf, bytes));

        for (int i = 0; i < totalN; ++i) {
            h_page[i] = 1.0f;
            h_pin[i] = 1.0f;
        }

        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));

        // Pageable H→D
        CUDA_CHECK(cudaEventRecord(t0));
        CUDA_CHECK(cudaMemcpy(d_buf, h_page, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms_page;
        CUDA_CHECK(cudaEventElapsedTime(&ms_page, t0, t1));

        // Pinned H→D
        CUDA_CHECK(cudaEventRecord(t0));
        CUDA_CHECK(cudaMemcpy(d_buf, h_pin, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms_pin;
        CUDA_CHECK(cudaEventElapsedTime(&ms_pin, t0, t1));

        printf("  Pageable: %.2f ms (%.1f GB/s)\n",
               ms_page, bytes / ms_page / 1e6);
        printf("  Pinned:   %.2f ms (%.1f GB/s)\n",
               ms_pin, bytes / ms_pin / 1e6);
        printf("  加速: %.2fx\n\n", ms_page / ms_pin);

        CUDA_CHECK(cudaEventDestroy(t0));
        CUDA_CHECK(cudaEventDestroy(t1));
        CUDA_CHECK(cudaFree(d_buf));
        CUDA_CHECK(cudaFreeHost(h_pin));
        free(h_page);
    }

    // 清理
    CUDA_CHECK(cudaFreeHost(h_data));

    printf("课后思考：\n");
    printf("  1. 为什么 Stream 数从 2→4 有加速，但 8→16 可能没有？\n");
    printf("     提示：GPU 的 Copy Engine 数量有限\n");
    printf("  2. 如果 Kernel 执行时间远大于传输时间，Stream 还有加速效果吗？\n");
    printf("     提示：传输时间被计算完全隐藏\n");
    printf("  3. cudaStreamWaitEvent 和 cudaDeviceSynchronize 有什么区别？\n");
    printf("     提示：粒度不同，一个阻塞一个 Stream，一个阻塞所有\n");
    printf("  4. 如果用 Pageable Memory 调 cudaMemcpyAsync，会怎样？\n");
    printf("     提示：驱动会退化为同步操作，失去重叠效果\n");
    printf("  5. 如何用 nsys/nvprof 可视化 Stream 的并行执行？\n");
    printf("     提示：nsys profile ./06_streams.exe，然后用 nsys-ui 打开\n");

    return 0;
}
