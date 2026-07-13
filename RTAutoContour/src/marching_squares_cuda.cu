/**
 * @file marching_squares_cuda.cu
 * @brief GPU Marching Squares 实现 + CPU 线段链接
 *
 * ══════════════════════════════════════════════════════════════
 *  Marching Squares 核心原理
 * ══════════════════════════════════════════════════════════════
 *
 * 每个 2×2 cell 的 4 个角（左上 a, 右上 b, 右下 c, 左下 d）各自
 * 是前景(1)或背景(0)，组合出 4 位索引 (0~15)：
 *
 *     a ── b         caseIdx = a | (b<<1) | (c<<2) | (d<<3)
 *     |    |
 *     d ── c
 *
 * 16 种 case 中，0 和 15 不产生线段（全背景/全前景），
 * 其余 14 种各产生 1~2 条线段。
 *
 * 线段端点在 cell 的 4 条边上，通过插值获得亚像素坐标。
 * 对于二值掩膜，插值退化为中点（0.5 偏移）。
 *
 * 4 条边编号：
 *     0: 上边 (a─b)
 *     1: 右边 (b─c)
 *     2: 下边 (c─d)
 *     3: 左边 (a─d)
 *
 * ══════════════════════════════════════════════════════════════
 */

#include "marching_squares_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <deque>
#include <cmath>

namespace rtac {

// ═══════════════════════════════════════════════════════════════
//  查找表：16 种 case → 线段端点（边编号对）
//  每种 case 最多 2 条线段，每条线段用 (edge0, edge1) 表示
//  -1 = 不产生线段
// ═══════════════════════════════════════════════════════════════

struct CaseEntry {
    int seg[2][2];  // seg[i] = {edge0, edge1}，-1 表示无
};

//  a=bit0(左上), b=bit1(右上), c=bit2(右下), d=bit3(左下)
//
//  边编号:  0=上(a-b), 1=右(b-c), 2=下(d-c), 3=左(a-d)
//
static __constant__ int d_caseTable[16][4] = {
    // case 0:  □□□□ → 无线段
    {-1, -1, -1, -1},
    // case 1:  a=1 → 线段 (上,左)
    { 0,  3, -1, -1},
    // case 2:  b=1 → 线段 (上,右)
    { 0,  1, -1, -1},
    // case 3:  a,b=1 → 线段 (右,左)
    { 1,  3, -1, -1},
    // case 4:  c=1 → 线段 (右,下)
    { 1,  2, -1, -1},
    // case 5:  a,c=1（对角）→ 两条线段：(上,左),(右,下)   [歧义情况，选择一种]
    { 0,  3,  1,  2},
    // case 6:  b,c=1 → 线段 (上,下)
    { 0,  2, -1, -1},
    // case 7:  a,b,c=1 → 线段 (下,左)
    { 2,  3, -1, -1},
    // case 8:  d=1 → 线段 (下,左)
    { 2,  3, -1, -1},
    // case 9:  a,d=1 → 线段 (上,下)
    { 0,  2, -1, -1},
    // case 10: b,d=1（对角）→ 两条线段：(上,右),(下,左)  [歧义情况]
    { 0,  1,  2,  3},
    // case 11: a,b,d=1 → 线段 (右,下)
    { 1,  2, -1, -1},
    // case 12: c,d=1 → 线段 (右,左)
    { 1,  3, -1, -1},
    // case 13: a,c,d=1 → 线段 (上,右)
    { 0,  1, -1, -1},
    // case 14: b,c,d=1 → 线段 (上,左)
    { 0,  3, -1, -1},
    // case 15: ■■■■ → 无线段
    {-1, -1, -1, -1},
};

// host 侧同份查找表（供 CPU fallback）
static const int h_caseTable[16][4] = {
    {-1, -1, -1, -1},
    { 0,  3, -1, -1},
    { 0,  1, -1, -1},
    { 1,  3, -1, -1},
    { 1,  2, -1, -1},
    { 0,  3,  1,  2},
    { 0,  2, -1, -1},
    { 2,  3, -1, -1},
    { 2,  3, -1, -1},
    { 0,  2, -1, -1},
    { 0,  1,  2,  3},
    { 1,  2, -1, -1},
    { 1,  3, -1, -1},
    { 0,  1, -1, -1},
    { 0,  3, -1, -1},
    {-1, -1, -1, -1},
};

// ═══════════════════════════════════════════════════════════════
//  边编号 → 端点坐标（cell 左上角为原点，cell 大小 1×1）
//  二值 mask 时插值 = 边中点（0.5 偏移）
// ═══════════════════════════════════════════════════════════════

__device__ __host__
inline void EdgeMidpoint(int edge, float cellX, float cellY, float& px, float& py)
{
    switch (edge) {
    case 0: px = cellX + 0.5f; py = cellY;        break; // 上边中点
    case 1: px = cellX + 1.0f; py = cellY + 0.5f; break; // 右边中点
    case 2: px = cellX + 0.5f; py = cellY + 1.0f; break; // 下边中点
    case 3: px = cellX;        py = cellY + 0.5f; break; // 左边中点
    }
}

// ═══════════════════════════════════════════════════════════════
//  CUDA Kernel：每个线程处理一个 cell
// ═══════════════════════════════════════════════════════════════

__global__ void MarchingSquaresKernel(
    const uint8_t* __restrict__ mask,
    int W, int H,
    Segment2f* segments,
    int* segCount,
    int maxSegs)
{
    int cellX = blockIdx.x * blockDim.x + threadIdx.x;
    int cellY = blockIdx.y * blockDim.y + threadIdx.y;
    
    // cell 范围: [0, W-2] × [0, H-2]
    if (cellX >= W - 1 || cellY >= H - 1) return;

    // 读取 4 个角
    int a = (mask[cellY       * W + cellX]     != 0) ? 1 : 0;
    int b = (mask[cellY       * W + cellX + 1] != 0) ? 1 : 0;
    int c = (mask[(cellY + 1) * W + cellX + 1] != 0) ? 1 : 0;
    int d = (mask[(cellY + 1) * W + cellX]     != 0) ? 1 : 0;

    int caseIdx = a | (b << 1) | (c << 2) | (d << 3);

    // 无线段：case 0 和 15
    if (caseIdx == 0 || caseIdx == 15) return;

    float cx = static_cast<float>(cellX);
    float cy = static_cast<float>(cellY);

    // 第一条线段
    int e0 = d_caseTable[caseIdx][0];
    int e1 = d_caseTable[caseIdx][1];
    if (e0 >= 0 && e1 >= 0) {
        int idx = atomicAdd(segCount, 1);
        if (idx < maxSegs) {
            EdgeMidpoint(e0, cx, cy, segments[idx].x0, segments[idx].y0);
            EdgeMidpoint(e1, cx, cy, segments[idx].x1, segments[idx].y1);
        }
    }

    // 第二条线段（仅 case 5 和 10 有）
    int e2 = d_caseTable[caseIdx][2];
    int e3 = d_caseTable[caseIdx][3];
    if (e2 >= 0 && e3 >= 0) {
        int idx = atomicAdd(segCount, 1);
        if (idx < maxSegs) {
            EdgeMidpoint(e2, cx, cy, segments[idx].x0, segments[idx].y0);
            EdgeMidpoint(e3, cx, cy, segments[idx].x1, segments[idx].y1);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  CPU：将无序线段链接为有序闭合轮廓
// ═══════════════════════════════════════════════════════════════
//
//  原理：
//    每条线段有两个端点，用哈希表把端点相同的线段连起来。
//    因为 Marching Squares 的线段端点都在网格半整数位上
//    (x.0/x.5, y.0/y.5)，所以用整数编码做精确匹配。
//
//  端点编码：将坐标 ×2 取整，得到唯一整数键
//    key = (int)(y*2) * (W*2+1) + (int)(x*2)
// ═══════════════════════════════════════════════════════════════

void ChainSegmentsToContours(
    const Segment2f*           segments,
    int                        numSegs,
    std::vector<OrderedContour>& contours)
{
    contours.clear();
    if (numSegs == 0) return;

    // 找最大坐标范围，用于编码
    float maxCoord = 0;
    for (int i = 0; i < numSegs; ++i) {
        maxCoord = std::max(maxCoord, std::max(segments[i].x0, segments[i].x1));
        maxCoord = std::max(maxCoord, std::max(segments[i].y0, segments[i].y1));
    }
    int stride = static_cast<int>(maxCoord * 2) + 10;

    auto encodeKey = [stride](float x, float y) -> int64_t {
        int ix = static_cast<int>(std::round(x * 2.0f));
        int iy = static_cast<int>(std::round(y * 2.0f));
        return static_cast<int64_t>(iy) * stride + ix;
    };

    // 邻接表：endpoint_key → [(seg_index, which_end)]
    // which_end: 0 = 用 (x0,y0) 端, 1 = 用 (x1,y1) 端
    struct AdjEntry { int segIdx; int endIdx; };
    std::unordered_multimap<int64_t, AdjEntry> adj;
    adj.reserve(numSegs * 2);

    for (int i = 0; i < numSegs; ++i) {
        int64_t k0 = encodeKey(segments[i].x0, segments[i].y0);
        int64_t k1 = encodeKey(segments[i].x1, segments[i].y1);
        adj.insert({k0, {i, 0}});
        adj.insert({k1, {i, 1}});
    }

    std::vector<bool> used(numSegs, false);

    for (int i = 0; i < numSegs; ++i) {
        if (used[i]) continue;

        // 从线段 i 开始，向两端延伸
        std::deque<std::pair<float, float>> chain;

        // 放入第一条线段
        used[i] = true;
        chain.push_back({segments[i].x0, segments[i].y0});
        chain.push_back({segments[i].x1, segments[i].y1});

        // 向后端延伸
        while (true) {
            auto& [tailX, tailY] = chain.back();
            int64_t key = encodeKey(tailX, tailY);
            bool extended = false;
            auto range = adj.equal_range(key);
            for (auto it = range.first; it != range.second; ++it) {
                int si = it->second.segIdx;
                if (used[si]) continue;
                used[si] = true;
                // 找到该线段的"另一端"
                float nx, ny;
                if (it->second.endIdx == 0) {
                    nx = segments[si].x1; ny = segments[si].y1;
                } else {
                    nx = segments[si].x0; ny = segments[si].y0;
                }
                chain.push_back({nx, ny});
                extended = true;
                break;
            }
            if (!extended) break;
        }

        // 向前端延伸
        while (true) {
            auto& [headX, headY] = chain.front();
            int64_t key = encodeKey(headX, headY);
            bool extended = false;
            auto range = adj.equal_range(key);
            for (auto it = range.first; it != range.second; ++it) {
                int si = it->second.segIdx;
                if (used[si]) continue;
                used[si] = true;
                float nx, ny;
                if (it->second.endIdx == 0) {
                    nx = segments[si].x1; ny = segments[si].y1;
                } else {
                    nx = segments[si].x0; ny = segments[si].y0;
                }
                chain.push_front({nx, ny});
                extended = true;
                break;
            }
            if (!extended) break;
        }

        // 至少 3 个点才构成有意义的轮廓
        if (chain.size() >= 3) {
            OrderedContour oc;
            oc.xs.reserve(chain.size());
            oc.ys.reserve(chain.size());
            for (auto& [px, py] : chain) {
                oc.xs.push_back(px);
                oc.ys.push_back(py);
            }
            contours.push_back(std::move(oc));
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  CPU Fallback：当 CUDA 不可用时的纯 CPU 实现
// ═══════════════════════════════════════════════════════════════

static void MarchingSquaresCPU(
    const uint8_t* mask, int W, int H,
    std::vector<Segment2f>& outSegs)
{
    outSegs.clear();
    outSegs.reserve(W + H); // 粗估

    for (int cy = 0; cy < H - 1; ++cy)
    for (int cx = 0; cx < W - 1; ++cx) {
        int a = (mask[cy       * W + cx]     != 0) ? 1 : 0;
        int b = (mask[cy       * W + cx + 1] != 0) ? 1 : 0;
        int c = (mask[(cy + 1) * W + cx + 1] != 0) ? 1 : 0;
        int d = (mask[(cy + 1) * W + cx]     != 0) ? 1 : 0;
        int caseIdx = a | (b << 1) | (c << 2) | (d << 3);
        if (caseIdx == 0 || caseIdx == 15) continue;

        float fcx = static_cast<float>(cx);
        float fcy = static_cast<float>(cy);

        int e0 = h_caseTable[caseIdx][0];
        int e1 = h_caseTable[caseIdx][1];
        if (e0 >= 0 && e1 >= 0) {
            Segment2f s;
            EdgeMidpoint(e0, fcx, fcy, s.x0, s.y0);
            EdgeMidpoint(e1, fcx, fcy, s.x1, s.y1);
            outSegs.push_back(s);
        }
        int e2 = h_caseTable[caseIdx][2];
        int e3 = h_caseTable[caseIdx][3];
        if (e2 >= 0 && e3 >= 0) {
            Segment2f s;
            EdgeMidpoint(e2, fcx, fcy, s.x0, s.y0);
            EdgeMidpoint(e3, fcx, fcy, s.x1, s.y1);
            outSegs.push_back(s);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  公开接口
// ═══════════════════════════════════════════════════════════════

bool MarchingSquaresCUDA(
    const uint8_t*             mask,
    int                        width,
    int                        height,
    std::vector<OrderedContour>& contours)
{
    contours.clear();
    if (!mask || width < 2 || height < 2) return false;

    // 尝试 CUDA 路径
    int devCount = 0;
    cudaError_t err = cudaGetDeviceCount(&devCount);
    if (err != cudaSuccess || devCount == 0) {
        // CUDA 不可用 → CPU fallback
        std::vector<Segment2f> segs;
        MarchingSquaresCPU(mask, width, height, segs);
        ChainSegmentsToContours(segs.data(), static_cast<int>(segs.size()), contours);
        return true;
    }

    size_t maskSize = static_cast<size_t>(width) * height;
    // 最大线段数 = cell数 × 2（每个cell最多2条）
    int maxSegs = (width - 1) * (height - 1) * 2;

    // ── 分配 GPU 内存 ──────────────────────────────
    uint8_t*   d_mask     = nullptr;
    Segment2f* d_segments = nullptr;
    int*       d_count    = nullptr;

    cudaMalloc(&d_mask,     maskSize * sizeof(uint8_t));
    cudaMalloc(&d_segments, maxSegs * sizeof(Segment2f));
    cudaMalloc(&d_count,    sizeof(int));

    cudaMemcpy(d_mask, mask, maskSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemset(d_count, 0, sizeof(int));

    // ── 启动 kernel ───────────────────────────────
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width  - 1 + blockSize.x - 1) / blockSize.x,
        (height - 1 + blockSize.y - 1) / blockSize.y);

    MarchingSquaresKernel<<<gridSize, blockSize>>>(
        d_mask, width, height, d_segments, d_count, maxSegs);

    cudaDeviceSynchronize();

    // ── 读回结果 ──────────────────────────────────
    int hostCount = 0;
    cudaMemcpy(&hostCount, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    hostCount = std::min(hostCount, maxSegs);

    std::vector<Segment2f> hostSegs(hostCount);
    if (hostCount > 0) {
        cudaMemcpy(hostSegs.data(), d_segments,
                   hostCount * sizeof(Segment2f), cudaMemcpyDeviceToHost);
    }

    // ── 释放 GPU 内存 ─────────────────────────────
    cudaFree(d_mask);
    cudaFree(d_segments);
    cudaFree(d_count);

    // ── CPU 链接排序 ──────────────────────────────
    ChainSegmentsToContours(hostSegs.data(), hostCount, contours);

    return true;
}

} // namespace rtac
