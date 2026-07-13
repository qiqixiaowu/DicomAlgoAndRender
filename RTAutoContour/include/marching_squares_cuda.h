#pragma once
/**
 * ============================================================
 *  GPU Marching Squares — CUDA 加速的 Mask→轮廓提取
 * ============================================================
 *
 * 算法原理
 * ────────
 * 将掩膜视为 (W-1)×(H-1) 个 2×2 单元格（cell），每个 cell 有
 * 4 个角的前景/背景值，组成 4 位索引 (0~15)。
 * 查表得到该 cell 产生的 0~2 条边缘线段（端点在 cell 边上，
 * 通过线性插值获得亚像素位置）。
 *
 * GPU 阶段：
 *   每个线程处理一个 cell → 写出线段到全局数组（原子计数器）
 *   复杂度 O(1)/cell，全并行
 *
 * CPU 阶段：
 *   将无序线段集合链接为有序闭合轮廓
 *   使用端点哈希快速拼接，复杂度 O(N_segments)
 *
 * 性能参考（RTX 3060, 512×512 mask）：
 *   GPU kernel: ~0.1 ms
 *   CPU chain:  ~0.3 ms
 *   总计:       ~0.5 ms（vs CPU Moore ~2 ms）
 *
 * ============================================================
 */

#include <vector>
#include <cstdint>

namespace rtac {

/// 二维浮点线段
struct Segment2f {
    float x0, y0;   // 起点
    float x1, y1;   // 终点
};

/// 有序轮廓（闭合点序列）
struct OrderedContour {
    std::vector<float> xs;
    std::vector<float> ys;
};

/**
 * @brief GPU Marching Squares: 从二值掩膜提取有序轮廓
 *
 * @param mask       二值掩膜数据（host 指针），行优先，线性排列
 *                   mask[y * width + x]，非零 = 前景
 * @param width      掩膜宽度
 * @param height     掩膜高度
 * @param contours   [out] 提取到的有序闭合轮廓列表
 * @return true      成功
 * @return false     CUDA 不可用或参数非法
 */
bool MarchingSquaresCUDA(
    const uint8_t*             mask,
    int                        width,
    int                        height,
    std::vector<OrderedContour>& contours);

/**
 * @brief 将无序线段集合链接为有序闭合轮廓（纯 CPU）
 *
 * 此函数可独立使用，也被 MarchingSquaresCUDA 内部调用。
 *
 * @param segments   无序线段数组
 * @param numSegs    线段数量
 * @param contours   [out] 有序轮廓
 */
void ChainSegmentsToContours(
    const Segment2f*           segments,
    int                        numSegs,
    std::vector<OrderedContour>& contours);

} // namespace rtac
