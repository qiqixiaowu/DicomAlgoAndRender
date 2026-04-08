#pragma once
/**
 * @file pet_lesion_segment.h
 * @brief PET / 分子影像病灶分割接口
 *
 * 提供三种 SUVbw 阈值分割模式：
 *
 * ┌──────────────┬────────────────────────────────────────────┐
 * │  Fixed       │ 用户直接指定绝对 SUVbw 阈值                  │
 * │  Percent     │ 用户指定最大值百分比（如 42%SUVmax）          │
 * │  Adaptive    │ 自动调整权重，平衡灵敏度/特异性               │
 * └──────────────┴────────────────────────────────────────────┘
 *
 * 每种模式支持两个维度的变体：
 *   - SeedPoint 版本：用户点击单个种子点
 *   - WithVOI   版本：在预定义感兴趣区域内分割（支持多病灶）
 *
 * 另有 Hover 实时预览（鼠标跟随）模式，用于前端交互反馈。
 *
 * 算法原理
 * ─────────
 * 1. 从种子点出发，在邻域内找最大 SUVbw 值 (SUVmax)
 * 2. 根据模式计算分割阈值 T：
 *    Fixed:    T = rawAtGivenSUV  (通过 SUVInfo 线性映射)
 *    Percent:  T = rawAtSUV(SUVmax × dPercent/100)
 *    Adaptive: T = rawAtSUV(SUVmax × weight)，weight 自动调整
 *               使正常背景器官（肝右叶参考区）的误分率最小
 * 3. 区域生长：从种子点 BFS，纳入灰度 >= T 的连通体素
 */

#include "oncology_types.h"

namespace Onc {

// ─────────────────────────────────────────────
//  PET 分割结果（含 threshold 反馈）
// ─────────────────────────────────────────────
struct PETSegmentResult {
    SegmentResult voxels;         ///< 分割体素集合
    double        usedThreshold;  ///< 实际使用的原始灰度阈值
    int           maxCoord[3];    ///< SUVmax 体素坐标（VOI 模式使用）
};

// ─────────────────────────────────────────────
//  接口函数 — Fixed 模式
// ─────────────────────────────────────────────

/**
 * @brief 固定绝对 SUVbw 阈值分割（单种子点）
 * @param suvInfo      SUV 线性映射参数
 * @param data         原始图像数据指针
 * @param info         图像元信息
 * @param seedPoint    种子点（图像坐标）
 * @param fixedSUV     用户给定绝对 SUVbw 阈值
 * @param result       [out] 分割结果
 */
bool PETSegmentFixed(
    const SUVInfo&   suvInfo,
    const void*      data,
    const ImageInfo& info,
    const int        seedPoint[3],
    double           fixedSUV,
    PETSegmentResult& result);

/**
 * @brief 固定绝对 SUVbw 阈值分割（VOI 区域多病灶）
 * @param voiCoords    VOI 区域内所有体素坐标
 */
bool PETSegmentFixedWithVOI(
    const SUVInfo&         suvInfo,
    const void*            data,
    const ImageInfo&       info,
    const SegmentResult&   voiCoords,
    double                 fixedSUV,
    PETSegmentResult&      result);

// ─────────────────────────────────────────────
//  接口函数 — Percent 模式
// ─────────────────────────────────────────────

/**
 * @brief 百分比阈值分割（单种子点）
 * @param percentOfMax  阈值占 SUVmax 的百分比，例如 0.42 表示 42%
 */
bool PETSegmentPercent(
    const SUVInfo&   suvInfo,
    const void*      data,
    const ImageInfo& info,
    const int        seedPoint[3],
    double           percentOfMax,
    PETSegmentResult& result);

bool PETSegmentPercentWithVOI(
    const SUVInfo&         suvInfo,
    const void*            data,
    const ImageInfo&       info,
    const SegmentResult&   voiCoords,
    double                 percentOfMax,
    PETSegmentResult&      result);

// ─────────────────────────────────────────────
//  接口函数 — Adaptive 模式
// ─────────────────────────────────────────────

/**
 * @brief 自适应阈值分割（单种子点）
 * @param weight   [in/out] 自适应权重 [0~1]，默认 0.5，算法会更新此值
 */
bool PETSegmentAdaptive(
    const SUVInfo&   suvInfo,
    const void*      data,
    const ImageInfo& info,
    const int        seedPoint[3],
    double&          weight,
    PETSegmentResult& result);

bool PETSegmentAdaptiveWithVOI(
    const SUVInfo&         suvInfo,
    const void*            data,
    const ImageInfo&       info,
    const SegmentResult&   voiCoords,
    double&                weight,
    PETSegmentResult&      result);

// ─────────────────────────────────────────────
//  Hover 实时预览（鼠标跟随）
// ─────────────────────────────────────────────

/**
 * @brief Hover 模式：鼠标位置实时预览，使用自适应阈值
 * @param hoverPoint  鼠标当前位置（图像坐标）
 * @param weight      自适应权重（持久化，鼠标移动时保留上次值）
 * @param result      [out] 预览分割结果
 */
bool PETSegmentAdaptiveHover(
    const SUVInfo&   suvInfo,
    const void*      data,
    const ImageInfo& info,
    const int        hoverPoint[3],
    double&          weight,
    PETSegmentResult& result);

} // namespace Onc
