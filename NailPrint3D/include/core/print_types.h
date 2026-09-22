#pragma once
/**
 * @file    core/print_types.h
 * @brief   核心打印类型（无项目依赖）
 *
 * 包含：PrintConfig, PathSegment, SliceLayer, GCodeCmd
 */

#include <vector>
#include <string>

#include "core/geometry.h"     // InfillPattern, Vec2, Polygon
#include "core/color_types.h"  // ColorRGBA8

namespace NailPrint3D {

// ============================================================
// 打印路径与切片层
// ============================================================

/** @brief 打印路径段（含颜色） */
struct PathSegment {
    Vec2 start;
    Vec2 end;
    float speed = 50.0f;       ///< mm/s
    float extrusion = 0.0f;    ///< 挤出量
    int colorIndex = 0;        ///< 颜色索引（多色打印）
    bool travel = false;       ///< 是否为空行程
};

/** @brief 单层切片数据 */
struct SliceLayer {
    int layerIndex = 0;
    float zHeight = 0.0f;          ///< Z高度
    std::vector<Polygon> contours; ///< 外轮廓
    std::vector<Polygon> holes;    ///< 孔洞
    std::vector<PathSegment> infill;       ///< 填充路径
    std::vector<PathSegment> perimeters;   ///< 外壳路径
    std::vector<PathSegment> supportPaths; ///< 支撑路径
};

// ============================================================
// 打印配置
// ============================================================

/** @brief 打印配置参数 */
struct PrintConfig {
    // 层参数
    float layerHeight = 0.1f;        ///< 层高
    float firstLayerHeight = 0.15f;  ///< 首层层高
    int   adaptiveSlicing = 0;       ///< 自适应切片（0=关闭, 1=开启）

    // 挤出参数
    float nozzleDiameter = 0.2f;     ///< 喷嘴直径
    float filamentDiameter = 1.75f;  ///< 耗材直径
    float extrusionWidth = 0.22f;    ///< 挤出宽度
    float extrusionMultiplier = 1.0f;///< 挤出倍率

    // 填充参数
    InfillPattern infillPattern = InfillPattern::Grid;
    float infillDensity = 0.2f;      ///< 填充密度 0~1
    float infillAngle = 45.0f;      ///< 填充角度

    // 外壳参数
    int perimeters = 2;              ///< 外壳层数
    int topSolidLayers = 3;          ///< 顶层实心层数
    int bottomSolidLayers = 3;       ///< 底层实心层数

    // 支撑参数
    bool generateSupport = true;
    float supportAngle = 45.0f;      ///< 悬垂角度阈值
    float supportDensity = 0.15f;    ///< 支撑密度

    // 速度参数
    float printSpeed = 50.0f;        ///< 打印速度
    float travelSpeed = 120.0f;      ///< 空行程速度
    float firstLayerSpeed = 20.0f;   ///< 首层速度
    float infillSpeed = 60.0f;       ///< 填充速度
    float perimeterSpeed = 40.0f;    ///< 外壳速度

    // 多色参数
    int colorCount = 1;              ///< 颜色数量
    std::vector<ColorRGBA8> palette; ///< 颜色调色板
    float colorBlendDistance = 0.5f; ///< 颜色过渡距离

    // 美甲专用参数
    float nailBedWidth = 15.0f;      ///< 甲床宽度
    float nailBedLength = 20.0f;     ///< 甲床长度
    float nailCurvature = 0.3f;      ///< 甲面曲率
    float baseThickness = 0.3f;      ///< 底胶厚度
    float colorLayerHeight = 0.08f;  ///< 颜色层层高
    float topCoatThickness = 0.1f;   ///< 封层厚度
};

// ============================================================
// G-code 指令
// ============================================================

/** @brief G-code指令 */
struct GCodeCmd {
    char letter;        ///< G / M
    int number;         ///< 指令编号
    std::string params; ///< 参数字符串
};

} // namespace NailPrint3D
