#pragma once
/**
 * @file    nail_gcode.h
 * @brief   3D美甲打印 — G-code 生成模块
 */

#include "nail_types.h"
#include <string>
#include <vector>

namespace NailPrint3D {

// ============================================================
// G-code 配置
// ============================================================

/** @brief G-code 生成配置 */
struct GCodeConfig {
    // --- 挤出参数 ---
    float layerHeight       = 0.08f;   ///< 层高 (mm)
    float extrusionWidth    = 0.4f;    ///< 挤出宽度 (mm)
    float filamentDiameter  = 1.75f;   ///< 耗材直径 (mm)

    // --- 速度 ---
    float printSpeed        = 40.0f;   ///< 打印速度 (mm/s)
    float travelSpeed       = 120.0f;  ///< 空走速度 (mm/s)
    float zSpeed            = 5.0f;    ///< Z轴速度 (mm/s)
    float retractSpeed      = 40.0f;   ///< 回抽速度 (mm/s)

    // --- 回抽 ---
    float retractLength     = 2.0f;    ///< 回抽长度 (mm)

    // --- 温度 ---
    float nozzleTemp        = 60.0f;   ///< 喷嘴温度 (°C)
    float bedTemp           = 35.0f;   ///< 热床温度 (°C)

    // --- 美甲专用 ---
    float uvCureTime        = 30.0f;   ///< UV固化时间 (s)
    bool  uvCurePerLayer    = true;    ///< 每层UV固化
    float baseLayerSpeed    = 20.0f;   ///< 底胶层速度 (mm/s)
    float colorLayerSpeed   = 35.0f;   ///< 颜色层速度 (mm/s)
    float topCoatSpeed      = 25.0f;   ///< 封层速度 (mm/s)

    // --- 多色 ---
    bool  multiColor        = true;    ///< 多色打印
    int   colorCount        = 4;       ///< 颜色数量

    // --- 输出 ---
    std::string machineName = "NailPrinter-Pro";
    std::string filamentType = "UVResin";
};

// ============================================================
// G-code 生成器
// ============================================================

/** @brief G-code 生成器 */
class GCodeGenerator {
public:
    /** @brief 从切片层生成G-code */
    std::string generate(const std::vector<SliceLayer>& layers,
                          const GCodeConfig& config);

    /** @brief 保存G-code到文件 */
    bool saveToFile(const std::string& path, const std::string& gcode);

    /** @brief 生成G-code统计信息 */
    void printStats(const std::string& gcode);

private:
    // --- G-code指令生成 ---

    /** @brief 生成文件头 */
    std::string generateHeader(const GCodeConfig& config);

    /** @brief 生成文件尾 */
    std::string generateFooter(const GCodeConfig& config);

    /** @brief 生成单层G-code */
    std::string generateLayer(int layerIndex, float z,
                               const GCodeConfig& config);

    /** @brief 生成路径段G-code */
    std::string generatePath(const Vec2& start, const Vec2& end,
                              float z, float e, float speed,
                              const GCodeConfig& config);

    /** @brief 生成换色指令 */
    std::string generateColorChange(int colorIndex,
                                     const GCodeConfig& config);

    /** @brief 生成UV固化指令 */
    std::string generateUVCure(float duration,
                                const GCodeConfig& config);

    /** @brief 生成回抽指令 */
    std::string generateRetraction(float length, float speed);

    /** @brief 生成回抽恢复指令 */
    std::string generateUnretraction(float length, float speed);

    /** @brief 计算E值（挤出量） */
    float computeE(const Vec2& start, const Vec2& end,
                    const GCodeConfig& config);

    /** @brief 格式化移动指令 */
    std::string fmt(const std::string& cmd, float x, float y, float z);

    /** @brief 添加注释 */
    std::string comment(const std::string& text);

private:
    float totalExtrusion_ = 0.0f;  ///< 总挤出量
    int   layerCount_ = 0;         ///< 层数
};

} // namespace NailPrint3D
