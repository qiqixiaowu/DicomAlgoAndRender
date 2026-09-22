#pragma once
/**
 * @file    gcode_gen.h
 * @brief   G-code 生成模块
 *
 * 依赖：core/ (geometry, print_types)
 */

#include "core/geometry.h"
#include "core/print_types.h"
#include <string>
#include <vector>

namespace NailPrint3D {

// ============================================================
// G-code 配置
// ============================================================

/** @brief G-code 生成配置 */
struct GCodeConfig {
    // --- 挤出参数 ---
    float layerHeight       = 0.08f;
    float extrusionWidth    = 0.4f;
    float filamentDiameter  = 1.75f;

    // --- 速度 ---
    float printSpeed        = 40.0f;
    float travelSpeed       = 120.0f;
    float zSpeed            = 5.0f;
    float retractSpeed      = 40.0f;

    // --- 回抽 ---
    float retractLength     = 2.0f;

    // --- 温度 ---
    float nozzleTemp        = 60.0f;
    float bedTemp           = 35.0f;

    // --- 美甲专用 ---
    float uvCureTime        = 30.0f;
    bool  uvCurePerLayer    = true;
    float baseLayerSpeed    = 20.0f;
    float colorLayerSpeed   = 35.0f;
    float topCoatSpeed      = 25.0f;

    // --- 多色 ---
    bool  multiColor        = true;
    int   colorCount        = 4;

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
    std::string generate(const std::vector<SliceLayer>& layers,
                          const GCodeConfig& config);
    bool saveToFile(const std::string& path, const std::string& gcode);
    void printStats(const std::string& gcode);

private:
    std::string generateHeader(const GCodeConfig& config);
    std::string generateFooter(const GCodeConfig& config);
    std::string generateLayer(int layerIndex, float z,
                               const GCodeConfig& config);
    std::string generatePath(const Vec2& start, const Vec2& end,
                              float z, float e, float speed,
                              const GCodeConfig& config);
    std::string generateColorChange(int colorIndex,
                                     const GCodeConfig& config);
    std::string generateUVCure(float duration,
                                const GCodeConfig& config);
    std::string generateRetraction(float length, float speed);
    std::string generateUnretraction(float length, float speed);
    float computeE(const Vec2& start, const Vec2& end,
                    const GCodeConfig& config);
    std::string fmt(const std::string& cmd, float x, float y, float z);
    std::string comment(const std::string& text);

private:
    float totalExtrusion_ = 0.0f;
    int   layerCount_ = 0;
};

} // namespace NailPrint3D
