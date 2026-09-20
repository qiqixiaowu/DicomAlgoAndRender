/**
 * @file    nail_gcode.cpp
 * @brief   3D美甲打印 — G-code 生成器 实现
 */

#include "nail_gcode.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace NailPrint3D {

// ============================================================
// GCodeGenerator
// ============================================================

std::string GCodeGenerator::generate(const std::vector<SliceLayer>& layers,
                                       const GCodeConfig& config) {
    std::ostringstream gcode;

    gcode << generateHeader(config);

    float totalExtrusion = 0;
    float currentX = 0, currentY = 0, currentZ = 0;
    int currentColor = -1;

    for (int i = 0; i < (int)layers.size(); i++) {
        const auto& layer = layers[i];
        currentZ = layer.zHeight;

        gcode << generateLayer(i, layer.zHeight, config);

        // 颜色层标记
        bool isBaseLayer = (i == 0);
        bool isTopCoat = (i == (int)layers.size() - 1);

        if (isBaseLayer) {
            gcode << comment("底胶层 (Base Coat)");
        } else if (isTopCoat) {
            gcode << comment("封层 (Top Coat)");
        } else {
            gcode << comment("颜色层 " + std::to_string(i));
        }

        // 外壳路径
        for (const auto& seg : layer.perimeters) {
            // 颜色切换
            if (seg.colorIndex != currentColor && seg.colorIndex >= 0) {
                gcode << generateColorChange(seg.colorIndex, config);
                currentColor = seg.colorIndex;
            }

            if (seg.travel) {
                // 空行程
                gcode << fmt("G0", seg.start.x, seg.start.y, currentZ);
                currentX = seg.start.x;
                currentY = seg.start.y;
            } else {
                // 挤出
                float e = computeE(seg.start, seg.end, config);
                totalExtrusion += e;
                gcode << generatePath(seg.start, seg.end, currentZ, e, seg.speed, config);
                currentX = seg.end.x;
                currentY = seg.end.y;
            }
        }

        // 填充路径
        gcode << comment("填充");
        for (const auto& seg : layer.infill) {
            if (seg.colorIndex != currentColor && seg.colorIndex >= 0) {
                gcode << generateColorChange(seg.colorIndex, config);
                currentColor = seg.colorIndex;
            }

            if (seg.travel) {
                gcode << fmt("G0", seg.start.x, seg.start.y, currentZ);
            } else {
                float e = computeE(seg.start, seg.end, config);
                totalExtrusion += e;
                gcode << generatePath(seg.start, seg.end, currentZ, e, seg.speed, config);
            }
        }

        // 支撑路径
        if (!layer.supportPaths.empty()) {
            gcode << comment("支撑");
            for (const auto& seg : layer.supportPaths) {
                if (seg.travel) {
                    gcode << fmt("G0", seg.start.x, seg.start.y, currentZ);
                } else {
                    float e = computeE(seg.start, seg.end, config);
                    totalExtrusion += e;
                    gcode << generatePath(seg.start, seg.end, currentZ, e, seg.speed, config);
                }
            }
        }

        // UV 固化（每层结束后）
        if (config.uvCurePerLayer) {
            gcode << generateUVCure(config.uvCureTime, config);
        }
    }

    gcode << generateFooter(config);

    totalExtrusion_ = totalExtrusion;
    layerCount_ = (int)layers.size();

    return gcode.str();
}

bool GCodeGenerator::saveToFile(const std::string& path, const std::string& gcode) {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "[GCode] 无法写入文件: " << path << std::endl;
        return false;
    }
    file << gcode;
    std::cout << "[GCode] 已保存: " << path << " ("
              << gcode.size() << " 字节)" << std::endl;
    return true;
}

void GCodeGenerator::printStats(const std::string& gcode) {
    std::cout << "========================================" << std::endl;
    std::cout << " G-code 生成统计" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << " 总行数:     " << std::count(gcode.begin(), gcode.end(), '\n') << std::endl;
    std::cout << " 层数:       " << layerCount_ << std::endl;
    std::cout << " 总挤出量:   " << std::fixed << std::setprecision(2) << totalExtrusion_ << " mm" << std::endl;
    std::cout << " 文件大小:   " << gcode.size() << " 字节" << std::endl;
    std::cout << "========================================" << std::endl;
}

// ============================================================
// 内部生成函数
// ============================================================

std::string GCodeGenerator::generateHeader(const GCodeConfig& config) {
    std::ostringstream ss;
    ss << "; ============================================\n";
    ss << "; NailPrint3D G-code\n";
    ss << "; 生成时间: " << __DATE__ << " " << __TIME__ << "\n";
    ss << "; 耗材: " << config.filamentType << "\n";
    ss << "; 耗材直径: " << config.filamentDiameter << " mm\n";
    ss << "; 喷嘴温度: " << config.nozzleTemp << " °C\n";
    ss << "; 打印速度: " << config.printSpeed << " mm/s\n";
    ss << "; 多色打印: " << (config.multiColor ? "是" : "否");
    if (config.multiColor) ss << " (" << config.colorCount << " 色)";
    ss << "\n";
    ss << "; UV固化: " << (config.uvCurePerLayer ? "每层" : "否");
    if (config.uvCurePerLayer) ss << " (" << config.uvCureTime << "s/层)";
    ss << "\n";
    ss << "; ============================================\n\n";

    // 启动序列
    ss << "; 启动序列\n";
    ss << "G28 ; 归零\n";
    ss << "G21 ; 毫米单位\n";
    ss << "G90 ; 绝对定位\n";
    ss << "M82 ; 绝对挤出\n";
    ss << "M104 S" << config.nozzleTemp << " ; 设置喷嘴温度\n";
    ss << "M140 S" << config.bedTemp << " ; 设置热床温度\n";
    ss << "M106 S128 ; 风扇50%\n";
    ss << "G1 Z5 F300 ; 抬升Z轴\n";
    ss << "G1 X0 Y0 F" << (config.travelSpeed * 60) << " ; 移动到原点\n";
    ss << "M109 S" << config.nozzleTemp << " ; 等待喷嘴温度\n";
    ss << "M190 S" << config.bedTemp << " ; 等待热床温度\n";
    ss << "\n";

    return ss.str();
}

std::string GCodeGenerator::generateFooter(const GCodeConfig& config) {
    std::ostringstream ss;
    ss << "\n; ============================================\n";
    ss << "; 结束序列\n";
    ss << "; ============================================\n";
    ss << "M400 ; 等待完成\n";
    ss << "M107 ; 关闭风扇\n";
    ss << "M104 S0 ; 关闭喷嘴加热\n";
    ss << "M140 S0 ; 关闭热床加热\n";
    ss << "G1 Z" << (config.travelSpeed * 60) << " F300 ; 抬升Z轴\n";
    ss << "G1 X0 Y200 F" << (config.travelSpeed * 60) << " ; 移到前方\n";
    ss << "M84 ; 关闭电机\n";
    ss << "; ============================================\n";
    return ss.str();
}

std::string GCodeGenerator::generateLayer(int layerIndex, float z,
                                            const GCodeConfig& config) {
    std::ostringstream ss;
    ss << "\n; --- 层 " << layerIndex << " (Z=" << std::fixed << std::setprecision(3) << z << "mm) ---\n";
    ss << "G1 Z" << std::fixed << std::setprecision(3) << z
       << " F" << (config.zSpeed * 60) << "\n";
    return ss.str();
}

std::string GCodeGenerator::generatePath(const Vec2& start, const Vec2& end,
                                           float z, float e, float speed,
                                           const GCodeConfig& config) {
    std::ostringstream ss;
    ss << "G1 X" << std::fixed << std::setprecision(3) << end.x
       << " Y" << std::setprecision(3) << end.y
       << " E" << std::setprecision(5) << e
       << " F" << std::setprecision(0) << (speed * 60) << "\n";
    return ss.str();
}

std::string GCodeGenerator::generateColorChange(int colorIndex,
                                                  const GCodeConfig& config) {
    std::ostringstream ss;
    ss << "; --- 颜色切换 → 颜色 " << colorIndex << " ---\n";
    // 回抽
    ss << generateRetraction(config.retractLength, config.retractSpeed);
    // 移到换色位置
    ss << "G0 X0 Y0 F" << (config.travelSpeed * 60) << " ; 移到换色站\n";
    // 擦嘴
    ss << "G0 X10 Y0 F" << (config.travelSpeed * 60) << "\n";
    ss << "G0 X-10 Y0 F" << (config.travelSpeed * 60) << " ; 擦嘴\n";
    // 选择颜色
    ss << "M163 S" << colorIndex << " ; 选择颜色" << colorIndex << "\n";
    // 回退回抽
    ss << generateUnretraction(config.retractLength, config.retractSpeed);
    return ss.str();
}

std::string GCodeGenerator::generateUVCure(float duration,
                                             const GCodeConfig& config) {
    std::ostringstream ss;
    ss << "; --- UV固化 (" << duration << "s) ---\n";
    ss << "M240 ; 开启UV灯\n";
    ss << "G4 S" << duration << " ; 等待" << duration << "秒\n";
    ss << "M241 ; 关闭UV灯\n";
    return ss.str();
}

std::string GCodeGenerator::generateRetraction(float length, float speed) {
    std::ostringstream ss;
    ss << "G1 E" << std::fixed << std::setprecision(5) << (-length)
       << " F" << (speed * 60) << " ; 回抽\n";
    return ss.str();
}

std::string GCodeGenerator::generateUnretraction(float length, float speed) {
    std::string s = generateRetraction(-length, speed);
    // 替换注释
    size_t pos = s.find("; 回抽");
    if (pos != std::string::npos) s.replace(pos, 6, "; 回退回抽");
    return s;
}

float GCodeGenerator::computeE(const Vec2& start, const Vec2& end,
                                 const GCodeConfig& config) {
    float dx = end.x - start.x;
    float dy = end.y - start.y;
    float pathLength = std::sqrt(dx * dx + dy * dy);
    // E = (挤出截面积 / 耗材截面积) * 路径长度
    float extrusionArea = config.layerHeight * config.extrusionWidth;
    float filamentArea = 3.14159265f * (config.filamentDiameter * 0.5f) * (config.filamentDiameter * 0.5f);
    return (extrusionArea / filamentArea) * pathLength;
}

std::string GCodeGenerator::fmt(const std::string& cmd, float x, float y, float z) {
    std::ostringstream ss;
    ss << cmd << " X" << std::fixed << std::setprecision(3) << x
       << " Y" << std::setprecision(3) << y
       << " Z" << std::setprecision(3) << z << "\n";
    return ss.str();
}

std::string GCodeGenerator::comment(const std::string& text) {
    return "; " + text + "\n";
}

} // namespace NailPrint3D
