#pragma once
/**
 * @file    color_manager.h
 * @brief   色彩管理模块 — 颜色空间转换、ICC、LUT、抖动、校准
 *
 * 重构后的色彩层依赖：
 *   core/ (color_types, geometry)
 *   不依赖 mesh/ slicing/ render/ reconstruction/
 */

#include "core/color_types.h"
#include "core/geometry.h"
#include "core/print_types.h"   // PathSegment (MultiColorMapper 使用)
#include <map>
#include <string>
#include <vector>

namespace NailPrint3D {

// ============================================================
// ColorLUT 三线性插值实现（声明在 core/color_types.h）
// ============================================================

// ColorLUT::sample() 实现在 color/color_manager.cpp 中

// ============================================================
// 颜色空间转换
// ============================================================

/** @brief 颜色空间转换工具 */
class ColorSpaceConverter {
public:
    // --- sRGB ↔ 线性 ---
    static ColorRGBf sRGBToLinear(const ColorRGBf& c);
    static ColorRGBf linearTosRGB(const ColorRGBf& c);

    // --- RGB ↔ XYZ ---
    static void RGBToXYZ(const ColorRGBf& rgb, float& X, float& Y, float& Z);
    static ColorRGBf XYZToRGB(float X, float Y, float Z);

    // --- XYZ ↔ Lab ---
    static void XYZToLab(float X, float Y, float Z,
                         float& L, float& a, float& b);
    static void LabToXYZ(float L, float a, float b,
                         float& X, float& Y, float& Z);

    // --- Lab → RGB ---
    static ColorRGBf LabToRGB(float L, float a, float b);

    // --- RGB ↔ CMYK ---
    static void RGBToCMYK(const ColorRGBf& rgb, float& C, float& M, float& Y, float& K);
    static ColorRGBf CMYKToRGB(float C, float M, float Y, float K);

    // --- sRGB ↔ AdobeRGB ---
    static ColorRGBf sRGBToAdobeRGB(const ColorRGBf& rgb);

    // --- Gamma ---
    static ColorRGBf applyGamma(const ColorRGBf& c, float gamma);
    static ColorRGBf removeGamma(const ColorRGBf& c, float gamma);

    // --- Delta E ---
    static float deltaE(float L1, float a1, float b1,
                        float L2, float a2, float b2);
    static float deltaE2000(float L1, float a1, float b1,
                            float L2, float a2, float b2);
};

// ============================================================
// ICC色彩管理
// ============================================================

/** @brief ICC色彩管理器 */
class ICCColorManager {
public:
    /** @brief 加载预设ICC配置 (sRGB/AdobeRGB) 或从文件加载 */
    bool loadProfile(const std::string& name, const std::string& path = "");

    /** @brief 从文件加载ICC配置（.icc文件） */
    bool loadICCFile(const std::string& path, const std::string& name = "");

    /** @brief 颜色转换（源空间→目标空间） */
    ColorRGBf convert(const ColorRGBf& color,
                      const std::string& srcProfile,
                      const std::string& dstProfile) const;

    /** @brief 批量颜色转换 */
    std::vector<ColorRGBf> convertBatch(const std::vector<ColorRGBf>& colors,
                                         const std::string& srcProfile,
                                         const std::string& dstProfile) const;

    /** @brief 构建逆矩阵 */
    void buildMatrix(ICCProfile& profile);

private:
    std::map<std::string, ICCProfile> profiles_;
};

// ============================================================
// 3D LUT 色彩查找表
// ============================================================

/** @brief 3D LUT 管理器 */
class LUTManager {
public:
    /** @brief 加载 .cube 格式 3D LUT */
    bool loadCube(const std::string& path, ColorLUT& lut);

    /** @brief 保存 .cube 格式 3D LUT */
    bool saveCube(const std::string& path, const ColorLUT& lut);

    /** @brief 生成恒等 LUT */
    ColorLUT generateIdentity(int size = 17);

    /** @brief 生成伽马校正 LUT */
    ColorLUT generateGamma(float gamma, int size = 17);

    /** @brief 生成白平衡 LUT */
    ColorLUT generateWhiteBalance(const ColorRGBf& white, int size = 17);

    /** @brief 应用 LUT 到单个颜色 */
    ColorRGBf apply(const ColorRGBf& color, const ColorLUT& lut);
};

// ============================================================
// 抖动算法
// ============================================================

/** @brief 抖动算法类型 */
enum class DitherAlgorithm {
    None,
    FloydSteinberg,   ///< Floyd-Steinberg 误差扩散
    Bayer4x4,         ///< Bayer 4×4 有序抖动
    Bayer8x8,         ///< Bayer 8×8 有序抖动
    RandomDither,     ///< 随机抖动
    BlueNoise         ///< 蓝噪声抖动
};

/** @brief 抖动处理器 */
class DitherProcessor {
public:
    /** @brief 量化颜色到调色板最近色 */
    ColorRGBA8 quantize(const ColorRGBf& color,
                        const std::vector<ColorRGBf>& palette);

    /** @brief Floyd-Steinberg 误差扩散 */
    std::vector<ColorRGBA8> floydSteinberg(const std::vector<ColorRGBf>& pixels,
                                            int width, int height,
                                            const std::vector<ColorRGBf>& palette);

    /** @brief Bayer 有序抖动 */
    std::vector<ColorRGBA8> bayerDither(const std::vector<ColorRGBf>& pixels,
                                         int width, int height,
                                         const std::vector<ColorRGBf>& palette,
                                         int bayerSize = 4);

    /** @brief 查找调色板中最近颜色（Lab空间） */
    ColorRGBA8 findNearestColor(const ColorRGBf& color,
                                const std::vector<ColorRGBf>& palette);

private:
    static const float bayer4x4[16];
    static const float bayer8x8[64];
};

// ============================================================
// 多色打印颜色管理
// ============================================================

/** @brief 多色打印颜色映射器 */
class MultiColorMapper {
public:
    /** @brief 将纹理颜色映射到打印层 */
    std::vector<int> mapTextureToLayers(const std::vector<ColorRGBA8>& texture,
                                         int layerCount);

    /** @brief 生成颜色过渡路径 */
    std::vector<PathSegment> generateColorTransitions(
        const std::vector<PathSegment>& paths, int colorCount);

    /** @brief 计算颜色混合比例 */
    float computeBlendRatio(float z, float layerHeight,
                             int colorIndex1, int colorIndex2);

    /** @brief 混合两个颜色 */
    ColorRGBf blendColors(const ColorRGBf& c1, const ColorRGBf& c2, float t);
};

// ============================================================
// 颜色校准
// ============================================================

/** @brief 颜色校准器 */
class ColorCalibrator {
public:
    /** @brief 添加校准色块 (measured=实际测量值, reference=目标参考值) */
    void addPatch(const ColorRGBf& measured, const ColorRGBf& reference);

    /** @brief 执行校准 */
    bool calibrate();

    /** @brief 应用校准 */
    ColorRGBf applyCalibration(const ColorRGBf& color) const;

    /** @brief 计算平均Delta E */
    float computeAverageDeltaE() const;

private:
    std::vector<ColorRGBf> measured_;
    std::vector<ColorRGBf> reference_;
    ColorRGBf scale_   = { 1, 1, 1 };
    ColorRGBf offset_  = { 0, 0, 0 };
    bool calibrated_ = false;
};

} // namespace NailPrint3D
