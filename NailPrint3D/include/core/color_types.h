#pragma once
/**
 * @file    core/color_types.h
 * @brief   核心颜色数据结构（无项目依赖）
 *
 * 包含：ColorRGBf, ColorRGBA8, ColorSpace, ICCProfile, ColorLUT
 */

#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace NailPrint3D {

// ============================================================
// 颜色类型
// ============================================================

/** @brief RGB浮点颜色 */
struct ColorRGBf {
    float r = 0, g = 0, b = 0;

    ColorRGBf() = default;
    ColorRGBf(float r_, float g_, float b_) : r(r_), g(g_), b(b_) {}

    ColorRGBf operator*(float s) const { return { r * s, g * s, b * s }; }
    ColorRGBf operator/(float s) const { return { r / s, g / s, b / s }; }
    ColorRGBf operator+(const ColorRGBf& o) const { return { r + o.r, g + o.g, b + o.b }; }
    ColorRGBf operator-(const ColorRGBf& o) const { return { r - o.r, g - o.g, b - o.b }; }

    static ColorRGBf fromRGBA8(uint8_t r_, uint8_t g_, uint8_t b_) {
        return ColorRGBf(r_ / 255.0f, g_ / 255.0f, b_ / 255.0f);
    }
};

/** @brief RGBA颜色（8-bit） */
struct ColorRGBA8 {
    uint8_t r = 0, g = 0, b = 0, a = 255;

    ColorRGBA8() = default;
    ColorRGBA8(uint8_t r_, uint8_t g_, uint8_t b_, uint8_t a_ = 255)
        : r(r_), g(g_), b(b_), a(a_) {}

    ColorRGBf toFloatRGB() const {
        return { r / 255.0f, g / 255.0f, b / 255.0f };
    }
    static ColorRGBA8 fromColorRGBf(const ColorRGBf& c) {
        return ColorRGBA8{
            (uint8_t)std::clamp(c.r * 255.0f + 0.5f, 0.0f, 255.0f),
            (uint8_t)std::clamp(c.g * 255.0f + 0.5f, 0.0f, 255.0f),
            (uint8_t)std::clamp(c.b * 255.0f + 0.5f, 0.0f, 255.0f),
            255
        };
    }
};

// ============================================================
// 颜色空间
// ============================================================

/** @brief 颜色空间枚举 */
enum class ColorSpace {
    sRGB,       ///< 标准RGB
    AdobeRGB,   ///< Adobe RGB
    CMYK,       ///< 印刷四色
    Lab,        ///< CIE Lab
    XYZ         ///< CIE XYZ
};

// ============================================================
// ICC 配置
// ============================================================

/** @brief ICC配置信息 */
struct ICCProfile {
    std::string name;
    ColorSpace space = ColorSpace::sRGB;
    float m[9] = { 1,0,0, 0,1,0, 0,0,1 };       ///< 3x3 RGB→XYZ 矩阵
    float mInv[9] = { 1,0,0, 0,1,0, 0,0,1 };    ///< 3x3 XYZ→RGB 逆矩阵
    float whitePoint[3] = { 0.9505f, 1.0f, 1.089f };  ///< D65
    float gamma = 2.2f;
    std::string description;
    ColorSpace sourceSpace = ColorSpace::sRGB;
    ColorSpace targetSpace = ColorSpace::sRGB;
    float redPrimary[2]   = { 0.64f, 0.33f };
    float greenPrimary[2] = { 0.30f, 0.60f };
    float bluePrimary[2]  = { 0.15f, 0.06f };
    bool loaded = false;
};

// ============================================================
// 色彩查找表
// ============================================================

/** @brief 色彩查找表（3D LUT） */
struct ColorLUT {
    int size = 17;  ///< 每维采样点数
    std::vector<ColorRGBf> data;  ///< size^3 个颜色

    ColorLUT() { resize(17); }

    void resize(int n) {
        size = n;
        data.resize((size_t)n * n * n);
    }

    /** 三线性插值采样（实现在 color/color_manager.cpp） */
    ColorRGBf sample(float r, float g, float b) const;
};

} // namespace NailPrint3D
