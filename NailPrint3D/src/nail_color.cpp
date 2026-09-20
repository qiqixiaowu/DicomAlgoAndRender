/**
 * @file    nail_color.cpp
 * @brief   3D美甲打印 — 色彩管理模块 实现
 */

#include "nail_color.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>

namespace NailPrint3D {

// ============================================================
// ColorRGBf / ColorRGBA8 静态方法
// ============================================================

ColorRGBf ColorRGBf::fromRGBA8(uint8_t r, uint8_t g, uint8_t b) {
    return { r / 255.0f, g / 255.0f, b / 255.0f };
}

// ============================================================
// ColorSpaceConverter
// ============================================================

ColorRGBf ColorSpaceConverter::sRGBToLinear(const ColorRGBf& c) {
    auto transfer = [](float v) -> float {
        if (v <= 0.04045f) return v / 12.92f;
        return std::pow((v + 0.055f) / 1.055f, 2.4f);
    };
    return { transfer(c.r), transfer(c.g), transfer(c.b) };
}

ColorRGBf ColorSpaceConverter::linearTosRGB(const ColorRGBf& c) {
    auto transfer = [](float v) -> float {
        if (v <= 0.0031308f) return v * 12.92f;
        return 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
    };
    return { transfer(c.r), transfer(c.g), transfer(c.b) };
}

void ColorSpaceConverter::RGBToXYZ(const ColorRGBf& rgb, float& X, float& Y, float& Z) {
    // sRGB D65
    ColorRGBf lin = sRGBToLinear(rgb);
    X = 0.4124564f * lin.r + 0.3575761f * lin.g + 0.1804375f * lin.b;
    Y = 0.2126729f * lin.r + 0.7151522f * lin.g + 0.0721750f * lin.b;
    Z = 0.0193339f * lin.r + 0.1191920f * lin.g + 0.9503041f * lin.b;
}

ColorRGBf ColorSpaceConverter::XYZToRGB(float X, float Y, float Z) {
    float r =  3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z;
    float g = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
    float b =  0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z;
    return linearTosRGB({ r, g, b });
}

void ColorSpaceConverter::XYZToLab(float X, float Y, float Z,
                                     float& L, float& a, float& b) {
    // D65 参考白点
    const float Xn = 0.95047f, Yn = 1.00000f, Zn = 1.08883f;
    auto f = [](float t) -> float {
        return (t > 0.008856f) ? std::cbrt(t) : (7.787f * t + 16.0f / 116.0f);
    };

    float fx = f(X / Xn), fy = f(Y / Yn), fz = f(Z / Zn);
    L = 116.0f * fy - 16.0f;
    a = 500.0f * (fx - fy);
    b = 200.0f * (fy - fz);
}

void ColorSpaceConverter::LabToXYZ(float L, float a, float b,
                                     float& X, float& Y, float& Z) {
    const float Xn = 0.95047f, Yn = 1.00000f, Zn = 1.08883f;
    float fy = (L + 16.0f) / 116.0f;
    float fx = a / 500.0f + fy;
    float fz = fy - b / 200.0f;

    auto finv = [](float t) -> float {
        float t3 = t * t * t;
        return (t3 > 0.008856f) ? t3 : (t - 16.0f / 116.0f) / 7.787f;
    };

    X = Xn * finv(fx);
    Y = Yn * finv(fy);
    Z = Zn * finv(fz);
}

ColorRGBf ColorSpaceConverter::LabToRGB(float L, float a, float b) {
    float X, Y, Z;
    LabToXYZ(L, a, b, X, Y, Z);
    return XYZToRGB(X, Y, Z);
}

void ColorSpaceConverter::RGBToCMYK(const ColorRGBf& rgb, float& C, float& M,
                                      float& Y, float& K) {
    float r = rgb.r, g = rgb.g, bl = rgb.b;
    K = 1.0f - std::max({ r, g, bl });
    if (K >= 1.0f - 1e-10f) {
        C = M = Y = 0;
    } else {
        C = (1.0f - r - K) / (1.0f - K);
        M = (1.0f - g - K) / (1.0f - K);
        Y = (1.0f - bl - K) / (1.0f - K);
    }
}

ColorRGBf ColorSpaceConverter::CMYKToRGB(float C, float M, float Y, float K) {
    return {
        (1.0f - C) * (1.0f - K),
        (1.0f - M) * (1.0f - K),
        (1.0f - Y) * (1.0f - K)
    };
}

ColorRGBf ColorSpaceConverter::sRGBToAdobeRGB(const ColorRGBf& rgb) {
    // sRGB → linear → XYZ → Adobe RGB gamma
    float X, Y, Z;
    RGBToXYZ(rgb, X, Y, Z);
    // XYZ → Adobe RGB (D65)
    float r =  2.0413690f * X - 0.5649464f * Y - 0.3446944f * Z;
    float g = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
    float b =  0.0134474f * X - 0.1183897f * Y + 1.0154096f * Z;
    // Adobe RGB gamma = 2.2
    auto gamma = [](float v) { return std::pow(std::clamp(v, 0.0f, 1.0f), 1.0f / 2.2f); };
    return { gamma(r), gamma(g), gamma(b) };
}

float ColorSpaceConverter::deltaE(float L1, float a1, float b1,
                                    float L2, float a2, float b2) {
    float dL = L1 - L2, da = a1 - a2, db = b1 - b2;
    return std::sqrt(dL * dL + da * da + db * db);
}

float ColorSpaceConverter::deltaE2000(float L1, float a1, float b1,
                                        float L2, float a2, float b2) {
    // CIEDE2000
    float avgL = (L1 + L2) / 2.0f;
    float C1 = std::sqrt(a1 * a1 + b1 * b1);
    float C2 = std::sqrt(a2 * a2 + b2 * b2);
    float avgC = (C1 + C2) / 2.0f;
    float avgC7 = std::pow(avgC, 7);
    float G = 0.5f * (1.0f - std::sqrt(avgC7 / (avgC7 + std::pow(25.0f, 7))));
    float a1p = a1 * (1.0f + G);
    float a2p = a2 * (1.0f + G);
    float C1p = std::sqrt(a1p * a1p + b1 * b1);
    float C2p = std::sqrt(a2p * a2p + b2 * b2);
    float avgCp = (C1p + C2p) / 2.0f;
    float h1p = std::atan2(b1, a1p); if (h1p < 0) h1p += 2 * 3.14159265f;
    float h2p = std::atan2(b2, a2p); if (h2p < 0) h2p += 2 * 3.14159265f;
    float dLp = L2 - L1;
    float dCp = C2p - C1p;
    float dhp = h2p - h1p;
    if (dhp > 3.14159265f) dhp -= 2 * 3.14159265f;
    if (dhp < -3.14159265f) dhp += 2 * 3.14159265f;
    float dHp = 2.0f * std::sqrt(C1p * C2p) * std::sin(dhp / 2.0f);
    float avgLp = (avgL + L2) / 2.0f;  // simplified
    float avgHp = (h1p + h2p) / 2.0f;
    if (std::abs(h1p - h2p) > 3.14159265f) avgHp += 3.14159265f;
    float T = 1.0f - 0.17f * std::cos(avgHp - 0.5236f)
                  + 0.24f * std::cos(2.0f * avgHp)
                  + 0.32f * std::cos(3.0f * avgHp + 0.7854f);
    float dTheta = 30.0f * std::exp(-std::pow((avgHp * 180.0f / 3.14159265f - 275.0f) / 25.0f, 2));
    float RC = 2.0f * std::pow(avgCp, 7) / (std::pow(avgCp, 7) + std::pow(25.0f, 7));
    float SL = 1.0f + 0.015f * std::pow(avgLp - 50.0f, 2) / std::sqrt(20.0f + std::pow(avgLp - 50.0f, 2));
    float SC = 1.0f + 0.045f * avgCp;
    float SH = 1.0f + 0.015f * avgCp * T;
    float RT = -std::sin(2.0f * dTheta * 3.14159265f / 180.0f) * RC;
    float kL = 1.0f, kC = 1.0f, kH = 1.0f;
    float dE = std::sqrt(
        std::pow(dLp / (kL * SL), 2) +
        std::pow(dCp / (kC * SC), 2) +
        std::pow(dHp / (kH * SH), 2) +
        RT * (dCp / (kC * SC)) * (dHp / (kH * SH))
    );
    return dE;
}

// ============================================================
// ICCColorManager
// ============================================================

bool ICCColorManager::loadProfile(const std::string& name, const std::string& path) {
    // 简化版：从预设矩阵加载
    if (name == "sRGB") {
        ICCProfile p;
        p.name = "sRGB";
        p.space = ColorSpace::sRGB;
        p.m[0] = 0.4124564f; p.m[1] = 0.3575761f; p.m[2] = 0.1804375f;
        p.m[3] = 0.2126729f; p.m[4] = 0.7151522f; p.m[5] = 0.0721750f;
        p.m[6] = 0.0193339f; p.m[7] = 0.1191920f; p.m[8] = 0.9503041f;
        p.whitePoint[0] = 0.95047f; p.whitePoint[1] = 1.00000f; p.whitePoint[2] = 1.08883f;
        p.gamma = 2.2f;
        buildMatrix(p);
        profiles_[name] = p;
        return true;
    }
    if (name == "AdobeRGB") {
        ICCProfile p;
        p.name = "AdobeRGB";
        p.space = ColorSpace::AdobeRGB;
        p.m[0] = 0.5767309f; p.m[1] = 0.1855540f; p.m[2] = 0.1881852f;
        p.m[3] = 0.2973769f; p.m[4] = 0.6273491f; p.m[5] = 0.0752741f;
        p.m[6] = 0.0270342f; p.m[7] = 0.0706872f; p.m[8] = 0.9911085f;
        p.whitePoint[0] = 0.95047f; p.whitePoint[1] = 1.00000f; p.whitePoint[2] = 1.08883f;
        p.gamma = 2.2f;
        buildMatrix(p);
        profiles_[name] = p;
        return true;
    }
    // 尝试从文件加载
    return loadICCFile(path, name);
}

bool ICCColorManager::loadICCFile(const std::string& path, const std::string& name) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ICC] 无法打开文件: " << path << std::endl;
        return false;
    }
    // 简化：读取文件头，实际应解析ICC标签
    // 这里仅标记为已加载
    ICCProfile profile;
    profile.name = name;
    profile.space = ColorSpace::sRGB;
    profile.gamma = 2.2f;
    profiles_[name] = profile;
    std::cout << "[ICC] 已加载配置文件: " << name << " (" << path << ")" << std::endl;
    return true;
}

ColorRGBf ColorSpaceConverter::applyGamma(const ColorRGBf& c, float gamma) {
    auto g = [gamma](float v) { return std::pow(std::clamp(v, 0.0f, 1.0f), gamma); };
    return { g(c.r), g(c.g), g(c.b) };
}

ColorRGBf ColorSpaceConverter::removeGamma(const ColorRGBf& c, float gamma) {
    auto g = [gamma](float v) { return std::pow(std::clamp(v, 0.0f, 1.0f), 1.0f / gamma); };
    return { g(c.r), g(c.g), g(c.b) };
}

ColorRGBf ICCColorManager::convert(const ColorRGBf& color, const std::string& srcProfile,
                                     const std::string& dstProfile) const {
    auto srcIt = profiles_.find(srcProfile);
    auto dstIt = profiles_.find(dstProfile);
    if (srcIt == profiles_.end() || dstIt == profiles_.end()) {
        std::cerr << "[ICC] 配置文件未找到: " << srcProfile << " → " << dstProfile << std::endl;
        return color;
    }

    const auto& src = srcIt->second;
    const auto& dst = dstIt->second;

    // 1. 源空间 → 线性
    ColorRGBf linear = ColorSpaceConverter::removeGamma(color, src.gamma);
    // 2. 源 RGB → XYZ
    float X = src.m[0] * linear.r + src.m[1] * linear.g + src.m[2] * linear.b;
    float Y = src.m[3] * linear.r + src.m[4] * linear.g + src.m[5] * linear.b;
    float Z = src.m[6] * linear.r + src.m[7] * linear.g + src.m[8] * linear.b;
    // 3. XYZ → 目标 RGB (假设白点相同，简化)
    float r =  dst.mInv[0] * X + dst.mInv[1] * Y + dst.mInv[2] * Z;
    float g =  dst.mInv[3] * X + dst.mInv[4] * Y + dst.mInv[5] * Z;
    float b =  dst.mInv[6] * X + dst.mInv[7] * Y + dst.mInv[8] * Z;
    // 4. 应用目标 gamma
    return ColorSpaceConverter::applyGamma({ r, g, b }, dst.gamma);
}

std::vector<ColorRGBf> ICCColorManager::convertBatch(const std::vector<ColorRGBf>& colors,
                                                       const std::string& srcProfile,
                                                       const std::string& dstProfile) const {
    std::vector<ColorRGBf> result;
    result.reserve(colors.size());
    for (const auto& c : colors)
        result.push_back(convert(c, srcProfile, dstProfile));
    return result;
}

void ICCColorManager::buildMatrix(ICCProfile& profile) {
    // 构建 3x3 逆矩阵 (简化：使用伴随矩阵法)
    float m[9] = { profile.m[0], profile.m[1], profile.m[2],
                   profile.m[3], profile.m[4], profile.m[5],
                   profile.m[6], profile.m[7], profile.m[8] };
    float det = m[0] * (m[4] * m[8] - m[5] * m[7])
              - m[1] * (m[3] * m[8] - m[5] * m[6])
              + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if (std::abs(det) < 1e-10f) {
        std::fill(profile.mInv, profile.mInv + 9, 0.0f);
        return;
    }
    float invDet = 1.0f / det;
    profile.mInv[0] = (m[4] * m[8] - m[5] * m[7]) * invDet;
    profile.mInv[1] = (m[2] * m[7] - m[1] * m[8]) * invDet;
    profile.mInv[2] = (m[1] * m[5] - m[2] * m[4]) * invDet;
    profile.mInv[3] = (m[5] * m[6] - m[3] * m[8]) * invDet;
    profile.mInv[4] = (m[0] * m[8] - m[2] * m[6]) * invDet;
    profile.mInv[5] = (m[2] * m[3] - m[0] * m[5]) * invDet;
    profile.mInv[6] = (m[3] * m[7] - m[4] * m[6]) * invDet;
    profile.mInv[7] = (m[1] * m[6] - m[0] * m[7]) * invDet;
    profile.mInv[8] = (m[0] * m[4] - m[1] * m[3]) * invDet;
}

// ============================================================
// LUTManager
// ============================================================

bool LUTManager::loadCube(const std::string& path, ColorLUT& lut) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[LUT] 无法打开: " << path << std::endl;
        return false;
    }
    std::string line;
    int size = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line.substr(0, 10) == "LUT_3DSIZE") {
            size = std::atoi(line.substr(10).c_str());
            lut.size = size;
            lut.data.resize(size * size * size);
        } else if (line[0] >= '0' && line[0] <= '9' || line[0] == '-' || line[0] == '.') {
            std::istringstream iss(line);
            float r, g, b;
            if (iss >> r >> g >> b) {
                lut.data.push_back({ r, g, b });
            }
        }
    }
    std::cout << "[LUT] 已加载: " << path << " (size=" << size << ")" << std::endl;
    return true;
}

bool LUTManager::saveCube(const std::string& path, const ColorLUT& lut) {
    std::ofstream file(path);
    if (!file.is_open()) return false;
    file << "TITLE \"NailPrint3D_LUT\"\n";
    file << "LUT_3DSIZE " << lut.size << "\n";
    file << "# Generated by NailPrint3D\n";
    for (const auto& c : lut.data) {
        file << c.r << " " << c.g << " " << c.b << "\n";
    }
    return true;
}

ColorLUT LUTManager::generateIdentity(int size) {
    ColorLUT lut;
    lut.size = size;
    lut.data.resize(size * size * size);
    for (int b = 0; b < size; b++)
        for (int g = 0; g < size; g++)
            for (int r = 0; r < size; r++) {
                int idx = r + g * size + b * size * size;
                lut.data[idx] = { (float)r / (size - 1),
                                  (float)g / (size - 1),
                                  (float)b / (size - 1) };
            }
    return lut;
}

ColorLUT LUTManager::generateGamma(float gamma, int size) {
    ColorLUT lut = generateIdentity(size);
    for (auto& c : lut.data) {
        c.r = std::pow(c.r, gamma);
        c.g = std::pow(c.g, gamma);
        c.b = std::pow(c.b, gamma);
    }
    return lut;
}

ColorLUT LUTManager::generateWhiteBalance(const ColorRGBf& white, int size) {
    ColorLUT lut = generateIdentity(size);
    for (auto& c : lut.data) {
        c.r *= white.r;
        c.g *= white.g;
        c.b *= white.b;
    }
    return lut;
}

ColorRGBf LUTManager::apply(const ColorRGBf& color, const ColorLUT& lut) {
    return lut.sample(color.r, color.g, color.b);
}

// ============================================================
// DitherProcessor
// ============================================================

const float DitherProcessor::bayer4x4[16] = {
    0,  8,  2, 10,
    12, 4, 14, 6,
    3, 11, 1, 9,
    15, 7, 13, 5
};

const float DitherProcessor::bayer8x8[64] = {
     0, 32,  8, 40,  2, 34, 10, 42,
    48, 16, 56, 24, 50, 18, 58, 26,
    12, 44,  4, 36, 14, 46,  6, 38,
    60, 28, 52, 20, 62, 30, 54, 22,
     3, 35, 11, 43,  1, 33,  9, 41,
    51, 19, 59, 27, 49, 17, 57, 25,
    15, 47,  7, 39, 13, 45,  5, 37,
    63, 31, 55, 23, 61, 29, 53, 21
};

ColorRGBA8 DitherProcessor::quantize(const ColorRGBf& color,
                                       const std::vector<ColorRGBf>& palette) {
    return findNearestColor(color, palette);
}

std::vector<ColorRGBA8> DitherProcessor::floydSteinberg(
    const std::vector<ColorRGBf>& pixels, int width, int height,
    const std::vector<ColorRGBf>& palette) {
    std::vector<ColorRGBf> work = pixels;
    std::vector<ColorRGBA8> result(width * height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            ColorRGBf old = work[idx];
            ColorRGBA8 quantized = findNearestColor(old, palette);
            result[idx] = quantized;

            ColorRGBf quantizedF = quantized.toFloatRGB();
            ColorRGBf error = old - quantizedF;

            // Floyd-Steinberg 误差扩散
            if (x + 1 < width)
                work[idx + 1] = work[idx + 1] + error * 7.0f / 16.0f;
            if (y + 1 < height) {
                if (x > 0)
                    work[idx + width - 1] = work[idx + width - 1] + error * 3.0f / 16.0f;
                work[idx + width] = work[idx + width] + error * 5.0f / 16.0f;
                if (x + 1 < width)
                    work[idx + width + 1] = work[idx + width + 1] + error * 1.0f / 16.0f;
            }
        }
    }

    return result;
}

std::vector<ColorRGBA8> DitherProcessor::bayerDither(
    const std::vector<ColorRGBf>& pixels, int width, int height,
    const std::vector<ColorRGBf>& palette, int bayerSize) {
    std::vector<ColorRGBA8> result(width * height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            ColorRGBf c = pixels[idx];

            // 添加 Bayer 抖动偏移
            float threshold = 0;
            if (bayerSize == 4) {
                threshold = bayer4x4[(y % 4) * 4 + (x % 4)] / 16.0f - 0.5f;
            } else {
                threshold = bayer8x8[(y % 8) * 8 + (x % 8)] / 64.0f - 0.5f;
            }
            c.r = std::clamp(c.r + threshold * 0.1f, 0.0f, 1.0f);
            c.g = std::clamp(c.g + threshold * 0.1f, 0.0f, 1.0f);
            c.b = std::clamp(c.b + threshold * 0.1f, 0.0f, 1.0f);

            result[idx] = findNearestColor(c, palette);
        }
    }

    return result;
}

ColorRGBA8 DitherProcessor::findNearestColor(const ColorRGBf& color,
                                               const std::vector<ColorRGBf>& palette) {
    if (palette.empty()) return ColorRGBA8{ 0, 0, 0, 255 };

    float bestDist = 1e30f;
    int bestIdx = 0;

    float L, a, b;
    ColorSpaceConverter::RGBToXYZ(color, L, a, b);
    ColorSpaceConverter::XYZToLab(L, a, b, L, a, b);

    for (int i = 0; i < (int)palette.size(); i++) {
        float pL, pa, pb;
        ColorSpaceConverter::RGBToXYZ(palette[i], pL, pa, pb);
        ColorSpaceConverter::XYZToLab(pL, pa, pb, pL, pa, pb);
        float d = ColorSpaceConverter::deltaE(L, a, b, pL, pa, pb);
        if (d < bestDist) {
            bestDist = d;
            bestIdx = i;
        }
    }

    return ColorRGBA8::fromColorRGBf(palette[bestIdx]);
}

// ============================================================
// MultiColorMapper
// ============================================================

std::vector<int> MultiColorMapper::mapTextureToLayers(
    const std::vector<ColorRGBA8>& texture, int layerCount) {
    std::vector<int> mapping(texture.size());
    for (size_t i = 0; i < texture.size(); i++) {
        // 根据颜色的亮度分配到不同层
        float lum = (texture[i].r * 0.299f + texture[i].g * 0.587f + texture[i].b * 0.114f) / 255.0f;
        mapping[i] = (int)(lum * layerCount);
        mapping[i] = std::clamp(mapping[i], 0, layerCount - 1);
    }
    return mapping;
}

std::vector<PathSegment> MultiColorMapper::generateColorTransitions(
    const std::vector<PathSegment>& paths, int colorCount) {
    std::vector<PathSegment> transitions;
    for (const auto& seg : paths) {
        if (seg.colorIndex >= 0 && seg.colorIndex < colorCount) {
            transitions.push_back(seg);
        }
    }
    return transitions;
}

float MultiColorMapper::computeBlendRatio(float z, float layerHeight,
                                            int colorIndex1, int colorIndex2) {
    // 在颜色切换层计算混合比例
    float transitionLayers = 2.0f;  // 过渡层数
    float layerPos = z / layerHeight;
    float t = std::clamp(layerPos / transitionLayers, 0.0f, 1.0f);
    return t;
}

ColorRGBf MultiColorMapper::blendColors(const ColorRGBf& c1, const ColorRGBf& c2, float t) {
    return c1 * (1.0f - t) + c2 * t;
}

// ============================================================
// ColorCalibrator
// ============================================================

void ColorCalibrator::addPatch(const ColorRGBf& measured, const ColorRGBf& reference) {
    measured_.push_back(measured);
    reference_.push_back(reference);
}

bool ColorCalibrator::calibrate() {
    if (measured_.size() < 4) {
        std::cerr << "[Calibrate] 需要至少4个色块" << std::endl;
        return false;
    }
    // 简化：计算平均缩放因子
    scale_ = { 1, 1, 1 };
    offset_ = { 0, 0, 0 };
    for (size_t i = 0; i < measured_.size(); i++) {
        if (measured_[i].r > 0.01f) scale_.r += (reference_[i].r / measured_[i].r - 1) / measured_.size();
        if (measured_[i].g > 0.01f) scale_.g += (reference_[i].g / measured_[i].g - 1) / measured_.size();
        if (measured_[i].b > 0.01f) scale_.b += (reference_[i].b / measured_[i].b - 1) / measured_.size();
    }
    calibrated_ = true;
    std::cout << "[Calibrate] 校准完成: scale=(" << scale_.r << "," << scale_.g << "," << scale_.b << ")" << std::endl;
    return true;
}

ColorRGBf ColorCalibrator::applyCalibration(const ColorRGBf& color) const {
    if (!calibrated_) return color;
    return { std::clamp(color.r * scale_.r + offset_.r, 0.0f, 1.0f),
             std::clamp(color.g * scale_.g + offset_.g, 0.0f, 1.0f),
             std::clamp(color.b * scale_.b + offset_.b, 0.0f, 1.0f) };
}

float ColorCalibrator::computeAverageDeltaE() const {
    if (measured_.empty()) return 0;
    float totalDeltaE = 0;
    for (size_t i = 0; i < measured_.size(); i++) {
        ColorRGBf corrected = applyCalibration(measured_[i]);
        float L1, a1, b1, L2, a2, b2;
        float X, Y, Z;
        ColorSpaceConverter::RGBToXYZ(corrected, X, Y, Z);
        ColorSpaceConverter::XYZToLab(X, Y, Z, L1, a1, b1);
        ColorSpaceConverter::RGBToXYZ(reference_[i], X, Y, Z);
        ColorSpaceConverter::XYZToLab(X, Y, Z, L2, a2, b2);
        totalDeltaE += ColorSpaceConverter::deltaE(L1, a1, b1, L2, a2, b2);
    }
    return totalDeltaE / measured_.size();
}

// ============================================================
// VoxelColorManager
// ============================================================

void VoxelColorManager::fromMeshTexture(const Mesh& mesh,
                                          const std::vector<ColorRGBA8>& texture,
                                          int textureWidth, int textureHeight) {
    // 简化：为每个顶点分配颜色
    voxelColors_.resize(mesh.vertices.size());
    for (size_t i = 0; i < mesh.vertices.size(); i++) {
        // 使用顶点位置映射到纹理坐标（简化）
        float u = (mesh.vertices[i].x - mesh.bbox.min.x) / (mesh.bbox.max.x - mesh.bbox.min.x + 1e-10f);
        float v = (mesh.vertices[i].y - mesh.bbox.min.y) / (mesh.bbox.max.y - mesh.bbox.min.y + 1e-10f);
        int tx = std::clamp((int)(u * textureWidth), 0, textureWidth - 1);
        int ty = std::clamp((int)(v * textureHeight), 0, textureHeight - 1);
        voxelColors_[i] = texture[ty * textureWidth + tx];
    }
}

void VoxelColorManager::applyICC(ICCColorManager& icc,
                                   const std::string& srcProfile,
                                   const std::string& dstProfile) {
    for (auto& c : voxelColors_) {
        ColorRGBf rgb = c.toFloatRGB();
        rgb = icc.convert(rgb, srcProfile, dstProfile);
        c = ColorRGBA8::fromColorRGBf(rgb);
    }
}

void VoxelColorManager::applyLUT(const ColorLUT& lut) {
    for (auto& c : voxelColors_) {
        ColorRGBf rgb = c.toFloatRGB();
        rgb = lut.sample(rgb.r, rgb.g, rgb.b);
        c = ColorRGBA8::fromColorRGBf(rgb);
    }
}

void VoxelColorManager::quantizeToPalette(const std::vector<ColorRGBf>& palette) {
    DitherProcessor dither;
    for (auto& c : voxelColors_) {
        c = dither.findNearestColor(c.toFloatRGB(), palette);
    }
}

void VoxelColorManager::exportTo3DTexture(const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return;
    int count = (int)voxelColors_.size();
    file.write(reinterpret_cast<const char*>(&count), sizeof(int));
    file.write(reinterpret_cast<const char*>(voxelColors_.data()),
               voxelColors_.size() * sizeof(ColorRGBA8));
    std::cout << "[VoxelColor] 导出 " << count << " 个体素颜色到 " << path << std::endl;
}

} // namespace NailPrint3D
