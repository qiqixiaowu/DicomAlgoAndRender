#pragma once
/**
 * @file    nail_types.h
 * @brief   3D美甲打印 — 核心数据结构与类型定义
 *
 * 涵盖：
 *  - 顶点 / 三角面片 / 网格
 *  - 2D轮廓 / 切片层 / 填充路径
 *  - 颜色 / ICC配置 / 色彩查找表
 *  - 打印参数 / G-code指令
 */

#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <map>
#include <set>

namespace NailPrint3D {

// ============================================================
// 1. 几何数据结构
// ============================================================

/** @brief 3D顶点 */
struct Vec3 {
    float x = 0, y = 0, z = 0;

    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& o) const { return { x + o.x, y + o.y, z + o.z }; }
    Vec3 operator-(const Vec3& o) const { return { x - o.x, y - o.y, z - o.z }; }
    Vec3 operator*(float s) const { return { x * s, y * s, z * s }; }
    Vec3 operator/(float s) const { return { x / s, y / s, z / s }; }

    float dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return { y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x };
    }
    float length() const { return std::sqrt(x * x + y * y + z * z); }
    float lengthSq() const { return x * x + y * y + z * z; }
    Vec3 normalized() const {
        float len = length();
        return len > 1e-12f ? *this / len : *this;
    }
};

/** @brief 2D点（切片平面坐标） */
struct Vec2 {
    float x = 0, y = 0;

    Vec2() = default;
    Vec2(float x_, float y_) : x(x_), y(y_) {}

    Vec2 operator+(const Vec2& o) const { return { x + o.x, y + o.y }; }
    Vec2 operator-(const Vec2& o) const { return { x - o.x, y - o.y }; }
    Vec2 operator*(float s) const { return { x * s, y * s }; }
    Vec2 operator/(float s) const { return { x / s, y / s }; }

    float dot(const Vec2& o) const { return x * o.x + y * o.y; }
    float cross(const Vec2& o) const { return x * o.y - y * o.x; }
    float length() const { return std::sqrt(x * x + y * y); }
    float lengthSq() const { return x * x + y * y; }
    Vec2 normalized() const {
        float len = length();
        return len > 1e-12f ? *this / len : *this;
    }
};

/** @brief 轴对齐包围盒 */
struct AABB {
    Vec3 min{ 1e30f, 1e30f, 1e30f };
    Vec3 max{ -1e30f, -1e30f, -1e30f };

    void expand(const Vec3& p) {
        min.x = std::min(min.x, p.x); min.y = std::min(min.y, p.y); min.z = std::min(min.z, p.z);
        max.x = std::max(max.x, p.x); max.y = std::max(max.y, p.y); max.z = std::max(max.z, p.z);
    }
    Vec3 size() const { return max - min; }
    Vec3 center() const { return (max + min) * 0.5f; }
    bool valid() const { return min.x <= max.x; }
};

/** @brief 2D纹理坐标 */
struct Vec2UV {
    float u = 0, v = 0;
    Vec2UV() = default;
    Vec2UV(float u_, float v_) : u(u_), v(v_) {}
};

/** @brief RGB浮点颜色 */
struct ColorRGBf {
    float r = 0, g = 0, b = 0;

    ColorRGBf() = default;
    ColorRGBf(float r_, float g_, float b_) : r(r_), g(g_), b(b_) {}

    ColorRGBf operator*(float s) const { return { r * s, g * s, b * s }; }
    ColorRGBf operator/(float s) const { return { r / s, g / s, b / s }; }
    ColorRGBf operator+(const ColorRGBf& o) const { return { r + o.r, g + o.g, b + o.b }; }
    ColorRGBf operator-(const ColorRGBf& o) const { return { r - o.r, g - o.g, b - o.b }; }

    static ColorRGBf fromRGBA8(uint8_t r, uint8_t g, uint8_t b);
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
// 渲染图案模式（DIY图案支持）
// ============================================================

/** @brief 渲染图案模式 */
enum class RenderPattern {
    Procedural = 0,   ///< 程序化纹理（shader内生成：渐变/条纹/月牙）
    Photo      = 1,   ///< 照片/头像（纹理采样 + 完整光照）
    Cartoon    = 2,   ///< 卡通风格（posterize + 描边）
    FlatColor  = 3,   ///< 纯色块（无渐变，仅环境光）
    Text       = 4    ///< 文字/SDF（锐利边缘）
};

/** @brief 纹理变换参数（让用户调整图案位置/大小/旋转） */
struct TextureTransform {
    float offsetX = 0.0f;   ///< UV平移 X
    float offsetY = 0.0f;   ///< UV平移 Y
    float scale   = 1.0f;   ///< UV缩放
    float rotation = 0.0f;  ///< UV旋转（弧度）
    float opacity = 1.0f;   ///< 图案不透明度
    int   blendMode = 0;    ///< 混合模式: 0=正常, 1=正片叠底, 2=滤色, 3=覆盖
};

/** @brief 三角面片（含法线 + 颜色属性） */
struct Triangle {
    Vec3 v[3];              ///< 三个顶点坐标
    Vec3 normal;            ///< 面法线
    Vec2UV uv[3];           ///< 三个顶点的UV坐标
    uint16_t attr = 0;      ///< 属性字节（可用于颜色索引）
    uint32_t idx[3] = {0,0,0}; ///< 顶点索引（用于渲染）
};

/** @brief 网格数据 */
struct Mesh {
    std::vector<Triangle> triangles;
    std::vector<Vec3> vertices;   ///< 去重后的顶点列表
    std::vector<Vec3> normals;    ///< 顶点法线
    std::vector<Vec2UV> uvs;      ///< 顶点UV坐标
    std::vector<ColorRGBf> colors; ///< 顶点颜色
    AABB bbox;

    size_t triangleCount() const { return triangles.size(); }

    void computeBBox() {
        bbox = AABB{};
        for (const auto& tri : triangles)
            for (int i = 0; i < 3; i++)
                bbox.expand(tri.v[i]);
    }

    void computeNormals() {
        for (auto& tri : triangles) {
            Vec3 e1 = tri.v[1] - tri.v[0];
            Vec3 e2 = tri.v[2] - tri.v[0];
            tri.normal = e1.cross(e2).normalized();
        }
        // 同步顶点法线
        if (!vertices.empty()) {
            normals.assign(vertices.size(), Vec3(0, 0, 0));
            for (const auto& tri : triangles) {
                for (int i = 0; i < 3; i++) {
                    if (tri.idx[i] < normals.size())
                        normals[tri.idx[i]] = normals[tri.idx[i]] + tri.normal;
                }
            }
            for (auto& n : normals) n = n.normalized();
        }
    }
};

// ============================================================
// 2. 切片数据结构
// ============================================================

/** @brief 2D线段 */
struct Segment2D {
    Vec2 p1, p2;
};

/** @brief 闭合多边形（轮廓） */
struct Polygon {
    std::vector<Vec2> points;
    bool closed = true;

    float area() const {
        if (points.size() < 3) return 0;
        float a = 0;
        size_t n = points.size();
        for (size_t i = 0; i < n; i++) {
            size_t j = (i + 1) % n;
            a += points[i].cross(points[j]);
        }
        return a * 0.5f;
    }

    bool isCCW() const { return area() > 0; }
};

/** @brief 填充路径类型 */
enum class InfillPattern {
    None,           ///< 无填充
    Lines,          ///< 直线填充
    Grid,           ///< 网格填充
    Triangles,      ///< 三角形填充
    Honeycomb,      ///< 蜂窝填充
    Concentric      ///< 同心圆填充
};

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
// 3. 颜色数据结构
// ============================================================

/** @brief 颜色空间枚举 */
enum class ColorSpace {
    sRGB,       ///< 标准RGB
    AdobeRGB,   ///< Adobe RGB
    CMYK,       ///< 印刷四色
    Lab,        ///< CIE Lab
    XYZ         ///< CIE XYZ
};

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

/** @brief 色彩查找表（3D LUT） */
struct ColorLUT {
    int size = 17;  ///< 每维采样点数
    std::vector<ColorRGBf> data;  ///< size^3 个颜色

    ColorLUT() { resize(17); }

    void resize(int n) {
        size = n;
        data.resize((size_t)n * n * n);
    }

    /** 三线性插值采样 */
    ColorRGBf sample(float r, float g, float b) const;
};

// ============================================================
// 4. 打印参数
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
// 5. G-code指令
// ============================================================

/** @brief G-code指令 */
struct GCodeCmd {
    char letter;        ///< G / M
    int number;         ///< 指令编号
    std::string params; ///< 参数字符串
};

} // namespace NailPrint3D
