#pragma once
/**
 * @file    core/geometry.h
 * @brief   核心几何数据结构（无项目依赖）
 *
 * 包含：Vec3, Vec2, Vec2UV, AABB, Triangle, Mesh,
 *       Segment2D, Polygon, InfillPattern
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

#include "core/color_types.h"   // ColorRGBf 用于 Triangle/Mesh

namespace NailPrint3D {

// ============================================================
// 基本向量
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

/** @brief 2D纹理坐标 */
struct Vec2UV {
    float u = 0, v = 0;
    Vec2UV() = default;
    Vec2UV(float u_, float v_) : u(u_), v(v_) {}
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

// ============================================================
// 三角面片与网格
// ============================================================

/** @brief 三角面片（含法线 + 颜色属性） */
struct Triangle {
    Vec3 v[3];              ///< 三个顶点坐标
    Vec3 normal;            ///< 面法线
    Vec2UV uv[3];           ///< 三个顶点的UV坐标
    ColorRGBf vcolor[3];    ///< 三个顶点的颜色（用于多色装饰物）
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
// 2D 切片几何
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

} // namespace NailPrint3D
