#pragma once
/**
 * ============================================================
 *  放射治疗自动勾画 — 公共数据类型
 * ============================================================
 *
 * 提炼自联影 McsfAlgoAutoContour 工程。
 * 纯 C++17 header-only，零外部依赖。
 *
 * 提供：
 *   - 2D/3D 图像容器（Image2D / Image3D）
 *   - 2D 点 / 轮廓类型
 *   - 金属定位点类型
 *   - PGM 读写工具
 * ============================================================
 */

#include <vector>
#include <array>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <string>
#include <functional>
#include <queue>
#include <stack>
#include <deque>
#include <cassert>
#include <stdexcept>
#include <limits>

namespace rtac {

// ============================================================
//  2D 浮点点
// ============================================================
struct Point2f {
    float x = 0.f, y = 0.f;
    Point2f() = default;
    Point2f(float x_, float y_) : x(x_), y(y_) {}
    Point2f operator+(const Point2f& o) const { return {x + o.x, y + o.y}; }
    Point2f operator-(const Point2f& o) const { return {x - o.x, y - o.y}; }
    Point2f operator*(float s) const { return {x * s, y * s}; }
    float dot(const Point2f& o) const { return x * o.x + y * o.y; }
    float norm() const { return std::sqrt(x * x + y * y); }
    float distanceTo(const Point2f& o) const { return (*this - o).norm(); }
    bool operator==(const Point2f& o) const { return x == o.x && y == o.y; }
};

// ============================================================
//  2D 整型点
// ============================================================
struct Point2i {
    int x = 0, y = 0;
    Point2i() = default;
    Point2i(int x_, int y_) : x(x_), y(y_) {}
    bool operator==(const Point2i& o) const { return x == o.x && y == o.y; }
};

// ============================================================
//  3D 浮点点
// ============================================================
struct Point3f {
    float x = 0.f, y = 0.f, z = 0.f;
    Point3f() = default;
    Point3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    Point3f operator-(const Point3f& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Point3f operator+(const Point3f& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Point3f operator*(float s) const { return {x * s, y * s, z * s}; }
    float norm() const { return std::sqrt(x * x + y * y + z * z); }
    float distanceTo(const Point3f& o) const { return (*this - o).norm(); }
};

// ============================================================
//  2D 图像
// ============================================================
template<typename T = float>
struct Image2D {
    std::vector<T> data;
    int width  = 0;
    int height = 0;

    Image2D() = default;
    Image2D(int w, int h, T val = T(0))
        : data(static_cast<size_t>(w) * h, val), width(w), height(h) {}

    size_t size() const { return static_cast<size_t>(width) * height; }
    T& at(int x, int y) { return data[y * width + x]; }
    const T& at(int x, int y) const { return data[y * width + x]; }
    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }

    bool inBounds(int x, int y) const {
        return x >= 0 && x < width && y >= 0 && y < height;
    }

    T atClamped(int x, int y) const {
        x = std::max(0, std::min(x, width - 1));
        y = std::max(0, std::min(y, height - 1));
        return data[y * width + x];
    }
};

// ============================================================
//  3D 体图像
// ============================================================
template<typename T = float>
struct Image3D {
    std::vector<T> data;
    int width  = 0;  // X
    int height = 0;  // Y
    int depth  = 0;  // Z
    float spacingX = 1.0f, spacingY = 1.0f, spacingZ = 1.0f;

    Image3D() = default;
    Image3D(int w, int h, int d, T val = T(0))
        : data(static_cast<size_t>(w) * h * d, val), width(w), height(h), depth(d) {}

    size_t size() const { return static_cast<size_t>(width) * height * depth; }
    size_t sliceSize() const { return static_cast<size_t>(width) * height; }
    int index(int x, int y, int z) const { return (z * height + y) * width + x; }
    T& at(int x, int y, int z) { return data[index(x, y, z)]; }
    const T& at(int x, int y, int z) const { return data[index(x, y, z)]; }
    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }

    bool inBounds(int x, int y, int z) const {
        return x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth;
    }

    T atClamped(int x, int y, int z) const {
        x = std::max(0, std::min(x, width - 1));
        y = std::max(0, std::min(y, height - 1));
        z = std::max(0, std::min(z, depth - 1));
        return data[index(x, y, z)];
    }

    // 获取第 z 层指针
    T* slicePtr(int z) { return data.data() + z * sliceSize(); }
    const T* slicePtr(int z) const { return data.data() + z * sliceSize(); }
};

// ============================================================
//  轮廓类型
// ============================================================

/// 单层 2D 轮廓 = 一组有序点
using Contour2D = std::vector<Point2f>;

/// 多层轮廓（每层可能有多个轮廓环）
using ContourSlice = std::vector<Contour2D>;

// ============================================================
//  金属定位点（Fiducial）类型
// ============================================================
enum class FiducialType { Left, Right, Top, Center };

struct FiducialPoint {
    Point3f pos;
    FiducialType type = FiducialType::Left;

    FiducialPoint() = default;
    FiducialPoint(float x, float y, float z, FiducialType t)
        : pos(x, y, z), type(t) {}
};

/// 一组 fiducial = 多个点（通常含 left/right/top + 计算得到的 center）
using FiducialGroup = std::vector<FiducialPoint>;

// ============================================================
//  区域属性
// ============================================================
struct RegionProps {
    int area = 0;                  ///< 像素面积
    float centroidX = 0.f;         ///< 质心 X
    float centroidY = 0.f;         ///< 质心 Y
    int bboxX0 = 0, bboxY0 = 0;   ///< 包围盒左上
    int bboxX1 = 0, bboxY1 = 0;   ///< 包围盒右下
    float majorAxisLen = 0.f;      ///< 主轴长度
    float minorAxisLen = 0.f;      ///< 短轴长度
    float eccentricity = 0.f;      ///< 偏心率
    float solidity     = 0.f;      ///< 紧致度 = area / convexHullArea
};

// ============================================================
//  进度回调
// ============================================================
using ProgressCallback = std::function<bool(double progress)>;

inline ProgressCallback NoProgress() {
    return [](double) -> bool { return true; };
}

// ============================================================
//  PGM 读写工具
// ============================================================
inline Image2D<uint8_t> loadPGM(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);
    std::string magic;
    f >> magic;
    if (magic != "P5") throw std::runtime_error("Not a P5 PGM");
    int w, h, maxval;
    f >> w >> h >> maxval;
    f.get();
    Image2D<uint8_t> img(w, h);
    f.read(reinterpret_cast<char*>(img.data.data()), w * h);
    return img;
}

inline void savePGM(const std::string& path, const Image2D<uint8_t>& img) {
    std::ofstream f(path, std::ios::binary);
    f << "P5\n" << img.width << " " << img.height << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
}

inline void savePGM(const std::string& path, const Image2D<float>& img) {
    float mn = *std::min_element(img.data.begin(), img.data.end());
    float mx = *std::max_element(img.data.begin(), img.data.end());
    float range = (mx - mn) > 1e-8f ? (mx - mn) : 1.0f;
    Image2D<uint8_t> out(img.width, img.height);
    for (size_t i = 0; i < img.size(); ++i)
        out[i] = static_cast<uint8_t>(255.0f * (img[i] - mn) / range);
    savePGM(path, out);
}

} // namespace rtac
