#pragma once
/**
 * ============================================================
 * 基础 2D/3D 图像容器
 * ============================================================
 * 
 * 为分割算法提供统一的图像数据结构，避免依赖 OpenCV 等外部库。
 * 支持任意类型（float/double/uint8_t 等），连续内存布局。
 */

#include <vector>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <string>
#include <functional>
#include <queue>
#include <stack>
#include <array>

namespace medseg {

// ============================================================
// 2D 图像
// ============================================================
template<typename T = float>
struct Image2D {
    std::vector<T> data;
    int width  = 0;
    int height = 0;

    Image2D() = default;
    Image2D(int w, int h, T val = T(0))
        : data(w * h, val), width(w), height(h) {}

    int size() const { return width * height; }
    T& at(int x, int y) { return data[y * width + x]; }
    const T& at(int x, int y) const { return data[y * width + x]; }
    T& operator[](int i) { return data[i]; }
    const T& operator[](int i) const { return data[i]; }

    bool inBounds(int x, int y) const {
        return x >= 0 && x < width && y >= 0 && y < height;
    }

    // 安全访问，越界返回边界值（Neumann 边界条件）
    T atClamped(int x, int y) const {
        x = std::max(0, std::min(x, width - 1));
        y = std::max(0, std::min(y, height - 1));
        return data[y * width + x];
    }
};

// ============================================================
// 3D 体积图像
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
};

// ============================================================
// PGM 文件读写（方便测试）
// ============================================================
inline Image2D<uint8_t> loadPGM(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);
    std::string magic;
    f >> magic;
    if (magic != "P5") throw std::runtime_error("Not a P5 PGM");
    int w, h, maxval;
    f >> w >> h >> maxval;
    f.get(); // consume newline
    Image2D<uint8_t> img(w, h);
    f.read(reinterpret_cast<char*>(img.data.data()), w * h);
    return img;
}

inline void savePGM(const std::string& path, const Image2D<uint8_t>& img) {
    std::ofstream f(path, std::ios::binary);
    f << "P5\n" << img.width << " " << img.height << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
}

// 把 float 图像归一化到 [0, 255] 后保存
inline void savePGM(const std::string& path, const Image2D<float>& img) {
    float mn = *std::min_element(img.data.begin(), img.data.end());
    float mx = *std::max_element(img.data.begin(), img.data.end());
    float range = (mx - mn) > 1e-8f ? (mx - mn) : 1.0f;
    Image2D<uint8_t> out(img.width, img.height);
    for (int i = 0; i < img.size(); ++i)
        out[i] = static_cast<uint8_t>(255.0f * (img[i] - mn) / range);
    savePGM(path, out);
}

} // namespace medseg
