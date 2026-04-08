#pragma once
/**
 * @file oncology_types.h
 * @brief 肿瘤学算法公共数据类型定义
 *
 * 本文件定义了整个 OncologyAlgorithms 工程所共用的基础数据类型，
 * 包括三维点/向量、分割结果容器、图像元信息结构及进度回调接口。
 * 设计上不依赖任何第三方框架，可直接在 C++17 环境中使用。
 */

#include <vector>
#include <functional>
#include <string>
#include <cmath>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <climits>

namespace Onc {

// ─────────────────────────────────────────────
//  基础几何类型
// ─────────────────────────────────────────────

/** 整型三维坐标点（体素坐标） */
struct Point3i {
    int x = 0, y = 0, z = 0;
    Point3i() = default;
    Point3i(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
    bool operator==(const Point3i& o) const { return x==o.x && y==o.y && z==o.z; }

    float distanceTo(const Point3i& o) const {
        float dx = static_cast<float>(x - o.x);
        float dy = static_cast<float>(y - o.y);
        float dz = static_cast<float>(z - o.z);
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
};

/** 浮点型三维坐标点（物理空间坐标，单位 mm） */
struct Point3f {
    float x = 0.f, y = 0.f, z = 0.f;
    Point3f() = default;
    Point3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    float distanceTo(const Point3f& o) const {
        float dx = x - o.x, dy = y - o.y, dz = z - o.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    Point3f operator-(const Point3f& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Point3f operator+(const Point3f& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Point3f operator*(float s) const { return {x*s, y*s, z*s}; }
    float dot(const Point3f& o) const { return x*o.x + y*o.y + z*o.z; }
    float norm() const { return std::sqrt(x*x + y*y + z*z); }
};

/** 三维图像元信息 */
struct ImageInfo {
    int    dim[3]     = {0, 0, 0};   ///< 体素维度 [X, Y, Z]
    double spacing[3] = {1.0, 1.0, 1.0}; ///< 体素间距 mm [X, Y, Z]

    int totalVoxels() const { return dim[0] * dim[1] * dim[2]; }
    int sliceSize()   const { return dim[0] * dim[1]; }
    int linearIndex(int x, int y, int z) const {
        return z * sliceSize() + y * dim[0] + x;
    }
    bool inBounds(int x, int y, int z) const {
        return x >= 0 && x < dim[0]
            && y >= 0 && y < dim[1]
            && z >= 0 && z < dim[2];
    }
};

/** 图像模态标志 */
enum class ImageModality : int {
    CT        = 0,  ///< CT
    MR        = 1,  ///< MR
    DECT_80kV = 3,  ///< 双能CT 80kV
    DECT_140kV= 4,  ///< 双能CT 140kV
    VMI_100keV= 5,  ///< 虚拟单能 100keV
    VMI_120keV= 6,  ///< 虚拟单能 120keV
    VMI_70keV = 7,  ///< 虚拟单能 70keV
};

/** 绘制方向（用户交互面）*/
enum class DrawDirection : int {
    Axial    = 0,  ///< 横断位
    Sagittal = 1,  ///< 矢状位
    Coronal  = 2,  ///< 冠状位
};

// ─────────────────────────────────────────────
//  分割结果类型
// ─────────────────────────────────────────────

/** 分割结果：体素坐标集合 */
using SegmentResult = std::vector<Point3i>;

/** 分割掩码（与图像等尺寸，0=背景 1=前景） */
using SegmentMask = std::vector<uint8_t>;

// ─────────────────────────────────────────────
//  进度回调
// ─────────────────────────────────────────────

/**
 * @brief 算法进度回调接口
 * @param progress  当前进度 [0.0, 1.0]
 * @return false 表示请求取消
 */
using ProgressCallback = std::function<bool(double progress)>;

/** 空进度回调（默认值，永不取消） */
inline ProgressCallback NoProgress() {
    return [](double) -> bool { return true; };
}

// ─────────────────────────────────────────────
//  PET SUV 信息
// ─────────────────────────────────────────────

/** PET 标准化摄取值（SUVbw）线性映射参数 */
struct SUVInfo {
    bool   hasValidSUV  = false;  ///< 是否可计算 SUVbw
    double rawAtSUV1    = 0.0;    ///< SUVbw=1.0 对应的原始灰度值
    double rawAtSUV2    = 0.0;    ///< SUVbw=2.0 对应的原始灰度值
    double suv1         = 1.0;
    double suv2         = 2.0;
    double slope        = 1.0;    ///< 线性映射斜率
    double intercept    = 0.0;    ///< 线性映射截距
    std::string ImgType = "MET_FLOAT"; ///< 像素类型("MET_SHORT"/"MET_USHORT"/"MET_FLOAT")

    /** 原始灰度值 → SUVbw */
    double rawToSUV(double rawValue) const {
        return slope * rawValue + intercept;
    }
    /** SUVbw → 原始灰度值 */
    double suvToRaw(double suv) const {
        return (slope != 0.0) ? (suv - intercept) / slope : 0.0;
    }
};

// ─────────────────────────────────────────────
//  VOI 综合统计信息
// ─────────────────────────────────────────────

/**
 * @brief VOI（Volume Of Interest）综合统计结构
 *
 * 对标参考工程 TissueCtrl::VOIStatistic，将以下信息合并为一体：
 *  - 灰度统计（体素数、均值、标准差、极值、灰度累加）
 *  - 包围盒（AABB）
 *  - SUV 统计（SUVmax / SUVmean / SUVmin / SUVpeak）
 *  - 径线测量（2D长径/垂直径、3D最长径 及各端点坐标）
 *  - 体积
 */
struct VOIStatistic {
    // ── 基础灰度统计 ──────────────────────────
    int    elemCount            = 0;        ///< 前景体素数
    int    maxDcm               = -32768;   ///< 最大原始灰度值
    int    minDcm               = 32767;    ///< 最小原始灰度值
    double sumDcm               = 0.0;      ///< 灰度值总和（用于计算均值）
    double squareSumDcm         = 0.0;      ///< 灰度值平方和（用于计算方差）
    size_t maxSlicePositionIndex= 0;        ///< 最大灰度值体素的线性索引

    // ── 包围盒（AABB）────────────────────────
    int    iMinRange[3]  = {INT_MAX, INT_MAX, INT_MAX};
    int    iMaxRange[3]  = {INT_MIN, INT_MIN, INT_MIN};

    // ── 派生统计 ────────────────────────────
    double volume_mm3    = 0.0;   ///< 体积 mm³
    double mean          = 0.0;   ///< 均值
    double stdDev        = 0.0;   ///< 标准差

    // ── SUV 统计（仅 PET 有效）──────────────
    double suvMax        = 0.0;   ///< SUVmax
    double suvMean       = 0.0;   ///< SUVmean
    double suvMin        = 0.0;   ///< SUVmin
    double suvPeak       = 0.0;   ///< SUVpeak（1 ml 球加权平均最大值）

    // ── 径线测量 ────────────────────────────
    double longAxis_mm   = 0.0;   ///< 2D最大截面长径 mm
    double shortAxis_mm  = 0.0;   ///< 2D垂直径 mm
    double maxDiameter_mm= 0.0;   ///< 3D最长径 mm

    Point3f ptLongAxisStart;      ///< 长径起点（体素坐标）
    Point3f ptLongAxisEnd;        ///< 长径终点
    Point3f ptShortAxisStart;     ///< 垂直径起点
    Point3f ptShortAxisEnd;       ///< 垂直径终点

    // ── 便捷方法 ────────────────────────────
    /** 计算均值（从累加量导出） */
    void finalize() {
        if (elemCount > 0) {
            mean = sumDcm / elemCount;
            double var = squareSumDcm / elemCount - mean * mean;
            stdDev = (var > 0.0) ? std::sqrt(var) : 0.0;
        }
    }
};

} // namespace Onc
