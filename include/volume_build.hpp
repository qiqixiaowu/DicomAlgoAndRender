#pragma once

#include "dicom_utils.hpp"
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimgle/dcmimage.h>
#include <vector>
#include <cstdint>

struct VolumeBuildResult {
    std::vector<uint8_t> buffer; // 体素原始数据 (8-bit)
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 0;
    double spacing[3] = {1,1,1}; // x,y,z 物理尺寸
    double orientationRow[3] = {1,0,0};
    double orientationCol[3] = {0,1,0};
    double origin[3] = {0,0,0};

    VolumeBuildResult() : width(0), height(0), depth(0) {
        spacing[0] = spacing[1] = spacing[2] = 1.0f;
        origin[0] = origin[1] = origin[2] = 0.0f;
        orientationRow[0] = 1.0f; orientationRow[1] = 0.0f; orientationRow[2] = 0.0f;
        orientationCol[0] = 0.0f; orientationCol[1] = 1.0f; orientationCol[2] = 0.0f;
    }
};

enum class InterpolationMethod {
    CUBIC_SPLINE,    // 三次样条插值
    TRILINEAR,       // 三线性插值
    NEAREST_NEIGHBOR // 最近邻插值
};

VolumeBuildResult buildVolume_none(const SeriesData& series);