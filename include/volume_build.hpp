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
VolumeBuildResult buildVolume_none(const SeriesData& series,
                                    bool enableResampling,
                                    float targetZSpacing,
                                    InterpolationMethod method);

/**
 * 床板伪影去除 (in-place)
 *
 * 对 8-bit 体数据逐轴向切片：
 *   1. 圆形 FOV 裁剪（去除重建域外的角落噪声）
 *   2. 阈值分割 + 最大连通域提取（保留患者身体，排除床板）
 *   3. 孔洞填充（保留肺、空腔等低密度区域不被误删）
 *   4. 非身体区域体素置为 0（与空气等价）
 *
 * @param volume     要处理的体数据（直接修改 buffer）
 * @param bodyThresh 8-bit 阈值（默认 10），大于此值视为"身体候选"
 */
void removeBedArtifact(VolumeBuildResult& volume, uint8_t bodyThresh = 10);