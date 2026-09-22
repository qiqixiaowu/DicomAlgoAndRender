#pragma once
/**
 * @file    voxel_color.h
 * @brief   体素级颜色管理（全彩3D打印）
 *
 * 依赖：
 *   core/ (color_types, geometry)
 *   color/ (color_manager — ICCColorManager, DitherProcessor)
 *   mesh/ (mesh_io — Mesh)
 */

#include "core/color_types.h"
#include "core/geometry.h"
#include "color/color_manager.h"
#include "mesh/mesh_io.h"
#include <string>
#include <vector>

namespace NailPrint3D {

// ============================================================
// 体素颜色数据
// ============================================================

/** @brief 体素颜色数据 */
struct VoxelColorGrid {
    std::vector<ColorRGBA8> colors;
    int width = 0, height = 0, depth = 0;
    float voxelSize = 0.1f;

    ColorRGBA8& at(int x, int y, int z) {
        return colors[(size_t)z * width * height + y * width + x];
    }
    ColorRGBA8 at(int x, int y, int z) const {
        return colors[(size_t)z * width * height + y * width + x];
    }
};

// ============================================================
// 体素颜色管理器
// ============================================================

/** @brief 体素颜色管理器 */
class VoxelColorManager {
public:
    /** @brief 从网格+纹理生成体素颜色 */
    void fromMeshTexture(const Mesh& mesh,
                         const std::vector<ColorRGBA8>& texture,
                         int texWidth, int texHeight);

    /** @brief 应用ICC色彩管理到体素颜色 */
    void applyICC(ICCColorManager& icc,
                  const std::string& srcProfile,
                  const std::string& dstProfile);

    /** @brief 应用3D LUT到体素颜色 */
    void applyLUT(const ColorLUT& lut);

    /** @brief 量化体素颜色到调色板 */
    void quantizeToPalette(const std::vector<ColorRGBf>& palette);

    /** @brief 导出体素颜色为3D纹理数据 */
    void exportTo3DTexture(const std::string& path);

private:
    std::vector<ColorRGBA8> voxelColors_;
};

} // namespace NailPrint3D
