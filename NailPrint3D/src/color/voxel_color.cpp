/**
 * @file    voxel_color.cpp
 * @brief   体素级颜色管理 实现
 */

#include "color/voxel_color.h"
#include <algorithm>
#include <fstream>
#include <iostream>

namespace NailPrint3D {

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
