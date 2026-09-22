#pragma once
/**
 * @file    voxelizer.h
 * @brief   网格体素化与SDF生成
 *
 * 依赖：core/ (geometry), reconstruction/ (reconstruction)
 */

#include "core/geometry.h"
#include "reconstruction/reconstruction.h"

namespace NailPrint3D {

/** @brief 网格 → 体素化 */
class Voxelizer {
public:
    static VoxelGrid voxelize(const Mesh& mesh, int resolution = 128);
    static SDFGrid generateSDF(const Mesh& mesh, int resolution = 128,
                               int bandWidth = 4);
};

} // namespace NailPrint3D
