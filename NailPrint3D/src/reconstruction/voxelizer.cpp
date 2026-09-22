/**
 * @file    voxelizer.cpp
 * @brief   网格体素化与SDF生成 实现
 */

#include "reconstruction/voxelizer.h"
#include <algorithm>
#include <cmath>

namespace NailPrint3D {

VoxelGrid Voxelizer::voxelize(const Mesh& mesh, int resolution) {
    VoxelGrid grid;
    grid.width = resolution;
    grid.height = resolution;
    grid.depth = resolution;
    grid.bbox = mesh.bbox;
    Vec3 size = mesh.bbox.size();
    float maxSize = std::max({ size.x, size.y, size.z });
    grid.voxelSize = maxSize / resolution;
    grid.data.assign((size_t)resolution * resolution * resolution, 0.0f);

    for (const auto& tri : mesh.triangles) {
        AABB triBBox;
        for (int i = 0; i < 3; i++) triBBox.expand(tri.v[i]);

        Vec3 minIdx = (triBBox.min - grid.bbox.min) / grid.voxelSize;
        Vec3 maxIdx = (triBBox.max - grid.bbox.min) / grid.voxelSize;

        for (int z = (int)minIdx.z; z <= (int)maxIdx.z; z++) {
            for (int y = (int)minIdx.y; y <= (int)maxIdx.y; y++) {
                for (int x = (int)minIdx.x; x <= (int)maxIdx.x; x++) {
                    if (!grid.valid(x, y, z)) continue;
                    grid.at(x, y, z) = 1.0f;
                }
            }
        }
    }

    return grid;
}

SDFGrid Voxelizer::generateSDF(const Mesh& mesh, int resolution, int bandWidth) {
    SDFGrid sdf;
    sdf.width = resolution;
    sdf.height = resolution;
    sdf.depth = resolution;
    sdf.bbox = mesh.bbox;
    Vec3 size = mesh.bbox.size();
    float maxSize = std::max({ size.x, size.y, size.z });
    sdf.cellSize = maxSize / resolution;
    sdf.distances.assign((size_t)resolution * resolution * resolution, 1e10f);

    for (int z = 0; z < resolution; z++) {
        for (int y = 0; y < resolution; y++) {
            for (int x = 0; x < resolution; x++) {
                Vec3 p = sdf.bbox.min + Vec3(x, y, z) * sdf.cellSize;
                float minDist = 1e10f;
                for (const auto& tri : mesh.triangles) {
                    Vec3 center = (tri.v[0] + tri.v[1] + tri.v[2]) * (1.0f / 3.0f);
                    float d = (p - center).length();
                    minDist = std::min(minDist, d);
                }
                sdf.at(x, y, z) = minDist;
            }
        }
    }

    return sdf;
}

} // namespace NailPrint3D
