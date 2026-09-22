#pragma once
/**
 * @file    reconstruction.h
 * @brief   3D重建模块 — Marching Cubes / Poisson / 网格修复
 *
 * 依赖：core/ (geometry, color_types), mesh/ (mesh_io)
 */

#include "core/geometry.h"
#include "core/color_types.h"
#include "mesh/mesh_io.h"

namespace NailPrint3D {

// ============================================================
// 体素数据
// ============================================================

/** @brief 3D体素网格 */
struct VoxelGrid {
    std::vector<float> data;    ///< 密度场
    int width = 0, height = 0, depth = 0;
    float voxelSize = 0.1f;     ///< 体素尺寸
    AABB bbox;

    float& at(int x, int y, int z) {
        return data[(size_t)z * width * height + y * width + x];
    }
    float at(int x, int y, int z) const {
        return data[(size_t)z * width * height + y * width + x];
    }

    bool valid(int x, int y, int z) const {
        return x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth;
    }

    size_t voxelCount() const { return (size_t)width * height * depth; }
};

/** @brief SDF（有向距离场） */
struct SDFGrid {
    std::vector<float> distances;
    int width = 0, height = 0, depth = 0;
    float cellSize = 0.05f;
    AABB bbox;

    float& at(int x, int y, int z) {
        return distances[(size_t)z * width * height + y * width + x];
    }
    float at(int x, int y, int z) const {
        return distances[(size_t)z * width * height + y * width + x];
    }
};

// ============================================================
// Marching Cubes 等值面提取
// ============================================================

/** @brief Marching Cubes算法 */
class MarchingCubes {
public:
    static Mesh extract(const VoxelGrid& grid, float isovalue);
    static Mesh extractSDF(const SDFGrid& sdf);

    static const int edgeTable[256];
    static const int triTable[256][16];

private:
    static Vec3 interpolate(const Vec3& p1, const Vec3& p2,
                            float val1, float val2, float isovalue);
    static Vec3 gradientNormal(const VoxelGrid& grid, int x, int y, int z);
};

// ============================================================
// Poisson表面重建
// ============================================================

/** @brief 3D点云数据 */
struct PointCloud {
    std::vector<Vec3> points;
    std::vector<Vec3> normals;
    AABB bbox;

    size_t size() const { return points.size(); }

    void computeBBox() {
        bbox = AABB{};
        for (const auto& p : points) bbox.expand(p);
    }
};

/** @brief Poisson重建 */
class PoissonReconstruction {
public:
    static Mesh reconstruct(const PointCloud& cloud, int depth = 8);

private:
    static void buildOctree(const PointCloud& cloud, int depth);
    static void estimateNormals(PointCloud& cloud, int kNeighbors = 20);
};

// ============================================================
// 网格修复与优化
// ============================================================

/** @brief 网格修复工具 */
class MeshRepair {
public:
    static void fillHoles(Mesh& mesh, float maxHoleSize = 5.0f);
    static void unifyNormals(Mesh& mesh);
    static int detectNonManifoldEdges(const Mesh& mesh);
    static void weldVertices(Mesh& mesh, float threshold = 1e-5f);
    static void laplacianSmooth(Mesh& mesh, int iterations = 3, float lambda = 0.5f);
    static void taubinSmooth(Mesh& mesh, int iterations = 5,
                             float lambda = 0.5f, float mu = -0.53f);
    static void simplify(Mesh& mesh, float targetRatio = 0.5f);
};

// ============================================================
// 重建管线
// ============================================================

/** @brief 重建配置 */
struct ReconstructionConfig {
    int   voxelResolution = 128;
    float isoValue = 0.5f;
    int   smoothIterations = 3;
    bool  fillHoles = true;
    bool  unifyNormals = true;
    bool  weldVertices = true;
    float simplifyRatio = 0.0f;
    int   poissonDepth = 8;
};

/** @brief 重建管线 */
class ReconstructionPipeline {
public:
    static Mesh reconstructFromVoxels(const VoxelGrid& grid,
                                      const ReconstructionConfig& config);
    static Mesh reconstructFromPointCloud(const PointCloud& cloud,
                                         const ReconstructionConfig& config);
    static Mesh repairAndOptimize(Mesh mesh,
                                  const ReconstructionConfig& config);
    static void printStats(const Mesh& mesh, const std::string& stage);
};

} // namespace NailPrint3D
