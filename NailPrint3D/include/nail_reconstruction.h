#pragma once
/**
 * @file    nail_reconstruction.h
 * @brief   3D美甲重建模块
 *
 * 功能：
 *  - 从多视角照片/深度图重建3D指甲表面
 *  - Marching Cubes等值面提取
 *  - Poisson表面重建
 *  - 网格修复与优化（孔洞填充、法线一致化、平滑）
 *  - 体素化与SDF（有向距离场）生成
 */

#include "nail_types.h"
#include "stl_loader.h"

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
    /** @brief 从体素数据提取等值面
     *  @param grid 体素网格
     *  @param isovalue 等值面阈值
     *  @return 提取的网格
     */
    static Mesh extract(const VoxelGrid& grid, float isovalue);

    /** @brief 从SDF提取等值面（isovalue=0） */
    static Mesh extractSDF(const SDFGrid& sdf);

    static const int edgeTable[256];
    static const int triTable[256][16];

private:
    /** @brief 线性插值交点 */
    static Vec3 interpolate(const Vec3& p1, const Vec3& p2,
                            float val1, float val2, float isovalue);

    /** @brief 计算梯度法线 */
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
    /** @brief 从点云重建表面
     *  @param cloud 输入点云（需含法线）
     *  @param depth 八叉树深度（5~12，越大越精细）
     *  @return 重建网格
     */
    static Mesh reconstruct(const PointCloud& cloud, int depth = 8);

private:
    /** @brief 构建八叉树 */
    static void buildOctree(const PointCloud& cloud, int depth);

    /** @brief 估计点云法线（如果未提供） */
    static void estimateNormals(PointCloud& cloud, int kNeighbors = 20);
};

// ============================================================
// 网格修复与优化
// ============================================================

/** @brief 网格修复工具 */
class MeshRepair {
public:
    /** @brief 填充孔洞 */
    static void fillHoles(Mesh& mesh, float maxHoleSize = 5.0f);

    /** @brief 法线一致化 */
    static void unifyNormals(Mesh& mesh);

    /** @brief 非流形边检测 */
    static int detectNonManifoldEdges(const Mesh& mesh);

    /** @brief 顶点焊接（去重） */
    static void weldVertices(Mesh& mesh, float threshold = 1e-5f);

    /** @brief Laplacian平滑 */
    static void laplacianSmooth(Mesh& mesh, int iterations = 3, float lambda = 0.5f);

    /** @brief Taubin平滑（保特征） */
    static void taubinSmooth(Mesh& mesh, int iterations = 5,
                             float lambda = 0.5f, float mu = -0.53f);

    /** @brief 网格简化（边折叠 QEM） */
    static void simplify(Mesh& mesh, float targetRatio = 0.5f);
};

// ============================================================
// 体素化与SDF
// ============================================================

/** @brief 网格 → 体素化 */
class Voxelizer {
public:
    /** @brief 将网格体素化为密度场
     *  @param mesh 输入网格
     *  @param resolution 体素分辨率
     *  @return 体素网格
     */
    static VoxelGrid voxelize(const Mesh& mesh, int resolution = 128);

    /** @brief 生成SDF（有向距离场）
     *  @param mesh 输入网格
     *  @param resolution 分辨率
     *  @param bandWidth 窄带宽度（体素数）
     *  @return SDF网格
     */
    static SDFGrid generateSDF(const Mesh& mesh, int resolution = 128,
                               int bandWidth = 4);
};

// ============================================================
// 重建管线
// ============================================================

/** @brief 重建配置 */
struct ReconstructionConfig {
    int   voxelResolution = 128;    ///< 体素分辨率
    float isoValue = 0.5f;          ///< 等值面阈值
    int   smoothIterations = 3;     ///< 平滑迭代次数
    bool  fillHoles = true;         ///< 填充孔洞
    bool  unifyNormals = true;      ///< 法线一致化
    bool  weldVertices = true;      ///< 顶点焊接
    float simplifyRatio = 0.0f;     ///< 简化比例（0=不简化）
    int   poissonDepth = 8;         ///< Poisson重建深度
};

/** @brief 重建管线 */
class ReconstructionPipeline {
public:
    /** @brief 从体素数据重建网格 */
    static Mesh reconstructFromVoxels(const VoxelGrid& grid,
                                      const ReconstructionConfig& config);

    /** @brief 从点云重建网格 */
    static Mesh reconstructFromPointCloud(const PointCloud& cloud,
                                         const ReconstructionConfig& config);

    /** @brief 从网格修复并优化 */
    static Mesh repairAndOptimize(Mesh mesh,
                                  const ReconstructionConfig& config);

    /** @brief 打印重建统计信息 */
    static void printStats(const Mesh& mesh, const std::string& stage);
};

} // namespace NailPrint3D
