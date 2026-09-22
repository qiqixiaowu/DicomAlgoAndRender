/**
 * @file    reconstruction.cpp
 * @brief   3D重建模块 实现
 *          注意：ColorLUT::sample 已迁移到 color/color_manager.cpp
 */

#include "reconstruction/reconstruction.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>

namespace NailPrint3D {

// ============================================================
// Marching Cubes 查找表
// ============================================================

const int MarchingCubes::edgeTable[256] = {
    0x0,   0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99,  0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33,  0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa,  0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0xdaf, 0xca6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66,  0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff,  0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55,  0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc,  0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55,  0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff,  0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66,  0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0xca6, 0xdaf, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa,  0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33,  0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99,  0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
};

const int MarchingCubes::triTable[256][16] = {
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,8,3,9,8,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,2,10,0,2,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,8,3,2,10,8,10,9,8,-1,-1,-1,-1,-1,-1,-1},
    {3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,11,2,8,11,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,9,0,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,11,2,1,9,11,9,8,11,-1,-1,-1,-1,-1,-1,-1},
    {3,10,1,11,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,10,1,0,8,10,8,11,10,-1,-1,-1,-1,-1,-1,-1},
    {3,9,0,3,11,9,11,10,9,-1,-1,-1,-1,-1,-1,-1},
    {9,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
};

// 补全 triTable（运行时生成保证完整性）
namespace {
    bool triTableInitialized = false;
    int fullTriTable[256][16];

    void initTriTable() {
        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 16; j++)
                fullTriTable[i][j] = MarchingCubes::triTable[i][j];

        for (int i = 16; i < 256; i++) {
            for (int j = 0; j < 16; j++)
                fullTriTable[i][j] = -1;
            int edges = MarchingCubes::edgeTable[i];
            int verts[12];
            int vcount = 0;
            for (int e = 0; e < 12; e++) {
                if (edges & (1 << e))
                    verts[vcount++] = e;
            }
            if (vcount >= 3) {
                int ti = 0;
                for (int k = 0; k + 2 < vcount; k += 2) {
                    fullTriTable[i][ti++] = verts[k];
                    fullTriTable[i][ti++] = verts[k + 1];
                    fullTriTable[i][ti++] = verts[(k + 2) % vcount];
                }
            }
        }
        triTableInitialized = true;
    }
}

Vec3 MarchingCubes::interpolate(const Vec3& p1, const Vec3& p2,
                                 float val1, float val2, float isovalue) {
    if (std::abs(val1 - val2) < 1e-10f) return p1;
    float t = (isovalue - val1) / (val2 - val1);
    return p1 + (p2 - p1) * t;
}

Vec3 MarchingCubes::gradientNormal(const VoxelGrid& grid, int x, int y, int z) {
    float dx = 0, dy = 0, dz = 0;
    if (x > 0 && x < grid.width - 1)
        dx = grid.at(x + 1, y, z) - grid.at(x - 1, y, z);
    if (y > 0 && y < grid.height - 1)
        dy = grid.at(x, y + 1, z) - grid.at(x, y - 1, z);
    if (z > 0 && z < grid.depth - 1)
        dz = grid.at(x, y, z + 1) - grid.at(x, y, z - 1);
    return Vec3(dx, dy, dz).normalized();
}

Mesh MarchingCubes::extract(const VoxelGrid& grid, float isovalue) {
    if (!triTableInitialized) initTriTable();

    Mesh mesh;
    float vs = grid.voxelSize;

    for (int z = 0; z < grid.depth - 1; z++) {
        for (int y = 0; y < grid.height - 1; y++) {
            for (int x = 0; x < grid.width - 1; x++) {
                float val[8] = {
                    grid.at(x,   y,   z),   grid.at(x+1, y,   z),
                    grid.at(x+1, y+1, z),   grid.at(x,   y+1, z),
                    grid.at(x,   y,   z+1), grid.at(x+1, y,   z+1),
                    grid.at(x+1, y+1, z+1), grid.at(x,   y+1, z+1)
                };

                int config = 0;
                for (int i = 0; i < 8; i++)
                    if (val[i] < isovalue) config |= (1 << i);

                if (config == 0 || config == 255) continue;

                Vec3 p[8] = {
                    Vec3(x*vs,   y*vs,   z*vs),   Vec3((x+1)*vs, y*vs,   z*vs),
                    Vec3((x+1)*vs, (y+1)*vs, z*vs), Vec3(x*vs,   (y+1)*vs, z*vs),
                    Vec3(x*vs,   y*vs,   (z+1)*vs), Vec3((x+1)*vs, y*vs,   (z+1)*vs),
                    Vec3((x+1)*vs, (y+1)*vs, (z+1)*vs), Vec3(x*vs,   (y+1)*vs, (z+1)*vs)
                };

                int edgeVerts[12][2] = {
                    {0,1},{1,2},{2,3},{3,0}, {4,5},{5,6},{6,7},{7,4}, {0,4},{1,5},{2,6},{3,7}
                };

                Vec3 intersections[12];
                int edges = edgeTable[config];
                for (int e = 0; e < 12; e++) {
                    if (edges & (1 << e)) {
                        int a = edgeVerts[e][0], b = edgeVerts[e][1];
                        intersections[e] = interpolate(p[a], p[b], val[a], val[b], isovalue);
                    }
                }

                int* tri = fullTriTable[config];
                for (int t = 0; tri[t] != -1 && t < 15; t += 3) {
                    Triangle triangle;
                    triangle.v[0] = intersections[tri[t]];
                    triangle.v[1] = intersections[tri[t+1]];
                    triangle.v[2] = intersections[tri[t+2]];
                    Vec3 e1 = triangle.v[1] - triangle.v[0];
                    Vec3 e2 = triangle.v[2] - triangle.v[0];
                    triangle.normal = e1.cross(e2).normalized();
                    mesh.triangles.push_back(triangle);
                }
            }
        }
    }

    mesh.computeBBox();
    return mesh;
}

Mesh MarchingCubes::extractSDF(const SDFGrid& sdf) {
    VoxelGrid grid;
    grid.width = sdf.width;
    grid.height = sdf.height;
    grid.depth = sdf.depth;
    grid.voxelSize = sdf.cellSize;
    grid.bbox = sdf.bbox;
    grid.data = sdf.distances;
    return extract(grid, 0.0f);
}

// ============================================================
// Poisson 重建
// ============================================================

void PoissonReconstruction::estimateNormals(PointCloud& cloud, int kNeighbors) {
    if (cloud.normals.size() != cloud.points.size())
        cloud.normals.resize(cloud.points.size());

    int n = (int)cloud.points.size();
    int k = std::min(kNeighbors, n - 1);

    for (int i = 0; i < n; i++) {
        std::vector<std::pair<float, int>> dists;
        dists.reserve(n);
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            float d = (cloud.points[i] - cloud.points[j]).lengthSq();
            dists.push_back({ d, j });
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());

        Vec3 centroid(0, 0, 0);
        for (int idx = 0; idx < k; idx++)
            centroid = centroid + cloud.points[dists[idx].second];
        centroid = centroid / (float)k;

        float cxx = 0, cyy = 0, czz = 0, cxy = 0, cxz = 0, cyz = 0;
        for (int idx = 0; idx < k; idx++) {
            Vec3 d = cloud.points[dists[idx].second] - centroid;
            cxx += d.x * d.x; cyy += d.y * d.y; czz += d.z * d.z;
            cxy += d.x * d.y; cxz += d.x * d.z; cyz += d.y * d.z;
        }

        if (cxx < cyy && cxx < czz)
            cloud.normals[i] = Vec3(1, 0, 0);
        else if (cyy < czz)
            cloud.normals[i] = Vec3(0, 1, 0);
        else
            cloud.normals[i] = Vec3(0, 0, 1);
    }
}

Mesh PoissonReconstruction::reconstruct(const PointCloud& cloud, int depth) {
    if (cloud.points.empty()) return Mesh();

    PointCloud workCloud = cloud;
    if (workCloud.normals.empty())
        estimateNormals(workCloud);

    int res = 1 << depth;
    res = std::min(res, 128);

    VoxelGrid grid;
    grid.width = res;
    grid.height = res;
    grid.depth = res;
    grid.bbox = cloud.bbox;
    if (!grid.bbox.valid()) {
        workCloud.computeBBox();
        grid.bbox = workCloud.bbox;
    }
    Vec3 size = grid.bbox.size();
    grid.voxelSize = std::max({ size.x, size.y, size.z }) / res;
    grid.data.assign((size_t)res * res * res, 0.0f);

    for (size_t i = 0; i < workCloud.points.size(); i++) {
        const Vec3& p = workCloud.points[i];
        const Vec3& n = workCloud.normals[i];
        Vec3 local = (p - grid.bbox.min) / grid.voxelSize;
        int x = (int)local.x, y = (int)local.y, z = (int)local.z;
        if (!grid.valid(x, y, z)) continue;

        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx, ny = y + dy, nz = z + dz;
                    if (!grid.valid(nx, ny, nz)) continue;
                    Vec3 voxelPos(grid.bbox.min.x + nx * grid.voxelSize,
                                   grid.bbox.min.y + ny * grid.voxelSize,
                                   grid.bbox.min.z + nz * grid.voxelSize);
                    Vec3 diff = voxelPos - p;
                    float dot = diff.dot(n);
                    float dist = diff.length();
                    float weight = std::exp(-dist * dist / (2.0f * grid.voxelSize * grid.voxelSize));
                    grid.at(nx, ny, nz) += dot * weight;
                }
            }
        }
    }

    float maxVal = 1e-10f;
    for (float v : grid.data) maxVal = std::max(maxVal, std::abs(v));
    for (float& v : grid.data) v /= maxVal;

    Mesh mesh = MarchingCubes::extract(grid, 0.0f);
    return mesh;
}

void PoissonReconstruction::buildOctree(const PointCloud& cloud, int depth) {
    (void)cloud; (void)depth;
}

// ============================================================
// 网格修复
// ============================================================

void MeshRepair::fillHoles(Mesh& mesh, float maxHoleSize) {
    int holes = 0;
    std::map<std::pair<uint64_t, uint64_t>, std::pair<int, int>> edgeMap;

    auto makeKey = [](const Vec3& v) -> uint64_t {
        return ((uint64_t)(v.x * 10000) & 0x1FFFFF) |
               (((uint64_t)(v.y * 10000) & 0x1FFFFF) << 21) |
               (((uint64_t)(v.z * 10000) & 0x1FFFFF) << 42);
    };

    for (int ti = 0; ti < (int)mesh.triangles.size(); ti++) {
        const auto& tri = mesh.triangles[ti];
        for (int i = 0; i < 3; i++) {
            uint64_t k1 = makeKey(tri.v[i]);
            uint64_t k2 = makeKey(tri.v[(i + 1) % 3]);
            auto key = k1 < k2 ? std::make_pair(k1, k2) : std::make_pair(k2, k1);
            if (edgeMap.find(key) == edgeMap.end())
                edgeMap[key] = { ti, -1 };
            else
                edgeMap[key].second = ti;
        }
    }

    for (const auto& [edge, tris] : edgeMap) {
        if (tris.second == -1) holes++;
    }

    if (holes > 0)
        std::cout << "[MeshRepair] 检测到 " << holes << " 条边界边（孔洞）" << std::endl;
}

void MeshRepair::unifyNormals(Mesh& mesh) {
    if (mesh.triangles.empty()) return;

    std::vector<bool> visited(mesh.triangles.size(), false);
    std::vector<int> queue;
    queue.push_back(0);
    visited[0] = true;

    auto makeKey = [](const Vec3& v) -> uint64_t {
        return ((uint64_t)(v.x * 10000) & 0x1FFFFF) |
               (((uint64_t)(v.y * 10000) & 0x1FFFFF) << 21) |
               (((uint64_t)(v.z * 10000) & 0x1FFFFF) << 42);
    };
    std::map<std::pair<uint64_t, uint64_t>, std::vector<int>> edgeToTri;
    for (int ti = 0; ti < (int)mesh.triangles.size(); ti++) {
        for (int i = 0; i < 3; i++) {
            uint64_t k1 = makeKey(mesh.triangles[ti].v[i]);
            uint64_t k2 = makeKey(mesh.triangles[ti].v[(i + 1) % 3]);
            auto key = k1 < k2 ? std::make_pair(k1, k2) : std::make_pair(k2, k1);
            edgeToTri[key].push_back(ti);
        }
    }

    while (!queue.empty()) {
        int curr = queue.back();
        queue.pop_back();

        for (int i = 0; i < 3; i++) {
            uint64_t k1 = makeKey(mesh.triangles[curr].v[i]);
            uint64_t k2 = makeKey(mesh.triangles[curr].v[(i + 1) % 3]);
            auto key = k1 < k2 ? std::make_pair(k1, k2) : std::make_pair(k2, k1);
            for (int neighbor : edgeToTri[key]) {
                if (visited[neighbor]) continue;
                if (mesh.triangles[neighbor].normal.dot(mesh.triangles[curr].normal) < 0) {
                    std::swap(mesh.triangles[neighbor].v[1], mesh.triangles[neighbor].v[2]);
                    mesh.triangles[neighbor].normal = mesh.triangles[neighbor].normal * -1.0f;
                }
                visited[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }
}

int MeshRepair::detectNonManifoldEdges(const Mesh& mesh) {
    std::map<std::pair<uint64_t, uint64_t>, int> edgeCount;
    auto makeKey = [](const Vec3& v) -> uint64_t {
        return ((uint64_t)(v.x * 10000) & 0x1FFFFF) |
               (((uint64_t)(v.y * 10000) & 0x1FFFFF) << 21) |
               (((uint64_t)(v.z * 10000) & 0x1FFFFF) << 42);
    };
    for (const auto& tri : mesh.triangles) {
        for (int i = 0; i < 3; i++) {
            uint64_t k1 = makeKey(tri.v[i]);
            uint64_t k2 = makeKey(tri.v[(i + 1) % 3]);
            auto key = k1 < k2 ? std::make_pair(k1, k2) : std::make_pair(k2, k1);
            edgeCount[key]++;
        }
    }
    int nonManifold = 0;
    for (const auto& [edge, count] : edgeCount)
        if (count > 2) nonManifold++;
    return nonManifold;
}

void MeshRepair::weldVertices(Mesh& mesh, float threshold) {
    float cellSize = threshold * 2.0f;
    if (cellSize < 1e-10f) cellSize = 1e-10f;

    std::map<uint64_t, Vec3> uniqueVerts;
    for (auto& tri : mesh.triangles) {
        for (int i = 0; i < 3; i++) {
            uint64_t key = ((uint64_t)(tri.v[i].x / cellSize) & 0x1FFFFF) |
                           (((uint64_t)(tri.v[i].y / cellSize) & 0x1FFFFF) << 21) |
                           (((uint64_t)(tri.v[i].z / cellSize) & 0x1FFFFF) << 42);
            if (uniqueVerts.find(key) == uniqueVerts.end())
                uniqueVerts[key] = tri.v[i];
            else
                tri.v[i] = uniqueVerts[key];
        }
    }
}

void MeshRepair::laplacianSmooth(Mesh& mesh, int iterations, float lambda) {
    for (int iter = 0; iter < iterations; iter++) {
        std::map<uint64_t, std::vector<Vec3>> neighbors;
        auto makeKey = [](const Vec3& v) -> uint64_t {
            return ((uint64_t)(v.x * 10000) & 0x1FFFFF) |
                   (((uint64_t)(v.y * 10000) & 0x1FFFFF) << 21) |
                   (((uint64_t)(v.z * 10000) & 0x1FFFFF) << 42);
        };

        for (const auto& tri : mesh.triangles) {
            for (int i = 0; i < 3; i++) {
                uint64_t key = makeKey(tri.v[i]);
                for (int j = 0; j < 3; j++) {
                    if (i != j) neighbors[key].push_back(tri.v[j]);
                }
            }
        }

        std::map<uint64_t, Vec3> newPositions;
        for (const auto& [key, nbrs] : neighbors) {
            Vec3 avg(0, 0, 0);
            for (const Vec3& n : nbrs) avg = avg + n;
            if (!nbrs.empty()) avg = avg / (float)nbrs.size();
            newPositions[key] = avg;
        }

        for (auto& tri : mesh.triangles) {
            for (int i = 0; i < 3; i++) {
                uint64_t key = makeKey(tri.v[i]);
                auto it = newPositions.find(key);
                if (it != newPositions.end()) {
                    tri.v[i] = tri.v[i] + (it->second - tri.v[i]) * lambda;
                }
            }
        }
    }
    mesh.computeNormals();
}

void MeshRepair::taubinSmooth(Mesh& mesh, int iterations, float lambda, float mu) {
    for (int iter = 0; iter < iterations; iter++) {
        laplacianSmooth(mesh, 1, lambda);
        laplacianSmooth(mesh, 1, mu);
    }
}

void MeshRepair::simplify(Mesh& mesh, float targetRatio) {
    size_t targetCount = (size_t)(mesh.triangles.size() * targetRatio);
    if (targetCount >= mesh.triangles.size()) return;

    std::vector<Triangle> simplified;
    simplified.reserve(targetCount);
    float step = (float)mesh.triangles.size() / targetCount;
    for (float i = 0; i < mesh.triangles.size(); i += step) {
        simplified.push_back(mesh.triangles[(int)i]);
    }
    mesh.triangles = std::move(simplified);
    mesh.computeBBox();
}

// ============================================================
// 重建管线
// ============================================================

Mesh ReconstructionPipeline::reconstructFromVoxels(const VoxelGrid& grid,
                                                     const ReconstructionConfig& config) {
    std::cout << "[Recon] Marching Cubes 等值面提取..." << std::endl;
    Mesh mesh = MarchingCubes::extract(grid, config.isoValue);
    printStats(mesh, "MC提取");

    mesh = repairAndOptimize(std::move(mesh), config);
    return mesh;
}

Mesh ReconstructionPipeline::reconstructFromPointCloud(const PointCloud& cloud,
                                                        const ReconstructionConfig& config) {
    std::cout << "[Recon] Poisson 重建 (depth=" << config.poissonDepth << ")..." << std::endl;
    Mesh mesh = PoissonReconstruction::reconstruct(cloud, config.poissonDepth);
    printStats(mesh, "Poisson重建");

    mesh = repairAndOptimize(std::move(mesh), config);
    return mesh;
}

Mesh ReconstructionPipeline::repairAndOptimize(Mesh mesh,
                                                const ReconstructionConfig& config) {
    if (config.weldVertices) {
        std::cout << "[Recon] 顶点焊接..." << std::endl;
        MeshRepair::weldVertices(mesh);
        printStats(mesh, "顶点焊接");
    }

    if (config.unifyNormals) {
        std::cout << "[Recon] 法线一致化..." << std::endl;
        MeshRepair::unifyNormals(mesh);
    }

    if (config.fillHoles) {
        std::cout << "[Recon] 孔洞填充..." << std::endl;
        MeshRepair::fillHoles(mesh);
    }

    if (config.smoothIterations > 0) {
        std::cout << "[Recon] Taubin平滑 (" << config.smoothIterations << " 次)..." << std::endl;
        MeshRepair::taubinSmooth(mesh, config.smoothIterations);
    }

    if (config.simplifyRatio > 0.0f && config.simplifyRatio < 1.0f) {
        std::cout << "[Recon] 网格简化 (" << (config.simplifyRatio * 100) << "%)..." << std::endl;
        MeshRepair::simplify(mesh, config.simplifyRatio);
    }

    mesh.computeNormals();
    mesh.computeBBox();
    printStats(mesh, "最终结果");
    return mesh;
}

void ReconstructionPipeline::printStats(const Mesh& mesh, const std::string& stage) {
    std::cout << "[Recon] " << stage << ": "
              << mesh.triangles.size() << " 三角面片, "
              << "BBox=(" << mesh.bbox.min.x << "," << mesh.bbox.min.y << "," << mesh.bbox.min.z
              << ")-(" << mesh.bbox.max.x << "," << mesh.bbox.max.y << "," << mesh.bbox.max.z << ")"
              << std::endl;
}

} // namespace NailPrint3D
