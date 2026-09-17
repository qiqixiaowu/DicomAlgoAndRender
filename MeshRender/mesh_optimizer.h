#pragma once
/**
 * @file    mesh_optimizer.h
 * @brief   网格优化算法
 *
 * 包含：
 *  - 边折叠网格简化（Quadric Error Metric）
 *  - 网格平滑（Laplacian / Taubin）
 *  - 顶点去重（焊接）
 *  - LOD 生成
 */

#include "mesh.h"
#include <map>
#include <set>
#include <queue>
#include <unordered_map>

namespace MeshRender {

// ============================================================
// 顶点焊接（去重）
// ============================================================

/** @brief 合并位置相近的顶点
 * @param mesh 输入网格
 * @param threshold 距离阈值
 */
inline void weldVertices(MeshData& mesh, float threshold = 1e-5f) {
    if (mesh.vertices.empty()) return;

    // 使用网格哈希加速空间查找
    std::unordered_map<int64_t, std::vector<uint32_t>> grid;
    float cellSize = threshold * 2.0f;
    if (cellSize < 1e-10f) cellSize = 1e-10f;

    auto cellKey = [](float x, float y, float z, float cs) -> int64_t {
        int32_t cx = (int32_t)(x / cs);
        int32_t cy = (int32_t)(y / cs);
        int32_t cz = (int32_t)(z / cs);
        return ((int64_t)cx & 0xFFFFF) | (((int64_t)cy & 0xFFFFF) << 20) | (((int64_t)cz & 0xFFFFF) << 40);
    };

    std::vector<uint32_t> remap(mesh.vertices.size());
    std::vector<Vertex> uniqueVerts;
    std::vector<bool> visited(mesh.vertices.size(), false);

    for (uint32_t i = 0; i < mesh.vertices.size(); i++) {
        if (visited[i]) { remap[i] = remap[i]; continue; }

        int64_t key = cellKey(mesh.vertices[i].position[0],
                              mesh.vertices[i].position[1],
                              mesh.vertices[i].position[2], cellSize);
        auto& cell = grid[key];

        bool found = false;
        for (uint32_t j : cell) {
            float dx = mesh.vertices[i].position[0] - mesh.vertices[j].position[0];
            float dy = mesh.vertices[i].position[1] - mesh.vertices[j].position[1];
            float dz = mesh.vertices[i].position[2] - mesh.vertices[j].position[2];
            if (dx*dx + dy*dy + dz*dz < threshold * threshold) {
                remap[i] = remap[j];
                visited[i] = true;
                found = true;
                break;
            }
        }

        if (!found) {
            uint32_t newIdx = (uint32_t)uniqueVerts.size();
            uniqueVerts.push_back(mesh.vertices[i]);
            remap[i] = newIdx;
            visited[i] = true;
            cell.push_back(i);
        }
    }

    // 重映射索引
    for (auto& idx : mesh.indices)
        idx = remap[idx];

    mesh.vertices = std::move(uniqueVerts);
}

// ============================================================
// 网格平滑
// ============================================================

/** @brief Laplacian 平滑
 * @param mesh 网格
 * @param iterations 迭代次数
 * @param lambda 平滑因子 (0~1)
 */
inline void laplacianSmooth(MeshData& mesh, int iterations = 3, float lambda = 0.5f) {
    // 构建邻接表
    std::vector<std::set<uint32_t>> adjacency(mesh.vertices.size());
    for (size_t i = 0; i < mesh.indices.size(); i += 3) {
        uint32_t a = mesh.indices[i], b = mesh.indices[i+1], c = mesh.indices[i+2];
        adjacency[a].insert(b); adjacency[a].insert(c);
        adjacency[b].insert(a); adjacency[b].insert(c);
        adjacency[c].insert(a); adjacency[c].insert(b);
    }

    for (int iter = 0; iter < iterations; iter++) {
        std::vector<float> newPos(mesh.vertices.size() * 3);
        for (uint32_t i = 0; i < mesh.vertices.size(); i++) {
            if (adjacency[i].empty()) {
                newPos[i*3]   = mesh.vertices[i].position[0];
                newPos[i*3+1] = mesh.vertices[i].position[1];
                newPos[i*3+2] = mesh.vertices[i].position[2];
                continue;
            }
            float cx = 0, cy = 0, cz = 0;
            for (uint32_t n : adjacency[i]) {
                cx += mesh.vertices[n].position[0];
                cy += mesh.vertices[n].position[1];
                cz += mesh.vertices[n].position[2];
            }
            float inv = 1.0f / adjacency[i].size();
            cx *= inv; cy *= inv; cz *= inv;
            newPos[i*3]   = mesh.vertices[i].position[0] + lambda * (cx - mesh.vertices[i].position[0]);
            newPos[i*3+1] = mesh.vertices[i].position[1] + lambda * (cy - mesh.vertices[i].position[1]);
            newPos[i*3+2] = mesh.vertices[i].position[2] + lambda * (cz - mesh.vertices[i].position[2]);
        }
        for (uint32_t i = 0; i < mesh.vertices.size(); i++) {
            mesh.vertices[i].position[0] = newPos[i*3];
            mesh.vertices[i].position[1] = newPos[i*3+1];
            mesh.vertices[i].position[2] = newPos[i*3+2];
        }
    }
    computeNormals(mesh);
}

/** @brief Taubin 平滑（双向滤波，保持体积）
 * @param mesh 网格
 * @param iterations 迭代次数
 * @param lambda 平滑因子
 * @param mu 反向因子 (mu < 0, |mu| > lambda)
 */
inline void taubinSmooth(MeshData& mesh, int iterations = 10,
                         float lambda = 0.5f, float mu = -0.53f) {
    std::vector<std::set<uint32_t>> adjacency(mesh.vertices.size());
    for (size_t i = 0; i < mesh.indices.size(); i += 3) {
        uint32_t a = mesh.indices[i], b = mesh.indices[i+1], c = mesh.indices[i+2];
        adjacency[a].insert(b); adjacency[a].insert(c);
        adjacency[b].insert(a); adjacency[b].insert(c);
        adjacency[c].insert(a); adjacency[c].insert(b);
    }

    auto smoothStep = [&](float factor) {
        std::vector<float> newPos(mesh.vertices.size() * 3);
        for (uint32_t i = 0; i < mesh.vertices.size(); i++) {
            if (adjacency[i].empty()) {
                for (int k = 0; k < 3; k++) newPos[i*3+k] = mesh.vertices[i].position[k];
                continue;
            }
            float cx = 0, cy = 0, cz = 0;
            for (uint32_t n : adjacency[i]) {
                cx += mesh.vertices[n].position[0];
                cy += mesh.vertices[n].position[1];
                cz += mesh.vertices[n].position[2];
            }
            float inv = 1.0f / adjacency[i].size();
            cx *= inv; cy *= inv; cz *= inv;
            newPos[i*3]   = mesh.vertices[i].position[0] + factor * (cx - mesh.vertices[i].position[0]);
            newPos[i*3+1] = mesh.vertices[i].position[1] + factor * (cy - mesh.vertices[i].position[1]);
            newPos[i*3+2] = mesh.vertices[i].position[2] + factor * (cz - mesh.vertices[i].position[2]);
        }
        for (uint32_t i = 0; i < mesh.vertices.size(); i++)
            for (int k = 0; k < 3; k++)
                mesh.vertices[i].position[k] = newPos[i*3+k];
    };

    for (int iter = 0; iter < iterations; iter++) {
        smoothStep(lambda);
        smoothStep(mu);
    }
    computeNormals(mesh);
}

// ============================================================
// Quadric Error Metric 网格简化
// ============================================================

/** @brief 4×4 对称矩阵（QEM 用） */
struct SymMat4 {
    float m[10] = {0}; // 上三角：m00,m01,m02,m03,m11,m12,m13,m22,m23,m33

    void add(const SymMat4& o) { for (int i = 0; i < 10; i++) m[i] += o.m[i]; }

    float operator()(float x, float y, float z) const {
        // Q(v) = v^T M v = m00*x² + 2*m01*xy + 2*m02*xz + 2*m03*x
        //      + m11*y² + 2*m12*yz + 2*m13*y + m22*z² + 2*m23*z + m33
        return m[0]*x*x + 2*m[1]*x*y + 2*m[2]*x*z + 2*m[3]*x
             + m[4]*y*y + 2*m[5]*y*z + 2*m[6]*y
             + m[7]*z*z + 2*m[8]*z + m[9];
    }
};

/** @brief 从三角形平面构建 QEM 矩阵 */
inline SymMat4 quadricFromTriangle(const float p0[3], const float p1[3], const float p2[3]) {
    // 平面方程 ax + by + cz + d = 0
    float e1x = p1[0]-p0[0], e1y = p1[1]-p0[1], e1z = p1[2]-p0[2];
    float e2x = p2[0]-p0[0], e2y = p2[1]-p0[1], e2z = p2[2]-p0[2];
    float a = e1y*e2z - e1z*e2y;
    float b = e1z*e2x - e1x*e2z;
    float c = e1x*e2y - e1y*e2x;
    float len = std::sqrt(a*a + b*b + c*c);
    if (len < 1e-12f) len = 1e-12f;
    a /= len; b /= len; c /= len;
    float d = -(a*p0[0] + b*p0[1] + c*p0[2]);

    SymMat4 q;
    q.m[0]=a*a; q.m[1]=a*b; q.m[2]=a*c; q.m[3]=a*d;
    q.m[4]=b*b; q.m[5]=b*c; q.m[6]=b*d;
    q.m[7]=c*c; q.m[8]=c*d;
    q.m[9]=d*d;
    return q;
}

/** @brief 边折叠网格简化（基于 QEM）
 * @param mesh 输入网格
 * @param targetRatio 目标三角形比例 (0~1, 如 0.5 = 简化到 50%)
 * @return 简化后的网格
 */
inline MeshData simplifyMesh(const MeshData& mesh, float targetRatio) {
    if (mesh.indices.empty() || targetRatio >= 1.0f) return mesh;

    // 创建可变副本（后续折叠操作需要修改顶点位置）
    MeshData working = mesh;

    size_t targetTris = (size_t)(mesh.triangleCount() * targetRatio);
    if (targetTris < 2) targetTris = 2;

    // 1. 计算每个顶点的 QEM 矩阵
    std::vector<SymMat4> vertexQuadrics(working.vertices.size());
    for (size_t i = 0; i < working.indices.size(); i += 3) {
        uint32_t i0 = working.indices[i], i1 = working.indices[i+1], i2 = working.indices[i+2];
        SymMat4 q = quadricFromTriangle(
            working.vertices[i0].position, working.vertices[i1].position, working.vertices[i2].position);
        vertexQuadrics[i0].add(q);
        vertexQuadrics[i1].add(q);
        vertexQuadrics[i2].add(q);
    }

    // 2. 构建边列表
    struct Edge {
        uint32_t v0, v1;
        float error;
        float newPos[3];
        bool operator<(const Edge& o) const { return error > o.error; } // 最小堆
    };

    std::set<std::pair<uint32_t,uint32_t>> edgeSet;
    std::vector<std::set<uint32_t>> adjacency(working.vertices.size());
    for (size_t i = 0; i < working.indices.size(); i += 3) {
        uint32_t a = working.indices[i], b = working.indices[i+1], c = working.indices[i+2];
        uint32_t e0 = std::min(a,b), e1 = std::max(a,b);
        uint32_t e2 = std::min(b,c), e3 = std::max(b,c);
        uint32_t e4 = std::min(a,c), e5 = std::max(a,c);
        edgeSet.insert({e0,e1}); edgeSet.insert({e2,e3}); edgeSet.insert({e4,e5});
        adjacency[a].insert(b); adjacency[a].insert(c);
        adjacency[b].insert(a); adjacency[b].insert(c);
        adjacency[c].insert(a); adjacency[c].insert(b);
    }

    // 3. 计算每条边的折叠误差
    std::priority_queue<Edge> edgeQueue;
    for (auto& [v0, v1] : edgeSet) {
        SymMat4 q = vertexQuadrics[v0];
        q.add(vertexQuadrics[v1]);

        // 最优折叠位置：解 ∇Q = 0
        // 简化：取中点或 v0/v1，选误差最小的
        float mid[3] = {
            (working.vertices[v0].position[0] + working.vertices[v1].position[0]) * 0.5f,
            (working.vertices[v0].position[1] + working.vertices[v1].position[1]) * 0.5f,
            (working.vertices[v0].position[2] + working.vertices[v1].position[2]) * 0.5f,
        };
        float errMid = q(mid[0], mid[1], mid[2]);
        float errV0  = q(working.vertices[v0].position[0], working.vertices[v0].position[1], working.vertices[v0].position[2]);
        float errV1  = q(working.vertices[v1].position[0], working.vertices[v1].position[1], working.vertices[v1].position[2]);

        Edge edge;
        edge.v0 = v0; edge.v1 = v1;
        if (errMid <= errV0 && errMid <= errV1) {
            edge.error = errMid;
            edge.newPos[0] = mid[0]; edge.newPos[1] = mid[1]; edge.newPos[2] = mid[2];
        } else if (errV0 <= errV1) {
            edge.error = errV0;
            edge.newPos[0] = working.vertices[v0].position[0];
            edge.newPos[1] = working.vertices[v0].position[1];
            edge.newPos[2] = working.vertices[v0].position[2];
        } else {
            edge.error = errV1;
            edge.newPos[0] = working.vertices[v1].position[0];
            edge.newPos[1] = working.vertices[v1].position[1];
            edge.newPos[2] = working.vertices[v1].position[2];
        }
        edgeQueue.push(edge);
    }

    // 4. 逐步折叠
    std::vector<bool> vertexAlive(working.vertices.size(), true);
    std::vector<uint32_t> vertexRemap(working.vertices.size());
    for (uint32_t i = 0; i < working.vertices.size(); i++) vertexRemap[i] = i;

    size_t currentTris = working.triangleCount();

    while (currentTris > targetTris && !edgeQueue.empty()) {
        Edge edge = edgeQueue.top();
        edgeQueue.pop();

        if (!vertexAlive[edge.v0] || !vertexAlive[edge.v1]) continue;
        if (vertexRemap[edge.v0] != edge.v0 || vertexRemap[edge.v1] != edge.v1) continue;

        // 折叠 v1 → v0
        uint32_t keep = edge.v0, remove = edge.v1;
        working.vertices[keep].position[0] = edge.newPos[0];
        working.vertices[keep].position[1] = edge.newPos[1];
        working.vertices[keep].position[2] = edge.newPos[2];

        vertexQuadrics[keep].add(vertexQuadrics[remove]);
        vertexRemap[remove] = keep;
        vertexAlive[remove] = false;

        // 更新邻接
        for (uint32_t n : adjacency[remove]) {
            if (vertexAlive[n] && n != keep) {
                adjacency[keep].insert(n);
                adjacency[n].insert(keep);
                adjacency[n].erase(remove);
            }
        }
        adjacency[remove].clear();
    }

    // 5. 重建索引
    MeshData result;
    result.name = working.name + "_simplified";

    // 重映射索引并移除退化三角形
    std::unordered_map<uint32_t, uint32_t> indexRemap;
    for (size_t i = 0; i < working.indices.size(); i += 3) {
        uint32_t a = vertexRemap[working.indices[i]];
        uint32_t b = vertexRemap[working.indices[i+1]];
        uint32_t c = vertexRemap[working.indices[i+2]];
        if (a == b || b == c || a == c) {
            currentTris--;
            continue;
        }
        for (uint32_t v : {a, b, c}) {
            if (indexRemap.find(v) == indexRemap.end()) {
                indexRemap[v] = (uint32_t)result.vertices.size();
                result.vertices.push_back(working.vertices[v]);
            }
        }
        result.indices.push_back(indexRemap[a]);
        result.indices.push_back(indexRemap[b]);
        result.indices.push_back(indexRemap[c]);
    }

    computeNormals(result);
    return result;
}

// ============================================================
// LOD 生成
// ============================================================

/** @brief LOD 层级数据 */
struct LODLevel {
    MeshData mesh;
    float screenRatio;  ///< 屏幕占比阈值
};

/** @brief 生成多级 LOD
 * @param baseMesh 原始网格
 * @param levels LOD 层数（含原始）
 * @return LOD 链
 */
inline std::vector<LODLevel> generateLODChain(const MeshData& baseMesh, int levels = 4) {
    std::vector<LODLevel> lodChain;
    if (levels < 1) levels = 1;

    lodChain.push_back({baseMesh, 0.0f});

    for (int l = 1; l < levels; l++) {
        float ratio = 1.0f / (1 << l);  // 1/2, 1/4, 1/8...
        MeshData simplified = simplifyMesh(lodChain.back().mesh, ratio);
        if (simplified.triangleCount() < 2) break;

        LODLevel lod;
        lod.mesh = simplified;
        lod.screenRatio = (float)l * 0.15f;  // 简化：按层级递增阈值
        lodChain.push_back(lod);
    }
    return lodChain;
}

} // namespace MeshRender
