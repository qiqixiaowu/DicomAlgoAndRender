#pragma once
/**
 * @file    mesh.h
 * @brief   Mesh 数据结构与几何生成（程序化网格）
 *
 * 包含：
 *  - Vertex / MeshData 数据结构
 *  - 程序化网格生成：立方体、球体、圆环、平面、圆柱、茶壶近似
 *  - 网格工具：计算法线、切线、AABB、合并
 */

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace MeshRender {

// ============================================================
// 数据结构
// ============================================================

/** @brief 顶点属性（交错布局，32 字节对齐友好） */
struct Vertex {
    float position[3];   ///< 位置
    float normal[3];     ///< 法线
    float texcoord[2];   ///< 纹理坐标
    float tangent[3];    ///< 切线（用于法线映射）
    float bitangent[3];  ///< 副切线

    Vertex() {
        for (int i = 0; i < 3; i++) { position[i] = 0; normal[i] = 0; tangent[i] = 0; bitangent[i] = 0; }
        texcoord[0] = texcoord[1] = 0;
    }

    Vertex(float px, float py, float pz) {
        position[0] = px; position[1] = py; position[2] = pz;
        for (int i = 0; i < 3; i++) { normal[i] = 0; tangent[i] = 0; bitangent[i] = 0; }
        texcoord[0] = texcoord[1] = 0;
    }
};

/** @brief 网格数据（CPU 端） */
struct MeshData {
    std::vector<Vertex>  vertices;
    std::vector<uint32_t> indices;
    std::string name;

    void clear() { vertices.clear(); indices.clear(); name.clear(); }
    size_t vertexCount() const { return vertices.size(); }
    size_t indexCount()  const { return indices.size(); }
    size_t triangleCount() const { return indices.size() / 3; }

    /** 合并另一个网格 */
    void append(const MeshData& other) {
        uint32_t offset = (uint32_t)vertices.size();
        vertices.insert(vertices.end(), other.vertices.begin(), other.vertices.end());
        for (uint32_t idx : other.indices)
            indices.push_back(idx + offset);
    }
};

/** @brief AABB 包围盒 */
struct AABB {
    float min[3] = { 1e30f, 1e30f, 1e30f };
    float max[3] = { -1e30f, -1e30f, -1e30f };

    void expand(const float p[3]) {
        for (int i = 0; i < 3; i++) {
            min[i] = std::min(min[i], p[i]);
            max[i] = std::max(max[i], p[i]);
        }
    }

    void expand(const AABB& other) {
        for (int i = 0; i < 3; i++) {
            min[i] = std::min(min[i], other.min[i]);
            max[i] = std::max(max[i], other.max[i]);
        }
    }

    float* center() const {
        static float c[3];
        for (int i = 0; i < 3; i++) c[i] = (min[i] + max[i]) * 0.5f;
        return c;
    }

    float diagonal() const {
        float d = 0;
        for (int i = 0; i < 3; i++) {
            float t = max[i] - min[i];
            d += t * t;
        }
        return std::sqrt(d);
    }
};

// ============================================================
// 程序化网格生成
// ============================================================

/** @brief 生成立方体网格 */
inline MeshData createCube(float size = 1.0f) {
    MeshData mesh;
    mesh.name = "Cube";
    float s = size * 0.5f;

    // 6 面 × 4 顶点
    // 每面：position, normal, texcoord
    struct FaceDef {
        float dir[3];       // 法线方向
        float positions[4][3];
        float uvs[4][2];
    };

    FaceDef faces[6] = {
        // +X
        {{1,0,0}, {{s,-s,-s},{s,s,-s},{s,s,s},{s,-s,s}}, {{0,0},{1,0},{1,1},{0,1}}},
        // -X
        {{-1,0,0}, {{-s,-s,s},{-s,s,s},{-s,s,-s},{-s,-s,-s}}, {{0,0},{1,0},{1,1},{0,1}}},
        // +Y
        {{0,1,0}, {{-s,s,-s},{-s,s,s},{s,s,s},{s,s,-s}}, {{0,0},{1,0},{1,1},{0,1}}},
        // -Y
        {{0,-1,0}, {{-s,-s,s},{-s,-s,-s},{s,-s,-s},{s,-s,s}}, {{0,0},{1,0},{1,1},{0,1}}},
        // +Z
        {{0,0,1}, {{-s,-s,s},{s,-s,s},{s,s,s},{-s,s,s}}, {{0,0},{1,0},{1,1},{0,1}}},
        // -Z
        {{0,0,-1}, {{s,-s,-s},{-s,-s,-s},{-s,s,-s},{s,s,-s}}, {{0,0},{1,0},{1,1},{0,1}}},
    };

    for (int f = 0; f < 6; f++) {
        uint32_t base = (uint32_t)mesh.vertices.size();
        for (int v = 0; v < 4; v++) {
            Vertex vert;
            for (int i = 0; i < 3; i++) {
                vert.position[i] = faces[f].positions[v][i];
                vert.normal[i] = faces[f].dir[i];
            }
            vert.texcoord[0] = faces[f].uvs[v][0];
            vert.texcoord[1] = faces[f].uvs[v][1];
            mesh.vertices.push_back(vert);
        }
        mesh.indices.push_back(base + 0);
        mesh.indices.push_back(base + 1);
        mesh.indices.push_back(base + 2);
        mesh.indices.push_back(base + 0);
        mesh.indices.push_back(base + 2);
        mesh.indices.push_back(base + 3);
    }
    return mesh;
}

/** @brief 生成球体网格（UV 球） */
inline MeshData createSphere(float radius = 0.5f, int segments = 32, int rings = 16) {
    MeshData mesh;
    mesh.name = "Sphere";

    for (int r = 0; r <= rings; r++) {
        float v = (float)r / rings;
        float phi = v * 3.14159265358979f;
        float y = radius * std::cos(phi);
        float rxz = radius * std::sin(phi);

        for (int s = 0; s <= segments; s++) {
            float u = (float)s / segments;
            float theta = u * 2.0f * 3.14159265358979f;
            float x = rxz * std::cos(theta);
            float z = rxz * std::sin(theta);

            Vertex vert;
            vert.position[0] = x; vert.position[1] = y; vert.position[2] = z;
            vert.normal[0] = x / radius; vert.normal[1] = y / radius; vert.normal[2] = z / radius;
            vert.texcoord[0] = u; vert.texcoord[1] = v;
            mesh.vertices.push_back(vert);
        }
    }

    for (int r = 0; r < rings; r++) {
        for (int s = 0; s < segments; s++) {
            uint32_t a = r * (segments + 1) + s;
            uint32_t b = a + segments + 1;
            mesh.indices.push_back(a);
            mesh.indices.push_back(b);
            mesh.indices.push_back(a + 1);
            mesh.indices.push_back(a + 1);
            mesh.indices.push_back(b);
            mesh.indices.push_back(b + 1);
        }
    }
    return mesh;
}

/** @brief 生成圆环（Torus）网格 */
inline MeshData createTorus(float majorRadius = 0.5f, float minorRadius = 0.15f,
                            int majorSegments = 48, int minorSegments = 24) {
    MeshData mesh;
    mesh.name = "Torus";

    const float PI = 3.14159265358979f;
    for (int i = 0; i <= majorSegments; i++) {
        float u = (float)i / majorSegments;
        float theta = u * 2.0f * PI;
        float ct = std::cos(theta), st = std::sin(theta);

        for (int j = 0; j <= minorSegments; j++) {
            float v = (float)j / minorSegments;
            float phi = v * 2.0f * PI;
            float cp = std::cos(phi), sp = std::sin(phi);

            Vertex vert;
            vert.position[0] = (majorRadius + minorRadius * cp) * ct;
            vert.position[1] = minorRadius * sp;
            vert.position[2] = (majorRadius + minorRadius * cp) * st;

            // 法线指向圆环中心轴外
            vert.normal[0] = cp * ct;
            vert.normal[1] = sp;
            vert.normal[2] = cp * st;

            vert.texcoord[0] = u * 4.0f;  // 纹理重复
            vert.texcoord[1] = v * 2.0f;
            mesh.vertices.push_back(vert);
        }
    }

    for (int i = 0; i < majorSegments; i++) {
        for (int j = 0; j < minorSegments; j++) {
            uint32_t a = i * (minorSegments + 1) + j;
            uint32_t b = a + minorSegments + 1;
            mesh.indices.push_back(a);
            mesh.indices.push_back(b);
            mesh.indices.push_back(a + 1);
            mesh.indices.push_back(a + 1);
            mesh.indices.push_back(b);
            mesh.indices.push_back(b + 1);
        }
    }
    return mesh;
}

/** @brief 生成平面网格（细分） */
inline MeshData createPlane(float size = 2.0f, int subdivisions = 1) {
    MeshData mesh;
    mesh.name = "Plane";

    int grid = std::max(1, subdivisions);
    float step = size / grid;
    float half = size * 0.5f;

    for (int z = 0; z <= grid; z++) {
        for (int x = 0; x <= grid; x++) {
            Vertex vert;
            vert.position[0] = -half + x * step;
            vert.position[1] = 0.0f;
            vert.position[2] = -half + z * step;
            vert.normal[0] = 0; vert.normal[1] = 1; vert.normal[2] = 0;
            vert.texcoord[0] = (float)x / grid;
            vert.texcoord[1] = (float)z / grid;
            mesh.vertices.push_back(vert);
        }
    }

    for (int z = 0; z < grid; z++) {
        for (int x = 0; x < grid; x++) {
            uint32_t a = z * (grid + 1) + x;
            uint32_t b = a + grid + 1;
            mesh.indices.push_back(a);
            mesh.indices.push_back(b);
            mesh.indices.push_back(a + 1);
            mesh.indices.push_back(a + 1);
            mesh.indices.push_back(b);
            mesh.indices.push_back(b + 1);
        }
    }
    return mesh;
}

/** @brief 生成圆柱网格 */
inline MeshData createCylinder(float radius = 0.5f, float height = 1.0f,
                               int segments = 32, bool capped = true) {
    MeshData mesh;
    mesh.name = "Cylinder";
    float halfH = height * 0.5f;
    const float PI = 3.14159265358979f;

    // 侧面
    for (int s = 0; s <= segments; s++) {
        float u = (float)s / segments;
        float theta = u * 2.0f * PI;
        float ct = std::cos(theta), st = std::sin(theta);

        Vertex bottom, top;
        bottom.position[0] = radius * ct; bottom.position[1] = -halfH; bottom.position[2] = radius * st;
        top.position[0]    = radius * ct; top.position[1] =  halfH; top.position[2] = radius * st;
        bottom.normal[0] = ct; bottom.normal[1] = 0; bottom.normal[2] = st;
        top.normal[0]    = ct; top.normal[1] = 0; top.normal[2] = st;
        bottom.texcoord[0] = u; bottom.texcoord[1] = 0;
        top.texcoord[0]    = u; top.texcoord[1] = 1;
        mesh.vertices.push_back(bottom);
        mesh.vertices.push_back(top);
    }

    for (int s = 0; s < segments; s++) {
        uint32_t base = s * 2;
        mesh.indices.push_back(base);
        mesh.indices.push_back(base + 1);
        mesh.indices.push_back(base + 2);
        mesh.indices.push_back(base + 1);
        mesh.indices.push_back(base + 3);
        mesh.indices.push_back(base + 2);
    }

    // 封盖
    if (capped) {
        uint32_t centerBottom = (uint32_t)mesh.vertices.size();
        Vertex cb; cb.position[1] = -halfH; cb.normal[1] = -1; mesh.vertices.push_back(cb);
        uint32_t centerTop = (uint32_t)mesh.vertices.size();
        Vertex ct2; ct2.position[1] = halfH; ct2.normal[1] = 1; mesh.vertices.push_back(ct2);

        for (int s = 0; s < segments; s++) {
            float u1 = (float)s / segments, u2 = (float)(s + 1) / segments;
            float t1 = u1 * 2.0f * PI, t2 = u2 * 2.0f * PI;

            // 底盖
            Vertex b1, b2;
            b1.position[0] = radius * std::cos(t1); b1.position[1] = -halfH; b1.position[2] = radius * std::sin(t1);
            b2.position[0] = radius * std::cos(t2); b2.position[1] = -halfH; b2.position[2] = radius * std::sin(t2);
            b1.normal[1] = -1; b2.normal[1] = -1;
            b1.texcoord[0] = u1; b1.texcoord[1] = 1;
            b2.texcoord[0] = u2; b2.texcoord[1] = 1;
            uint32_t i1 = (uint32_t)mesh.vertices.size();
            mesh.vertices.push_back(b1);
            uint32_t i2 = (uint32_t)mesh.vertices.size();
            mesh.vertices.push_back(b2);
            mesh.indices.push_back(centerBottom);
            mesh.indices.push_back(i2);
            mesh.indices.push_back(i1);

            // 顶盖
            Vertex t1v, t2v;
            t1v.position[0] = radius * std::cos(t1); t1v.position[1] = halfH; t1v.position[2] = radius * std::sin(t1);
            t2v.position[0] = radius * std::cos(t2); t2v.position[1] = halfH; t2v.position[2] = radius * std::sin(t2);
            t1v.normal[1] = 1; t2v.normal[1] = 1;
            t1v.texcoord[0] = u1; t1v.texcoord[1] = 0;
            t2v.texcoord[0] = u2; t2v.texcoord[1] = 0;
            uint32_t j1 = (uint32_t)mesh.vertices.size();
            mesh.vertices.push_back(t1v);
            uint32_t j2 = (uint32_t)mesh.vertices.size();
            mesh.vertices.push_back(t2v);
            mesh.indices.push_back(centerTop);
            mesh.indices.push_back(j1);
            mesh.indices.push_back(j2);
        }
    }
    return mesh;
}

/** @brief 生成 Icosahedron（正二十面体，适合作为球体细分基础） */
inline MeshData createIcosahedron(float radius = 0.5f) {
    MeshData mesh;
    mesh.name = "Icosahedron";

    const float t = (1.0f + std::sqrt(5.0f)) * 0.5f;  // 黄金比
    float len = std::sqrt(1.0f + t * t);
    float n = radius / len;

    struct Vec3 { float x, y, z; };
    Vec3 verts[12] = {
        {-n, n*t, 0}, { n, n*t, 0}, {-n,-n*t, 0}, { n,-n*t, 0},
        {0,-n, n*t}, {0, n, n*t}, {0,-n,-n*t}, {0, n,-n*t},
        { n*t, 0,-n}, { n*t, 0, n}, {-n*t, 0,-n}, {-n*t, 0, n},
    };

    int tris[20][3] = {
        {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},
        {1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
        {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},
        {4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1},
    };

    for (auto& v : verts) {
        Vertex vert;
        vert.position[0] = v.x; vert.position[1] = v.y; vert.position[2] = v.z;
        vert.normal[0] = v.x / radius; vert.normal[1] = v.y / radius; vert.normal[2] = v.z / radius;
        mesh.vertices.push_back(vert);
    }
    for (auto& tri : tris) {
        mesh.indices.push_back(tri[0]);
        mesh.indices.push_back(tri[1]);
        mesh.indices.push_back(tri[2]);
    }
    return mesh;
}

/** @brief 生成程序化地形网格（基于多层正弦波） */
inline MeshData createTerrain(float size = 4.0f, int gridN = 64,
                              float heightScale = 0.3f) {
    MeshData mesh;
    mesh.name = "Terrain";
    float half = size * 0.5f;
    float step = size / gridN;

    auto heightFunc = [](float x, float z) -> float {
        return 0.5f * std::sin(x * 1.5f) * std::cos(z * 1.2f)
             + 0.3f * std::sin(x * 3.0f + 1.0f) * std::cos(z * 2.5f)
             + 0.15f * std::sin(x * 6.0f) * std::cos(z * 5.5f);
    };

    for (int z = 0; z <= gridN; z++) {
        for (int x = 0; x <= gridN; x++) {
            float wx = -half + x * step;
            float wz = -half + z * step;
            float h = heightFunc(wx, wz) * heightScale;

            Vertex vert;
            vert.position[0] = wx; vert.position[1] = h; vert.position[2] = wz;

            // 数值法线（中心差分）
            float eps = step * 0.5f;
            float hL = heightFunc(wx - eps, wz) * heightScale;
            float hR = heightFunc(wx + eps, wz) * heightScale;
            float hD = heightFunc(wx, wz - eps) * heightScale;
            float hU = heightFunc(wx, wz + eps) * heightScale;
            float nx = hL - hR;
            float nz = hD - hU;
            float ny = 2.0f * eps;
            float nlen = std::sqrt(nx * nx + ny * ny + nz * nz) + 1e-12f;
            vert.normal[0] = nx / nlen; vert.normal[1] = ny / nlen; vert.normal[2] = nz / nlen;

            vert.texcoord[0] = (float)x / gridN * 4.0f;
            vert.texcoord[1] = (float)z / gridN * 4.0f;
            mesh.vertices.push_back(vert);
        }
    }

    for (int z = 0; z < gridN; z++) {
        for (int x = 0; x < gridN; x++) {
            uint32_t a = z * (gridN + 1) + x;
            uint32_t b = a + gridN + 1;
            mesh.indices.push_back(a);
            mesh.indices.push_back(b);
            mesh.indices.push_back(a + 1);
            mesh.indices.push_back(a + 1);
            mesh.indices.push_back(b);
            mesh.indices.push_back(b + 1);
        }
    }
    return mesh;
}

// ============================================================
// 网格工具函数
// ============================================================

/** @brief 计算法线（如果未设置或需要重新计算） */
inline void computeNormals(MeshData& mesh) {
    // 清零法线
    for (auto& v : mesh.vertices)
        for (int i = 0; i < 3; i++) v.normal[i] = 0;

    // 累加面法线
    for (size_t i = 0; i < mesh.indices.size(); i += 3) {
        uint32_t i0 = mesh.indices[i];
        uint32_t i1 = mesh.indices[i + 1];
        uint32_t i2 = mesh.indices[i + 2];

        const float* p0 = mesh.vertices[i0].position;
        const float* p1 = mesh.vertices[i1].position;
        const float* p2 = mesh.vertices[i2].position;

        // e1 = p1-p0, e2 = p2-p0, normal = cross(e1, e2)
        float e1x = p1[0] - p0[0], e1y = p1[1] - p0[1], e1z = p1[2] - p0[2];
        float e2x = p2[0] - p0[0], e2y = p2[1] - p0[1], e2z = p2[2] - p0[2];
        float nx = e1y * e2z - e1z * e2y;
        float ny = e1z * e2x - e1x * e2z;
        float nz = e1x * e2y - e1y * e2x;

        mesh.vertices[i0].normal[0] += nx; mesh.vertices[i0].normal[1] += ny; mesh.vertices[i0].normal[2] += nz;
        mesh.vertices[i1].normal[0] += nx; mesh.vertices[i1].normal[1] += ny; mesh.vertices[i1].normal[2] += nz;
        mesh.vertices[i2].normal[0] += nx; mesh.vertices[i2].normal[1] += ny; mesh.vertices[i2].normal[2] += nz;
    }

    // 归一化
    for (auto& v : mesh.vertices) {
        float len = std::sqrt(v.normal[0]*v.normal[0] + v.normal[1]*v.normal[1] + v.normal[2]*v.normal[2]);
        if (len > 1e-12f) {
            float inv = 1.0f / len;
            v.normal[0] *= inv; v.normal[1] *= inv; v.normal[2] *= inv;
        }
    }
}

/** @brief 计算切线和副切线（用于法线映射）
 *
 * 算法：基于纹理坐标梯度
 *   T = (ΔU·E1 - ΔU2·E2) / det
 *   B = (-ΔV2·E1 + ΔV1·E2) / det
 */
inline void computeTangents(MeshData& mesh) {
    // 清零
    for (auto& v : mesh.vertices) {
        for (int i = 0; i < 3; i++) { v.tangent[i] = 0; v.bitangent[i] = 0; }
    }

    for (size_t i = 0; i < mesh.indices.size(); i += 3) {
        uint32_t i0 = mesh.indices[i];
        uint32_t i1 = mesh.indices[i + 1];
        uint32_t i2 = mesh.indices[i + 2];

        const Vertex& v0 = mesh.vertices[i0];
        const Vertex& v1 = mesh.vertices[i1];
        const Vertex& v2 = mesh.vertices[i2];

        float e1x = v1.position[0] - v0.position[0];
        float e1y = v1.position[1] - v0.position[1];
        float e1z = v1.position[2] - v0.position[2];
        float e2x = v2.position[0] - v0.position[0];
        float e2y = v2.position[1] - v0.position[1];
        float e2z = v2.position[2] - v0.position[2];

        float du1 = v1.texcoord[0] - v0.texcoord[0];
        float dv1 = v1.texcoord[1] - v0.texcoord[1];
        float du2 = v2.texcoord[0] - v0.texcoord[0];
        float dv2 = v2.texcoord[1] - v0.texcoord[1];

        float det = du1 * dv2 - du2 * dv1;
        if (std::abs(det) < 1e-10f) continue;
        float invDet = 1.0f / det;

        float tx = invDet * (dv2 * e1x - dv1 * e2x);
        float ty = invDet * (dv2 * e1y - dv1 * e2y);
        float tz = invDet * (dv2 * e1z - dv1 * e2z);
        float bx = invDet * (-du2 * e1x + du1 * e2x);
        float by = invDet * (-du2 * e1y + du1 * e2y);
        float bz = invDet * (-du2 * e1z + du1 * e2z);

        for (int k = 0; k < 3; k++) {
            uint32_t idx = (k == 0) ? i0 : (k == 1) ? i1 : i2;
            mesh.vertices[idx].tangent[0] += tx;
            mesh.vertices[idx].tangent[1] += ty;
            mesh.vertices[idx].tangent[2] += tz;
            mesh.vertices[idx].bitangent[0] += bx;
            mesh.vertices[idx].bitangent[1] += by;
            mesh.vertices[idx].bitangent[2] += bz;
        }
    }

    // 归一化 + Gram-Schmidt 正交化
    for (auto& v : mesh.vertices) {
        float tlen = std::sqrt(v.tangent[0]*v.tangent[0] + v.tangent[1]*v.tangent[1] + v.tangent[2]*v.tangent[2]);
        if (tlen > 1e-12f) {
            float inv = 1.0f / tlen;
            v.tangent[0] *= inv; v.tangent[1] *= inv; v.tangent[2] *= inv;
        }
        float blen = std::sqrt(v.bitangent[0]*v.bitangent[0] + v.bitangent[1]*v.bitangent[1] + v.bitangent[2]*v.bitangent[2]);
        if (blen > 1e-12f) {
            float inv = 1.0f / blen;
            v.bitangent[0] *= inv; v.bitangent[1] *= inv; v.bitangent[2] *= inv;
        }
    }
}

/** @brief 计算 AABB */
inline AABB computeAABB(const MeshData& mesh) {
    AABB box;
    for (const auto& v : mesh.vertices)
        box.expand(v.position);
    return box;
}

} // namespace MeshRender
