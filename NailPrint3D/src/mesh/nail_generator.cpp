/**
 * @file    nail_generator.cpp
 * @brief   程序化美甲网格生成器 实现
 */

#include "mesh/nail_generator.h"
#include <cmath>
#include <iostream>
#include <map>
#include <vector>

namespace NailPrint3D {

// ============================================================
// 内部辅助：带UV的四边形拆分
// ============================================================

static void addQuadUV(Mesh& mesh, const Vec3& v0, const Vec3& v1,
                       const Vec3& v2, const Vec3& v3,
                       const Vec2UV& uv0, const Vec2UV& uv1,
                       const Vec2UV& uv2, const Vec2UV& uv3) {
    Triangle t1, t2;
    t1.v[0] = v0; t1.v[1] = v1; t1.v[2] = v2;
    t2.v[0] = v0; t2.v[1] = v2; t2.v[2] = v3;
    t1.uv[0] = uv0; t1.uv[1] = uv1; t1.uv[2] = uv2;
    t2.uv[0] = uv0; t2.uv[1] = uv2; t2.uv[2] = uv3;
    Vec3 e1 = v1 - v0, e2 = v2 - v0;
    Vec3 n = e1.cross(e2).normalized();
    t1.normal = n; t2.normal = n;
    mesh.triangles.push_back(t1);
    mesh.triangles.push_back(t2);
}

// ============================================================
// NailMeshGenerator
// ============================================================

void NailMeshGenerator::buildVertexIndex(Mesh& mesh) {
    mesh.vertices.clear();
    mesh.normals.clear();
    mesh.uvs.clear();
    mesh.colors.clear();
    std::map<uint64_t, uint32_t> vertMap;
    auto makeKey = [](const Vec3& v) -> uint64_t {
        return ((uint64_t)(v.x * 10000) & 0x1FFFFF) |
               (((uint64_t)(v.y * 10000) & 0x1FFFFF) << 21) |
               (((uint64_t)(v.z * 10000) & 0x1FFFFF) << 42);
    };
    for (auto& tri : mesh.triangles) {
        for (int i = 0; i < 3; i++) {
            uint64_t key = makeKey(tri.v[i]);
            auto it = vertMap.find(key);
            if (it == vertMap.end()) {
                uint32_t idx = (uint32_t)mesh.vertices.size();
                vertMap[key] = idx;
                mesh.vertices.push_back(tri.v[i]);
                mesh.uvs.push_back(tri.uv[i]);
                // 保留三角形顶点颜色，无则默认美甲粉色
                ColorRGBf c = tri.vcolor[i];
                if (c.r == 0 && c.g == 0 && c.b == 0)
                    c = ColorRGBf{0.92f, 0.82f, 0.78f};
                mesh.colors.push_back(c);
                tri.idx[i] = idx;
            } else {
                tri.idx[i] = it->second;
            }
        }
    }
}

void NailMeshGenerator::addQuad(Mesh& mesh, const Vec3& v0, const Vec3& v1,
                                 const Vec3& v2, const Vec3& v3) {
    Triangle t1, t2;
    t1.v[0] = v0; t1.v[1] = v1; t1.v[2] = v2;
    t2.v[0] = v0; t2.v[1] = v2; t2.v[2] = v3;
    Vec3 e1 = v1 - v0, e2 = v2 - v0;
    Vec3 n = e1.cross(e2).normalized();
    t1.normal = n; t2.normal = n;
    mesh.triangles.push_back(t1);
    mesh.triangles.push_back(t2);
}

Mesh NailMeshGenerator::generateNailPatch(float width, float length,
                                           float curvature, float thickness,
                                           int segmentsU, int segmentsV) {
    Mesh mesh;
    mesh.triangles.clear();

    float halfW = width * 0.5f;
    float halfL = length * 0.5f;

    std::vector<std::vector<Vec3>> topVerts(segmentsV + 1, std::vector<Vec3>(segmentsU + 1));
    std::vector<std::vector<Vec3>> botVerts(segmentsV + 1, std::vector<Vec3>(segmentsU + 1));
    std::vector<std::vector<Vec2UV>> uvVerts(segmentsV + 1, std::vector<Vec2UV>(segmentsU + 1));

    for (int j = 0; j <= segmentsV; j++) {
        float v = (float)j / segmentsV;
        float y = -halfL + v * length;

        // 前端收窄：宽度随 v 变化（后端全宽，前端收窄到 70%）
        float widthScale = 1.0f - 0.3f * v * v;
        float curHalfW = halfW * widthScale;

        for (int i = 0; i <= segmentsU; i++) {
            float u = (float)i / segmentsU;
            float x = -curHalfW + u * (2.0f * curHalfW);

            // 横向弧度：抛物线穹起（中间高，两侧低）
            float lateralArc = curvature * (1.0f - (2.0f * u - 1.0f) * (2.0f * u - 1.0f));

            // 纵向弧度：从前到后微微拱起
            float longArc = curvature * 0.4f * (1.0f - (2.0f * v - 1.0f) * (2.0f * v - 1.0f));

            float z = lateralArc + longArc;

            topVerts[j][i] = Vec3(x, y, z);
            botVerts[j][i] = Vec3(x, y, z - thickness);

            uvVerts[j][i] = Vec2UV(u, v);
        }
    }

    // 顶面三角形（带UV）
    for (int j = 0; j < segmentsV; j++) {
        for (int i = 0; i < segmentsU; i++) {
            addQuadUV(mesh, topVerts[j][i], topVerts[j][i + 1],
                      topVerts[j + 1][i + 1], topVerts[j + 1][i],
                      uvVerts[j][i], uvVerts[j][i + 1],
                      uvVerts[j + 1][i + 1], uvVerts[j + 1][i]);
        }
    }

    // 底面三角形（翻转法线，UV相同）
    for (int j = 0; j < segmentsV; j++) {
        for (int i = 0; i < segmentsU; i++) {
            addQuadUV(mesh, botVerts[j + 1][i], botVerts[j + 1][i + 1],
                      botVerts[j][i + 1], botVerts[j][i],
                      uvVerts[j + 1][i], uvVerts[j + 1][i + 1],
                      uvVerts[j][i + 1], uvVerts[j][i]);
        }
    }

    // 四条侧边（带UV）
    for (int j = 0; j < segmentsV; j++) {
        // 左边
        addQuadUV(mesh, botVerts[j][0], topVerts[j][0],
                  topVerts[j + 1][0], botVerts[j + 1][0],
                  Vec2UV(0, (float)j / segmentsV), Vec2UV(0, (float)j / segmentsV),
                  Vec2UV(0, (float)(j + 1) / segmentsV), Vec2UV(0, (float)(j + 1) / segmentsV));
        // 右边
        addQuadUV(mesh, botVerts[j + 1][segmentsU], topVerts[j + 1][segmentsU],
                  topVerts[j][segmentsU], botVerts[j][segmentsU],
                  Vec2UV(1, (float)(j + 1) / segmentsV), Vec2UV(1, (float)(j + 1) / segmentsV),
                  Vec2UV(1, (float)j / segmentsV), Vec2UV(1, (float)j / segmentsV));
    }
    for (int i = 0; i < segmentsU; i++) {
        // 前边（v=0，指甲根部）
        addQuadUV(mesh, botVerts[0][i], topVerts[0][i],
                  topVerts[0][i + 1], botVerts[0][i + 1],
                  Vec2UV((float)i / segmentsU, 0), Vec2UV((float)i / segmentsU, 0),
                  Vec2UV((float)(i + 1) / segmentsU, 0), Vec2UV((float)(i + 1) / segmentsU, 0));
        // 后边（v=1，指尖）
        addQuadUV(mesh, botVerts[segmentsV][i + 1], topVerts[segmentsV][i + 1],
                  topVerts[segmentsV][i], botVerts[segmentsV][i],
                  Vec2UV((float)(i + 1) / segmentsU, 1), Vec2UV((float)(i + 1) / segmentsU, 1),
                  Vec2UV((float)i / segmentsU, 1), Vec2UV((float)i / segmentsU, 1));
    }

    mesh.computeBBox();
    mesh.triangles[0].attr = 0;

    buildVertexIndex(mesh);
    mesh.computeNormals();
    return mesh;
}

Mesh NailMeshGenerator::addReliefPattern(Mesh& base, int patternType, float patternHeight) {
    const float PI = 3.14159265358979f;

    for (size_t vi = 0; vi < base.vertices.size(); vi++) {
        float u = base.uvs[vi].u;
        float v = base.uvs[vi].v;
        float zOffset = 0.0f;
        ColorRGBf color{0.92f, 0.82f, 0.78f};

        if (patternType == 0) {
            // === 花朵图案 ===
            float cx = 0.5f, cy = 0.4f;
            float dx = u - cx, dy = v - cy;
            float dist = std::sqrt(dx * dx + dy * dy);
            float angle = std::atan2(dy, dx);
            float petalR = 0.12f * std::abs(std::cos(2.5f * angle));
            if (dist < petalR + 0.03f) {
                float t = dist / (petalR + 0.03f);
                zOffset = patternHeight * (1.0f - t) * 0.8f;
                color = ColorRGBf{0.95f - t * 0.15f, 0.55f + t * 0.2f, 0.62f + t * 0.15f};
            }
            for (int k = 0; k < 3; k++) {
                float la = k * 2.0f * PI / 3.0f + PI / 6.0f;
                float lx = cx + 0.25f * std::cos(la);
                float ly = cy + 0.25f * std::sin(la);
                float ld = std::sqrt((u - lx) * (u - lx) + (v - ly) * (v - ly));
                if (ld < 0.06f) {
                    zOffset += patternHeight * 0.3f * (1.0f - ld / 0.06f);
                    color = ColorRGBf{0.45f, 0.65f, 0.50f};
                }
            }
            float edgeDist = std::min(u, std::min(1.0f - u, std::min(v, 1.0f - v)));
            if (edgeDist < 0.05f) {
                zOffset += patternHeight * 0.2f;
                color = ColorRGBf{0.88f, 0.72f, 0.70f};
            }
        } else if (patternType == 1) {
            // === 几何菱格图案 ===
            float scale = 8.0f;
            float gu = u * scale;
            float gv = v * scale;
            float fu = gu - std::floor(gu);
            float fv = gv - std::floor(gv);
            float diamond = std::abs(fu - 0.5f) + std::abs(fv - 0.5f);
            if (diamond < 0.25f) {
                zOffset = patternHeight * 0.6f;
                int checker = ((int)std::floor(gu) + (int)std::floor(gv)) % 2;
                color = checker ? ColorRGBf{0.88f, 0.60f, 0.65f} : ColorRGBf{0.60f, 0.72f, 0.82f};
            }
            float cdist = std::sqrt((u - 0.5f) * (u - 0.5f) + (v - 0.4f) * (v - 0.4f));
            if (cdist < 0.12f) {
                zOffset = patternHeight * (1.0f - cdist / 0.12f);
                color = ColorRGBf{0.95f, 0.82f, 0.45f};
            }
        } else if (patternType == 2) {
            // === 星空图案 ===
            for (int s = 0; s < 12; s++) {
                float seed = (float)s * 7.13f;
                float sx = 0.1f + 0.8f * std::fmod(seed * 1.7f, 1.0f);
                float sy = 0.1f + 0.8f * std::fmod(seed * 2.3f, 1.0f);
                float sd = std::sqrt((u - sx) * (u - sx) + (v - sy) * (v - sy));
                if (sd < 0.04f) {
                    float t = sd / 0.04f;
                    zOffset = std::max(zOffset, patternHeight * (1.0f - t));
                    color = ColorRGBf{0.95f, 0.90f, 0.65f};
                }
            }
            color.r = 0.35f + u * 0.25f;
            color.g = 0.30f + v * 0.20f;
            color.b = 0.50f + (1.0f - u) * 0.25f;
        }

        base.vertices[vi].z += zOffset;
        base.colors[vi] = color;
    }

    base.computeNormals();
    base.computeBBox();
    return base;
}

void NailMeshGenerator::applyDisplacementMap(Mesh& base,
                                             const unsigned char* imageData,
                                             int imgW, int imgH,
                                             float displacementHeight,
                                             float uvScaleU, float uvScaleV) {
    if (!imageData || imgW <= 0 || imgH <= 0 || base.vertices.empty()) {
        std::cerr << "[Displacement] 无效参数" << std::endl;
        return;
    }

    std::cout << "[Displacement] 开始位移映射: " << base.vertices.size()
              << " 顶点, 图片 " << imgW << "x" << imgH
              << ", 高度=" << displacementHeight << "mm" << std::endl;

    int displacedCount = 0;

    for (size_t vi = 0; vi < base.vertices.size(); vi++) {
        if (vi >= base.uvs.size()) continue;

        float u = base.uvs[vi].u * uvScaleU;
        float v = base.uvs[vi].v * uvScaleV;

        u = u - std::floor(u);
        v = v - std::floor(v);
        v = 1.0f - v;

        int px = std::min((int)(u * imgW), imgW - 1);
        int py = std::min((int)(v * imgH), imgH - 1);
        int pixelIdx = (py * imgW + px) * 4;

        float r = imageData[pixelIdx] / 255.0f;
        float g = imageData[pixelIdx + 1] / 255.0f;
        float b = imageData[pixelIdx + 2] / 255.0f;
        float luminance = 0.299f * r + 0.587f * g + 0.114f * b;

        float zOffset = luminance * displacementHeight;

        if (base.vertices[vi].z > -0.01f) {
            base.vertices[vi].z += zOffset;
            displacedCount++;
            base.colors[vi] = ColorRGBf{r, g, b};
        }
    }

    base.computeNormals();
    base.computeBBox();

    std::cout << "[Displacement] 完成: " << displacedCount << " 顶点已位移"
              << ", Z范围: [" << base.bbox.min.z << ", " << base.bbox.max.z << "]"
              << std::endl;
}

Mesh NailMeshGenerator::generateCylinder(float radius, float height, int segments) {
    Mesh mesh;
    std::vector<Vec3> topRing(segments), botRing(segments);
    for (int i = 0; i < segments; i++) {
        float angle = (float)i / segments * 2.0f * 3.14159265f;
        float x = radius * std::cos(angle);
        float y = radius * std::sin(angle);
        topRing[i] = Vec3(x, y, height);
        botRing[i] = Vec3(x, y, 0);
    }
    for (int i = 0; i < segments; i++) {
        int next = (i + 1) % segments;
        addQuad(mesh, botRing[i], botRing[next], topRing[next], topRing[i]);
    }
    Vec3 centerBot(0, 0, 0), centerTop(0, 0, height);
    for (int i = 0; i < segments; i++) {
        int next = (i + 1) % segments;
        Triangle t1;
        t1.v[0] = centerBot; t1.v[1] = botRing[i]; t1.v[2] = botRing[next];
        mesh.triangles.push_back(t1);
        Triangle t2;
        t2.v[0] = centerTop; t2.v[1] = topRing[next]; t2.v[2] = topRing[i];
        mesh.triangles.push_back(t2);
    }
    mesh.computeBBox();
    buildVertexIndex(mesh);
    mesh.computeNormals();
    return mesh;
}

Mesh NailMeshGenerator::generateBox(float w, float h, float d) {
    Mesh mesh;
    float hw = w * 0.5f, hh = h * 0.5f, hd = d * 0.5f;
    Vec3 v000(-hw, -hh, -hd), v100(hw, -hh, -hd), v110(hw, hh, -hd), v010(-hw, hh, -hd);
    Vec3 v001(-hw, -hh, hd), v101(hw, -hh, hd), v111(hw, hh, hd), v011(-hw, hh, hd);

    addQuad(mesh, v000, v100, v110, v010);
    addQuad(mesh, v001, v011, v111, v101);
    addQuad(mesh, v000, v010, v011, v001);
    addQuad(mesh, v100, v101, v111, v110);
    addQuad(mesh, v000, v001, v101, v100);
    addQuad(mesh, v010, v110, v111, v011);

    mesh.computeBBox();
    buildVertexIndex(mesh);
    mesh.computeNormals();
    return mesh;
}

} // namespace NailPrint3D
