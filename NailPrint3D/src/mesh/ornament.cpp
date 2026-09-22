/**
 * @file    ornament.cpp
 * @brief   3D立体装饰物生成器 实现
 */

#include "mesh/ornament.h"
#include "mesh/nail_generator.h"   // NailMeshGenerator::addQuad, buildVertexIndex
#include <cmath>
#include <iostream>
#include <vector>

namespace NailPrint3D {

// ============================================================
// 基础体生成
// ============================================================

Mesh OrnamentGenerator::generateSphere(float radius, int segments, int rings) {
    Mesh mesh;
    mesh.triangles.clear();
    std::vector<std::vector<Vec3>> verts(rings + 1, std::vector<Vec3>(segments + 1));
    for (int r = 0; r <= rings; r++) {
        float phi = (float)r / rings * 3.14159265f;
        for (int s = 0; s <= segments; s++) {
            float theta = (float)s / segments * 2.0f * 3.14159265f;
            verts[r][s] = Vec3(
                radius * std::sin(phi) * std::cos(theta),
                radius * std::sin(phi) * std::sin(theta),
                radius * std::cos(phi));
        }
    }
    for (int r = 0; r < rings; r++) {
        for (int s = 0; s < segments; s++) {
            NailMeshGenerator::addQuad(mesh, verts[r][s], verts[r][s + 1],
                    verts[r + 1][s + 1], verts[r + 1][s]);
        }
    }
    mesh.computeBBox();
    NailMeshGenerator::buildVertexIndex(mesh);
    mesh.computeNormals();
    return mesh;
}

Mesh OrnamentGenerator::generateCone(float radius, float height, int segments) {
    Mesh mesh;
    Vec3 apex(0, 0, height);
    std::vector<Vec3> base(segments);
    for (int i = 0; i < segments; i++) {
        float a = (float)i / segments * 2.0f * 3.14159265f;
        base[i] = Vec3(radius * std::cos(a), radius * std::sin(a), 0);
    }
    for (int i = 0; i < segments; i++) {
        int next = (i + 1) % segments;
        Triangle t;
        t.v[0] = base[i]; t.v[1] = base[next]; t.v[2] = apex;
        mesh.triangles.push_back(t);
    }
    Vec3 center(0, 0, 0);
    for (int i = 0; i < segments; i++) {
        int next = (i + 1) % segments;
        Triangle t;
        t.v[0] = center; t.v[1] = base[next]; t.v[2] = base[i];
        mesh.triangles.push_back(t);
    }
    mesh.computeBBox();
    NailMeshGenerator::buildVertexIndex(mesh);
    mesh.computeNormals();
    return mesh;
}

Mesh OrnamentGenerator::generateTorus(float majorR, float minorR, int majorSeg, int minorSeg) {
    Mesh mesh;
    std::vector<std::vector<Vec3>> verts(majorSeg, std::vector<Vec3>(minorSeg));
    for (int i = 0; i < majorSeg; i++) {
        float u = (float)i / majorSeg * 2.0f * 3.14159265f;
        for (int j = 0; j < minorSeg; j++) {
            float v = (float)j / minorSeg * 2.0f * 3.14159265f;
            verts[i][j] = Vec3(
                (majorR + minorR * std::cos(v)) * std::cos(u),
                (majorR + minorR * std::cos(v)) * std::sin(u),
                minorR * std::sin(v));
        }
    }
    for (int i = 0; i < majorSeg; i++) {
        int ni = (i + 1) % majorSeg;
        for (int j = 0; j < minorSeg; j++) {
            int nj = (j + 1) % minorSeg;
            NailMeshGenerator::addQuad(mesh, verts[i][j], verts[ni][j],
                    verts[ni][nj], verts[i][nj]);
        }
    }
    mesh.computeBBox();
    NailMeshGenerator::buildVertexIndex(mesh);
    mesh.computeNormals();
    return mesh;
}

Mesh OrnamentGenerator::mergeMeshes(const Mesh& a, const Mesh& b) {
    Mesh result = a;
    for (const auto& tri : b.triangles)
        result.triangles.push_back(tri);
    result.computeBBox();
    NailMeshGenerator::buildVertexIndex(result);
    result.computeNormals();
    return result;
}

void OrnamentGenerator::translateMesh(Mesh& m, float x, float y, float z) {
    for (auto& tri : m.triangles)
        for (int i = 0; i < 3; i++) {
            tri.v[i].x += x; tri.v[i].y += y; tri.v[i].z += z;
        }
    for (auto& v : m.vertices) { v.x += x; v.y += y; v.z += z; }
    m.computeBBox();
}

void OrnamentGenerator::rotateMeshZ(Mesh& m, float angle) {
    float c = std::cos(angle), s = std::sin(angle);
    for (auto& tri : m.triangles)
        for (int i = 0; i < 3; i++) {
            float x = tri.v[i].x, y = tri.v[i].y;
            tri.v[i].x = x * c - y * s;
            tri.v[i].y = x * s + y * c;
        }
    for (auto& v : m.vertices) {
        float x = v.x, y = v.y;
        v.x = x * c - y * s;
        v.y = x * s + y * c;
    }
    m.computeBBox();
}

void OrnamentGenerator::scaleMesh(Mesh& m, float s) {
    for (auto& tri : m.triangles)
        for (int i = 0; i < 3; i++)
            tri.v[i] = tri.v[i] * s;
    for (auto& v : m.vertices) v = v * s;
    m.computeBBox();
}

void OrnamentGenerator::colorMesh(Mesh& m, const ColorRGBf& c) {
    for (auto& tri : m.triangles)
        for (int i = 0; i < 3; i++)
            tri.vcolor[i] = c;
    m.colors.assign(m.vertices.size(), c);
}

// ============================================================
// 小蜘蛛
// ============================================================
Mesh OrnamentGenerator::generateSpider(float size, ColorRGBf color) {
    Mesh spider;
    float s = size * 0.1f;
    ColorRGBf bodyColor(0.15f, 0.08f, 0.08f);
    ColorRGBf legColor(0.2f, 0.1f, 0.1f);
    ColorRGBf eyeColor(1.0f, 0.9f, 0.2f);
    (void)color;

    Mesh head = generateSphere(s * 0.6f, 12, 8);
    translateMesh(head, 0, s * 0.5f, s * 0.3f);
    colorMesh(head, bodyColor);
    Mesh abdomen = generateSphere(s * 1.0f, 12, 8);
    translateMesh(abdomen, 0, -s * 0.6f, s * 0.4f);
    colorMesh(abdomen, bodyColor);
    spider = mergeMeshes(head, abdomen);

    for (int side = 0; side < 2; side++) {
        float dir = (side == 0) ? 1.0f : -1.0f;
        for (int i = 0; i < 4; i++) {
            Mesh leg = generateCone(s * 0.08f, s * 1.5f, 6);
            float angle = (float)(i - 1.5) * 0.4f;
            rotateMeshZ(leg, angle);
            translateMesh(leg, dir * s * 0.5f, s * 0.3f - i * s * 0.3f, s * 0.2f);
            colorMesh(leg, legColor);
            spider = mergeMeshes(spider, leg);
        }
    }

    Mesh eyeL = generateSphere(s * 0.12f, 8, 6);
    translateMesh(eyeL, -s * 0.2f, s * 0.7f, s * 0.7f);
    colorMesh(eyeL, eyeColor);
    Mesh eyeR = generateSphere(s * 0.12f, 8, 6);
    translateMesh(eyeR, s * 0.2f, s * 0.7f, s * 0.7f);
    colorMesh(eyeR, eyeColor);
    spider = mergeMeshes(spider, mergeMeshes(eyeL, eyeR));

    return spider;
}

// ============================================================
// 立体花朵
// ============================================================
Mesh OrnamentGenerator::generateFlower(float size, ColorRGBf color) {
    Mesh flower;
    float s = size * 0.1f;
    ColorRGBf centerColor(1.0f, 0.75f, 0.1f);
    ColorRGBf petalColor = color;
    ColorRGBf leafColor(0.2f, 0.6f, 0.2f);

    Mesh center = generateSphere(s * 0.4f, 12, 8);
    colorMesh(center, centerColor);
    flower = center;

    for (int i = 0; i < 6; i++) {
        float angle = (float)i / 6.0f * 2.0f * 3.14159265f;
        Mesh petal = generateSphere(s * 0.5f, 10, 6);
        for (auto& tri : petal.triangles)
            for (int j = 0; j < 3; j++) tri.v[j].z *= 0.3f;
        for (auto& v : petal.vertices) v.z *= 0.3f;
        translateMesh(petal, s * 0.7f * std::cos(angle), s * 0.7f * std::sin(angle), s * 0.1f);
        ColorRGBf pc(petalColor.r * (0.85f + 0.15f * i / 6.0f),
                     petalColor.g * (0.85f + 0.15f * i / 6.0f),
                     petalColor.b);
        colorMesh(petal, pc);
        flower = mergeMeshes(flower, petal);
    }

    for (int side = 0; side < 2; side++) {
        float dir = (side == 0) ? -1.0f : 1.0f;
        Mesh leaf = generateSphere(s * 0.35f, 8, 5);
        for (auto& tri : leaf.triangles)
            for (int j = 0; j < 3; j++) { tri.v[j].z *= 0.2f; tri.v[j].y *= 1.8f; }
        for (auto& v : leaf.vertices) { v.z *= 0.2f; v.y *= 1.8f; }
        translateMesh(leaf, dir * s * 0.9f, -s * 0.3f, s * 0.05f);
        rotateMeshZ(leaf, dir * 0.5f);
        colorMesh(leaf, leafColor);
        flower = mergeMeshes(flower, leaf);
    }

    return flower;
}

// ============================================================
// 五角星
// ============================================================
Mesh OrnamentGenerator::generateStar(float size, ColorRGBf color) {
    Mesh star;
    float s = size * 0.1f;
    float thickness = s * 0.3f;
    ColorRGBf starColor = color;
    ColorRGBf edgeColor(1.0f, 0.95f, 0.3f);

    std::vector<Vec3> outerPts, innerPts;
    for (int i = 0; i < 5; i++) {
        float outerAngle = (float)i / 5.0f * 2.0f * 3.14159265f - 1.5708f;
        float innerAngle = outerAngle + 3.14159265f / 5.0f;
        outerPts.push_back(Vec3(s * std::cos(outerAngle), s * std::sin(outerAngle), 0));
        innerPts.push_back(Vec3(s * 0.4f * std::cos(innerAngle), s * 0.4f * std::sin(innerAngle), 0));
    }

    Vec3 centerTop(0, 0, thickness * 0.5f);
    Vec3 centerBot(0, 0, -thickness * 0.5f);
    (void)centerTop; (void)centerBot;
    for (int i = 0; i < 5; i++) {
        int ni = (i + 1) % 5;
        Triangle top1, top2;
        top1.v[0] = Vec3(outerPts[i].x, outerPts[i].y, thickness * 0.5f);
        top1.v[1] = Vec3(innerPts[i].x, innerPts[i].y, thickness * 0.5f);
        top1.v[2] = Vec3(outerPts[ni].x, outerPts[ni].y, thickness * 0.5f);
        for (int j = 0; j < 3; j++) top1.vcolor[j] = starColor;
        star.triangles.push_back(top1);
        top2.v[0] = Vec3(outerPts[i].x, outerPts[i].y, -thickness * 0.5f);
        top2.v[1] = Vec3(outerPts[ni].x, outerPts[ni].y, -thickness * 0.5f);
        top2.v[2] = Vec3(innerPts[i].x, innerPts[i].y, -thickness * 0.5f);
        for (int j = 0; j < 3; j++) top2.vcolor[j] = starColor;
        star.triangles.push_back(top2);
        Triangle side1, side2;
        side1.v[0] = Vec3(outerPts[i].x, outerPts[i].y, -thickness * 0.5f);
        side1.v[1] = Vec3(outerPts[i].x, outerPts[i].y, thickness * 0.5f);
        side1.v[2] = Vec3(outerPts[ni].x, outerPts[ni].y, thickness * 0.5f);
        for (int j = 0; j < 3; j++) side1.vcolor[j] = edgeColor;
        star.triangles.push_back(side1);
        side2.v[0] = Vec3(outerPts[i].x, outerPts[i].y, -thickness * 0.5f);
        side2.v[1] = Vec3(outerPts[ni].x, outerPts[ni].y, thickness * 0.5f);
        side2.v[2] = Vec3(outerPts[ni].x, outerPts[ni].y, -thickness * 0.5f);
        for (int j = 0; j < 3; j++) side2.vcolor[j] = edgeColor;
        star.triangles.push_back(side2);
        Triangle side3, side4;
        side3.v[0] = Vec3(innerPts[i].x, innerPts[i].y, -thickness * 0.5f);
        side3.v[1] = Vec3(innerPts[i].x, innerPts[i].y, thickness * 0.5f);
        side3.v[2] = Vec3(outerPts[ni].x, outerPts[ni].y, thickness * 0.5f);
        for (int j = 0; j < 3; j++) side3.vcolor[j] = edgeColor;
        star.triangles.push_back(side3);
        side4.v[0] = Vec3(innerPts[i].x, innerPts[i].y, -thickness * 0.5f);
        side4.v[1] = Vec3(outerPts[ni].x, outerPts[ni].y, thickness * 0.5f);
        side4.v[2] = Vec3(outerPts[ni].x, outerPts[ni].y, -thickness * 0.5f);
        for (int j = 0; j < 3; j++) side4.vcolor[j] = edgeColor;
        star.triangles.push_back(side4);
    }

    star.computeBBox();
    NailMeshGenerator::buildVertexIndex(star);
    star.computeNormals();
    return star;
}

// ============================================================
// 蝴蝶
// ============================================================
Mesh OrnamentGenerator::generateButterfly(float size, ColorRGBf color) {
    Mesh butterfly;
    float s = size * 0.1f;
    ColorRGBf bodyColor(0.15f, 0.08f, 0.05f);
    ColorRGBf wingUpperColor = color;
    ColorRGBf wingLowerColor(color.r * 0.7f, color.g * 0.7f, color.b * 1.1f);
    ColorRGBf spotColor(1.0f, 0.85f, 0.2f);
    ColorRGBf antennaColor(0.2f, 0.1f, 0.05f);

    Mesh body = generateSphere(s * 0.15f, 10, 6);
    for (auto& tri : body.triangles)
        for (int i = 0; i < 3; i++) tri.v[i].y *= 2.5f;
    for (auto& v : body.vertices) v.y *= 2.5f;
    colorMesh(body, bodyColor);
    butterfly = body;

    float wingZ = s * 0.15f;
    Mesh wingLU = generateSphere(s * 0.8f, 12, 8);
    for (auto& tri : wingLU.triangles) for (int i = 0; i < 3; i++) { tri.v[i].z *= 0.15f; tri.v[i].x *= 0.7f; }
    for (auto& v : wingLU.vertices) { v.z *= 0.15f; v.x *= 0.7f; }
    translateMesh(wingLU, -s * 0.6f, s * 0.3f, wingZ);
    colorMesh(wingLU, wingUpperColor);
    butterfly = mergeMeshes(butterfly, wingLU);

    Mesh wingRU = generateSphere(s * 0.8f, 12, 8);
    for (auto& tri : wingRU.triangles) for (int i = 0; i < 3; i++) { tri.v[i].z *= 0.15f; tri.v[i].x *= 0.7f; }
    for (auto& v : wingRU.vertices) { v.z *= 0.15f; v.x *= 0.7f; }
    translateMesh(wingRU, s * 0.6f, s * 0.3f, wingZ);
    colorMesh(wingRU, wingUpperColor);
    butterfly = mergeMeshes(butterfly, wingRU);

    Mesh wingLD = generateSphere(s * 0.5f, 10, 6);
    for (auto& tri : wingLD.triangles) for (int i = 0; i < 3; i++) { tri.v[i].z *= 0.15f; tri.v[i].x *= 0.6f; }
    for (auto& v : wingLD.vertices) { v.z *= 0.15f; v.x *= 0.6f; }
    translateMesh(wingLD, -s * 0.4f, -s * 0.4f, wingZ);
    colorMesh(wingLD, wingLowerColor);
    butterfly = mergeMeshes(butterfly, wingLD);

    Mesh wingRD = generateSphere(s * 0.5f, 10, 6);
    for (auto& tri : wingRD.triangles) for (int i = 0; i < 3; i++) { tri.v[i].z *= 0.15f; tri.v[i].x *= 0.6f; }
    for (auto& v : wingRD.vertices) { v.z *= 0.15f; v.x *= 0.6f; }
    translateMesh(wingRD, s * 0.4f, -s * 0.4f, wingZ);
    colorMesh(wingRD, wingLowerColor);
    butterfly = mergeMeshes(butterfly, wingRD);

    Mesh spotLU = generateSphere(s * 0.12f, 8, 6);
    translateMesh(spotLU, -s * 0.7f, s * 0.4f, wingZ + s * 0.02f);
    colorMesh(spotLU, spotColor);
    butterfly = mergeMeshes(butterfly, spotLU);
    Mesh spotRU = generateSphere(s * 0.12f, 8, 6);
    translateMesh(spotRU, s * 0.7f, s * 0.4f, wingZ + s * 0.02f);
    colorMesh(spotRU, spotColor);
    butterfly = mergeMeshes(butterfly, spotRU);

    Mesh antL = generateCone(s * 0.03f, s * 0.5f, 4);
    translateMesh(antL, -s * 0.08f, s * 1.2f, s * 0.1f);
    rotateMeshZ(antL, -0.3f);
    colorMesh(antL, antennaColor);
    butterfly = mergeMeshes(butterfly, antL);
    Mesh antR = generateCone(s * 0.03f, s * 0.5f, 4);
    translateMesh(antR, s * 0.08f, s * 1.2f, s * 0.1f);
    rotateMeshZ(antR, 0.3f);
    colorMesh(antR, antennaColor);
    butterfly = mergeMeshes(butterfly, antR);

    return butterfly;
}

// ============================================================
// 爱心
// ============================================================
Mesh OrnamentGenerator::generateHeart(float size, ColorRGBf color) {
    Mesh heart;
    float s = size * 0.1f;
    ColorRGBf heartColor = color;
    ColorRGBf highlightColor(1.0f, 0.9f, 0.9f);

    int segments = 32;
    float thickness = s * 0.4f;
    std::vector<Vec3> topPts(segments), botPts(segments);

    for (int i = 0; i < segments; i++) {
        float t = (float)i / segments * 2.0f * 3.14159265f;
        float x = 16.0f * std::pow(std::sin(t), 3);
        float y = 13.0f * std::cos(t) - 5.0f * std::cos(2 * t) - 2.0f * std::cos(3 * t) - std::cos(4 * t);
        x *= s * 0.06f;
        y *= s * 0.06f;
        topPts[i] = Vec3(x, y, thickness * 0.5f);
        botPts[i] = Vec3(x, y, -thickness * 0.5f);
    }

    Vec3 centerTop(0, 0, thickness * 0.5f);
    Vec3 centerBot(0, 0, -thickness * 0.5f);
    for (int i = 0; i < segments; i++) {
        int ni = (i + 1) % segments;
        Triangle t1;
        t1.v[0] = centerTop; t1.v[1] = topPts[i]; t1.v[2] = topPts[ni];
        for (int j = 0; j < 3; j++) t1.vcolor[j] = highlightColor;
        heart.triangles.push_back(t1);
        Triangle t2;
        t2.v[0] = centerBot; t2.v[1] = botPts[ni]; t2.v[2] = botPts[i];
        for (int j = 0; j < 3; j++) t2.vcolor[j] = heartColor;
        heart.triangles.push_back(t2);
        Triangle s1, s2;
        s1.v[0] = botPts[i]; s1.v[1] = topPts[i]; s1.v[2] = topPts[ni];
        for (int j = 0; j < 3; j++) s1.vcolor[j] = heartColor;
        heart.triangles.push_back(s1);
        s2.v[0] = botPts[i]; s2.v[1] = topPts[ni]; s2.v[2] = botPts[ni];
        for (int j = 0; j < 3; j++) s2.vcolor[j] = heartColor;
        heart.triangles.push_back(s2);
    }

    heart.computeBBox();
    NailMeshGenerator::buildVertexIndex(heart);
    heart.computeNormals();
    return heart;
}

// ============================================================
// 蝴蝶结
// ============================================================
Mesh OrnamentGenerator::generateBow(float size, ColorRGBf color) {
    Mesh bow;
    float s = size * 0.1f;
    ColorRGBf centerColor(0.4f, 0.05f, 0.15f);
    ColorRGBf wingColor = color;
    ColorRGBf ribbonColor(color.r * 0.8f, color.g * 0.6f, color.b * 0.8f);
    ColorRGBf highlightColor(1.0f, 0.95f, 0.9f);

    Mesh center = generateSphere(s * 0.25f, 10, 8);
    colorMesh(center, centerColor);
    bow = center;

    for (int side = 0; side < 2; side++) {
        float dir = (side == 0) ? -1.0f : 1.0f;
        Mesh wing = generateSphere(s * 0.6f, 12, 8);
        for (auto& tri : wing.triangles)
            for (int i = 0; i < 3; i++) {
                tri.v[i].z *= 0.2f;
                tri.v[i].x *= 1.2f;
            }
        for (auto& v : wing.vertices) { v.z *= 0.2f; v.x *= 1.2f; }
        translateMesh(wing, dir * s * 0.6f, 0, s * 0.1f);
        colorMesh(wing, wingColor);
        bow = mergeMeshes(bow, wing);

        Mesh hl = generateSphere(s * 0.1f, 8, 6);
        translateMesh(hl, dir * s * 0.5f, s * 0.15f, s * 0.12f);
        colorMesh(hl, highlightColor);
        bow = mergeMeshes(bow, hl);
    }

    for (int side = 0; side < 2; side++) {
        float dir = (side == 0) ? -1.0f : 1.0f;
        Mesh ribbon = generateSphere(s * 0.15f, 8, 6);
        for (auto& tri : ribbon.triangles)
            for (int i = 0; i < 3; i++) {
                tri.v[i].z *= 0.15f;
                tri.v[i].y *= 3.0f;
            }
        for (auto& v : ribbon.vertices) { v.z *= 0.15f; v.y *= 3.0f; }
        translateMesh(ribbon, dir * s * 0.3f, -s * 0.8f, s * 0.05f);
        rotateMeshZ(ribbon, dir * 0.3f);
        colorMesh(ribbon, ribbonColor);
        bow = mergeMeshes(bow, ribbon);
    }

    return bow;
}

// ============================================================
// 小皇冠
// ============================================================
Mesh OrnamentGenerator::generateCrown(float size, ColorRGBf color) {
    Mesh crown;
    float s = size * 0.1f;
    ColorRGBf baseColor = color;
    ColorRGBf spikeColor(0.9f, 0.75f, 0.1f);
    ColorRGBf tipColor(0.9f, 0.2f, 0.3f);
    ColorRGBf gemColor(0.2f, 0.5f, 0.9f);

    Mesh baseRing = generateTorus(s * 0.6f, s * 0.12f, 20, 8);
    colorMesh(baseRing, baseColor);
    crown = baseRing;

    for (int i = 0; i < 4; i++) {
        float a = (float)i / 4.0f * 2.0f * 3.14159265f + 0.4f;
        Mesh gem = generateSphere(s * 0.08f, 8, 6);
        translateMesh(gem, s * 0.6f * std::cos(a), s * 0.6f * std::sin(a), s * 0.12f);
        colorMesh(gem, gemColor);
        crown = mergeMeshes(crown, gem);
    }

    for (int i = 0; i < 5; i++) {
        float angle = (float)i / 5.0f * 2.0f * 3.14159265f;
        Mesh spike = generateCone(s * 0.15f, s * 0.6f, 8);
        translateMesh(spike, s * 0.6f * std::cos(angle), s * 0.6f * std::sin(angle), s * 0.3f);
        colorMesh(spike, spikeColor);
        crown = mergeMeshes(crown, spike);

        Mesh tip = generateSphere(s * 0.1f, 8, 6);
        translateMesh(tip, s * 0.6f * std::cos(angle), s * 0.6f * std::sin(angle), s * 0.9f);
        colorMesh(tip, tipColor);
        crown = mergeMeshes(crown, tip);
    }

    return crown;
}

// ============================================================
// 宝石（多面切割）
// ============================================================
Mesh OrnamentGenerator::generateGem(float size, ColorRGBf color) {
    Mesh gem;
    float s = size * 0.1f;
    ColorRGBf tableColor = color;
    ColorRGBf crownColor(color.r * 1.1f, color.g * 1.05f, color.b * 1.1f);
    ColorRGBf pavilionColor(color.r * 0.6f, color.g * 0.55f, color.b * 0.7f);

    int n = 6;
    float topR = s * 0.7f;
    float midR = s * 0.9f;
    float topZ = s * 0.4f;
    float bottomZ = -s * 0.5f;

    std::vector<Vec3> topRing(n), midRing(n);

    for (int i = 0; i < n; i++) {
        float a = (float)i / n * 2.0f * 3.14159265f;
        topRing[i] = Vec3(topR * std::cos(a), topR * std::sin(a), topZ);
        midRing[i] = Vec3(midR * std::cos(a), midR * std::sin(a), 0);
    }
    Vec3 bottom(0, 0, bottomZ);
    Vec3 topCenter(0, 0, topZ);

    for (int i = 0; i < n; i++) {
        int ni = (i + 1) % n;
        Triangle t;
        t.v[0] = topCenter; t.v[1] = topRing[i]; t.v[2] = topRing[ni];
        for (int j = 0; j < 3; j++) t.vcolor[j] = tableColor;
        gem.triangles.push_back(t);
    }
    for (int i = 0; i < n; i++) {
        int ni = (i + 1) % n;
        Vec3 v0 = topRing[i], v1 = topRing[ni], v2 = midRing[ni], v3 = midRing[i];
        Triangle t1, t2;
        t1.v[0] = v0; t1.v[1] = v1; t1.v[2] = v2;
        t2.v[0] = v0; t2.v[1] = v2; t2.v[2] = v3;
        for (int j = 0; j < 3; j++) { t1.vcolor[j] = crownColor; t2.vcolor[j] = crownColor; }
        gem.triangles.push_back(t1);
        gem.triangles.push_back(t2);
    }
    for (int i = 0; i < n; i++) {
        int ni = (i + 1) % n;
        Triangle t1;
        t1.v[0] = midRing[i]; t1.v[1] = midRing[ni]; t1.v[2] = bottom;
        for (int j = 0; j < 3; j++) t1.vcolor[j] = pavilionColor;
        gem.triangles.push_back(t1);
    }

    gem.computeBBox();
    NailMeshGenerator::buildVertexIndex(gem);
    gem.computeNormals();
    return gem;
}

// ============================================================
// 统一生成入口
// ============================================================
Mesh OrnamentGenerator::generate(OrnamentType type, float size, ColorRGBf color) {
    switch (type) {
        case OrnamentType::Spider:    return generateSpider(size, color);
        case OrnamentType::Flower:    return generateFlower(size, color);
        case OrnamentType::Star:      return generateStar(size, color);
        case OrnamentType::Butterfly: return generateButterfly(size, color);
        case OrnamentType::Heart:     return generateHeart(size, color);
        case OrnamentType::Bow:       return generateBow(size, color);
        case OrnamentType::Crown:     return generateCrown(size, color);
        case OrnamentType::Gem:       return generateGem(size, color);
        default: return generateStar(size, color);
    }
}

// ============================================================
// 将装饰物放置到甲片上
// ============================================================
Mesh OrnamentGenerator::placeOnNail(const Mesh& nailMesh, const Mesh& ornament,
                                     float u, float v, float scale, float rotation) {
    Mesh result = nailMesh;
    Mesh orn = ornament;

    scaleMesh(orn, scale);
    rotateMeshZ(orn, rotation);

    float bestDist = 1e30f;
    float surfaceX = 0, surfaceY = 0, surfaceZ = 0;

    if (!nailMesh.uvs.empty()) {
        for (size_t i = 0; i < nailMesh.uvs.size(); i++) {
            float du = nailMesh.uvs[i].u - u;
            float dv = nailMesh.uvs[i].v - v;
            float dist = du * du + dv * dv;
            if (dist < bestDist) {
                bestDist = dist;
                surfaceX = nailMesh.vertices[i].x;
                surfaceY = nailMesh.vertices[i].y;
                surfaceZ = nailMesh.vertices[i].z;
            }
        }
    } else {
        float nailW = nailMesh.bbox.max.x - nailMesh.bbox.min.x;
        float nailL = nailMesh.bbox.max.y - nailMesh.bbox.min.y;
        surfaceX = nailMesh.bbox.min.x + u * nailW;
        surfaceY = nailMesh.bbox.min.y + v * nailL;
        for (size_t i = 0; i < nailMesh.vertices.size(); i++) {
            float dx = nailMesh.vertices[i].x - surfaceX;
            float dy = nailMesh.vertices[i].y - surfaceY;
            float dist = dx * dx + dy * dy;
            if (dist < bestDist) {
                bestDist = dist;
                surfaceZ = nailMesh.vertices[i].z;
            }
        }
    }

    float ornBotZ = orn.bbox.min.z;
    float ornCenterX = orn.bbox.center().x;
    float ornCenterY = orn.bbox.center().y;

    translateMesh(orn, surfaceX - ornCenterX,
                       surfaceY - ornCenterY,
                       surfaceZ - ornBotZ);

    result = mergeMeshes(result, orn);

    std::cout << "[装饰物] 已贴合甲面 (" << u << ", " << v << ")"
              << ", 表面坐标=(" << surfaceX << "," << surfaceY << "," << surfaceZ << ")"
              << ", 缩放=" << scale
              << ", 装饰物三角形数=" << orn.triangles.size()
              << std::endl;

    return result;
}

} // namespace NailPrint3D
