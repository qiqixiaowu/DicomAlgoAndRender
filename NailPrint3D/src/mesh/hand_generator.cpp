/**
 * @file    hand_generator.cpp
 * @brief   程序化手部网格生成器 实现
 *
 * 生成简化手部：手掌（圆角长方体）+ 5根手指（圆柱体）。
 * 每根手指有3节（掌骨/指骨），可弯曲。
 */

#include "mesh/hand_generator.h"
#include "mesh/nail_generator.h"
#include <cmath>
#include <iostream>

namespace NailPrint3D {

// 手指参数：位置(相对手掌根部)、长度、半径、弯曲角度
struct FingerParam {
    Vec3 basePos;       // 手指根部位置（相对手掌中心）
    float length;       // 手指总长
    float radius;       // 手指半径
    float bendAngle;    // 弯曲角度（度）
    float spreadAngle;  // 张开角度（度，绕Z轴）
};

// 5根手指参数（右手，手心朝下，手指朝+Y方向）
static const FingerParam fingerParams[5] = {
    // 拇指：偏左下方，较短粗，向外张开
    { Vec3(-12.0f, -5.0f, 2.0f),  35.0f, 5.5f, 15.0f, -45.0f },
    // 食指
    { Vec3(-7.0f,  10.0f, 0.0f),  45.0f, 4.5f, 10.0f, -8.0f },
    // 中指：最长
    { Vec3( 0.0f,  10.0f, 0.0f),  50.0f, 4.5f,  8.0f,  0.0f },
    // 无名指
    { Vec3( 7.0f,  10.0f, 0.0f),  46.0f, 4.5f, 12.0f,  8.0f },
    // 小指：最短细
    { Vec3(13.0f,  10.0f, 0.0f),  38.0f, 4.0f, 15.0f, 18.0f },
};

// 手掌参数
static const float PALM_WIDTH  = 40.0f;  // 手掌宽度
static const float PALM_LENGTH = 45.0f;  // 手掌长度
static const float PALM_THICK  = 12.0f;  // 手掌厚度

/// 生成圆柱体手指（3节，带弯曲）
static Mesh generateFinger(const FingerParam& param, float scale) {
    Mesh mesh;
    int segments = 12;

    float len = param.length * scale;
    float rad = param.radius * scale;
    float bendRad = param.bendAngle * 3.14159265f / 180.0f;
    float spreadRad = param.spreadAngle * 3.14159265f / 180.0f;

    // 3节手指长度比例
    float segLens[3] = { len * 0.4f, len * 0.3f, len * 0.3f };

    // 当前节的起点和方向
    Vec3 curPos = param.basePos * scale;
    Vec3 curDir = Vec3(
        std::sin(spreadRad),
        std::cos(spreadRad),
        0.0f
    ).normalized();

    for (int seg = 0; seg < 3; seg++) {
        float segLen = segLens[seg];

        // 弯曲：每节逐渐弯曲
        float segBend = bendRad * 0.33f * (seg + 1);

        // 生成圆柱体
        Vec3 segEnd = curPos + curDir * segLen;

        // 圆柱侧面
        for (int i = 0; i < segments; i++) {
            float a1 = (float)i / segments * 2.0f * 3.14159265f;
            float a2 = (float)(i + 1) / segments * 2.0f * 3.14159265f;

            // 计算垂直于方向的法线
            Vec3 axis = curDir;
            Vec3 ref = std::abs(axis.y) > 0.9f ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
            Vec3 right = axis.cross(ref).normalized();
            Vec3 up = right.cross(axis).normalized();

            Vec3 n1 = (right * std::cos(a1) + up * std::sin(a1)).normalized();
            Vec3 n2 = (right * std::cos(a2) + up * std::sin(a2)).normalized();

            Vec3 p1 = curPos + n1 * rad;
            Vec3 p2 = curPos + n2 * rad;
            Vec3 p3 = segEnd + n2 * rad;
            Vec3 p4 = segEnd + n1 * rad;

            NailMeshGenerator::addQuad(mesh, p1, p2, p3, p4);
        }

        // 指尖半球（最后一节末端）
        if (seg == 2) {
            for (int i = 0; i < segments; i++) {
                float a1 = (float)i / segments * 2.0f * 3.14159265f;
                float a2 = (float)(i + 1) / segments * 2.0f * 3.14159265f;

                Vec3 axis = curDir;
                Vec3 ref = std::abs(axis.y) > 0.9f ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
                Vec3 right = axis.cross(ref).normalized();
                Vec3 up = right.cross(axis).normalized();

                Vec3 n1 = (right * std::cos(a1) + up * std::sin(a1)).normalized();
                Vec3 n2 = (right * std::cos(a2) + up * std::sin(a2)).normalized();

                // 半球顶点
                Vec3 tip = segEnd + axis * rad;
                Vec3 p1 = segEnd + n1 * rad;
                Vec3 p2 = segEnd + n2 * rad;

                Triangle t;
                t.v[0] = p1; t.v[1] = p2; t.v[2] = tip;
                Vec3 e1 = p2 - p1, e2 = tip - p1;
                t.normal = e1.cross(e2).normalized();
                mesh.triangles.push_back(t);
            }
        }

        // 更新方向（弯曲）
        curDir = Vec3(
            curDir.x * std::cos(segBend) - curDir.z * std::sin(segBend),
            curDir.y,
            curDir.x * std::sin(segBend) + curDir.z * std::cos(segBend)
        ).normalized();

        curPos = segEnd;
    }

    return mesh;
}

/// 生成圆角手掌（简化为长方体 + 倒角）
static Mesh generatePalm(float scale) {
    Mesh mesh;
    float w = PALM_WIDTH * scale * 0.5f;
    float l = PALM_LENGTH * scale * 0.5f;
    float t = PALM_THICK * scale * 0.5f;
    float corner = 3.0f * scale; // 倒角半径

    // 8个顶点（带倒角）
    Vec3 verts[8] = {
        Vec3(-w + corner, -l + corner, -t), Vec3(w - corner, -l + corner, -t),
        Vec3(w - corner,  l - corner, -t), Vec3(-w + corner,  l - corner, -t),
        Vec3(-w + corner, -l + corner,  t), Vec3(w - corner, -l + corner,  t),
        Vec3(w - corner,  l - corner,  t), Vec3(-w + corner,  l - corner,  t),
    };

    // 底面
    NailMeshGenerator::addQuad(mesh, verts[3], verts[2], verts[1], verts[0]);
    // 顶面
    NailMeshGenerator::addQuad(mesh, verts[4], verts[5], verts[6], verts[7]);
    // 前面（手指方向）
    NailMeshGenerator::addQuad(mesh, verts[0], verts[1], verts[5], verts[4]);
    // 后面（手腕方向）
    NailMeshGenerator::addQuad(mesh, verts[2], verts[3], verts[7], verts[6]);
    // 左面
    NailMeshGenerator::addQuad(mesh, verts[3], verts[0], verts[4], verts[7]);
    // 右面
    NailMeshGenerator::addQuad(mesh, verts[1], verts[2], verts[6], verts[5]);

    // 设置皮肤颜色
    for (auto& tri : mesh.triangles) {
        for (int i = 0; i < 3; i++) {
            tri.vcolor[i] = ColorRGBf{0.85f, 0.72f, 0.62f};
        }
    }

    return mesh;
}

Mesh HandGenerator::generateHand(float scale) {
    Mesh hand;

    // 手掌
    Mesh palm = generatePalm(scale);
    hand.triangles.insert(hand.triangles.end(), palm.triangles.begin(), palm.triangles.end());

    // 5根手指
    for (int i = 0; i < 5; i++) {
        Mesh finger = generateFinger(fingerParams[i], scale);
        // 设置皮肤颜色
        for (auto& tri : finger.triangles) {
            for (int j = 0; j < 3; j++) {
                tri.vcolor[j] = ColorRGBf{0.85f, 0.72f, 0.62f};
            }
        }
        hand.triangles.insert(hand.triangles.end(), finger.triangles.begin(), finger.triangles.end());
    }

    // 构建顶点索引 + 法线
    NailMeshGenerator::buildVertexIndex(hand);
    hand.computeBBox();
    hand.computeNormals();

    std::cout << "[HandGenerator] 手部网格: " << hand.triangles.size() << " 三角形, "
              << hand.vertices.size() << " 顶点" << std::endl;

    return hand;
}

Vec3 HandGenerator::getFingerTip(FingerIndex finger, float scale) {
    const auto& p = fingerParams[finger];
    float len = p.length * scale;
    float spreadRad = p.spreadAngle * 3.14159265f / 180.0f;
    float bendRad = p.bendAngle * 3.14159265f / 180.0f;

    // 简化：指尖 = 基部 + 方向 * 长度（忽略逐节弯曲累积）
    Vec3 dir = Vec3(
        std::sin(spreadRad),
        std::cos(spreadRad),
        std::sin(bendRad)
    ).normalized();

    return p.basePos * scale + dir * len;
}

glm::mat4 HandGenerator::getNailTransform(FingerIndex finger, float scale) {
    Vec3 tip = getFingerTip(finger, scale);
    const auto& p = fingerParams[finger];

    float fingerRadius = p.radius * scale;
    // 指甲缩放：根据手指粗细自动调整
    // 美甲网格原始宽度15mm，缩放到 ≈ 手指直径 × 1.1
    float nailScale = (fingerRadius * 2.0f * 1.1f) / 15.0f;
    float nailHalfLen = 10.0f * nailScale;  // 原始半长20mm/2

    glm::mat4 transform = glm::mat4(1.0f);

    // 放在手指背面（+Z），从指尖向指根延伸
    transform = glm::translate(transform,
        glm::vec3(tip.x, tip.y - nailHalfLen, tip.z + fingerRadius * 0.9f));

    // 手指张开角度
    float spreadRad = p.spreadAngle * 3.14159265f / 180.0f;
    transform = glm::rotate(transform, spreadRad, glm::vec3(0, 0, 1));

    // 缩放（最后应用，使指甲匹配手指大小）
    transform = glm::scale(transform, glm::vec3(nailScale));

    return transform;
}

} // namespace NailPrint3D
