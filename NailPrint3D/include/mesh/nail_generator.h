#pragma once
/**
 * @file    mesh/nail_generator.h
 * @brief   程序化美甲网格生成器
 *
 * 生成标准指甲片网格、浮雕图案、位移映射、基础体（圆柱/立方体）
 */

#include "core/geometry.h"   // Mesh, Vec3

namespace NailPrint3D {

// ============================================================
// 美甲网格生成器
// ============================================================

/** @brief 美甲网格生成器 */
class NailMeshGenerator {
public:
    /** @brief 生成标准指甲片网格
     *  @param width  甲片宽度
     *  @param length 甲片长度
     *  @param curvature 甲面曲率（0=平, 1=高弧度）
     *  @param thickness 厚度
     *  @param segmentsU U方向分段
     *  @param segmentsV V方向分段
     */
    static Mesh generateNailPatch(float width, float length,
                                  float curvature, float thickness,
                                  int segmentsU = 32, int segmentsV = 24);

    /** @brief 生成带图案浮雕的美甲网格
     *  @param base 基础甲片网格
     *  @param patternType 图案类型 (0=花朵, 1=几何, 2=文字)
     *  @param patternHeight 浮雕高度
     */
    static Mesh addReliefPattern(Mesh& base, int patternType, float patternHeight);

    /** @brief 用图片亮度作为高度图，对网格做位移映射（真正的3D浮雕）
     *  @param base 基础甲片网格
     *  @param imageData RGBA像素数据
     *  @param imgW 图片宽度
     *  @param imgH 图片高度
     *  @param displacementHeight 最大位移高度（mm）
     *  @param uvScaleU UV横向缩放（图案重复次数）
     *  @param uvScaleV UV纵向缩放
     */
    static void applyDisplacementMap(Mesh& base,
                                     const unsigned char* imageData,
                                     int imgW, int imgH,
                                     float displacementHeight,
                                     float uvScaleU = 1.0f,
                                     float uvScaleV = 1.0f);

    /** @brief 生成圆柱体网格（用于支撑结构） */
    static Mesh generateCylinder(float radius, float height,
                                 int segments = 16);

    /** @brief 生成立方体网格 */
    static Mesh generateBox(float w, float h, float d);

    /** @brief 将四边形拆分为两个三角形 */
    static void addQuad(Mesh& mesh, const Vec3& v0, const Vec3& v1,
                        const Vec3& v2, const Vec3& v3);

    /** @brief 构建去重顶点列表 + 三角形索引 */
    static void buildVertexIndex(Mesh& mesh);
};

} // namespace NailPrint3D
