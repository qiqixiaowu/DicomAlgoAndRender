#pragma once
/**
 * @file    stl_loader.h
 * @brief   STL文件加载器（ASCII + 二进制）
 *
 * 支持：
 *  - 二进制STL解析（含属性字节颜色）
 *  - ASCII STL解析
 *  - 模型修复：法线一致化、孔洞检测
 *  - 边界框计算
 */

#include "nail_types.h"

namespace NailPrint3D {

// ============================================================
// STL加载器
// ============================================================

class STLLoader {
public:
    /** @brief 加载STL文件（自动检测ASCII/二进制） */
    static Mesh load(const std::string& filepath);

    /** @brief 加载二进制STL */
    static Mesh loadBinary(const std::string& filepath);

    /** @brief 加载ASCII STL */
    static Mesh loadASCII(const std::string& filepath);

    /** @brief 保存为二进制STL */
    static bool saveBinary(const std::string& filepath, const Mesh& mesh);

    /** @brief 保存为ASCII STL */
    static bool saveASCII(const std::string& filepath, const Mesh& mesh);

    /** @brief 检测文件是否为ASCII格式 */
    static bool isASCII(const std::string& filepath);

private:
    /** @brief 计算法线（如果STL法线为0） */
    static void ensureNormals(Mesh& mesh);

    /** @brief 法线一致化（使相邻三角形法线方向一致） */
    static void unifyNormals(Mesh& mesh);

    /** @brief 检测孔洞（边只被一个三角形引用） */
    static int detectHoles(const Mesh& mesh);
};

// ============================================================
// 程序化美甲网格生成
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

    /** @brief 生成圆柱体网格（用于支撑结构） */
    static Mesh generateCylinder(float radius, float height,
                                 int segments = 16);

    /** @brief 生成立方体网格 */
    static Mesh generateBox(float w, float h, float d);

private:
    /** @brief 将四边形拆分为两个三角形 */
    static void addQuad(Mesh& mesh, const Vec3& v0, const Vec3& v1,
                        const Vec3& v2, const Vec3& v3);

    /** @brief 构建去重顶点列表 + 三角形索引 */
    static void buildVertexIndex(Mesh& mesh);
};

} // namespace NailPrint3D
