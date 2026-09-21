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

// ============================================================
// 3D立体装饰物生成器（真正的3D几何体放在指甲面上）
// ============================================================

/** @brief 装饰物类型枚举 */
enum class OrnamentType {
    Spider      = 0,   ///< 小蜘蛛
    Flower      = 1,   ///< 立体花朵
    Star        = 2,   ///< 五角星
    Butterfly   = 3,   ///< 蝴蝶
    Heart       = 4,   ///< 爱心
    Bow         = 5,   ///< 蝴蝶结
    Crown       = 6,   ///< 小皇冠
    Gem         = 7    ///< 宝石（多面切割）
};

/** @brief 3D立体装饰物生成器 */
class OrnamentGenerator {
public:
    /** @brief 生成3D装饰物网格
     *  @param type 装饰物类型
     *  @param size 整体大小（mm）
     *  @param color 装饰物颜色
     */
    static Mesh generate(OrnamentType type, float size, ColorRGBf color = ColorRGBf(0.8f, 0.2f, 0.3f));

    /** @brief 将装饰物放置到甲片网格上
     *  @param nailMesh 甲片网格
     *  @param ornament 装饰物网格
     *  @param u U方向位置 (0~1)
     *  @param v V方向位置 (0~1)
     *  @param scale 缩放
     *  @param rotation 旋转角度（弧度）
     */
    static Mesh placeOnNail(const Mesh& nailMesh, const Mesh& ornament,
                            float u, float v, float scale = 1.0f,
                            float rotation = 0.0f);

    /// 单独生成各类装饰物
    static Mesh generateSpider(float size, ColorRGBf color);
    static Mesh generateFlower(float size, ColorRGBf color);
    static Mesh generateStar(float size, ColorRGBf color);
    static Mesh generateButterfly(float size, ColorRGBf color);
    static Mesh generateHeart(float size, ColorRGBf color);
    static Mesh generateBow(float size, ColorRGBf color);
    static Mesh generateCrown(float size, ColorRGBf color);
    static Mesh generateGem(float size, ColorRGBf color);

private:
    /// 生成UV球
    static Mesh generateSphere(float radius, int segments = 16, int rings = 12);
    /// 生成圆锥
    static Mesh generateCone(float radius, float height, int segments = 16);
    /// 生成圆环（甜甜圈）
    static Mesh generateTorus(float majorR, float minorR, int majorSeg = 24, int minorSeg = 12);
    /// 合并两个网格
    static Mesh mergeMeshes(const Mesh& a, const Mesh& b);
    /// 平移网格
    static void translateMesh(Mesh& m, float x, float y, float z);
    /// 旋转网格（绕Z轴）
    static void rotateMeshZ(Mesh& m, float angle);
    /// 缩放网格
    static void scaleMesh(Mesh& m, float s);
    /// 设置网格颜色
    static void colorMesh(Mesh& m, const ColorRGBf& c);
};

} // namespace NailPrint3D
