#pragma once
/**
 * @file    mesh/ornament.h
 * @brief   3D立体装饰物生成器（真正的3D几何体放在指甲面上）
 */

#include "core/geometry.h"   // Mesh, ColorRGBf

namespace NailPrint3D {

// ============================================================
// 装饰物类型
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

// ============================================================
// 3D立体装饰物生成器
// ============================================================

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
