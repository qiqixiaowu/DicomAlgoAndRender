#pragma once
/**
 * @file    slicer.h
 * @brief   切片引擎接口定义
 *
 * 重构后的切片层依赖：
 *   core/ (geometry, print_types) + mesh/ (mesh_io)
 *   不依赖 render/ 或 reconstruction/
 */

#include "core/geometry.h"
#include "core/print_types.h"
#include "mesh/mesh_io.h"

namespace NailPrint3D {

// ============================================================
// 切片配置
// ============================================================

/** @brief 切片配置 */
struct SliceConfig {
    float layerHeight = 0.1f;
    float firstLayerHeight = 0.15f;
    bool adaptiveSlicing = false;
    float adaptiveMinLayerHeight = 0.05f;
    float adaptiveMaxLayerHeight = 0.2f;
    float adaptiveAngleThreshold = 30.0f;  ///< 曲面角度阈值

    // 外壳
    int perimeters = 2;
    float perimeterOffset = 0.22f;  ///< 外壳间距

    // 填充
    InfillPattern infillPattern = InfillPattern::Grid;
    float infillDensity = 0.2f;
    float infillAngle = 45.0f;
    float infillOffset = 0.0f;

    // 顶层/底层
    int topSolidLayers = 3;
    int bottomSolidLayers = 3;

    // 支撑
    bool generateSupport = true;
    float supportAngle = 45.0f;
    float supportDensity = 0.15f;
    float supportOffset = 0.3f;  ///< 支撑与模型间距

    // 美甲专用
    bool nailMode = true;
    float baseLayerHeight = 0.3f;   ///< 底胶层厚
    float colorLayerHeight = 0.08f;///< 颜色层厚
    float topCoatHeight = 0.1f;    ///< 封层厚
};

// ============================================================
// 切片引擎
// ============================================================

/** @brief 切片引擎 */
class NailSlicer {
public:
    /** @brief 执行完整切片
     *  @param mesh 输入网格
     *  @param config 切片配置
     *  @return 切片层列表
     */
    static std::vector<SliceLayer> slice(const Mesh& mesh,
                                         const SliceConfig& config);

    /** @brief 等距切片 */
    static std::vector<SliceLayer> sliceUniform(const Mesh& mesh,
                                                 const SliceConfig& config);

    /** @brief 自适应切片 */
    static std::vector<SliceLayer> sliceAdaptive(const Mesh& mesh,
                                                  const SliceConfig& config);

    /** @brief 打印切片统计 */
    static void printStats(const std::vector<SliceLayer>& layers);

private:
    // --- 层切分 ---

    /** @brief 计算切片Z高度列表 */
    static std::vector<float> computeLayerHeights(const Mesh& mesh,
                                                   const SliceConfig& config);

    /** @brief 计算自适应层高度 */
    static std::vector<float> computeAdaptiveHeights(const Mesh& mesh,
                                                      const SliceConfig& config);

    // --- 求交 ---

    /** @brief 单个三角形与Z平面求交 */
    static bool intersectTriangleZ(const Triangle& tri, float z,
                                   Vec2& outP1, Vec2& outP2);

    /** @brief 所有三角形与Z平面求交 → 线段集合 */
    static std::vector<Segment2D> sliceAtZ(const Mesh& mesh, float z);

    // --- 轮廓连接 ---

    /** @brief 将线段连接为闭合多边形 */
    static std::vector<Polygon> connectContours(const std::vector<Segment2D>& segments,
                                                 float tolerance = 1e-4f);

    /** @brief 区分外轮廓与孔洞 */
    static void classifyContours(std::vector<Polygon>& contours,
                                 std::vector<Polygon>& holes);

    // --- 外壳 ---

    /** @brief 多边形偏移（生成外壳） */
    static Polygon offsetPolygon(const Polygon& poly, float offset);

    /** @brief 生成外壳路径 */
    static std::vector<PathSegment> generatePerimeters(const std::vector<Polygon>& contours,
                                                        const std::vector<Polygon>& holes,
                                                        int count, float offset,
                                                        float speed);

    // --- 填充 ---

    /** @brief 生成填充路径 */
    static std::vector<PathSegment> generateInfill(const std::vector<Polygon>& contours,
                                                    const std::vector<Polygon>& holes,
                                                    const SliceConfig& config,
                                                    float z, int layerIndex);

    /** @brief 直线填充 */
    static std::vector<PathSegment> infillLines(const std::vector<Polygon>& boundary,
                                                 float density, float angle,
                                                 float extrusionWidth, float speed);

    /** @brief 网格填充 */
    static std::vector<PathSegment> infillGrid(const std::vector<Polygon>& boundary,
                                                float density, float angle,
                                                float extrusionWidth, float speed);

    /** @brief 蜂窝填充 */
    static std::vector<PathSegment> infillHoneycomb(const std::vector<Polygon>& boundary,
                                                     float density, float cellSize,
                                                     float extrusionWidth, float speed);

    /** @brief 同心填充 */
    static std::vector<PathSegment> infillConcentric(const std::vector<Polygon>& boundary,
                                                      float density, float offset,
                                                      float speed);

    /** @brief 扫描线裁剪多边形 */
    static std::vector<Vec2> scanlineIntersect(const std::vector<Polygon>& polys,
                                                float y);

    // --- 支撑 ---

    /** @brief 检测悬垂区域 */
    static std::vector<Polygon> detectOverhang(const std::vector<SliceLayer>& layers,
                                                int layerIndex, float angleThreshold);

    /** @brief 生成支撑路径 */
    static std::vector<PathSegment> generateSupport(const std::vector<Polygon>& overhang,
                                                     float density, float offset,
                                                     float speed);

    // --- 路径优化 ---

    /** @brief 路径优化（最近邻 + TSP近似） */
    static void optimizePath(std::vector<PathSegment>& paths);

    /** @brief 计算挤出量 */
    static float computeExtrusion(float dx, float dy, float layerHeight,
                                  float extrusionWidth, float filamentDiameter);
};

// ============================================================
// 美甲专用切片
// ============================================================

/** @brief 美甲切片器（分层结构：底胶→颜色层→封层） */
class NailPrintSlicer {
public:
    /** @brief 美甲专用切片 */
    std::vector<SliceLayer> sliceNail(const Mesh& mesh,
                                       const PrintConfig& config);

    /** @brief 生成颜色映射 */
    void assignColors(std::vector<SliceLayer>& layers,
                      const std::vector<ColorRGBA8>& palette,
                      int colorCount);

    /** @brief 生成颜色过渡路径 */
    std::vector<PathSegment> generateColorBlend(
        const Vec2& start, const Vec2& end,
        int colorIndex1, int colorIndex2,
        float blendDistance, float speed);
};

} // namespace NailPrint3D
