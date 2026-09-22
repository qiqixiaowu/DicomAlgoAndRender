/**
 * @file    slicer.cpp
 * @brief   3D美甲打印 — RIP切片引擎 实现
 */

#include "slicing/slicer.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace NailPrint3D {

// ============================================================
// NailSlicer — 公开接口
// ============================================================

std::vector<SliceLayer> NailSlicer::slice(const Mesh& mesh,
                                           const SliceConfig& config) {
    if (config.adaptiveSlicing)
        return sliceAdaptive(mesh, config);
    return sliceUniform(mesh, config);
}

std::vector<SliceLayer> NailSlicer::sliceUniform(const Mesh& mesh,
                                                   const SliceConfig& config) {
    std::cout << "[Slice] 等距切片 (层高=" << config.layerHeight << "mm)..." << std::endl;

    std::vector<float> heights = computeLayerHeights(mesh, config);
    std::vector<SliceLayer> layers;
    layers.reserve(heights.size());

    for (int i = 0; i < (int)heights.size(); i++) {
        float z = heights[i];
        SliceLayer layer;
        layer.layerIndex = i;
        layer.zHeight = z;

        // 1. 三角形-平面求交 → 线段
        auto segments = sliceAtZ(mesh, z);

        // 2. 线段连接为闭合多边形
        auto contours = connectContours(segments);
        classifyContours(layer.contours, layer.holes);
        layer.contours = contours;

        // 3. 生成外壳路径
        layer.perimeters = generatePerimeters(layer.contours, layer.holes,
                                               config.perimeters, config.perimeterOffset,
                                               40.0f);

        // 4. 生成填充路径
        // 合并轮廓用于填充边界
        std::vector<Polygon> boundary = layer.contours;
        for (const auto& h : layer.holes) boundary.push_back(h);

        bool isTopOrBottom = (i < config.bottomSolidLayers) ||
                             (i >= (int)heights.size() - config.topSolidLayers);
        if (isTopOrBottom) {
            // 顶层/底层：实心填充
            SliceConfig solidConfig = config;
            solidConfig.infillDensity = 1.0f;
            layer.infill = generateInfill(boundary, layer.holes, solidConfig, z, i);
        } else {
            layer.infill = generateInfill(boundary, layer.holes, config, z, i);
        }

        layers.push_back(layer);
    }

    // 5. 支撑生成
    if (config.generateSupport) {
        std::cout << "[Slice] 生成支撑结构..." << std::endl;
        for (int i = 2; i < (int)layers.size(); i++) {
            auto overhang = detectOverhang(layers, i, config.supportAngle);
            if (!overhang.empty()) {
                layers[i].supportPaths = generateSupport(overhang, config.supportDensity,
                                                          config.supportOffset, 50.0f);
            }
        }
    }

    // 6. 路径优化
    for (auto& layer : layers) {
        optimizePath(layer.perimeters);
        optimizePath(layer.infill);
        optimizePath(layer.supportPaths);
    }

    printStats(layers);
    return layers;
}

std::vector<SliceLayer> NailSlicer::sliceAdaptive(const Mesh& mesh,
                                                    const SliceConfig& config) {
    std::cout << "[Slice] 自适应切片..." << std::endl;

    std::vector<float> heights = computeAdaptiveHeights(mesh, config);
    std::vector<SliceLayer> layers;
    layers.reserve(heights.size());

    for (int i = 0; i < (int)heights.size(); i++) {
        float z = heights[i];
        SliceLayer layer;
        layer.layerIndex = i;
        layer.zHeight = z;

        auto segments = sliceAtZ(mesh, z);
        layer.contours = connectContours(segments);

        layer.perimeters = generatePerimeters(layer.contours, layer.holes,
                                               config.perimeters, config.perimeterOffset, 40.0f);

        std::vector<Polygon> boundary = layer.contours;
        for (const auto& h : layer.holes) boundary.push_back(h);
        layer.infill = generateInfill(boundary, layer.holes, config, z, i);

        layers.push_back(layer);
    }

    printStats(layers);
    return layers;
}

void NailSlicer::printStats(const std::vector<SliceLayer>& layers) {
    int totalPaths = 0;
    float totalLength = 0;
    for (const auto& layer : layers) {
        totalPaths += (int)(layer.perimeters.size() + layer.infill.size() + layer.supportPaths.size());
        for (const auto& p : layer.perimeters) totalLength += (p.start - p.end).length();
        for (const auto& p : layer.infill) totalLength += (p.start - p.end).length();
    }
    std::cout << "[Slice] 完成: " << layers.size() << " 层, "
              << totalPaths << " 条路径, 总长度 " << totalLength << " mm" << std::endl;
}

// ============================================================
// 层切分
// ============================================================

std::vector<float> NailSlicer::computeLayerHeights(const Mesh& mesh,
                                                     const SliceConfig& config) {
    std::vector<float> heights;
    if (mesh.triangles.empty()) return heights;

    float zMin = mesh.bbox.min.z;
    float zMax = mesh.bbox.max.z;

    // 首层
    float z = zMin + config.firstLayerHeight;
    heights.push_back(z);

    while (z < zMax - config.layerHeight * 0.5f) {
        z += config.layerHeight;
        heights.push_back(z);
    }

    return heights;
}

std::vector<float> NailSlicer::computeAdaptiveHeights(const Mesh& mesh,
                                                        const SliceConfig& config) {
    std::vector<float> heights;
    if (mesh.triangles.empty()) return heights;

    float zMin = mesh.bbox.min.z;
    float zMax = mesh.bbox.max.z;

    float z = zMin + config.firstLayerHeight;
    heights.push_back(z);

    while (z < zMax) {
        // 采样当前层的轮廓复杂度，决定下一层高度
        auto segs = sliceAtZ(mesh, z);
        auto contours = connectContours(segs);

        // 计算轮廓总周长作为复杂度指标
        float totalPerimeter = 0;
        for (const auto& c : contours) {
            for (size_t i = 0; i < c.points.size(); i++) {
                size_t j = (i + 1) % c.points.size();
                totalPerimeter += (c.points[i] - c.points[j]).length();
            }
        }

        // 简化自适应：周长变化大时用薄层
        float layerH = config.layerHeight;
        if (totalPerimeter > 50.0f)
            layerH = config.adaptiveMinLayerHeight;
        else
            layerH = config.adaptiveMaxLayerHeight;

        z += layerH;
        if (z < zMax) heights.push_back(z);
    }

    return heights;
}

// ============================================================
// 求交
// ============================================================

bool NailSlicer::intersectTriangleZ(const Triangle& tri, float z,
                                      Vec2& outP1, Vec2& outP2) {
    // 计算三个顶点相对于Z平面的符号
    float d[3] = { tri.v[0].z - z, tri.v[1].z - z, tri.v[2].z - z };

    // 全同号 → 无交点
    if ((d[0] > 0 && d[1] > 0 && d[2] > 0) ||
        (d[0] < 0 && d[1] < 0 && d[2] < 0))
        return false;

    // 找到两条与平面相交的边
    Vec2 intersections[2];
    int count = 0;

    for (int i = 0; i < 3; i++) {
        int j = (i + 1) % 3;
        if (d[i] * d[j] < 0) {
            // 边 vi-vj 与平面相交
            float t = d[i] / (d[i] - d[j]);
            float x = tri.v[i].x + t * (tri.v[j].x - tri.v[i].x);
            float y = tri.v[i].y + t * (tri.v[j].y - tri.v[i].y);
            intersections[count++] = Vec2(x, y);
        }
    }

    // 处理顶点在平面上的情况
    if (count < 2) {
        for (int i = 0; i < 3; i++) {
            if (std::abs(d[i]) < 1e-8f) {
                intersections[count++] = Vec2(tri.v[i].x, tri.v[i].y);
                if (count == 2) break;
            }
        }
    }

    if (count < 2) return false;

    outP1 = intersections[0];
    outP2 = intersections[1];
    return true;
}

std::vector<Segment2D> NailSlicer::sliceAtZ(const Mesh& mesh, float z) {
    std::vector<Segment2D> segments;
    segments.reserve(mesh.triangles.size() / 10);  // 估计只有部分三角形相交

    for (const auto& tri : mesh.triangles) {
        // 快速排除：三角形Z范围不包含z
        float zMin = std::min({ tri.v[0].z, tri.v[1].z, tri.v[2].z });
        float zMax = std::max({ tri.v[0].z, tri.v[1].z, tri.v[2].z });
        if (z < zMin || z > zMax) continue;

        Segment2D seg;
        if (intersectTriangleZ(tri, z, seg.p1, seg.p2)) {
            segments.push_back(seg);
        }
    }

    return segments;
}

// ============================================================
// 轮廓连接
// ============================================================

std::vector<Polygon> NailSlicer::connectContours(const std::vector<Segment2D>& segments,
                                                   float tolerance) {
    if (segments.empty()) return {};

    std::vector<bool> used(segments.size(), false);
    std::vector<Polygon> polygons;

    for (size_t i = 0; i < segments.size(); i++) {
        if (used[i]) continue;

        Polygon poly;
        poly.points.push_back(segments[i].p1);
        poly.points.push_back(segments[i].p2);
        used[i] = true;

        // 从 segments[i].p2 开始连接
        Vec2 current = segments[i].p2;
        bool found = true;

        while (found) {
            found = false;
            float bestDist = tolerance;
            int bestIdx = -1;
            bool bestReversed = false;

            for (size_t j = 0; j < segments.size(); j++) {
                if (used[j]) continue;

                // 检查 segments[j].p1 是否接近 current
                float d1 = (segments[j].p1 - current).length();
                if (d1 < bestDist) {
                    bestDist = d1;
                    bestIdx = (int)j;
                    bestReversed = false;
                }

                // 检查 segments[j].p2 是否接近 current（需要翻转）
                float d2 = (segments[j].p2 - current).length();
                if (d2 < bestDist) {
                    bestDist = d2;
                    bestIdx = (int)j;
                    bestReversed = true;
                }
            }

            if (bestIdx >= 0) {
                used[bestIdx] = true;
                if (bestReversed) {
                    current = segments[bestIdx].p1;
                    poly.points.push_back(segments[bestIdx].p1);
                } else {
                    current = segments[bestIdx].p2;
                    poly.points.push_back(segments[bestIdx].p2);
                }
                found = true;

                // 检查是否闭合
                if ((current - poly.points[0]).length() < tolerance) {
                    poly.closed = true;
                    break;
                }
            }
        }

        polygons.push_back(poly);
    }

    return polygons;
}

void NailSlicer::classifyContours(std::vector<Polygon>& contours,
                                    std::vector<Polygon>& holes) {
    // 面积为负的为孔洞（CW方向）
    holes.clear();
    std::vector<Polygon> outer;

    for (auto& poly : contours) {
        if (poly.area() < 0) {
            // 翻转为CCW
            std::reverse(poly.points.begin(), poly.points.end());
            holes.push_back(poly);
        } else {
            outer.push_back(poly);
        }
    }
    contours = outer;
}

// ============================================================
// 外壳生成
// ============================================================

Polygon NailSlicer::offsetPolygon(const Polygon& poly, float offset) {
    // 简化版多边形偏移：沿法线方向移动每个顶点
    // 实际实现应使用 Clipper 库或更精确的偏移算法
    Polygon result;
    result.closed = poly.closed;
    size_t n = poly.points.size();

    for (size_t i = 0; i < n; i++) {
        size_t prev = (i == 0) ? n - 1 : i - 1;
        size_t next = (i + 1) % n;

        Vec2 edge1 = (poly.points[i] - poly.points[prev]).normalized();
        Vec2 edge2 = (poly.points[next] - poly.points[i]).normalized();

        // 计算角平分线方向
        Vec2 bisector = (edge1 + edge2).normalized();
        // 垂直方向（内偏移）
        Vec2 normal(-bisector.y, bisector.x);

        // 偏移量按角度调整
        float dot = edge1.dot(edge2);
        float angleFactor = 1.0f / std::sqrt((1.0f + dot) * 0.5f + 1e-10f);

        result.points.push_back(poly.points[i] + normal * offset * angleFactor);
    }

    return result;
}

std::vector<PathSegment> NailSlicer::generatePerimeters(const std::vector<Polygon>& contours,
                                                          const std::vector<Polygon>& holes,
                                                          int count, float offset,
                                                          float speed) {
    std::vector<PathSegment> paths;

    for (int ring = 0; ring < count; ring++) {
        float ringOffset = offset * (ring + 1);

        for (const auto& poly : contours) {
            Polygon offsetPoly = offsetPolygon(poly, -ringOffset);  // 内偏移
            for (size_t i = 0; i < offsetPoly.points.size(); i++) {
                size_t j = (i + 1) % offsetPoly.points.size();
                PathSegment seg;
                seg.start = offsetPoly.points[i];
                seg.end = offsetPoly.points[j];
                seg.speed = speed;
                seg.extrusion = 1.0f;
                seg.travel = false;
                paths.push_back(seg);
            }
        }

        for (const auto& hole : holes) {
            Polygon offsetHole = offsetPolygon(hole, ringOffset);  // 孔洞外偏移
            for (size_t i = 0; i < offsetHole.points.size(); i++) {
                size_t j = (i + 1) % offsetHole.points.size();
                PathSegment seg;
                seg.start = offsetHole.points[i];
                seg.end = offsetHole.points[j];
                seg.speed = speed;
                seg.extrusion = 1.0f;
                seg.travel = false;
                paths.push_back(seg);
            }
        }
    }

    return paths;
}

// ============================================================
// 填充生成
// ============================================================

std::vector<PathSegment> NailSlicer::generateInfill(const std::vector<Polygon>& contours,
                                                      const std::vector<Polygon>& holes,
                                                      const SliceConfig& config,
                                                      float z, int layerIndex) {
    std::vector<PathSegment> paths;

    // 合并轮廓和孔洞作为边界
    std::vector<Polygon> boundary = contours;
    for (const auto& h : holes) boundary.push_back(h);

    float angle = config.infillAngle + (layerIndex % 2) * 90.0f;  // 交替角度

    switch (config.infillPattern) {
        case InfillPattern::Lines:
            paths = infillLines(boundary, config.infillDensity, angle,
                                config.perimeterOffset, 60.0f);
            break;
        case InfillPattern::Grid:
            paths = infillGrid(boundary, config.infillDensity, angle,
                               config.perimeterOffset, 60.0f);
            break;
        case InfillPattern::Honeycomb:
            paths = infillHoneycomb(boundary, config.infillDensity, 3.0f,
                                    config.perimeterOffset, 60.0f);
            break;
        case InfillPattern::Concentric:
            paths = infillConcentric(boundary, config.infillDensity, config.perimeterOffset, 60.0f);
            break;
        case InfillPattern::Triangles:
            paths = infillGrid(boundary, config.infillDensity, angle,
                               config.perimeterOffset, 60.0f);  // 简化：用网格代替
            break;
        default:
            break;
    }

    return paths;
}

std::vector<PathSegment> NailSlicer::infillLines(const std::vector<Polygon>& boundary,
                                                   float density, float angle,
                                                   float extrusionWidth, float speed) {
    std::vector<PathSegment> paths;
    if (boundary.empty() || density <= 0) return paths;

    // 计算边界框
    Vec2 bMin(1e30f, 1e30f), bMax(-1e30f, -1e30f);
    for (const auto& poly : boundary) {
        for (const auto& p : poly.points) {
            bMin.x = std::min(bMin.x, p.x); bMin.y = std::min(bMin.y, p.y);
            bMax.x = std::max(bMax.x, p.x); bMax.y = std::max(bMax.y, p.y);
        }
    }

    // 扫描线间距
    float spacing = extrusionWidth / std::max(density, 0.01f);
    float rad = angle * 3.14159265f / 180.0f;
    float cosA = std::cos(rad), sinA = std::sin(rad);

    // 生成扫描线
    float diag = (bMax - bMin).length();
    for (float d = -diag; d <= diag; d += spacing) {
        // 扫描线方向向量
        Vec2 dir(cosA, sinA);
        Vec2 perp(-sinA, cosA);
        // 扫描线起点
        Vec2 center = Vec2((bMin.x + bMax.x) * 0.5f, (bMin.y + bMax.y) * 0.5f);
        Vec2 start = center + perp * d - dir * diag;
        Vec2 end = center + perp * d + dir * diag;

        // 与所有多边形求交
        auto intersections = scanlineIntersect(boundary, 0);  // 简化：使用参数化交点
        // 直接计算线段与多边形边的交点
        std::vector<float> tValues;
        for (const auto& poly : boundary) {
            for (size_t i = 0; i < poly.points.size(); i++) {
                size_t j = (i + 1) % poly.points.size();
                // 线段-线段求交
                Vec2 p1 = poly.points[i], p2 = poly.points[j];
                Vec2 r = end - start;
                Vec2 s = p2 - p1;
                float rxs = r.cross(s);
                if (std::abs(rxs) < 1e-10f) continue;
                float t = (p1 - start).cross(s) / rxs;
                float u = (p1 - start).cross(r) / rxs;
                if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
                    tValues.push_back(t);
                }
            }
        }

        std::sort(tValues.begin(), tValues.end());

        // 成对生成路径段
        for (size_t i = 0; i + 1 < tValues.size(); i += 2) {
            PathSegment seg;
            seg.start = start + (end - start) * tValues[i];
            seg.end = start + (end - start) * tValues[i + 1];
            seg.speed = speed;
            seg.extrusion = 1.0f;
            paths.push_back(seg);
        }
    }

    return paths;
}

std::vector<PathSegment> NailSlicer::infillGrid(const std::vector<Polygon>& boundary,
                                                  float density, float angle,
                                                  float extrusionWidth, float speed) {
    // 网格 = 两组正交直线
    auto paths = infillLines(boundary, density, angle, extrusionWidth, speed);
    auto paths2 = infillLines(boundary, density, angle + 90.0f, extrusionWidth, speed);
    paths.insert(paths.end(), paths2.begin(), paths2.end());
    return paths;
}

std::vector<PathSegment> NailSlicer::infillHoneycomb(const std::vector<Polygon>& boundary,
                                                       float density, float cellSize,
                                                       float extrusionWidth, float speed) {
    // 简化蜂窝：用六边形图案近似
    // 实际实现需要完整的六边形铺贴
    return infillGrid(boundary, density, 30.0f, extrusionWidth, speed);
}

std::vector<PathSegment> NailSlicer::infillConcentric(const std::vector<Polygon>& boundary,
                                                        float density, float offset,
                                                        float speed) {
    std::vector<PathSegment> paths;
    float spacing = offset / std::max(density, 0.01f);

    for (const auto& poly : boundary) {
        Polygon current = poly;
        for (float d = spacing; d < 100.0f; d += spacing) {
            current = offsetPolygon(current, -d);
            if (current.points.size() < 3) break;

            for (size_t i = 0; i < current.points.size(); i++) {
                size_t j = (i + 1) % current.points.size();
                PathSegment seg;
                seg.start = current.points[i];
                seg.end = current.points[j];
                seg.speed = speed;
                seg.extrusion = 1.0f;
                paths.push_back(seg);
            }
        }
    }

    return paths;
}

std::vector<Vec2> NailSlicer::scanlineIntersect(const std::vector<Polygon>& polys, float y) {
    std::vector<Vec2> intersections;
    for (const auto& poly : polys) {
        for (size_t i = 0; i < poly.points.size(); i++) {
            size_t j = (i + 1) % poly.points.size();
            const Vec2& p1 = poly.points[i];
            const Vec2& p2 = poly.points[j];
            if ((p1.y <= y && p2.y > y) || (p2.y <= y && p1.y > y)) {
                float t = (y - p1.y) / (p2.y - p1.y);
                intersections.push_back(Vec2(p1.x + t * (p2.x - p1.x), y));
            }
        }
    }
    return intersections;
}

// ============================================================
// 支撑生成
// ============================================================

std::vector<Polygon> NailSlicer::detectOverhang(const std::vector<SliceLayer>& layers,
                                                  int layerIndex, float angleThreshold) {
    std::vector<Polygon> overhang;
    if (layerIndex < 2 || layerIndex >= (int)layers.size()) return overhang;

    // 比较当前层与上一层的轮廓
    // 当前层有但上一层没有的区域 = 悬垂
    // 简化版：检查当前层轮廓面积是否大于上一层
    const auto& current = layers[layerIndex];
    const auto& previous = layers[layerIndex - 1];

    float currentArea = 0, previousArea = 0;
    for (const auto& c : current.contours) currentArea += std::abs(c.area());
    for (const auto& c : previous.contours) previousArea += std::abs(c.area());

    if (currentArea > previousArea * 1.1f) {
        // 有悬垂，返回当前层轮廓作为支撑区域
        overhang = current.contours;
    }

    return overhang;
}

std::vector<PathSegment> NailSlicer::generateSupport(const std::vector<Polygon>& overhang,
                                                       float density, float offset,
                                                       float speed) {
    // 支撑 = 稀疏填充
    return infillLines(overhang, density, 0.0f, 0.3f, speed);
}

// ============================================================
// 路径优化
// ============================================================

void NailSlicer::optimizePath(std::vector<PathSegment>& paths) {
    if (paths.size() < 2) return;

    // 最近邻贪心 TSP 近似
    std::vector<PathSegment> optimized;
    std::vector<bool> used(paths.size(), false);

    // 从第一条路径开始
    int current = 0;
    used[current] = true;
    optimized.push_back(paths[current]);

    Vec2 currentPos = paths[current].end;

    for (size_t i = 1; i < paths.size(); i++) {
        float bestDist = 1e30f;
        int bestIdx = -1;
        bool bestReversed = false;

        for (size_t j = 0; j < paths.size(); j++) {
            if (used[j]) continue;

            float d1 = (paths[j].start - currentPos).length();
            if (d1 < bestDist) {
                bestDist = d1;
                bestIdx = (int)j;
                bestReversed = false;
            }

            float d2 = (paths[j].end - currentPos).length();
            if (d2 < bestDist) {
                bestDist = d2;
                bestIdx = (int)j;
                bestReversed = true;
            }
        }

        if (bestIdx >= 0) {
            used[bestIdx] = true;
            PathSegment seg = paths[bestIdx];
            if (bestReversed) {
                std::swap(seg.start, seg.end);
            }
            // 添加空行程
            if (bestDist > 0.1f) {
                PathSegment travel;
                travel.start = currentPos;
                travel.end = seg.start;
                travel.travel = true;
                travel.speed = 120.0f;
                optimized.push_back(travel);
            }
            optimized.push_back(seg);
            currentPos = seg.end;
        }
    }

    paths = std::move(optimized);
}

float NailSlicer::computeExtrusion(float dx, float dy, float layerHeight,
                                     float extrusionWidth, float filamentDiameter) {
    float pathLength = std::sqrt(dx * dx + dy * dy);
    // E = (挤出截面积 / 耗材截面积) * 路径长度
    float extrusionArea = layerHeight * extrusionWidth;
    float filamentArea = 3.14159265f * (filamentDiameter * 0.5f) * (filamentDiameter * 0.5f);
    return (extrusionArea / filamentArea) * pathLength;
}

// ============================================================
// NailPrintSlicer — 美甲专用切片
// ============================================================

std::vector<SliceLayer> NailPrintSlicer::sliceNail(const Mesh& mesh,
                                                     const PrintConfig& config) {
    std::cout << "[NailSlice] 美甲专用切片..." << std::endl;

    SliceConfig sliceConfig;
    sliceConfig.layerHeight = config.colorLayerHeight;
    sliceConfig.firstLayerHeight = config.baseThickness;
    sliceConfig.perimeters = config.perimeters;
    sliceConfig.infillDensity = 0.3f;
    sliceConfig.infillPattern = InfillPattern::Grid;
    sliceConfig.nailMode = true;
    sliceConfig.baseLayerHeight = config.baseThickness;
    sliceConfig.colorLayerHeight = config.colorLayerHeight;
    sliceConfig.topCoatHeight = config.topCoatThickness;

    auto layers = NailSlicer::slice(mesh, sliceConfig);

    // 分层标记：底胶层 / 颜色层 / 封层
    int baseLayers = (int)std::ceil(config.baseThickness / config.colorLayerHeight);
    int topCoatLayers = (int)std::ceil(config.topCoatThickness / config.colorLayerHeight);

    std::cout << "[NailSlice] 底胶层: " << baseLayers << " 层, "
              << "颜色层: " << (layers.size() - baseLayers - topCoatLayers) << " 层, "
              << "封层: " << topCoatLayers << " 层" << std::endl;

    // 分配颜色
    if (config.colorCount > 1 && !config.palette.empty()) {
        assignColors(layers, config.palette, config.colorCount);
    }

    return layers;
}

void NailPrintSlicer::assignColors(std::vector<SliceLayer>& layers,
                                    const std::vector<ColorRGBA8>& palette,
                                    int colorCount) {
    std::cout << "[NailSlice] 分配颜色 (" << colorCount << " 色)..." << std::endl;

    int baseLayers = 1;  // 底胶层
    int topCoatLayers = 1;  // 封层

    for (int i = 0; i < (int)layers.size(); i++) {
        int colorIndex = 0;

        if (i < baseLayers) {
            colorIndex = 0;  // 底胶（透明/白色）
        } else if (i >= (int)layers.size() - topCoatLayers) {
            colorIndex = 0;  // 封层（透明）
        } else {
            // 颜色层：按层数分配颜色
            int colorLayer = i - baseLayers;
            colorIndex = (colorLayer / 3) % colorCount;  // 每3层换一次颜色
        }

        for (auto& seg : layers[i].perimeters)
            seg.colorIndex = colorIndex;
        for (auto& seg : layers[i].infill)
            seg.colorIndex = colorIndex;
    }
}

std::vector<PathSegment> NailPrintSlicer::generateColorBlend(
    const Vec2& start, const Vec2& end,
    int colorIndex1, int colorIndex2,
    float blendDistance, float speed) {
    std::vector<PathSegment> paths;

    float totalLength = (end - start).length();
    if (totalLength < 1e-6f) return paths;

    Vec2 dir = (end - start).normalized();
    int segments = (int)(totalLength / blendDistance);
    if (segments < 1) segments = 1;

    for (int i = 0; i < segments; i++) {
        float t1 = (float)i / segments;
        float t2 = (float)(i + 1) / segments;
        PathSegment seg;
        seg.start = start + dir * (t1 * totalLength);
        seg.end = start + dir * (t2 * totalLength);
        seg.speed = speed;
        seg.extrusion = 1.0f;
        // 在过渡区域混合颜色
        seg.colorIndex = (t1 < 0.5f) ? colorIndex1 : colorIndex2;
        paths.push_back(seg);
    }

    return paths;
}

} // namespace NailPrint3D
