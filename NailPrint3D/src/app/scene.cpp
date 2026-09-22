/**
 * @file    scene.cpp
 * @brief   场景状态 实现
 */

#include "app/scene.h"

namespace NailPrint3D {

void Scene::uploadSlicePaths() {
    std::vector<LineSegment3D> segments;
    for (int li = 0; li < (int)sliceLayers.size(); li++) {
        const auto& layer = sliceLayers[li];
        float z = layer.zHeight;
        // 外轮廓
        for (const auto& contour : layer.contours) {
            int n = (int)contour.points.size();
            for (int i = 0; i < n; i++) {
                int j = contour.closed ? (i + 1) % n : i + 1;
                if (!contour.closed && i == n - 1) break;
                LineSegment3D seg;
                seg.start = Vec3(contour.points[i].x, contour.points[i].y, z);
                seg.end   = Vec3(contour.points[j].x, contour.points[j].y, z);
                segments.push_back(seg);
            }
        }
        // 孔洞
        for (const auto& hole : layer.holes) {
            int n = (int)hole.points.size();
            for (int i = 0; i < n; i++) {
                int j = hole.closed ? (i + 1) % n : i + 1;
                if (!hole.closed && i == n - 1) break;
                LineSegment3D seg;
                seg.start = Vec3(hole.points[i].x, hole.points[i].y, z);
                seg.end   = Vec3(hole.points[j].x, hole.points[j].y, z);
                segments.push_back(seg);
            }
        }
        // 填充路径
        for (const auto& seg2d : layer.infill) {
            LineSegment3D seg;
            seg.start = Vec3(seg2d.start.x, seg2d.start.y, z);
            seg.end   = Vec3(seg2d.end.x,   seg2d.end.y,   z);
            segments.push_back(seg);
        }
        // 外壳路径
        for (const auto& seg2d : layer.perimeters) {
            LineSegment3D seg;
            seg.start = Vec3(seg2d.start.x, seg2d.start.y, z);
            seg.end   = Vec3(seg2d.end.x,   seg2d.end.y,   z);
            segments.push_back(seg);
        }
        // 支撑路径
        for (const auto& seg2d : layer.supportPaths) {
            LineSegment3D seg;
            seg.start = Vec3(seg2d.start.x, seg2d.start.y, z);
            seg.end   = Vec3(seg2d.end.x,   seg2d.end.y,   z);
            segments.push_back(seg);
        }
    }
    glSliceMesh.uploadLines(segments);
}

} // namespace NailPrint3D
