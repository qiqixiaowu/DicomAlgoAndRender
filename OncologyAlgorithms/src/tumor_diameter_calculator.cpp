/**
 * @file tumor_diameter_calculator.cpp
 * @brief 肿瘤径线计算实现
 *
 * 算法流程
 * ─────────
 * 2D 最大截面长径
 *   Step1: 统计每个 Z 切片含有的分割体素数，取最多的切片
 *   Step2: 在该切片上收集所有前景点的 (x,y) 坐标
 *   Step3: 使用旋转卡壳法（Rotating Calipers）计算凸包上的最大距离对
 *          → 得到长径端点对 (ldStart, ldEnd)
 *   Step4: 在凸包上搜索与长径方向垂直、跨度最大的投影
 *          → 得到垂直径端点对 (pdStart, pdEnd)
 *
 * 3D 最长径（近似）
 *   采用迭代最远点搜索（类 powerMethod）：
 *   - 从任意体素出发，找距离最远的体素 A
 *   - 再从 A 出发，找距离最远的体素 B
 *   - (A, B) 近似为直径端点
 *   对大型结节还支持距离限时保护。
 */

#include "tumor_diameter_calculator.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <chrono>
#include <cassert>
#include <cstring>

namespace Onc {

// ──────────────────────────────────────
//  内部工具函数
// ──────────────────────────────────────

/** Graham scan 凸包，返回逆时针顺序的凸包顶点 */
static std::vector<Point3f> ConvexHull2D(std::vector<Point3f> pts)
{
    if (pts.size() < 3) return pts;
    // 按 (y,x) 排序，取最下方最左点作为极点
    std::sort(pts.begin(), pts.end(), [](const Point3f& a, const Point3f& b){
        return (a.y != b.y) ? a.y < b.y : a.x < b.x;
    });
    Point3f pivot = pts[0];

    // 按极角排序
    std::sort(pts.begin()+1, pts.end(), [&pivot](const Point3f& a, const Point3f& b){
        float ax = a.x - pivot.x, ay = a.y - pivot.y;
        float bx = b.x - pivot.x, by = b.y - pivot.y;
        float cross = ax*by - ay*bx;
        if (cross != 0.f) return cross > 0.f;
        // 共线则距近的在前
        return (ax*ax+ay*ay) < (bx*bx+by*by);
    });

    std::vector<Point3f> hull;
    auto cross2D = [](const Point3f& O, const Point3f& A, const Point3f& B){
        return (A.x-O.x)*(B.y-O.y) - (A.y-O.y)*(B.x-O.x);
    };
    for (const auto& p : pts) {
        while (hull.size() >= 2 && cross2D(hull[hull.size()-2], hull.back(), p) <= 0.f)
            hull.pop_back();
        hull.push_back(p);
    }
    return hull;
}

/** 旋转卡壳：在 2D 凸包上找最大跨度点对 */
static std::pair<Point3f,Point3f> RotatingCalipers(const std::vector<Point3f>& hull)
{
    int n = static_cast<int>(hull.size());
    if (n == 0) return {};
    if (n == 1) return {hull[0], hull[0]};
    if (n == 2) return {hull[0], hull[1]};

    float maxDist2 = 0.f;
    int ai = 0, bi = 0;
    int j = 1;
    for (int i = 0; i < n; ++i) {
        const Point3f& A = hull[i];
        const Point3f& B = hull[(i+1)%n];
        float edgeX = B.x - A.x, edgeY = B.y - A.y;
        while (true) {
            const Point3f& C = hull[j];
            const Point3f& D = hull[(j+1)%n];
            // 叉积判断 j+1 是否把与边 AB 的垂直分量增大
            float cross = edgeX*(D.y-C.y) - edgeY*(D.x-C.x);
            if (cross > 0.f) j = (j+1)%n;
            else break;
        }
        float dx = hull[j].x - A.x, dy = hull[j].y - A.y;
        float d2 = dx*dx + dy*dy;
        if (d2 > maxDist2) { maxDist2 = d2; ai = i; bi = j; }
    }
    return {hull[ai], hull[bi]};
}

// ──────────────────────────────────────
//  2D 最大截面长径/垂直径
// ──────────────────────────────────────

static bool Calc2DDiameters(
    const SegmentResult& segPoints,
    const ImageInfo&     info,
    TumorDiameterResult& result)
{
    // 按 Z 切片统计体素数
    std::map<int, std::vector<Point3f>> sliceMap;
    for (const auto& p : segPoints) {
        float px = static_cast<float>(p.x) * static_cast<float>(info.spacing[0]);
        float py = static_cast<float>(p.y) * static_cast<float>(info.spacing[1]);
        sliceMap[p.z].push_back({px, py, 0.f});
    }
    if (sliceMap.empty()) return false;

    // 找体素数最多的切片
    auto maxIt = std::max_element(sliceMap.begin(), sliceMap.end(),
        [](const auto& a, const auto& b){ return a.second.size() < b.second.size(); });
    const auto& pts2D = maxIt->second;
    int bestZ = maxIt->first;

    // 凸包 + 旋转卡壳
    auto hull = ConvexHull2D(pts2D);
    auto [ldA, ldB] = RotatingCalipers(hull);

    float ldLen = ldA.distanceTo(ldB); // mm
    result.longestDiameter2D_cm = ldLen / 10.f;
    // 还原为体素坐标（粗略）
    float spX = static_cast<float>(info.spacing[0]);
    float spY = static_cast<float>(info.spacing[1]);
    result.ldStart = {ldA.x / spX, ldA.y / spY, static_cast<float>(bestZ)};
    result.ldEnd   = {ldB.x / spX, ldB.y / spY, static_cast<float>(bestZ)};

    // 垂直径方向：与长径垂直，在凸包上搜索最大投影跨度
    float dirX = ldB.x - ldA.x, dirY = ldB.y - ldA.y;
    float len  = std::sqrt(dirX*dirX + dirY*dirY);
    if (len < 1e-6f) { result.perpendicularDiameter_cm = 0.f; return true; }
    // 垂直方向
    float perpX = -dirY / len, perpY = dirX / len;

    float minProj = std::numeric_limits<float>::max();
    float maxProj = -std::numeric_limits<float>::max();
    int   minIdx = 0, maxIdx = 0;
    for (int i = 0; i < (int)hull.size(); ++i) {
        float proj = hull[i].x * perpX + hull[i].y * perpY;
        if (proj < minProj) { minProj = proj; minIdx = i; }
        if (proj > maxProj) { maxProj = proj; maxIdx = i; }
    }
    float pdLen = maxProj - minProj; // mm
    result.perpendicularDiameter_cm = pdLen / 10.f;
    result.pdStart = {hull[minIdx].x / spX, hull[minIdx].y / spY, static_cast<float>(bestZ)};
    result.pdEnd   = {hull[maxIdx].x / spX, hull[maxIdx].y / spY, static_cast<float>(bestZ)};

    return true;
}

// ──────────────────────────────────────
//  3D 最长径（迭代最远点近似）
// ──────────────────────────────────────

static float Calc3DDiameter(
    const SegmentResult& segPoints,
    const ImageInfo&     info,
    Point3f&             p3dA,
    Point3f&             p3dB,
    double               maxCalTimeMs)
{
    if (segPoints.empty()) return -1.f;

    auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() -> double {
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    // 将体素坐标转为物理坐标（mm）
    const float sx = static_cast<float>(info.spacing[0]);
    const float sy = static_cast<float>(info.spacing[1]);
    const float sz = static_cast<float>(info.spacing[2]);

    // 采样上限，避免超大结节耗时过长
    constexpr int MAX_SAMPLE = 2000;
    std::vector<Point3f> phys;
    phys.reserve(std::min<int>(segPoints.size(), MAX_SAMPLE));
    int step = std::max<int>(1, static_cast<int>(segPoints.size()) / MAX_SAMPLE);
    for (int i = 0; i < (int)segPoints.size(); i += step)
        phys.push_back({segPoints[i].x * sx,
                        segPoints[i].y * sy,
                        segPoints[i].z * sz});

    if (phys.empty()) return -1.f;

    // 迭代最远点
    auto findFarthest = [&](const Point3f& from) -> int {
        float maxD2 = -1.f; int idx = 0;
        for (int i = 0; i < (int)phys.size(); ++i) {
            float dx = phys[i].x - from.x;
            float dy = phys[i].y - from.y;
            float dz = phys[i].z - from.z;
            float d2 = dx*dx + dy*dy + dz*dz;
            if (d2 > maxD2) { maxD2 = d2; idx = i; }
        }
        return idx;
    };

    if (elapsed() > maxCalTimeMs) return -1.f;
    int idxA = findFarthest(phys[0]);
    if (elapsed() > maxCalTimeMs) return -1.f;
    int idxB = findFarthest(phys[idxA]);
    if (elapsed() > maxCalTimeMs) return -1.f;
    // 再迭代一次提升精度
    idxA = findFarthest(phys[idxB]);
    if (elapsed() > maxCalTimeMs) return -1.f;

    p3dA = phys[idxA];
    p3dB = phys[idxB];
    return phys[idxA].distanceTo(phys[idxB]);
}

// ──────────────────────────────────────
//  公开接口实现
// ──────────────────────────────────────

bool CalculateTumorDiameters(
    const SegmentResult& segPoints,
    const ImageInfo&     info,
    TumorDiameterResult& result,
    double               maxCalTimeMs,
    ProgressCallback     progress)
{
    if (segPoints.empty()) return false;
    if (info.dim[0] <= 0 || info.dim[1] <= 0 || info.dim[2] <= 0) return false;

    result = TumorDiameterResult{};

    // Step1: 2D 长短径
    if (!Calc2DDiameters(segPoints, info, result)) return false;
    if (progress) progress(0.5);

    // Step2: 3D 最长径
    Point3f a3d, b3d;
    float d3d = Calc3DDiameter(segPoints, info, a3d, b3d, maxCalTimeMs * 0.5);
    result.longestDiameter3D_cm = (d3d >= 0.f) ? d3d / 10.f : -1.f;
    if (progress) progress(1.0);

    return true;
}

} // namespace Onc
