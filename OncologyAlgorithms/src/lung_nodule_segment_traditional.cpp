/**
 * @file lung_nodule_segment_traditional.cpp
 * @brief CT 肺结节传统分割实现
 *
 * 分阶段流程：胸腔提取 → 初始分割 → Hessian精化 → 凸包平滑
 * 贴壁/血管型结节有专用精化分支。
 */

#include "lung_nodule_segment_traditional.h"
#include <queue>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <cstring>

namespace Onc {

LungNoduleSegmentTraditional::LungNoduleSegmentTraditional(const LungNoduleSegParams& p)
    : m_params(p) {}

// ─────────────────────────────────────────────
//  Step1: 胸腔肺野提取
//  算法：
//   a. 阈值 < -400 HU → 肺野候选
//   b. 从图像四角出发 BFS 找最大连通背景（体外空气）
//   c. 排除体外空气后剩余低密度连通区域即为肺野
// ─────────────────────────────────────────────
bool LungNoduleSegmentTraditional::extractLungMask(
    const short*      data,
    const ImageInfo&  info,
    std::vector<uint8_t>& lungMask)
{
    int N = info.totalVoxels();
    lungMask.assign(N, 0);

    // a. 低密度二值化
    std::vector<uint8_t> lowDens(N, 0);
    for (int i = 0; i < N; ++i)
        lowDens[i] = (data[i] < static_cast<short>(m_params.lungHUThresh)) ? 1 : 0;

    // b. 从四角 BFS 标记体外空气
    const int dx6[] = {1,-1,0,0,0,0};
    const int dy6[] = {0,0,1,-1,0,0};
    const int dz6[] = {0,0,0,0,1,-1};
    std::vector<uint8_t> airMask(N, 0);
    std::queue<Point3i> q;

    auto tryAdd = [&](int x, int y, int z) {
        if (!info.inBounds(x,y,z)) return;
        int i = info.linearIndex(x,y,z);
        if (lowDens[i] && !airMask[i]) {
            airMask[i] = 1; q.push({x,y,z});
        }
    };
    // 加入四条 Z 边界上的所有体外体素
    for (int z = 0; z < info.dim[2]; z += info.dim[2]-1)
    for (int y = 0; y < info.dim[1]; ++y)
    for (int x = 0; x < info.dim[0]; ++x)
        tryAdd(x, y, z);
    for (int z = 0; z < info.dim[2]; ++z)
    for (int y = 0; y < info.dim[1]; y += info.dim[1]-1)
    for (int x = 0; x < info.dim[0]; ++x)
        tryAdd(x, y, z);

    while (!q.empty()) {
        auto cur = q.front(); q.pop();
        for (int d = 0; d < 6; ++d) {
            int nx = cur.x+dx6[d], ny = cur.y+dy6[d], nz = cur.z+dz6[d];
            if (!info.inBounds(nx,ny,nz)) continue;
            int ni = info.linearIndex(nx,ny,nz);
            if (lowDens[ni] && !airMask[ni]) {
                airMask[ni] = 1; q.push({nx,ny,nz});
            }
        }
    }

    // c. 低密度 且 非体外空气 → 肺野候选（两肺）
    for (int i = 0; i < N; ++i)
        lungMask[i] = (lowDens[i] && !airMask[i]) ? 1 : 0;

    // 轻度膨胀：把紧贴胸壁的结节也包含在内（dilate 1 pixel）
    std::vector<uint8_t> dilated(N, 0);
    for (int z = 0; z < info.dim[2]; ++z)
    for (int y = 0; y < info.dim[1]; ++y)
    for (int x = 0; x < info.dim[0]; ++x) {
        if (!lungMask[info.linearIndex(x,y,z)]) continue;
        for (int d = 0; d < 6; ++d) {
            int nx = x+dx6[d], ny = y+dy6[d], nz = z+dz6[d];
            if (info.inBounds(nx,ny,nz))
                dilated[info.linearIndex(nx,ny,nz)] = 1;
        }
    }
    // 合并
    for (int i = 0; i < N; ++i)
        if (dilated[i]) lungMask[i] = 1;

    return true;
}

// ─────────────────────────────────────────────
//  Step2: 阈值+连通初始分割
// ─────────────────────────────────────────────
bool LungNoduleSegmentTraditional::initialSegment(
    const short*            data,
    const ImageInfo&        info,
    const std::vector<uint8_t>& lungMask,
    const Point3i&          seedPoint,
    SegmentResult&          coarseResult)
{
    if (!info.inBounds(seedPoint.x, seedPoint.y, seedPoint.z)) return false;
    int seedIdx = info.linearIndex(seedPoint.x, seedPoint.y, seedPoint.z);
    if (!lungMask[seedIdx]) return false;   // 种子不在肺野内

    const int dx6[] = {1,-1,0,0,0,0};
    const int dy6[] = {0,0,1,-1,0,0};
    const int dz6[] = {0,0,0,0,1,-1};

    float huMin = m_params.noduleHUMin;
    float huMax = m_params.noduleHUMax;
    short tMin  = static_cast<short>(huMin);
    short tMax  = static_cast<short>(huMax);

    // 在 ROI 范围内 BFS
    int roiVox = static_cast<int>(m_params.roiRadiusMM / info.spacing[0]) + 2;
    int x0 = std::max(0, seedPoint.x - roiVox), x1 = std::min(info.dim[0]-1, seedPoint.x + roiVox);
    int y0 = std::max(0, seedPoint.y - roiVox), y1 = std::min(info.dim[1]-1, seedPoint.y + roiVox);
    int z0 = std::max(0, seedPoint.z - roiVox), z1 = std::min(info.dim[2]-1, seedPoint.z + roiVox);

    std::vector<uint8_t> visited(info.totalVoxels(), 0);
    std::queue<Point3i> q;

    if (data[seedIdx] < tMin || data[seedIdx] > tMax) {
        // 种子 HU 不在结节范围，寻找邻域最近合适体素
        bool found = false;
        for (int dz = -3; dz <= 3 && !found; ++dz)
        for (int dy = -3; dy <= 3 && !found; ++dy)
        for (int dx = -3; dx <= 3 && !found; ++dx) {
            int nx = seedPoint.x+dx, ny = seedPoint.y+dy, nz = seedPoint.z+dz;
            if (!info.inBounds(nx,ny,nz)) continue;
            int ni = info.linearIndex(nx,ny,nz);
            if (data[ni] >= tMin && data[ni] <= tMax && lungMask[ni]) {
                q.push({nx,ny,nz}); visited[ni] = 1; found = true;
            }
        }
        if (!found) return false;
    } else {
        q.push(seedPoint); visited[seedIdx] = 1;
    }

    while (!q.empty()) {
        auto cur = q.front(); q.pop();
        coarseResult.push_back(cur);
        for (int d = 0; d < 6; ++d) {
            int nx = cur.x+dx6[d], ny = cur.y+dy6[d], nz = cur.z+dz6[d];
            if (nx<x0||nx>x1||ny<y0||ny>y1||nz<z0||nz>z1) continue;
            if (!info.inBounds(nx,ny,nz)) continue;
            int ni = info.linearIndex(nx,ny,nz);
            if (visited[ni] || !lungMask[ni]) continue;
            if (data[ni] < tMin || data[ni] > tMax) continue;
            visited[ni] = 1;
            q.push({nx,ny,nz});
        }
    }
    return !coarseResult.empty();
}

// ─────────────────────────────────────────────
//  Step3a: Hessian 引导精化（实性结节）
// ─────────────────────────────────────────────
bool LungNoduleSegmentTraditional::refineWithHessian(
    const short*           data,
    const ImageInfo&       info,
    const SegmentResult&   coarse,
    const Point3i&         seedPoint,
    SegmentResult&         refined)
{
    if (coarse.empty()) return false;

    // 计算 ROI 边界
    int xMin=info.dim[0], xMax=0, yMin=info.dim[1], yMax=0, zMin=info.dim[2], zMax=0;
    for (const auto& p : coarse) {
        xMin=std::min(xMin,p.x); xMax=std::max(xMax,p.x);
        yMin=std::min(yMin,p.y); yMax=std::max(yMax,p.y);
        zMin=std::min(zMin,p.z); zMax=std::max(zMax,p.z);
    }
    // 加边距
    int mg = 3;
    xMin=std::max(0,xMin-mg); xMax=std::min(info.dim[0]-1,xMax+mg);
    yMin=std::max(0,yMin-mg); yMax=std::min(info.dim[1]-1,yMax+mg);
    zMin=std::max(0,zMin-mg); zMax=std::min(info.dim[2]-1,zMax+mg);

    // 构建 ROI 子图像
    ImageInfo roiInfo;
    roiInfo.dim[0] = xMax-xMin+1;
    roiInfo.dim[1] = yMax-yMin+1;
    roiInfo.dim[2] = zMax-zMin+1;
    for (int k=0;k<3;++k) roiInfo.spacing[k] = info.spacing[k];

    int roiN = roiInfo.totalVoxels();
    std::vector<short> roiData(roiN);
    for (int z=zMin; z<=zMax; ++z)
    for (int y=yMin; y<=yMax; ++y)
    for (int x=xMin; x<=xMax; ++x) {
        int ri = roiInfo.linearIndex(x-xMin, y-yMin, z-zMin);
        roiData[ri] = data[info.linearIndex(x,y,z)];
    }

    // Hessian 滤波（使用两个尺度）
    HessianFilterParams hp;
    hp.scales = {m_params.hessianScale1, m_params.hessianScale2};
    HessianFilter hf(hp);
    std::vector<float> hdot, hline, hplane;
    hf.compute(roiData.data(), roiInfo, hdot, hline, hplane);

    // 以 hdot 响应 > 阈值 处为血管/高曲率区域，从分割中排除
    // 结节区域：hdot 较低（圆球状）
    // 阈值选取：hdot 均值 + 0.3*(max-mean)
    float sumH = 0.f, maxH = 0.f;
    for (float v : hdot) { sumH += v; maxH = std::max(maxH, v); }
    float meanH = sumH / roiN;
    float hThresh = meanH + 0.3f * (maxH - meanH);

    // 构建初始掩码（粗分割结果）
    std::vector<uint8_t> mask(info.totalVoxels(), 0);
    for (const auto& p : coarse)
        mask[info.linearIndex(p.x,p.y,p.z)] = 1;

    // 精化：在粗分割内，移除 hdot > hThresh 的体素（血管区域）
    refined.clear();
    for (const auto& p : coarse) {
        if (!info.inBounds(p.x,p.y,p.z)) continue;
        int ri = roiInfo.linearIndex(p.x-xMin, p.y-yMin, p.z-zMin);
        if (ri < 0 || ri >= roiN) continue;
        if (hdot[ri] <= hThresh) refined.push_back(p);
    }

    // 若精化后过小，退回粗分割
    if (refined.size() < 5) refined = coarse;
    return !refined.empty();
}

// ─────────────────────────────────────────────
//  Step3b: 贴壁结节精化
// ─────────────────────────────────────────────
bool LungNoduleSegmentTraditional::refineJuxtaWall(
    const short*           data,
    const ImageInfo&       info,
    const std::vector<uint8_t>& lungMask,
    const SegmentResult&   coarse,
    const Point3i&         seedPoint,
    SegmentResult&         refined)
{
    // 贴壁结节分离策略：
    //   通过腐蚀肺野掩码 3 次（移除贴壁边缘），
    //   然后在剩余肺野内重新做阈值分割
    if (coarse.empty()) return false;

    int N = info.totalVoxels();
    const int dx6[] = {1,-1,0,0,0,0};
    const int dy6[] = {0,0,1,-1,0,0};
    const int dz6[] = {0,0,0,0,1,-1};

    // 腐蚀 lungMask 2 次
    std::vector<uint8_t> erodedMask = lungMask;
    for (int iter = 0; iter < 2; ++iter) {
        std::vector<uint8_t> tmp(N, 0);
        for (int z=0; z<info.dim[2]; ++z)
        for (int y=0; y<info.dim[1]; ++y)
        for (int x=0; x<info.dim[0]; ++x) {
            if (!erodedMask[info.linearIndex(x,y,z)]) continue;
            bool allOn = true;
            for (int d=0;d<6;++d) {
                int nx=x+dx6[d], ny=y+dy6[d], nz=z+dz6[d];
                if (!info.inBounds(nx,ny,nz) || !erodedMask[info.linearIndex(nx,ny,nz)]) {
                    allOn = false; break;
                }
            }
            if (allOn) tmp[info.linearIndex(x,y,z)] = 1;
        }
        erodedMask = std::move(tmp);
    }

    // 在腐蚀后的掩码内重新初始分割
    return initialSegment(data, info, erodedMask, seedPoint, refined);
}

// ─────────────────────────────────────────────
//  Step3c: 血管附着结节精化
// ─────────────────────────────────────────────
bool LungNoduleSegmentTraditional::refineJuxtaVessel(
    const short*           data,
    const ImageInfo&       info,
    const SegmentResult&   coarse,
    const Point3i&         seedPoint,
    SegmentResult&         refined)
{
    // 血管附着精化：
    //  通过 Hessian 管状响应（hline 较高）标记血管体素，
    //  然后在粗分割内排除血管体素，再做连通域过滤
    return refineWithHessian(data, info, coarse, seedPoint, refined);
}

// ─────────────────────────────────────────────
//  Auto 判断结节类型
// ─────────────────────────────────────────────
NoduleType LungNoduleSegmentTraditional::detectNoduleType(
    const short*      data,
    const ImageInfo&  info,
    const Point3i&    seed,
    const std::vector<uint8_t>& lungMask) const
{
    // 简单规则：
    //   检查种子点5像素范围内是否有肺野边界体素（贴壁检测）
    const int R = 5;
    bool nearWall = false;
    for (int dz=-R; dz<=R && !nearWall; ++dz)
    for (int dy=-R; dy<=R && !nearWall; ++dy)
    for (int dx=-R; dx<=R && !nearWall; ++dx) {
        int nx=seed.x+dx, ny=seed.y+dy, nz=seed.z+dz;
        if (!info.inBounds(nx,ny,nz)) continue;
        // 若邻域体素不在肺野但高密度，说明贴壁
        int ni = info.linearIndex(nx,ny,nz);
        if (!lungMask[ni] && data[ni] > 0) nearWall = true;
    }
    if (nearWall) return NoduleType::JuxtaWall;

    // 磨玻璃：种子点 HU 在 [-600, -200] 范围
    short seedHU = data[info.linearIndex(seed.x, seed.y, seed.z)];
    if (seedHU >= -600 && seedHU <= -200) return NoduleType::GGO;

    return NoduleType::Solid;
}

// ─────────────────────────────────────────────
//  简单 3D 凸包平滑（近似版：填充凸包包围盒内的凹陷体素）
// ─────────────────────────────────────────────
void LungNoduleSegmentTraditional::applyConvexHull(
    SegmentResult& inout, const ImageInfo& info)
{
    if (inout.empty()) return;

    // 找 bounding box
    int xMin=info.dim[0], xMax=0, yMin=info.dim[1], yMax=0, zMin=info.dim[2], zMax=0;
    std::vector<uint8_t> mask(info.totalVoxels(), 0);
    for (const auto& p : inout) {
        xMin=std::min(xMin,p.x); xMax=std::max(xMax,p.x);
        yMin=std::min(yMin,p.y); yMax=std::max(yMax,p.y);
        zMin=std::min(zMin,p.z); zMax=std::max(zMax,p.z);
        mask[info.linearIndex(p.x,p.y,p.z)] = 1;
    }

    // 按 Z 切片做 2D 凸包填充（近似 3D 凸包）
    for (int z = zMin; z <= zMax; ++z) {
        // 找该切片所有前景点的 x、y 范围
        int sxMin=info.dim[0], sxMax=0, syMin=info.dim[1], syMax=0;
        bool hasAny = false;
        for (int y=yMin; y<=yMax; ++y)
        for (int x=xMin; x<=xMax; ++x) {
            if (!mask[info.linearIndex(x,y,z)]) continue;
            hasAny = true;
            sxMin=std::min(sxMin,x); sxMax=std::max(sxMax,x);
            syMin=std::min(syMin,y); syMax=std::max(syMax,y);
        }
        if (!hasAny) continue;
        // 按行填充（简单的"行扫描凸包"近似）
        for (int y=syMin; y<=syMax; ++y) {
            int rowXMin = sxMax+1, rowXMax = sxMin-1;
            for (int x=sxMin; x<=sxMax; ++x) {
                if (mask[info.linearIndex(x,y,z)]) {
                    rowXMin=std::min(rowXMin,x); rowXMax=std::max(rowXMax,x);
                }
            }
            for (int x=rowXMin; x<=rowXMax; ++x) {
                int idx = info.linearIndex(x,y,z);
                if (!mask[idx]) { mask[idx] = 1; }
            }
        }
    }

    // 重建 result
    SegmentResult newResult;
    for (int z=zMin; z<=zMax; ++z)
    for (int y=yMin; y<=yMax; ++y)
    for (int x=xMin; x<=xMax; ++x) {
        if (mask[info.linearIndex(x,y,z)])
            newResult.push_back({x,y,z});
    }
    inout = std::move(newResult);
}

// ─────────────────────────────────────────────
//  主接口
// ─────────────────────────────────────────────
bool LungNoduleSegmentTraditional::segment(
    const short*      data,
    const ImageInfo&  info,
    const Point3i&    seedPoint,
    SegmentResult&    result,
    ProgressCallback  progress)
{
    if (!data || info.totalVoxels() == 0) return false;
    if (!info.inBounds(seedPoint.x, seedPoint.y, seedPoint.z)) return false;

    // Step1: 提取肺野
    std::vector<uint8_t> lungMask;
    if (!extractLungMask(data, info, lungMask)) return false;
    if (progress) progress(0.2);

    // 确定结节类型
    NoduleType type = m_params.noduleType;
    if (type == NoduleType::Auto)
        type = detectNoduleType(data, info, seedPoint, lungMask);

    // Step2: 初始分割
    SegmentResult coarse;
    if (!initialSegment(data, info, lungMask, seedPoint, coarse)) return false;
    if (progress) progress(0.5);

    // Step3: 精化
    SegmentResult refined;
    bool ok = false;
    switch (type) {
    case NoduleType::JuxtaWall:
        ok = refineJuxtaWall(data, info, lungMask, coarse, seedPoint, refined);
        break;
    case NoduleType::JuxtaVessel:
        ok = refineJuxtaVessel(data, info, coarse, seedPoint, refined);
        break;
    case NoduleType::GGO:
    case NoduleType::Solid:
    default:
        ok = refineWithHessian(data, info, coarse, seedPoint, refined);
        break;
    }
    if (!ok) refined = coarse; // 精化失败退回粗分割
    if (progress) progress(0.8);

    // Step4: 凸包平滑
    if (m_params.enableConvexHull)
        applyConvexHull(refined, info);

    result = std::move(refined);
    if (progress) progress(1.0);
    return !result.empty();
}

} // namespace Onc
