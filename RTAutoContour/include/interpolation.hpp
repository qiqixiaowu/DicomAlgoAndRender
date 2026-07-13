#pragma once
/**
 * ============================================================
 *  体数据插值重采样 (Volume Interpolation & Resampling)
 * ============================================================
 *
 * 提炼自 McsfAlgoAutoContour 工程的 McsfAlgoInterpolation。
 *
 * 【三线性插值 (Trilinear Interpolation)】
 *   在 3D 体数据中，给定连续坐标 (xf, yf, zf)，通过
 *   8 个最近体素的加权插值获得亚体素精度的灰度值。
 *
 *   权重 = (1-dx)*(1-dy)*(1-dz) * V[x0,y0,z0]
 *        + dx*(1-dy)*(1-dz) * V[x1,y0,z0]
 *        + ...（共 8 项）
 *
 * 【最近邻插值 (Nearest Neighbor)】
 *   直接四舍五入到最近体素。
 *
 * 【重采样】
 *   按目标 spacing 或目标 size 将体数据重采样到新分辨率。
 *
 * 【参考】
 *   原工程: McsfAlgoAutoContourInterpolation.cpp
 * ============================================================
 */

#include "rt_types.hpp"

namespace rtac {

// ============================================================
//  三线性插值（单点查询）
// ============================================================
/**
 * 在 3D 体数据中进行三线性插值采样。
 *
 * @param volume    体数据
 * @param xf,yf,zf  连续坐标（体素单位）
 * @param background 越界时的默认值
 * @return           插值后的灰度值
 */
template<typename T>
inline float TrilinearSample(
    const Image3D<T>& volume,
    float xf, float yf, float zf,
    float background = 0.f)
{
    int W = volume.width, H = volume.height, D = volume.depth;

    int x0 = static_cast<int>(std::floor(xf));
    int y0 = static_cast<int>(std::floor(yf));
    int z0 = static_cast<int>(std::floor(zf));
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;

    // 越界检查
    if (x1 < 0 || x0 >= W || y1 < 0 || y0 >= H || z1 < 0 || z0 >= D)
        return background;

    float dx = xf - x0;
    float dy = yf - y0;
    float dz = zf - z0;

    // 安全采样（clamp）
    auto safeGet = [&](int x, int y, int z) -> float {
        x = std::max(0, std::min(x, W - 1));
        y = std::max(0, std::min(y, H - 1));
        z = std::max(0, std::min(z, D - 1));
        return static_cast<float>(volume.at(x, y, z));
    };

    // 8 个顶点
    float c000 = safeGet(x0, y0, z0);
    float c100 = safeGet(x1, y0, z0);
    float c010 = safeGet(x0, y1, z0);
    float c110 = safeGet(x1, y1, z0);
    float c001 = safeGet(x0, y0, z1);
    float c101 = safeGet(x1, y0, z1);
    float c011 = safeGet(x0, y1, z1);
    float c111 = safeGet(x1, y1, z1);

    // 三线性插值
    float c00 = c000 * (1 - dx) + c100 * dx;
    float c01 = c001 * (1 - dx) + c101 * dx;
    float c10 = c010 * (1 - dx) + c110 * dx;
    float c11 = c011 * (1 - dx) + c111 * dx;

    float c0 = c00 * (1 - dy) + c10 * dy;
    float c1 = c01 * (1 - dy) + c11 * dy;

    return c0 * (1 - dz) + c1 * dz;
}

// ============================================================
//  最近邻插值（单点查询）
// ============================================================
template<typename T>
inline T NearestSample(
    const Image3D<T>& volume,
    float xf, float yf, float zf,
    T background = T(0))
{
    int x = static_cast<int>(std::round(xf));
    int y = static_cast<int>(std::round(yf));
    int z = static_cast<int>(std::round(zf));
    if (!volume.inBounds(x, y, z)) return background;
    return volume.at(x, y, z);
}

// ============================================================
//  重采样模式
// ============================================================
enum class InterpolationMode {
    NearestNeighbor,
    Trilinear
};

// ============================================================
//  按目标尺寸重采样
// ============================================================
/**
 * 将体数据重采样到目标尺寸。
 *
 * @param input   输入体数据
 * @param targetW/H/D 目标尺寸
 * @param mode    插值模式
 * @return        重采样后的体数据
 */
template<typename T>
inline Image3D<float> ResampleToSize(
    const Image3D<T>& input,
    int targetW, int targetH, int targetD,
    InterpolationMode mode = InterpolationMode::Trilinear)
{
    Image3D<float> output(targetW, targetH, targetD, 0.f);

    // 计算缩放比
    float scaleX = static_cast<float>(input.width)  / targetW;
    float scaleY = static_cast<float>(input.height) / targetH;
    float scaleZ = static_cast<float>(input.depth)  / targetD;

    // 更新 spacing
    output.spacingX = input.spacingX * scaleX;
    output.spacingY = input.spacingY * scaleY;
    output.spacingZ = input.spacingZ * scaleZ;

    for (int z = 0; z < targetD; ++z)
    for (int y = 0; y < targetH; ++y)
    for (int x = 0; x < targetW; ++x) {
        float srcX = (x + 0.5f) * scaleX - 0.5f;
        float srcY = (y + 0.5f) * scaleY - 0.5f;
        float srcZ = (z + 0.5f) * scaleZ - 0.5f;

        if (mode == InterpolationMode::Trilinear)
            output.at(x, y, z) = TrilinearSample(input, srcX, srcY, srcZ);
        else
            output.at(x, y, z) = static_cast<float>(NearestSample(input, srcX, srcY, srcZ));
    }
    return output;
}

// ============================================================
//  按目标 spacing 重采样
// ============================================================
/**
 * 将体数据重采样到指定的体素间距。
 */
template<typename T>
inline Image3D<float> ResampleToSpacing(
    const Image3D<T>& input,
    float targetSpX, float targetSpY, float targetSpZ,
    InterpolationMode mode = InterpolationMode::Trilinear)
{
    float physW = input.width  * input.spacingX;
    float physH = input.height * input.spacingY;
    float physD = input.depth  * input.spacingZ;

    int targetW = std::max(1, static_cast<int>(std::round(physW / targetSpX)));
    int targetH = std::max(1, static_cast<int>(std::round(physH / targetSpY)));
    int targetD = std::max(1, static_cast<int>(std::round(physD / targetSpZ)));

    auto output = ResampleToSize(input, targetW, targetH, targetD, mode);
    output.spacingX = targetSpX;
    output.spacingY = targetSpY;
    output.spacingZ = targetSpZ;
    return output;
}

// ============================================================
//  等间距重采样（仅 Z 轴）
// ============================================================
/**
 * 将非等间距 Z 轴重采样为等间距。
 * 输入提供每层 Z 位置，输出按均匀 spacing 采样。
 *
 * @param input       原始体数据
 * @param zPositions  每层的 Z 物理位置（长度 = input.depth）
 * @param targetSpZ   目标 Z 间距
 * @param mode        插值模式
 */
template<typename T>
inline Image3D<float> ResampleZUniform(
    const Image3D<T>& input,
    const std::vector<float>& zPositions,
    float targetSpZ,
    InterpolationMode mode = InterpolationMode::Trilinear)
{
    if (zPositions.size() != static_cast<size_t>(input.depth))
        throw std::runtime_error("zPositions size != input depth");

    float zMin = *std::min_element(zPositions.begin(), zPositions.end());
    float zMax = *std::max_element(zPositions.begin(), zPositions.end());
    int targetD = std::max(1, static_cast<int>(std::ceil((zMax - zMin) / targetSpZ)));

    Image3D<float> output(input.width, input.height, targetD, 0.f);
    output.spacingX = input.spacingX;
    output.spacingY = input.spacingY;
    output.spacingZ = targetSpZ;

    for (int z = 0; z < targetD; ++z) {
        float targetZ = zMin + z * targetSpZ;

        // 找 targetZ 在 zPositions 中的位置
        // 二分查找
        int lo = 0, hi = input.depth - 1;
        while (lo < hi - 1) {
            int mid = (lo + hi) / 2;
            if (zPositions[mid] <= targetZ) lo = mid;
            else hi = mid;
        }

        float t = 0.f;
        if (std::abs(zPositions[hi] - zPositions[lo]) > 1e-6f)
            t = (targetZ - zPositions[lo]) / (zPositions[hi] - zPositions[lo]);
        float srcZ = lo + t;

        for (int y = 0; y < input.height; ++y)
        for (int x = 0; x < input.width; ++x) {
            if (mode == InterpolationMode::Trilinear)
                output.at(x, y, z) = TrilinearSample(input, static_cast<float>(x),
                                                      static_cast<float>(y), srcZ);
            else
                output.at(x, y, z) = static_cast<float>(
                    NearestSample(input, static_cast<float>(x),
                                  static_cast<float>(y), srcZ));
        }
    }
    return output;
}

} // namespace rtac
