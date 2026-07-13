#pragma once
/**
 * ============================================================
 *  床板检测与校准 (Bed Board Detection & Calibration)
 * ============================================================
 *
 * 提炼自 McsfAlgoAutoContour 工程。
 *
 * 【背景】
 *   放疗 CT 图像中需要自动检测治疗床（couch）的位置，
 *   用于自动分割时排除床板区域、或在剂量计算中单独建模。
 *
 * 【床板检测 (DetectBedBoard)】
 *   1. 沿 Z 方向做 MIP（最大密度投影）生成 2D 投影
 *   2. 阈值分割获得候选床板区域（默认 -800 HU）
 *   3. 形态学净化（去噪）
 *   4. 按行统计床板行位置（仰卧: 底部；俯卧: 顶部）
 *
 * 【床板校准 (CalibrateBedBoard)】
 *   多级精确定位策略：
 *   1. 粗定位: MIP + 阈值获得初始位置
 *   2. 逐层追踪: 从已知好层开始向两端扩展
 *   3. 亚像素精确定位: 高斯拟合
 *   4. 全局平滑: 多项式拟合 y = c + bx + ax²
 *
 * 【参考】
 *   原工程: McsfAlgoBedBoardDetection.cpp, McsfAlgoBedBoardCalibration.cpp
 * ============================================================
 */

#include "rt_types.hpp"
#include "morphology.hpp"
#include "connected_component.hpp"

namespace rtac {

// ============================================================
//  检测参数
// ============================================================
struct BedBoardParams {
    short threshold = -800;     ///< 床板 HU 阈值
    bool  isSupine  = true;     ///< true=仰卧（床板在底部），false=俯卧
    int   polyOrder = 2;        ///< 拟合多项式阶数（默认二次）
};

// ============================================================
//  检测结果
// ============================================================
struct BedBoardResult {
    bool detected = false;
    int  colPosition = 0;             ///< 床板 X 中心列位置
    std::vector<double> rowPositions; ///< 每层的床板 Y 行位置
    double polyCoeffs[3] = {0,0,0};   ///< 多项式系数 y = c + bx + ax²
};

// ============================================================
//  辅助：计算 Z 方向 MIP
// ============================================================
namespace detail {

inline Image2D<short> ComputeZMIP(const Image3D<short>& ct) {
    int W = ct.width, H = ct.height, D = ct.depth;
    Image2D<short> mip(W, H, -32768);
    for (int z = 0; z < D; ++z)
    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
        mip.at(x, y) = std::max(mip.at(x, y), ct.at(x, y, z));
    }
    return mip;
}

/// 检测行方向突变位置（统计每行的前景像素数）
inline int DetectCutline(
    const Image2D<uint8_t>& mask,
    bool fromBottom)
{
    int W = mask.width, H = mask.height;
    std::vector<int> rowCount(H, 0);

    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
        if (mask.at(x, y)) ++rowCount[y];
    }

    // 找突变（前景像素数突然增大的位置）
    int threshold = W / 4; // 行前景像素 > 25% 宽度认为是床板
    if (fromBottom) {
        for (int y = H - 1; y >= 0; --y) {
            if (rowCount[y] >= threshold) return y;
        }
    } else {
        for (int y = 0; y < H; ++y) {
            if (rowCount[y] >= threshold) return y;
        }
    }
    return -1;
}

/// 检测列中心位置
inline int DetectColCenter(const Image2D<uint8_t>& mask, int bedRow) {
    int W = mask.width;
    int leftMost = W, rightMost = 0;

    // 在床板行附近几行统计列范围
    int rowStart = std::max(0, bedRow - 5);
    int rowEnd   = std::min(mask.height - 1, bedRow + 5);

    for (int y = rowStart; y <= rowEnd; ++y)
    for (int x = 0; x < W; ++x) {
        if (mask.at(x, y)) {
            leftMost  = std::min(leftMost, x);
            rightMost = std::max(rightMost, x);
        }
    }
    return (leftMost + rightMost) / 2;
}

/// 多项式拟合（最小二乘法）
/// 拟合 y = coeffs[0] + coeffs[1]*x + coeffs[2]*x^2
inline void PolyFit2(
    const std::vector<double>& xData,
    const std::vector<double>& yData,
    double coeffs[3])
{
    int n = static_cast<int>(xData.size());
    if (n < 3) {
        coeffs[0] = coeffs[1] = coeffs[2] = 0;
        if (n > 0) coeffs[0] = yData[0];
        return;
    }

    // 正规方程 A^T A x = A^T b
    // A = [1, xi, xi^2]
    double S[5] = {};  // S[k] = sum(xi^k)
    double T[3] = {};  // T[k] = sum(yi * xi^k)

    for (int i = 0; i < n; ++i) {
        double xi = xData[i], yi = yData[i];
        double xp = 1.0;
        for (int k = 0; k < 5; ++k) {
            S[k] += xp;
            if (k < 3) T[k] += yi * xp;
            xp *= xi;
        }
    }

    // 3x3 线性方程组 (Cramer 法则)
    // | S0 S1 S2 |   |c0|   |T0|
    // | S1 S2 S3 | × |c1| = |T1|
    // | S2 S3 S4 |   |c2|   |T2|
    double a[3][3] = {
        {S[0], S[1], S[2]},
        {S[1], S[2], S[3]},
        {S[2], S[3], S[4]}
    };
    double b[3] = {T[0], T[1], T[2]};

    // 高斯消元
    for (int col = 0; col < 3; ++col) {
        // 选主元
        int maxRow = col;
        for (int row = col + 1; row < 3; ++row)
            if (std::abs(a[row][col]) > std::abs(a[maxRow][col]))
                maxRow = row;
        std::swap(a[col], a[maxRow]);
        std::swap(b[col], b[maxRow]);

        if (std::abs(a[col][col]) < 1e-12) continue;

        for (int row = col + 1; row < 3; ++row) {
            double factor = a[row][col] / a[col][col];
            for (int k = col; k < 3; ++k)
                a[row][k] -= factor * a[col][k];
            b[row] -= factor * b[col];
        }
    }

    // 回代
    for (int i = 2; i >= 0; --i) {
        double sum = b[i];
        for (int j = i + 1; j < 3; ++j)
            sum -= a[i][j] * coeffs[j];
        coeffs[i] = (std::abs(a[i][i]) > 1e-12) ? sum / a[i][i] : 0.0;
    }
}

} // namespace detail

// ============================================================
//  检测床板位置
// ============================================================
/**
 * @param ct     CT 体数据 (short, HU 值)
 * @param params 检测参数
 * @return       检测结果
 */
inline BedBoardResult DetectBedBoard(
    const Image3D<short>& ct,
    const BedBoardParams& params = {})
{
    BedBoardResult result;
    int W = ct.width, H = ct.height, D = ct.depth;

    // ── 第1步: Z-MIP ──
    auto mip = detail::ComputeZMIP(ct);

    // ── 第2步: 阈值分割 ──
    Image2D<uint8_t> mipMask(W, H, 0);
    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
        if (mip.at(x, y) >= params.threshold)
            mipMask.at(x, y) = 255;
    }

    // ── 第3步: 形态学净化 ──
    Erosion2D(mipMask);
    Dilation2D(mipMask);

    // ── 第4步: 检测床板行位置 ──
    int bedRow = detail::DetectCutline(mipMask, params.isSupine);
    if (bedRow < 0) return result;

    result.detected = true;
    result.colPosition = detail::DetectColCenter(mipMask, bedRow);

    // ── 第5步: 逐层精确定位 ──
    result.rowPositions.resize(D);
    std::vector<double> xData, yData;

    for (int z = 0; z < D; ++z) {
        // 对每层做阈值分割并统计
        int bestRow = -1;
        int maxCount = 0;

        // 在粗定位附近搜索
        int searchStart = std::max(0, bedRow - 20);
        int searchEnd   = std::min(H - 1, bedRow + 20);

        for (int y = searchStart; y <= searchEnd; ++y) {
            int count = 0;
            int colStart = std::max(0, result.colPosition - W / 3);
            int colEnd   = std::min(W - 1, result.colPosition + W / 3);
            for (int x = colStart; x <= colEnd; ++x) {
                if (ct.at(x, y, z) >= params.threshold) ++count;
            }
            if (count > maxCount) {
                maxCount = count;
                bestRow = y;
            }
        }

        if (bestRow >= 0 && maxCount > W / 10) {
            result.rowPositions[z] = bestRow;
            xData.push_back(z);
            yData.push_back(bestRow);
        } else {
            result.rowPositions[z] = bedRow; // 用粗定位值
        }
    }

    // ── 第6步: 多项式拟合平滑 ──
    if (xData.size() >= 3) {
        detail::PolyFit2(xData, yData, result.polyCoeffs);

        // 用拟合结果平滑
        for (int z = 0; z < D; ++z) {
            double x = z;
            result.rowPositions[z] = result.polyCoeffs[0]
                                   + result.polyCoeffs[1] * x
                                   + result.polyCoeffs[2] * x * x;
        }
    }

    return result;
}

// ============================================================
//  生成床板掩膜
// ============================================================
/**
 * 根据检测结果生成 3D 床板掩膜。
 * 床板行位置以下（仰卧）或以上（俯卧）的区域标记为床板。
 */
inline Image3D<uint8_t> GenerateBedBoardMask(
    const BedBoardResult& result,
    int width, int height, int depth,
    bool isSupine = true,
    uint8_t maskID = 255)
{
    Image3D<uint8_t> mask(width, height, depth, 0);
    if (!result.detected) return mask;

    for (int z = 0; z < depth; ++z) {
        int bedRow = static_cast<int>(std::round(result.rowPositions[z]));
        for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            if (isSupine) {
                if (y >= bedRow) mask.at(x, y, z) = maskID;
            } else {
                if (y <= bedRow) mask.at(x, y, z) = maskID;
            }
        }
    }
    return mask;
}

} // namespace rtac
