#pragma once
/**
 * ============================================================
 *  Snake 活动轮廓 + GVF (Gradient Vector Flow)
 * ============================================================
 *
 * 【算法原理】
 *   Snake（参数化活动轮廓）将分割轮廓表示为一组有序控制点
 *   {v₁, v₂, ..., vₙ}，通过最小化能量泛函驱动轮廓向目标边缘运动。
 *
 *   总能量：
 *     E(v) = ∫[ α|v'(s)|² + β|v''(s)|² ] ds    // 内力：弹性 + 刚度
 *          + ∫ E_ext(v(s)) ds                      // 外力：图像驱动力
 *
 *   其中：
 *     α: 弹性系数，控制轮廓的伸展能力（越大越不容易拉长）
 *     β: 刚度系数，控制轮廓的弯曲程度（越大越光滑）
 *     E_ext: 外部能量，通常是图像边缘图的负值
 *
 * 【GVF (Gradient Vector Flow)】
 *   传统 Snake 的外力只在边缘附近有效，GVF 通过扩散边缘信息
 *   生成作用于整个图像域的向量场，解决捕获范围问题。
 *
 *   GVF 场 g=(u,v) 通过最小化以下能量得到：
 *     E = ∫∫ μ(u_x² + u_y² + v_x² + v_y²) + |∇f|²|g - ∇f|² dxdy
 *
 *   迭代更新：
 *     u ← u + μ·∇²u - |∇f|²·(u - f_x)
 *     v ← v + μ·∇²v - |∇f|²·(v - f_y)
 *
 *   其中 f 是边缘图（如 Canny 边缘或梯度幅值），μ 控制平滑程度。
 *
 * 【Snake 形变求解】
 *   将连续方程离散化后，每步求解线性系统：
 *     (A + γI)·x_new = γ·x_old + f_x(x_old, y_old)
 *     (A + γI)·y_new = γ·y_old + f_y(x_old, y_old)
 *
 *   其中 A 是由 α, β 构造的五对角环形矩阵（因为轮廓是闭合的）。
 *   γ 是步长参数。实际实现中预计算 (A + γI)⁻¹ 以加速。
 *
 * 【参考文献】
 *   Kass M, Witkin A, Terzopoulos D. "Snakes: Active contour models." IJCV, 1988.
 *   Xu C, Prince J. "Snakes, shapes, and gradient vector flow." IEEE TIP, 1998.
 * ============================================================
 */

#include "image2d.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace medseg {

// ============================================================
// 参数
// ============================================================
struct SnakeParams {
    float alpha    = 0.05f;   // 弹性系数（一阶导权重）
    float beta     = 0.1f;    // 刚度系数（二阶导权重）
    float gamma    = 1.0f;    // 时间步长
    int   iterations = 200;   // Snake 迭代次数
    int   gvfIterations = 80; // GVF 迭代次数
    float gvfMu    = 0.2f;    // GVF 平滑系数
};

// ============================================================
// 轮廓点
// ============================================================
struct ContourPoint {
    float x, y;
};

// ============================================================
// Snake 活动轮廓分割器
// ============================================================
class SnakeContour {
public:
    /**
     * 执行 Snake 分割
     * @param image       灰度图像 (float)
     * @param initContour 初始轮廓点序列（闭合，顺时针或逆时针）
     * @param params      算法参数
     * @return            演化后的轮廓点序列
     */
    std::vector<ContourPoint> segment(
        const Image2D<float>& image,
        const std::vector<ContourPoint>& initContour,
        const SnakeParams& params = {});

    /**
     * 从圆形初始化轮廓
     * @param cx, cy 圆心
     * @param radius 半径
     * @param nPoints 采样点数
     */
    static std::vector<ContourPoint> circleContour(
        float cx, float cy, float radius, int nPoints = 100);

    /**
     * 将轮廓转换为二值掩码
     */
    static Image2D<uint8_t> contourToMask(
        const std::vector<ContourPoint>& contour, int width, int height);

private:
    // 计算图像边缘图 f = |∇I|
    Image2D<float> computeEdgeMap(const Image2D<float>& image);

    // 计算 GVF 场
    void computeGVF(const Image2D<float>& edgeMap,
                    Image2D<float>& gvfU, Image2D<float>& gvfV,
                    float mu, int iterations);

    // 构造内力矩阵的逆 (A + γI)⁻¹
    // A 是环形五对角矩阵，由 α, β 参数决定
    std::vector<std::vector<float>> computeInverseMatrix(
        int n, float alpha, float beta, float gamma);

    // 双线性插值
    float bilinear(const Image2D<float>& img, float x, float y);
};

// ============================================================
//                      实现
// ============================================================

inline std::vector<ContourPoint> SnakeContour::circleContour(
    float cx, float cy, float radius, int nPoints)
{
    std::vector<ContourPoint> pts(nPoints);
    for (int i = 0; i < nPoints; ++i) {
        float theta = 2.0f * 3.14159265f * i / nPoints;
        pts[i] = { cx + radius * std::cos(theta),
                    cy + radius * std::sin(theta) };
    }
    return pts;
}

inline Image2D<uint8_t> SnakeContour::contourToMask(
    const std::vector<ContourPoint>& contour, int width, int height)
{
    Image2D<uint8_t> mask(width, height, 0);
    int n = static_cast<int>(contour.size());
    // 扫描线填充
    for (int y = 0; y < height; ++y) {
        std::vector<float> intersections;
        for (int i = 0; i < n; ++i) {
            int j = (i + 1) % n;
            float y0 = contour[i].y, y1 = contour[j].y;
            float x0 = contour[i].x, x1 = contour[j].x;
            if ((y0 <= y && y1 > y) || (y1 <= y && y0 > y)) {
                float t = (y - y0) / (y1 - y0);
                intersections.push_back(x0 + t * (x1 - x0));
            }
        }
        std::sort(intersections.begin(), intersections.end());
        for (size_t k = 0; k + 1 < intersections.size(); k += 2) {
            int x0 = std::max(0, static_cast<int>(std::ceil(intersections[k])));
            int x1 = std::min(width - 1, static_cast<int>(std::floor(intersections[k+1])));
            for (int x = x0; x <= x1; ++x) {
                mask.at(x, y) = 255;
            }
        }
    }
    return mask;
}

inline Image2D<float> SnakeContour::computeEdgeMap(const Image2D<float>& image) {
    int W = image.width, H = image.height;
    Image2D<float> edgeMap(W, H, 0.0f);

    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            float gx = (image.at(x+1, y) - image.at(x-1, y)) * 0.5f;
            float gy = (image.at(x, y+1) - image.at(x, y-1)) * 0.5f;
            edgeMap.at(x, y) = std::sqrt(gx * gx + gy * gy);
        }
    }

    // 归一化到 [0, 1]
    float maxE = *std::max_element(edgeMap.data.begin(), edgeMap.data.end());
    if (maxE > 1e-8f) {
        for (auto& v : edgeMap.data) v /= maxE;
    }
    return edgeMap;
}

inline void SnakeContour::computeGVF(
    const Image2D<float>& edgeMap,
    Image2D<float>& gvfU, Image2D<float>& gvfV,
    float mu, int iterations)
{
    int W = edgeMap.width, H = edgeMap.height;

    // 计算边缘图梯度
    Image2D<float> fx(W, H, 0.0f), fy(W, H, 0.0f);
    Image2D<float> gradSq(W, H, 0.0f);

    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            fx.at(x, y) = (edgeMap.at(x+1, y) - edgeMap.at(x-1, y)) * 0.5f;
            fy.at(x, y) = (edgeMap.at(x, y+1) - edgeMap.at(x, y-1)) * 0.5f;
            gradSq.at(x, y) = fx.at(x, y) * fx.at(x, y) + fy.at(x, y) * fy.at(x, y);
        }
    }

    // 初始化 GVF 场
    gvfU = fx;
    gvfV = fy;

    // 迭代扩散
    // u ← u + μ·∇²u - |∇f|²·(u - f_x)
    // v ← v + μ·∇²v - |∇f|²·(v - f_y)
    for (int iter = 0; iter < iterations; ++iter) {
        Image2D<float> uNew = gvfU;
        Image2D<float> vNew = gvfV;

        for (int y = 1; y < H - 1; ++y) {
            for (int x = 1; x < W - 1; ++x) {
                // 拉普拉斯算子 ∇²u ≈ u(x+1)+u(x-1)+u(y+1)+u(y-1) - 4u
                float lapU = gvfU.at(x+1, y) + gvfU.at(x-1, y)
                           + gvfU.at(x, y+1) + gvfU.at(x, y-1)
                           - 4.0f * gvfU.at(x, y);
                float lapV = gvfV.at(x+1, y) + gvfV.at(x-1, y)
                           + gvfV.at(x, y+1) + gvfV.at(x, y-1)
                           - 4.0f * gvfV.at(x, y);

                float b = gradSq.at(x, y);
                uNew.at(x, y) = gvfU.at(x, y) + mu * lapU
                               - b * (gvfU.at(x, y) - fx.at(x, y));
                vNew.at(x, y) = gvfV.at(x, y) + mu * lapV
                               - b * (gvfV.at(x, y) - fy.at(x, y));
            }
        }
        gvfU = uNew;
        gvfV = vNew;
    }
}

inline std::vector<std::vector<float>> SnakeContour::computeInverseMatrix(
    int n, float alpha, float beta, float gamma)
{
    /*
     * 内力矩阵 A 的对角元素（环形五对角带状矩阵）：
     *   对角:     2α + 6β
     *   ±1 对角:  -(α + 4β)
     *   ±2 对角:  β
     *
     * 实际需要 (A + γI)⁻¹ 用于隐式求解。
     * 这里用 LU 分解于稠密矩阵（点数通常 < 500，性能可接受）。
     */
    std::vector<std::vector<float>> M(n, std::vector<float>(n, 0.0f));

    float d0 = 2.0f * alpha + 6.0f * beta + gamma;
    float d1 = -(alpha + 4.0f * beta);
    float d2 = beta;

    for (int i = 0; i < n; ++i) {
        M[i][i] = d0;
        M[i][(i+1) % n] += d1;
        M[i][(i-1+n) % n] += d1;
        M[i][(i+2) % n] += d2;
        M[i][(i-2+n) % n] += d2;
    }

    // 高斯-约旦消元求逆
    std::vector<std::vector<float>> inv(n, std::vector<float>(n, 0.0f));
    for (int i = 0; i < n; ++i) inv[i][i] = 1.0f;

    for (int col = 0; col < n; ++col) {
        // 寻找主元
        int pivot = col;
        for (int row = col + 1; row < n; ++row) {
            if (std::abs(M[row][col]) > std::abs(M[pivot][col]))
                pivot = row;
        }
        std::swap(M[col], M[pivot]);
        std::swap(inv[col], inv[pivot]);

        float diagVal = M[col][col];
        if (std::abs(diagVal) < 1e-12f) continue;

        for (int j = 0; j < n; ++j) {
            M[col][j] /= diagVal;
            inv[col][j] /= diagVal;
        }

        for (int row = 0; row < n; ++row) {
            if (row == col) continue;
            float factor = M[row][col];
            for (int j = 0; j < n; ++j) {
                M[row][j] -= factor * M[col][j];
                inv[row][j] -= factor * inv[col][j];
            }
        }
    }

    return inv;
}

inline float SnakeContour::bilinear(const Image2D<float>& img, float x, float y) {
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    float dx = x - x0, dy = y - y0;

    float v00 = img.atClamped(x0, y0);
    float v10 = img.atClamped(x0+1, y0);
    float v01 = img.atClamped(x0, y0+1);
    float v11 = img.atClamped(x0+1, y0+1);

    return v00 * (1-dx) * (1-dy) + v10 * dx * (1-dy)
         + v01 * (1-dx) * dy     + v11 * dx * dy;
}

inline std::vector<ContourPoint> SnakeContour::segment(
    const Image2D<float>& image,
    const std::vector<ContourPoint>& initContour,
    const SnakeParams& params)
{
    int n = static_cast<int>(initContour.size());
    if (n < 3) return initContour;

    // 1. 计算边缘图
    auto edgeMap = computeEdgeMap(image);

    // 2. 计算 GVF 场
    Image2D<float> gvfU, gvfV;
    computeGVF(edgeMap, gvfU, gvfV, params.gvfMu, params.gvfIterations);

    // 3. 预计算内力矩阵的逆 (A + γI)⁻¹
    auto invM = computeInverseMatrix(n, params.alpha, params.beta, params.gamma);

    // 4. 初始化轮廓
    std::vector<float> cx(n), cy(n);
    for (int i = 0; i < n; ++i) {
        cx[i] = initContour[i].x;
        cy[i] = initContour[i].y;
    }

    // 5. 迭代演化
    for (int iter = 0; iter < params.iterations; ++iter) {
        // 计算外力（在当前轮廓点处插值 GVF 场）
        std::vector<float> fxVec(n), fyVec(n);
        for (int i = 0; i < n; ++i) {
            fxVec[i] = bilinear(gvfU, cx[i], cy[i]);
            fyVec[i] = bilinear(gvfV, cx[i], cy[i]);
        }

        // 右侧向量: γ·x_old + f_x
        std::vector<float> rhsX(n), rhsY(n);
        for (int i = 0; i < n; ++i) {
            rhsX[i] = params.gamma * cx[i] + fxVec[i];
            rhsY[i] = params.gamma * cy[i] + fyVec[i];
        }

        // 矩阵乘法: x_new = (A+γI)⁻¹ · rhs
        std::vector<float> newX(n, 0.0f), newY(n, 0.0f);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                newX[i] += invM[i][j] * rhsX[j];
                newY[i] += invM[i][j] * rhsY[j];
            }
        }

        // 限制在图像范围内
        for (int i = 0; i < n; ++i) {
            cx[i] = std::max(1.0f, std::min(static_cast<float>(image.width - 2), newX[i]));
            cy[i] = std::max(1.0f, std::min(static_cast<float>(image.height - 2), newY[i]));
        }
    }

    // 6. 输出结果
    std::vector<ContourPoint> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = { cx[i], cy[i] };
    }
    return result;
}

} // namespace medseg
