#pragma once
/**
 * @file hessian_filter.h
 * @brief Hessian 矩阵血管增强滤波器
 *
 * 基于 Frangi (1998) 血管增强滤波器：
 *   通过对图像在不同尺度 σ 下计算 Hessian 矩阵的特征值
 *   来增强管状/点状/板状结构：
 *
 *     V_o(s) = { 0                              if λ₂ > 0 or λ₃ > 0
 *               { (1-exp(-R_A²/2α²)) · exp(-R_B²/2β²) · (1-exp(-S²/2c²))
 *
 *   其中：
 *     R_A = |λ₂|/|λ₃|          — 截面形状比（板状 vs 管状）
 *     R_B = |λ₁|/√|λ₂·λ₃|     — 斑点抑制比
 *     S   = √(λ₁² + λ₂² + λ₃²) — Frobenius 范数（结构强度）
 *
 * 输出三种响应图：
 *   - hdot  : 点状增强（肺结节等圆形结构）
 *   - hline : 线状增强（血管等管状结构）
 *   - hplane: 板状增强
 */

#include "oncology_types.h"
#include <vector>

namespace Onc {

struct HessianFilterParams {
    std::vector<float> scales = {1.0f, 2.0f, 3.0f}; ///< Gaussian 尺度（mm）
    float alpha = 0.5f;  ///< 控制板状抑制
    float beta  = 0.5f;  ///< 控制斑点抑制
    float gamma = 500.f; ///< 结构强度阈值（归一化常数）
    bool  dark  = false; ///< true=增强暗管状（胆管等），false=增强亮管状（血管）
};

class HessianFilter {
public:
    explicit HessianFilter(const HessianFilterParams& p = {}) : m_params(p) {}

    /**
     * @brief 计算多尺度 Hessian 增强响应
     * @param data      输入图像（short，CT HU 值）
     * @param info      图像元信息
     * @param hdot      [out] 点状增强响应（与图像等尺寸）
     * @param hline     [out] 线状增强响应
     * @param hplane    [out] 板状增强响应
     */
    void compute(const short*        data,
                 const ImageInfo&    info,
                 std::vector<float>& hdot,
                 std::vector<float>& hline,
                 std::vector<float>& hplane);

private:
    HessianFilterParams m_params;

    // 计算单尺度 Hessian 特征值
    void computeEigenvalues(const std::vector<float>& smooth,
                            const ImageInfo&          info,
                            float                     sigma,
                            std::vector<float>&       ev1,   // |smallest|
                            std::vector<float>&       ev2,
                            std::vector<float>&       ev3);  // |largest|

    // Gaussian 平滑（分离卷积）
    void gaussianSmooth(const short*          input,
                        const ImageInfo&      info,
                        float                 sigma,
                        std::vector<float>&   output);

    // 1D Gaussian 核
    std::vector<float> makeGaussianKernel(float sigma, int& halfW);

    // 1D 卷积
    void convolve1D(const std::vector<float>& src,
                    const ImageInfo&          info,
                    const std::vector<float>& kernel,
                    int                       halfW,
                    int                       axis,
                    std::vector<float>&       dst);
};

} // namespace Onc
