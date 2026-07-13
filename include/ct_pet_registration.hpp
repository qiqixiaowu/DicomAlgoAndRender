#pragma once
/**
 * @file    ct_pet_registration.hpp
 * @brief   CT-PET 刚体配准算法库（Header-Only，无外部依赖）
 *
 * 忠实实现 CT_PET刚体配准详细文档.md 中描述的完整管线：
 *  §1  数据结构：VolumeData, VersorTransform, RegConfig
 *  §2  B-spline 核函数：β², β³, β³'（Parzen 窗口）
 *  §3  插值与梯度：三线性插值、体数据缩放
 *  §4  Versor 变换：四元数↔旋转矩阵、点变换、四元数乘法
 *  §5  NMI 度量：Parzen 窗口联合直方图 + 解析梯度（Mattes MI）
 *  §6  优化器：正则化梯度下降 + Versor 四元数乘法更新
 *  §7  预处理：CT 去床板+HU裁剪；PET 自适应百分位裁剪
 *  §8  多分辨率配准：粗→细金字塔
 *  §9  重采样
 *  §10 Phantom 生成：多模态 CT/PET 头部体模
 *
 * 对应源码：
 *  algo-fusion-app  → FusionRegistrationCT_PET
 *  algo-registration → CMutualInformationMetric, coptimization
 *  algo-registrationgpu → ImageRegistrationRigidG
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace CTPETReg {

// ============================================================
// §1  数据结构
// ============================================================

/** @brief 3D 体数据（float，存储顺序 Z-slice major: idx = z*W*H + y*W + x） */
struct VolumeData {
    std::vector<float> data;
    int width = 0, height = 0, depth = 0;
    float spacingX = 1.0f, spacingY = 1.0f, spacingZ = 1.0f;

    int  size()  const { return width * height * depth; }
    bool empty() const { return data.empty(); }
    int  idx(int x, int y, int z) const { return z * width * height + y * width + x; }
    float  at(int x, int y, int z) const { return data[idx(x, y, z)]; }
    float& at(int x, int y, int z)       { return data[idx(x, y, z)]; }

    void resize(int w, int h, int d) {
        width = w; height = h; depth = d;
        data.assign((size_t)w * h * d, 0.0f);
    }

    /** 获取第 z 层轴位切片 */
    std::vector<float> getAxialSlice(int z) const {
        z = std::clamp(z, 0, depth - 1);
        std::vector<float> s(width * height);
        std::copy(data.data() + (size_t)z * width * height,
                  data.data() + (size_t)(z + 1) * width * height, s.begin());
        return s;
    }

    /** 归一化到 [0,1] */
    void normalize() {
        if (data.empty()) return;
        float mn = *std::min_element(data.begin(), data.end());
        float mx = *std::max_element(data.begin(), data.end());
        float rng = mx - mn + 1e-8f;
        for (auto& v : data) v = (v - mn) / rng;
    }
};

/**
 * @brief Versor（四元数）三维刚体变换
 *
 * 存储 (qx, qy, qz)，qw = sqrt(1 - qx²-qy²-qz²)
 * 对应源码 PARAMETER_t 中的旋转+平移参数
 */
struct VersorTransform {
    float qx = 0.0f, qy = 0.0f, qz = 0.0f;  ///< 四元数虚部
    float tx = 0.0f, ty = 0.0f, tz = 0.0f;  ///< 平移（体素单位）

    static VersorTransform identity() { return {}; }

    /** 从 ZYX 欧拉角构造 */
    static VersorTransform fromEuler(float rx, float ry, float rz,
                                      float tx, float ty, float tz) {
        // 先构建旋转矩阵
        float cx = std::cos(rx), sx = std::sin(rx);
        float cy = std::cos(ry), sy = std::sin(ry);
        float cz = std::cos(rz), sz = std::sin(rz);
        float R[3][3] = {
            {cy*cz,           cz*sx*sy - cx*sz, cx*cz*sy + sx*sz},
            {cy*sz,           cx*cz + sx*sy*sz, cx*sy*sz - cz*sx},
            {-sy,             cy*sx,             cx*cy}
        };
        // 旋转矩阵 → 四元数
        float tr = R[0][0] + R[1][1] + R[2][2];
        VersorTransform t; t.tx = tx; t.ty = ty; t.tz = tz;
        if (tr > 0) {
            float S = 2.0f * std::sqrt(tr + 1.0f);
            t.qx = (R[2][1] - R[1][2]) / S;
            t.qy = (R[0][2] - R[2][0]) / S;
            t.qz = (R[1][0] - R[0][1]) / S;
        } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
            float S = 2.0f * std::sqrt(1.0f + R[0][0] - R[1][1] - R[2][2]);
            t.qx = 0.25f * S;
            t.qy = (R[0][1] + R[1][0]) / S;
            t.qz = (R[0][2] + R[2][0]) / S;
        } else if (R[1][1] > R[2][2]) {
            float S = 2.0f * std::sqrt(1.0f + R[1][1] - R[0][0] - R[2][2]);
            t.qx = (R[0][1] + R[1][0]) / S;
            t.qy = 0.25f * S;
            t.qz = (R[1][2] + R[2][1]) / S;
        } else {
            float S = 2.0f * std::sqrt(1.0f + R[2][2] - R[0][0] - R[1][1]);
            t.qx = (R[0][2] + R[2][0]) / S;
            t.qy = (R[1][2] + R[2][1]) / S;
            t.qz = 0.25f * S;
        }
        return t;
    }

    /** 转换为欧拉角（用于显示/比较） */
    void toEuler(float& rx, float& ry, float& rz) const {
        float qw = std::sqrt(std::max(0.0f, 1.0f - qx*qx - qy*qy - qz*qz));
        // 四元数 → 旋转矩阵
        float R00 = 1 - 2*(qy*qy + qz*qz);
        float R10 = 2*(qx*qy + qw*qz);
        float R20 = 2*(qx*qz - qw*qy);
        float R21 = 2*(qy*qz + qw*qx);
        float R22 = 1 - 2*(qx*qx + qy*qy);
        ry = std::asin(std::clamp(-R20, -1.0f, 1.0f));
        rx = std::atan2(R21, R22);
        rz = std::atan2(R10, R00);
    }
};

/** @brief 配准参数配置（对应源码 PARAMETER_t） */
struct RegConfig {
    // 度量参数
    int   numBins          = 32;       ///< 直方图 bin 数（源码 50~80）
    int   numSamples       = 3000;     ///< 采样点数
    // 优化器参数
    int   maxIterations    = 200;      ///< 最大迭代次数
    float maxStep          = 3.0f;     ///< 最大步长
    float minStep          = 0.01f;    ///< 最小步长（终止条件）
    float relaxFactor      = 0.9f;     ///< 松弛因子（方向反转时缩小步长）
    float magnitudeTol     = 0.001f;   ///< 梯度模长容忍度
    // 参数缩放（源码 Init(3, 1000, 0.01)）
    float transScale       = 0.01f;    ///< 平移缩放
    float rotScale         = 1000.0f;  ///< 旋转缩放
    // 多分辨率
    int   numLevels        = 2;        ///< 金字塔层数
    // 先验矩阵
    bool  useCenterAlign   = true;     ///< 使用中心对齐先验
};

// ============================================================
// §2  B-spline 核函数（Parzen 窗口）
// ============================================================

/** @brief 二阶 B-spline β²(x)
 * 对应源码 Evaluate(2, x)
 */
inline float bspline2(float x) {
    float ax = std::abs(x);
    if (ax < 0.5f)      return 0.75f - x * x;
    else if (ax < 1.5f) { float t = 1.5f - ax; return 0.5f * t * t; }
    else                 return 0.0f;
}

/** @brief 三阶 B-spline β³(x)
 * 对应源码 Evaluate(3, x)
 * - 参考图像用零阶（盒函数），直接计数
 * - 浮动图像用三阶 B-spline 平滑
 */
inline float bspline3(float x) {
    float ax = std::abs(x);
    if (ax < 1.0f)      return (2.0f/3.0f) - ax*ax + 0.5f*ax*ax*ax;
    else if (ax < 2.0f) { float t = 2.0f - ax; return t*t*t / 6.0f; }
    else                 return 0.0f;
}

/** @brief 三阶 B-spline 的导数 β³'(x) = β²(x+0.5) - β²(x-0.5)
 * 对应源码: sd = Evaluate(2, arg+0.5) - Evaluate(2, arg-0.5)
 */
inline float bspline3_deriv(float x) {
    return bspline2(x + 0.5f) - bspline2(x - 0.5f);
}

// ============================================================
// §3  插值与梯度
// ============================================================

/** @brief 三线性插值（越界返回 0）
 * 对应源码 ReSampleLinear3D
 */
inline float trilinear(const VolumeData& v, float x, float y, float z) {
    if (x < 0 || x > v.width - 1.001f ||
        y < 0 || y > v.height - 1.001f ||
        z < 0 || z > v.depth - 1.001f) return 0.0f;

    int xi = std::min((int)x, v.width - 2);
    int yi = std::min((int)y, v.height - 2);
    int zi = std::min((int)z, v.depth - 2);
    float fx = x - xi, fy = y - yi, fz = z - zi;

    float v000 = v.at(xi,   yi,   zi  ), v100 = v.at(xi+1, yi,   zi  );
    float v010 = v.at(xi,   yi+1, zi  ), v110 = v.at(xi+1, yi+1, zi  );
    float v001 = v.at(xi,   yi,   zi+1), v101 = v.at(xi+1, yi,   zi+1);
    float v011 = v.at(xi,   yi+1, zi+1), v111 = v.at(xi+1, yi+1, zi+1);

    return ((v000*(1-fx)+v100*fx)*(1-fy) + (v010*(1-fx)+v110*fx)*fy) * (1-fz)
         + ((v001*(1-fx)+v101*fx)*(1-fy) + (v011*(1-fx)+v111*fx)*fy) *    fz;
}

/** @brief 体数据缩放（三线性插值） */
inline VolumeData resizeVolume(const VolumeData& src, int dW, int dH, int dD) {
    VolumeData dst; dst.width = dW; dst.height = dH; dst.depth = dD;
    dst.spacingX = src.spacingX * src.width / dW;
    dst.spacingY = src.spacingY * src.height / dH;
    dst.spacingZ = src.spacingZ * src.depth / dD;
    dst.data.resize((size_t)dW * dH * dD);
    float sx = (float)src.width / dW, sy = (float)src.height / dH, sz = (float)src.depth / dD;
    for (int z = 0; z < dD; z++)
        for (int y = 0; y < dH; y++)
            for (int x = 0; x < dW; x++)
                dst.data[z * dW * dH + y * dW + x] = trilinear(src, x * sx, y * sy, z * sz);
    return dst;
}

// ============================================================
// §4  Versor 变换
// ============================================================

/** @brief 四元数 → 3×3 旋转矩阵
 * R = | 1-2(qy²+qz²)   2(qx·qy-qw·qz)   2(qx·qz+qw·qy) |
 *     | 2(qx·qy+qw·qz)   1-2(qx²+qz²)   2(qy·qz-qw·qx) |
 *     | 2(qx·qz-qw·qy)   2(qy·qz+qw·qx)   1-2(qx²+qy²) |
 */
inline void quatToMatrix(float qx, float qy, float qz, float R[3][3]) {
    float qx2 = qx*qx, qy2 = qy*qy, qz2 = qz*qz;
    float qw = std::sqrt(std::max(0.0f, 1.0f - qx2 - qy2 - qz2));
    R[0][0] = 1 - 2*(qy2 + qz2);  R[0][1] = 2*(qx*qy - qw*qz);  R[0][2] = 2*(qx*qz + qw*qy);
    R[1][0] = 2*(qx*qy + qw*qz);  R[1][1] = 1 - 2*(qx2 + qz2);  R[1][2] = 2*(qy*qz - qw*qx);
    R[2][0] = 2*(qx*qz - qw*qy);  R[2][1] = 2*(qy*qz + qw*qx);  R[2][2] = 1 - 2*(qx2 + qy2);
}

/** @brief Versor 变换一个点
 * p' = R × (p - center) + center + t
 * 对应源码 TransformPoint
 */
inline void transformPoint(const VersorTransform& t,
                            float cx, float cy, float cz,
                            float x, float y, float z,
                            float& ox, float& oy, float& oz) {
    float R[3][3]; quatToMatrix(t.qx, t.qy, t.qz, R);
    float dx = x - cx, dy = y - cy, dz = z - cz;
    ox = cx + t.tx + R[0][0]*dx + R[0][1]*dy + R[0][2]*dz;
    oy = cy + t.ty + R[1][0]*dx + R[1][1]*dy + R[1][2]*dz;
    oz = cz + t.tz + R[2][0]*dx + R[2][1]*dy + R[2][2]*dz;
}

/** @brief 四元数乘法 q_out = q_a × q_b
 * 对应源码 Optimization.cpp 中的 Versor 更新
 */
inline void quatMultiply(float aw, float ax, float ay, float az,
                          float bw, float bx, float by, float bz,
                          float& ow, float& ox, float& oy, float& oz) {
    ow = aw*bw - ax*bx - ay*by - az*bz;
    ox = aw*bx + ax*bw + ay*bz - az*by;
    oy = aw*by - ax*bz + ay*bw + az*bx;
    oz = aw*bz + ax*by - ay*bx + az*bw;
}

// ============================================================
// §5  NMI 度量（Parzen 窗口 + 解析梯度）
// ============================================================

/**
 * @brief NMI 采样点
 * 对应源码 SampleFixedImageDomain 的输出
 */
struct NMISample {
    float px, py, pz;   ///< 参考图像中的位置
    float refValue;      ///< 参考图像灰度值
    int   refBin;        ///< 预计算的参考 bin 索引（零阶 B-spline）
};

/** @brief NMI 评估结果 */
struct NMIResult {
    float value;          ///< NMI 值（越大越好）
    float gradient[6];    ///< 梯度 [tx, ty, tz, θx, θy, θz]（已缩放）
};

/**
 * @brief 归一化互信息度量（Mattes MI 实现）
 *
 * 算法：
 *  1. 采样参考图像域（随机采样）
 *  2. 预计算参考图像 bin 索引（零阶 B-spline）
 *  3. 预计算浮动图像梯度场（中心差分）
 *  4. 每次迭代：
 *     a. 变换采样点 → 插值浮动图像值
 *     b. 用三阶 B-spline 构建联合直方图
 *     c. 计算 NMI = Σ p(i,j)·log(p(i,j)/(p_A(i)·p_B(j)))
 *     d. 解析梯度：链式法则 d(MI)/d(T) = Σ d(p)/d(T)·log(p/p_B)
 *
 * 对应源码：CMutualInformationMetric::GetValueAndDerivative
 */
class NMIMetric {
public:
    /// 初始化：采样、预计算梯度场
    void initialize(const VolumeData& ref, const VolumeData& mov, const RegConfig& cfg) {
        numBins_    = cfg.numBins;
        transScale_ = cfg.transScale;
        rotScale_   = cfg.rotScale;

        // 计算灰度范围
        refMin_ = *std::min_element(ref.data.begin(), ref.data.end());
        refMax_ = *std::max_element(ref.data.begin(), ref.data.end());
        movMin_ = *std::min_element(mov.data.begin(), mov.data.end());
        movMax_ = *std::max_element(mov.data.begin(), mov.data.end());
        refBinSize_ = (refMax_ - refMin_) / (numBins_ - 1) + 1e-8f;
        movBinSize_ = (movMax_ - movMin_) / (numBins_ - 1) + 1e-8f;

        // 随机采样参考图像
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dx(1, ref.width - 2);
        std::uniform_int_distribution<int> dy(1, ref.height - 2);
        std::uniform_int_distribution<int> dz(1, ref.depth - 2);

        samples_.clear();
        samples_.reserve(cfg.numSamples);
        for (int i = 0; i < cfg.numSamples; i++) {
            NMISample s;
            s.px = (float)dx(rng);
            s.py = (float)dy(rng);
            s.pz = (float)dz(rng);
            s.refValue = trilinear(ref, s.px, s.py, s.pz);
            s.refBin = std::clamp((int)((s.refValue - refMin_) / refBinSize_), 0, numBins_ - 1);
            samples_.push_back(s);
        }

        // 预计算浮动图像梯度场（中心差分）
        movGradX_.resize(mov.width, mov.height, mov.depth);
        movGradY_.resize(mov.width, mov.height, mov.depth);
        movGradZ_.resize(mov.width, mov.height, mov.depth);
        for (int z = 1; z < mov.depth - 1; z++)
            for (int y = 1; y < mov.height - 1; y++)
                for (int x = 1; x < mov.width - 1; x++) {
                    movGradX_.at(x, y, z) = 0.5f * (mov.at(x+1, y, z) - mov.at(x-1, y, z));
                    movGradY_.at(x, y, z) = 0.5f * (mov.at(x, y+1, z) - mov.at(x, y-1, z));
                    movGradZ_.at(x, y, z) = 0.5f * (mov.at(x, y, z+1) - mov.at(x, y, z-1));
                }
    }

    /**
     * @brief 计算 NMI 值和梯度
     *
     * 两遍扫描：
     *   Pass 1: 变换采样点 → 构建联合 PDF + 边缘 PDF
     *   Pass 2: 用预计算 PDF 和链式法则计算梯度
     */
    NMIResult evaluate(const VolumeData& mov, const VersorTransform& t,
                        float cx, float cy, float cz) const {
        int B = numBins_;
        std::vector<float> jointPDF((size_t)B * B, 0.0f);
        std::vector<float> refPDF(B, 0.0f);
        std::vector<float> movPDF(B, 0.0f);

        // 缓存每个采样点的中间结果（避免 Pass 2 重复计算）
        struct Cache {
            int refBin;
            float movVal;
            int movBin;
            float movTerm;
            float mgx, mgy, mgz;  // 浮动图像梯度
            float vx, vy, vz;     // R×(p-center)，用于旋转 Jacobian
            bool valid;
        };
        std::vector<Cache> cache(samples_.size());

        float R[3][3]; quatToMatrix(t.qx, t.qy, t.qz, R);

        // ---- Pass 1: 构建 PDF ----
        int validCount = 0;
        for (int i = 0; i < (int)samples_.size(); i++) {
            const auto& s = samples_[i];
            auto& c = cache[i];
            c.refBin = s.refBin;
            c.valid = false;

            // 变换点
            float ox, oy, oz;
            transformPoint(t, cx, cy, cz, s.px, s.py, s.pz, ox, oy, oz);

            // 边界检查
            if (ox < 1 || ox > mov.width - 2.001f ||
                oy < 1 || oy > mov.height - 2.001f ||
                oz < 1 || oz > mov.depth - 2.001f) continue;

            // 插值浮动图像值
            c.movVal = trilinear(mov, ox, oy, oz);

            // Parzen 窗口索引（三阶 B-spline）
            float mNorm = (c.movVal - movMin_) / movBinSize_ - 0.5f;
            c.movBin = (int)std::floor(mNorm);
            c.movTerm = mNorm - c.movBin;
            if (c.movBin < 0 || c.movBin >= B) continue;

            // 累积联合 PDF（参考=零阶, 浮动=三阶 B-spline）
            for (int k = -1; k <= 2; k++) {
                int j = c.movBin + k;
                if (j < 0 || j >= B) continue;
                float w = bspline3((float)k - c.movTerm);
                jointPDF[(size_t)s.refBin * B + j] += w;
                movPDF[j] += w;
            }
            refPDF[s.refBin] += 1.0f;

            // 插值浮动图像梯度
            c.mgx = trilinear(movGradX_, ox, oy, oz);
            c.mgy = trilinear(movGradY_, ox, oy, oz);
            c.mgz = trilinear(movGradZ_, ox, oy, oz);

            // 旋转后的向量 v = R × (p - center)
            float dx = s.px - cx, dy = s.py - cy, dz = s.pz - cz;
            c.vx = R[0][0]*dx + R[0][1]*dy + R[0][2]*dz;
            c.vy = R[1][0]*dx + R[1][1]*dy + R[1][2]*dz;
            c.vz = R[2][0]*dx + R[2][1]*dy + R[2][2]*dz;

            c.valid = true;
            validCount++;
        }

        if (validCount == 0) {
            NMIResult r; r.value = 0; std::fill(r.gradient, r.gradient+6, 0.0f);
            return r;
        }

        // 归一化
        float invN = 1.0f / validCount;
        for (auto& v : jointPDF) v *= invN;
        for (auto& v : refPDF) v *= invN;
        for (auto& v : movPDF) v *= invN;

        // 计算 NMI 值: MI = Σ p(i,j)·log(p(i,j)/(p_A(i)·p_B(j)))
        float nmi = 0.0f;
        for (int i = 0; i < B; i++) {
            if (refPDF[i] < 1e-12f) continue;
            for (int j = 0; j < B; j++) {
                float p = jointPDF[(size_t)i * B + j];
                if (p < 1e-12f || movPDF[j] < 1e-12f) continue;
                nmi += p * std::log(p / (refPDF[i] * movPDF[j]));
            }
        }

        // 预计算 log 比率: logRatio[i][j] = log(p(i,j) / p_B(j))
        std::vector<float> logRatio((size_t)B * B, 0.0f);
        for (int i = 0; i < B; i++)
            for (int j = 0; j < B; j++) {
                float p = jointPDF[(size_t)i * B + j];
                if (p > 1e-12f && movPDF[j] > 1e-12f)
                    logRatio[(size_t)i * B + j] = std::log(p / movPDF[j]);
            }

        // ---- Pass 2: 计算梯度 ----
        // d(-MI)/d(T_k) = (1/N)·Σ_s Σ_k B3'(k-term)/binSize · d(m)/d(T_k) · logRatio
        NMIResult result;
        result.value = nmi;
        std::fill(result.gradient, result.gradient + 6, 0.0f);

        for (int i = 0; i < (int)samples_.size(); i++) {
            const auto& c = cache[i];
            if (!c.valid) continue;

            // d(mov_value)/d(T_k) = ∇I · d(p')/d(T_k)
            // 平移: d(p')/d(t) = e_k  →  d(m)/d(t_k) = ∇I_k
            // 旋转: d(p')/d(θ_k) = e_k × v  →  d(m)/d(θ_k) = ∇I × v
            float dM[6];
            dM[0] = c.mgx;                                          // tx
            dM[1] = c.mgy;                                          // ty
            dM[2] = c.mgz;                                          // tz
            dM[3] = c.mgy * (-c.vz) + c.mgz * c.vy;                 // θx
            dM[4] = c.mgx * c.vz    - c.mgz * c.vx;                 // θy
            dM[5] = -c.mgx * c.vy   + c.mgy * c.vx;                 // θz

            for (int k = -1; k <= 2; k++) {
                int j = c.movBin + k;
                if (j < 0 || j >= B) continue;
                float bDeriv = bspline3_deriv((float)k - c.movTerm) / movBinSize_;
                float lr = logRatio[(size_t)c.refBin * B + j];
                for (int p = 0; p < 6; p++)
                    result.gradient[p] += bDeriv * dM[p] * lr * invN;
            }
        }

        // 参数缩放（旋转和平移量纲统一）
        for (int p = 0; p < 3; p++) result.gradient[p] /= transScale_;
        for (int p = 3; p < 6; p++) result.gradient[p] /= rotScale_;

        return result;
    }

    float getInitialNMI(const VolumeData& mov, float cx, float cy, float cz) const {
        VersorTransform id = VersorTransform::identity();
        return evaluate(mov, id, cx, cy, cz).value;
    }

private:
    int numBins_ = 32;
    float refMin_, refMax_, movMin_, movMax_;
    float refBinSize_, movBinSize_;
    float transScale_ = 0.01f, rotScale_ = 1000.0f;
    std::vector<NMISample> samples_;
    VolumeData movGradX_, movGradY_, movGradZ_;
};

// ============================================================
// §6  优化器（正则化梯度下降 + Versor 更新）
// ============================================================

struct OptimizationResult {
    VersorTransform transform;
    float finalNMI = 0;
    std::vector<float> nmiHistory;
    double timeMs = 0;
};

/**
 * @brief 正则化梯度下降优化器
 *
 * 算法（对应源码 coptimization::ResumeOptimization）：
 *  1. 采样参考图像域
 *  2. 循环：
 *     a. 计算 NMI 值和梯度
 *     b. 梯度缩放（旋转/平移不同量纲）
 *     c. 收敛判断（梯度模长 < 容忍度）
 *     d. 方向反转检测 → 缩小步长
 *     e. 平移参数直接更新
 *     f. 旋转参数通过四元数乘法更新
 */
inline OptimizationResult optimize(const VolumeData& ref, const VolumeData& mov,
                                    const RegConfig& cfg, const VersorTransform& init) {
    OptimizationResult result;
    result.transform = init;

    float cx = ref.width * 0.5f, cy = ref.height * 0.5f, cz = ref.depth * 0.5f;

    NMIMetric metric;
    metric.initialize(ref, mov, cfg);

    float step = cfg.maxStep;
    float prevGrad[6] = {};
    bool hasPrev = false;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < cfg.maxIterations && step > cfg.minStep; it++) {
        NMIResult res = metric.evaluate(mov, result.transform, cx, cy, cz);
        result.nmiHistory.push_back(res.value);

        // 梯度模长
        float mag = 0.0f;
        for (int k = 0; k < 6; k++) mag += res.gradient[k] * res.gradient[k];
        mag = std::sqrt(mag);

        if (mag < cfg.magnitudeTol) break;

        // 方向反转检测
        if (hasPrev) {
            float dot = 0.0f;
            for (int k = 0; k < 6; k++) dot += res.gradient[k] * prevGrad[k];
            if (dot < 0) step *= cfg.relaxFactor;
        }
        for (int k = 0; k < 6; k++) prevGrad[k] = res.gradient[k];
        hasPrev = true;

        // 参数更新
        float factor = step / mag;

        // 平移：直接更新  t -= step · ∇/|∇|
        result.transform.tx -= factor * res.gradient[0];
        result.transform.ty -= factor * res.gradient[1];
        result.transform.tz -= factor * res.gradient[2];

        // 旋转：四元数乘法更新
        // 构造增量四元数 dq = (cos(θ/2), sin(θ/2)·axis)
        // q_new = q × dq
        float grot[3] = {res.gradient[3], res.gradient[4], res.gradient[5]};
        float grotMag = std::sqrt(grot[0]*grot[0] + grot[1]*grot[1] + grot[2]*grot[2]);
        if (grotMag > 1e-10f) {
            float angle = -factor * grotMag;  // 负号：最小化 -MI = 最大化 MI
            float half = angle * 0.5f;
            float sa = std::sin(half) / grotMag;
            float dqw = std::cos(half);
            float dqx = grot[0] * sa;
            float dqy = grot[1] * sa;
            float dqz = grot[2] * sa;

            // 当前四元数
            float qw = std::sqrt(std::max(0.0f, 1.0f
                - result.transform.qx * result.transform.qx
                - result.transform.qy * result.transform.qy
                - result.transform.qz * result.transform.qz));

            // q_new = q × dq
            float nqw, nqx, nqy, nqz;
            quatMultiply(qw, result.transform.qx, result.transform.qy, result.transform.qz,
                         dqw, dqx, dqy, dqz,
                         nqw, nqx, nqy, nqz);

            // 归一化
            float qn = std::sqrt(nqw*nqw + nqx*nqx + nqy*nqy + nqz*nqz);
            result.transform.qx = nqx / qn;
            result.transform.qy = nqy / qn;
            result.transform.qz = nqz / qn;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.timeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 最终评估
    NMIResult finalRes = metric.evaluate(mov, result.transform, cx, cy, cz);
    result.finalNMI = finalRes.value;

    return result;
}

// ============================================================
// §7  预处理
// ============================================================

struct CTPreprocessResult {
    VolumeData volume;
    int bedVoxelsRemoved = 0;
    float huMin = 0, huMax = 0;
};

/**
 * @brief CT 预处理：去床板 + HU 裁剪 [-1024, 1024]
 *
 * 对应源码 FusionRegistrationCT_PET::RegistrationTool:
 *   1. 床板区域设为 -1024（空气）
 *   2. DICOM 像素值 → HU 值（本 demo 已在 HU 空间）
 *   3. 窗位裁剪 [-1024, 1024]
 */
inline CTPreprocessResult preprocessCT(const VolumeData& ct) {
    CTPreprocessResult result;
    result.volume = ct;
    int W = ct.width, H = ct.height, D = ct.depth;

    // 模拟床板：底部 ~6% 行设为空气
    int bedRows = std::max(1, H / 16);
    for (int z = 0; z < D; z++)
        for (int y = H - bedRows; y < H; y++)
            for (int x = 0; x < W; x++) {
                result.volume.at(x, y, z) = -1024.0f;
                result.bedVoxelsRemoved++;
            }

    // HU 裁剪
    float mn = 1e30f, mx = -1e30f;
    for (int i = 0; i < ct.size(); i++) {
        float v = std::clamp(result.volume.data[i], -1024.0f, 1024.0f);
        result.volume.data[i] = v;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    result.huMin = mn;
    result.huMax = mx;
    return result;
}

struct PETPreprocessResult {
    VolumeData volume;
    float clipLow = 0, clipUp = 0;
    float backRatio = 0;
};

/**
 * @brief PET 预处理：自适应百分位窗位裁剪
 *
 * 对应源码 FusionRegistrationCT_PET::RegistrationTool:
 *   1. 排序所有像素值
 *   2. 初始百分位 [0.1%, 100%] → 初始范围
 *   3. 统计背景比例 back_ratio
 *   4. 自适应上界: up_ratio = back_ratio + (1-back_ratio)×0.99
 *   5. 裁剪原始图像到 [low, up]
 */
inline PETPreprocessResult preprocessPET(const VolumeData& pet) {
    PETPreprocessResult result;

    std::vector<float> sorted(pet.data.begin(), pet.data.end());
    std::sort(sorted.begin(), sorted.end());
    int N = (int)sorted.size();

    // 初始百分位
    float low = sorted[std::max(0, (int)(N * 0.001f))];
    float up  = sorted[N - 1];

    // 统计背景比例
    int backCount = 0;
    for (int i = 0; i < N; i++)
        if (sorted[i] <= low) backCount++;
    float backRatio = (float)backCount / N;

    // 自适应上界
    float upRatio = backRatio + (1.0f - backRatio) * 0.99f;
    low = sorted[std::max(0, (int)(N * 0.001f))];
    up  = sorted[std::min(N - 1, (int)(N * upRatio))];

    result.clipLow = low;
    result.clipUp = up;
    result.backRatio = backRatio;

    // 裁剪
    result.volume = pet;
    for (int i = 0; i < pet.size(); i++)
        result.volume.data[i] = std::clamp(pet.data[i], low, up);

    return result;
}

// ============================================================
// §8  多分辨率配准
// ============================================================

struct RegistrationResult {
    VersorTransform transform;
    float initialNMI = 0;
    float finalNMI = 0;
    std::vector<float> nmiHistory;
    double timeMs = 0;
    CTPreprocessResult ctPre;
    PETPreprocessResult petPre;
    VolumeData registeredPET;
};

/**
 * @brief CT-PET 刚体配准完整管线
 *
 * 流程（对应文档第 4 章）：
 *  1. CT 预处理（去床板 + HU 裁剪）
 *  2. PET 预处理（自适应百分位裁剪）
 *  3. 中心对齐先验矩阵
 *  4. 多分辨率优化（粗→细金字塔）
 *  5. 重采样 PET → CT 空间
 */
inline RegistrationResult registerCTPET(const VolumeData& ct, const VolumeData& pet,
                                         const RegConfig& cfg = RegConfig()) {
    RegistrationResult result;

    // 1. 预处理
    result.ctPre  = preprocessCT(ct);
    result.petPre = preprocessPET(pet);
    auto& ref = result.ctPre.volume;
    auto& mov = result.petPre.volume;

    // 2. 初始 NMI
    float cx = ref.width * 0.5f, cy = ref.height * 0.5f, cz = ref.depth * 0.5f;
    NMIMetric initMetric;
    initMetric.initialize(ref, mov, cfg);
    result.initialNMI = initMetric.getInitialNMI(mov, cx, cy, cz);

    // 3. 中心对齐先验
    VersorTransform transform = VersorTransform::identity();
    if (cfg.useCenterAlign) {
        transform.tx = (mov.width  * mov.spacingX - ref.width  * ref.spacingX) / 2.0f / ref.spacingX;
        transform.ty = (mov.height * mov.spacingY - ref.height * ref.spacingY) / 2.0f / ref.spacingY;
        transform.tz = (mov.depth  * mov.spacingZ - ref.depth  * ref.spacingZ) / 2.0f / ref.spacingZ;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // 4. 多分辨率优化
    int L = std::max(1, cfg.numLevels);
    for (int lvl = L - 1; lvl >= 0; lvl--) {
        float scale = 1.0f / (1 << lvl);  // lvl=1 → 0.5, lvl=0 → 1.0

        int rw = std::max(8, (int)(ref.width  * scale));
        int rh = std::max(8, (int)(ref.height * scale));
        int rd = std::max(4, (int)(ref.depth  * scale));

        VolumeData refLevel = resizeVolume(ref, rw, rh, rd);
        VolumeData movLevel = resizeVolume(mov, rw, rh, rd);

        // 缩放平移参数
        VersorTransform init = transform;
        init.tx *= scale; init.ty *= scale; init.tz *= scale;

        auto opt = optimize(refLevel, movLevel, cfg, init);

        // 缩放回原始分辨率
        transform = opt.transform;
        transform.tx /= scale; transform.ty /= scale; transform.tz /= scale;

        for (float v : opt.nmiHistory) result.nmiHistory.push_back(v);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.timeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    result.transform = transform;
    result.finalNMI = result.nmiHistory.empty() ? 0 : result.nmiHistory.back();

    // 5. 重采样
    float R[3][3]; quatToMatrix(transform.qx, transform.qy, transform.qz, R);
    result.registeredPET.resize(ref.width, ref.height, ref.depth);
    result.registeredPET.spacingX = ref.spacingX;
    result.registeredPET.spacingY = ref.spacingY;
    result.registeredPET.spacingZ = ref.spacingZ;

    for (int z = 0; z < ref.depth; z++)
        for (int y = 0; y < ref.height; y++)
            for (int x = 0; x < ref.width; x++) {
                float dx = x - cx, dy = y - cy, dz = z - cz;
                float mx = cx + transform.tx + R[0][0]*dx + R[0][1]*dy + R[0][2]*dz;
                float my = cy + transform.ty + R[1][0]*dx + R[1][1]*dy + R[1][2]*dz;
                float mz = cz + transform.tz + R[2][0]*dx + R[2][1]*dy + R[2][2]*dz;
                result.registeredPET.at(x, y, z) = trilinear(mov, mx, my, mz);
            }

    return result;
}

// ============================================================
// §9  重采样（独立函数，用于施加已知变换）
// ============================================================

inline VolumeData resample(const VolumeData& mov, const VolumeData& ref,
                            const VersorTransform& t) {
    float R[3][3]; quatToMatrix(t.qx, t.qy, t.qz, R);
    float cx = ref.width * 0.5f, cy = ref.height * 0.5f, cz = ref.depth * 0.5f;

    VolumeData out; out.resize(ref.width, ref.height, ref.depth);
    out.spacingX = ref.spacingX; out.spacingY = ref.spacingY; out.spacingZ = ref.spacingZ;

    for (int z = 0; z < ref.depth; z++)
        for (int y = 0; y < ref.height; y++)
            for (int x = 0; x < ref.width; x++) {
                float dx = x - cx, dy = y - cy, dz = z - cz;
                float mx = cx + t.tx + R[0][0]*dx + R[0][1]*dy + R[0][2]*dz;
                float my = cy + t.ty + R[1][0]*dx + R[1][1]*dy + R[1][2]*dz;
                float mz = cz + t.tz + R[2][0]*dx + R[2][1]*dy + R[2][2]*dz;
                out.at(x, y, z) = trilinear(mov, mx, my, mz);
            }
    return out;
}

// ============================================================
// §10  Phantom 生成（多模态 CT/PET 头部体模）
// ============================================================

/**
 * @brief 生成 CT 头部体模
 *
 * 解剖结构（HU 值）：
 *   - 颅骨: 1000 HU（亮骨）
 *   - 脑组织: 40 HU（软组织）
 *   - 脑室: 0 HU（脑脊液）
 *   - 肿瘤: 60 HU（与组织差异小，CT 难以辨识）
 *   - 床板: 300 HU（底部矩形区域）
 *   - 背景: -1000 HU（空气）
 */
inline VolumeData createCTPhantom(int W, int H, int D) {
    VolumeData img; img.resize(W, H, D);
    img.spacingX = img.spacingY = 1.0f; img.spacingZ = 1.5f;
    std::fill(img.data.begin(), img.data.end(), -1000.0f);

    float cx = W * 0.5f, cy = H * 0.5f, cz = D * 0.5f;

    auto ellipsoid = [&](float ex, float ey, float ez,
                          float rx, float ry, float rz, float val) {
        for (int z = 0; z < D; z++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) {
                    float dx = (x - ex) / rx, dy = (y - ey) / ry, dz = (z - ez) / rz;
                    if (dx*dx + dy*dy + dz*dz < 1.0f) img.at(x, y, z) = val;
                }
    };

    ellipsoid(cx, cy, cz, W*0.42f, H*0.40f, D*0.42f, 1000.0f);  // 颅骨
    ellipsoid(cx, cy, cz, W*0.36f, H*0.34f, D*0.36f, 40.0f);    // 脑组织
    ellipsoid(cx, cy, cz, W*0.12f, H*0.10f, D*0.12f, 0.0f);     // 脑室
    ellipsoid(cx + W*0.12f, cy - H*0.05f, cz + D*0.08f,
              W*0.05f, H*0.05f, D*0.05f, 60.0f);                 // 肿瘤（CT 难见）

    // 床板
    for (int z = 0; z < D; z++)
        for (int y = H - 3; y < H; y++)
            for (int x = W / 4; x < 3 * W / 4; x++)
                img.at(x, y, z) = 300.0f;

    // 噪声
    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 15.0f);
    for (auto& v : img.data) v += noise(rng);

    return img;
}

/**
 * @brief 生成 PET 头部体模（相同解剖结构，不同灰度映射）
 *
 * 解剖结构（SUV 值，缩放为 short 范围）：
 *   - 颅骨: 100（低代谢）
 *   - 脑组织: 500（中等代谢）
 *   - 脑室: 50（极低代谢）
 *   - 肿瘤: 2000（高代谢 — "热点"！PET 清晰可见）
 *   - 背景: 0
 *   - 无床板（PET 不显示床板）
 *
 * 关键：肿瘤在 CT 中几乎不可见（60 vs 40 HU），
 *       但在 PET 中极其明亮（2000 vs 500），体现多模态特性。
 */
inline VolumeData createPETPhantom(int W, int H, int D) {
    VolumeData img; img.resize(W, H, D);
    img.spacingX = img.spacingY = 1.0f; img.spacingZ = 1.5f;
    std::fill(img.data.begin(), img.data.end(), 0.0f);

    float cx = W * 0.5f, cy = H * 0.5f, cz = D * 0.5f;

    auto ellipsoid = [&](float ex, float ey, float ez,
                          float rx, float ry, float rz, float val) {
        for (int z = 0; z < D; z++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) {
                    float dx = (x - ex) / rx, dy = (y - ey) / ry, dz = (z - ez) / rz;
                    if (dx*dx + dy*dy + dz*dz < 1.0f) img.at(x, y, z) = val;
                }
    };

    ellipsoid(cx, cy, cz, W*0.42f, H*0.40f, D*0.42f, 100.0f);   // 颅骨（低代谢）
    ellipsoid(cx, cy, cz, W*0.36f, H*0.34f, D*0.36f, 500.0f);   // 脑组织
    ellipsoid(cx, cy, cz, W*0.12f, H*0.10f, D*0.12f, 50.0f);    // 脑室
    ellipsoid(cx + W*0.12f, cy - H*0.05f, cz + D*0.08f,
              W*0.05f, H*0.05f, D*0.05f, 2000.0f);               // 肿瘤（热点！）

    // 噪声
    std::mt19937 rng(123);
    std::normal_distribution<float> noise(0.0f, 20.0f);
    for (auto& v : img.data) v = std::max(0.0f, v + noise(rng));

    return img;
}

} // namespace CTPETReg
