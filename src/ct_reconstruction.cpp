#include "ct_reconstruction.hpp"
#include <cstring>
#include <numeric>
#include <random>
#include <iostream>
#include <fstream>
#include <cassert>

// ============================================================================
// ReconResult 实现
// ============================================================================

void ReconResult::normalize() {
    if (image.empty()) return;
    float minVal = *std::min_element(image.begin(), image.end());
    float maxVal = *std::max_element(image.begin(), image.end());
    float range = maxVal - minVal;
    if (range < 1e-10f) range = 1.0f;
    for (auto& v : image) {
        v = (v - minVal) / range;
    }
}

std::vector<uint8_t> ReconResult::toUint8() const {
    std::vector<uint8_t> result(image.size());
    for (size_t i = 0; i < image.size(); ++i) {
        float v = std::clamp(image[i], 0.0f, 1.0f);
        result[i] = static_cast<uint8_t>(v * 255.0f + 0.5f);
    }
    return result;
}

// ============================================================================
// Shepp-Logan 体模
// ============================================================================

std::vector<float> CTReconstructor::generateSheppLoganPhantom(int size) {
    // 经典 Shepp-Logan 体模参数 (10 个椭圆)
    // 参考: Shepp & Logan, "The Fourier Reconstruction of a Head Section", 1974
    std::vector<PhantomEllipse> ellipses = {
        {  2.0f,    0.0f,     0.0f,    0.69f,   0.92f,    0.0f  },  // 外椭圆 (头骨)
        { -0.98f,   0.0f,    -0.0184f, 0.6624f, 0.8740f,  0.0f  },  // 内部
        { -0.02f,   0.22f,    0.0f,    0.11f,   0.31f,   -18.0f },  // 右侧
        { -0.02f,  -0.22f,    0.0f,    0.16f,   0.41f,    18.0f },  // 左侧
        {  0.01f,   0.0f,     0.35f,   0.21f,   0.25f,    0.0f  },  // 上部小椭圆
        {  0.01f,   0.0f,     0.1f,    0.046f,  0.046f,   0.0f  },  // 小圆 1
        {  0.01f,   0.0f,    -0.1f,    0.046f,  0.046f,   0.0f  },  // 小圆 2
        {  0.01f,  -0.08f,   -0.605f,  0.046f,  0.023f,   0.0f  },  // 底部 1
        {  0.01f,   0.0f,    -0.605f,  0.023f,  0.023f,   0.0f  },  // 底部 2
        {  0.01f,   0.06f,   -0.605f,  0.023f,  0.046f,   0.0f  },  // 底部 3
    };
    return generatePhantom(size, ellipses);
}

std::vector<float> CTReconstructor::generatePhantom(
    int size, const std::vector<PhantomEllipse>& ellipses)
{
    std::vector<float> image(size * size, 0.0f);
    float halfSize = size / 2.0f;

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            // 归一化坐标 [-1, 1]
            float px = (x - halfSize + 0.5f) / halfSize;
            float py = (y - halfSize + 0.5f) / halfSize;

            for (const auto& e : ellipses) {
                float cosT = std::cos(e.theta * static_cast<float>(M_PI) / 180.0f);
                float sinT = std::sin(e.theta * static_cast<float>(M_PI) / 180.0f);

                // 平移后旋转
                float dx = px - e.cx;
                float dy = py - e.cy;
                float rx =  cosT * dx + sinT * dy;
                float ry = -sinT * dx + cosT * dy;

                // 椭圆方程: (rx/a)^2 + (ry/b)^2 <= 1
                float val = (rx * rx) / (e.a * e.a) + (ry * ry) / (e.b * e.b);
                if (val <= 1.0f) {
                    image[y * size + x] += e.intensity;
                }
            }
        }
    }
    return image;
}

// ============================================================================
// 正向投影 (Radon 变换) - 平行束
// ============================================================================

Sinogram CTReconstructor::forwardProject(
    const std::vector<float>& image, int imageSize,
    int numAngles, int numDetectors)
{
    if (numDetectors <= 0) {
        // 默认探测器数 = 图像对角线长度 (保证完整覆盖)
        numDetectors = static_cast<int>(std::ceil(imageSize * std::sqrt(2.0f)));
    }

    Sinogram sino;
    sino.numAngles = numAngles;
    sino.numDetectors = numDetectors;
    sino.angleStart = 0.0f;
    sino.angleEnd = static_cast<float>(M_PI);
    sino.data.resize(numAngles * numDetectors, 0.0f);

    float halfImage = imageSize / 2.0f;
    float halfDet = numDetectors / 2.0f;
    // 探测器间距: 覆盖图像对角线
    float detectorSpacing = (imageSize * std::sqrt(2.0f)) / numDetectors;

    for (int a = 0; a < numAngles; ++a) {
        float theta = sino.angleStart +
            a * (sino.angleEnd - sino.angleStart) / numAngles;
        float cosTheta = std::cos(theta);
        float sinTheta = std::sin(theta);

        for (int d = 0; d < numDetectors; ++d) {
            // 探测器位置 (沿 t 轴)
            float t = (d - halfDet + 0.5f) * detectorSpacing;
            float sum = 0.0f;

            // 沿射线方向积分 (s 方向)
            // 射线参数: x = t*cos(theta) - s*sin(theta)
            //          y = t*sin(theta) + s*cos(theta)
            float sMax = imageSize * std::sqrt(2.0f) / 2.0f;
            float ds = 1.0f; // 步长 = 1 像素

            for (float s = -sMax; s <= sMax; s += ds) {
                float x = t * cosTheta - s * sinTheta;
                float y = t * sinTheta + s * cosTheta;

                // 转换到图像坐标
                float ix = x + halfImage - 0.5f;
                float iy = y + halfImage - 0.5f;

                sum += bilinearSample(image, imageSize, ix, iy) * ds;
            }
            sino.at(a, d) = sum;
        }
    }
    return sino;
}

// ============================================================================
// FFT 实现 (Cooley-Tukey 基2)
// ============================================================================

int CTReconstructor::nextPowerOf2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

void CTReconstructor::fft1D(std::vector<std::complex<float>>& data, bool inverse) {
    int n = static_cast<int>(data.size());
    if (n <= 1) return;

    // 位逆序置换
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }

    // 蝶形运算
    for (int len = 2; len <= n; len <<= 1) {
        float angle = 2.0f * static_cast<float>(M_PI) / len * (inverse ? -1.0f : 1.0f);
        std::complex<float> wn(std::cos(angle), std::sin(angle));

        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (int j = 0; j < len / 2; ++j) {
                std::complex<float> u = data[i + j];
                std::complex<float> v = data[i + j + len / 2] * w;
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                w *= wn;
            }
        }
    }

    if (inverse) {
        for (auto& x : data) x /= static_cast<float>(n);
    }
}

// ============================================================================
// 滤波器
// ============================================================================

std::vector<float> CTReconstructor::createFilter(int size, FilterType type) {
    std::vector<float> filter(size);
    float halfSize = size / 2.0f;

    for (int i = 0; i < size; ++i) {
        // 频率归一化到 [0, 1]
        float freq = std::abs(i - halfSize) / halfSize;
        if (freq > 1.0f) freq = 1.0f;

        // 基础 ramp (Ram-Lak)
        float ramp = freq;

        switch (type) {
        case FilterType::RAM_LAK:
            filter[i] = ramp;
            break;
        case FilterType::SHEPP_LOGAN:
            if (freq < 1e-6f)
                filter[i] = 0.0f;
            else
                filter[i] = ramp * std::sin(static_cast<float>(M_PI) * freq / 2.0f)
                           / (static_cast<float>(M_PI) * freq / 2.0f);
            break;
        case FilterType::COSINE:
            filter[i] = ramp * std::cos(static_cast<float>(M_PI) * freq / 2.0f);
            break;
        case FilterType::HAMMING:
            filter[i] = ramp * (0.54f + 0.46f * std::cos(static_cast<float>(M_PI) * freq));
            break;
        case FilterType::HANN:
            filter[i] = ramp * 0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * freq));
            break;
        }
    }
    return filter;
}

std::vector<float> CTReconstructor::applyFilter(
    const std::vector<float>& projection, FilterType type)
{
    int n = static_cast<int>(projection.size());
    int N = nextPowerOf2(2 * n); // 零填充避免循环卷积

    // 转为复数
    std::vector<std::complex<float>> freq(N, {0.0f, 0.0f});
    for (int i = 0; i < n; ++i) {
        freq[i] = std::complex<float>(projection[i], 0.0f);
    }

    // FFT
    fft1D(freq, false);

    // 构造频域滤波器
    std::vector<float> filter = createFilter(N, type);

    // 应用滤波
    for (int i = 0; i < N; ++i) {
        freq[i] *= filter[i];
    }

    // IFFT
    fft1D(freq, true);

    // 取实部
    std::vector<float> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = freq[i].real();
    }
    return result;
}

// ============================================================================
// 滤波反投影 (FBP) 重建
// ============================================================================

ReconResult CTReconstructor::reconstructFBP(
    const Sinogram& sinogram, int outputSize, FilterType filter)
{
    ReconResult result;
    result.size = outputSize;
    result.image.resize(outputSize * outputSize, 0.0f);

    float halfOutput = outputSize / 2.0f;
    float halfDet = sinogram.numDetectors / 2.0f;
    float detectorSpacing = (outputSize * std::sqrt(2.0f)) / sinogram.numDetectors;

    // 步骤1: 对每个角度的投影进行滤波
    std::vector<std::vector<float>> filteredProjections(sinogram.numAngles);

    for (int a = 0; a < sinogram.numAngles; ++a) {
        // 提取该角度的投影
        std::vector<float> proj(sinogram.numDetectors);
        for (int d = 0; d < sinogram.numDetectors; ++d) {
            proj[d] = sinogram.at(a, d);
        }
        // 滤波
        filteredProjections[a] = applyFilter(proj, filter);
    }

    // 步骤2: 反投影 (Back Projection)
    float angleStep = (sinogram.angleEnd - sinogram.angleStart) / sinogram.numAngles;

    for (int a = 0; a < sinogram.numAngles; ++a) {
        float theta = sinogram.angleStart + a * angleStep;
        float cosTheta = std::cos(theta);
        float sinTheta = std::sin(theta);

        for (int y = 0; y < outputSize; ++y) {
            for (int x = 0; x < outputSize; ++x) {
                // 像素物理坐标
                float px = (x - halfOutput + 0.5f);
                float py = (y - halfOutput + 0.5f);

                // 计算该像素在当前角度下对应的探测器位置
                float t = px * cosTheta + py * sinTheta;

                // 转换为探测器索引
                float detIdx = t / detectorSpacing + halfDet - 0.5f;

                // 线性插值获取滤波后的投影值
                int d0 = static_cast<int>(std::floor(detIdx));
                int d1 = d0 + 1;
                float frac = detIdx - d0;

                float val = 0.0f;
                if (d0 >= 0 && d1 < sinogram.numDetectors) {
                    val = (1.0f - frac) * filteredProjections[a][d0]
                        + frac * filteredProjections[a][d1];
                } else if (d0 >= 0 && d0 < sinogram.numDetectors) {
                    val = filteredProjections[a][d0];
                }

                result.image[y * outputSize + x] += val * angleStep;
            }
        }
    }

    return result;
}

// ============================================================================
// SIRT 迭代重建
// ============================================================================

ReconResult CTReconstructor::reconstructSIRT(
    const Sinogram& sinogram, int outputSize,
    int iterations, float relaxation)
{
    ReconResult result;
    result.size = outputSize;
    result.image.resize(outputSize * outputSize, 0.0f);

    float halfOutput = outputSize / 2.0f;
    float halfDet = sinogram.numDetectors / 2.0f;
    float detectorSpacing = (outputSize * std::sqrt(2.0f)) / sinogram.numDetectors;

    for (int iter = 0; iter < iterations; ++iter) {
        // 1. 正向投影当前估计 -> 模拟正弦图
        Sinogram simSino;
        simSino.numAngles = sinogram.numAngles;
        simSino.numDetectors = sinogram.numDetectors;
        simSino.angleStart = sinogram.angleStart;
        simSino.angleEnd = sinogram.angleEnd;
        simSino.data.resize(sinogram.data.size(), 0.0f);

        float angleStep = (sinogram.angleEnd - sinogram.angleStart) / sinogram.numAngles;

        for (int a = 0; a < sinogram.numAngles; ++a) {
            float theta = sinogram.angleStart + a * angleStep;
            float cosTheta = std::cos(theta);
            float sinTheta = std::sin(theta);

            for (int d = 0; d < sinogram.numDetectors; ++d) {
                float t = (d - halfDet + 0.5f) * detectorSpacing;
                float sum = 0.0f;
                float sMax = outputSize * std::sqrt(2.0f) / 2.0f;

                for (float s = -sMax; s <= sMax; s += 1.0f) {
                    float x = t * cosTheta - s * sinTheta + halfOutput - 0.5f;
                    float y = t * sinTheta + s * cosTheta + halfOutput - 0.5f;
                    sum += bilinearSample(result.image, outputSize, x, y);
                }
                simSino.at(a, d) = sum;
            }
        }

        // 2. 计算残差 = 测量值 - 模拟值
        std::vector<float> residual(sinogram.data.size());
        float totalError = 0.0f;
        for (size_t i = 0; i < sinogram.data.size(); ++i) {
            residual[i] = sinogram.data[i] - simSino.data[i];
            totalError += residual[i] * residual[i];
        }

        // 3. 反投影残差并更新图像
        std::vector<float> correction(outputSize * outputSize, 0.0f);
        std::vector<float> weight(outputSize * outputSize, 0.0f);

        for (int a = 0; a < sinogram.numAngles; ++a) {
            float theta = sinogram.angleStart + a * angleStep;
            float cosTheta = std::cos(theta);
            float sinTheta = std::sin(theta);

            for (int y = 0; y < outputSize; ++y) {
                for (int x = 0; x < outputSize; ++x) {
                    float px = (x - halfOutput + 0.5f);
                    float py = (y - halfOutput + 0.5f);
                    float t = px * cosTheta + py * sinTheta;
                    float detIdx = t / detectorSpacing + halfDet - 0.5f;

                    int d0 = static_cast<int>(std::floor(detIdx));
                    float frac = detIdx - d0;

                    float res = 0.0f;
                    if (d0 >= 0 && d0 + 1 < sinogram.numDetectors) {
                        res = (1.0f - frac) * residual[a * sinogram.numDetectors + d0]
                            + frac * residual[a * sinogram.numDetectors + d0 + 1];
                    }

                    correction[y * outputSize + x] += res;
                    weight[y * outputSize + x] += 1.0f;
                }
            }
        }

        // 更新
        for (int i = 0; i < outputSize * outputSize; ++i) {
            if (weight[i] > 0.0f) {
                result.image[i] += relaxation * correction[i] / weight[i];
            }
            // 非负约束
            if (result.image[i] < 0.0f) result.image[i] = 0.0f;
        }

        result.convergenceError = totalError / sinogram.data.size();

        if ((iter + 1) % 10 == 0 || iter == 0) {
            std::cout << "  SIRT 迭代 " << (iter + 1)
                      << "/" << iterations
                      << "  MSE=" << result.convergenceError << std::endl;
        }
    }

    return result;
}

// ============================================================================
// ART 迭代重建
// ============================================================================

ReconResult CTReconstructor::reconstructART(
    const Sinogram& sinogram, int outputSize,
    int iterations, float relaxation)
{
    ReconResult result;
    result.size = outputSize;
    result.image.resize(outputSize * outputSize, 0.0f);

    float halfOutput = outputSize / 2.0f;
    float halfDet = sinogram.numDetectors / 2.0f;
    float detectorSpacing = (outputSize * std::sqrt(2.0f)) / sinogram.numDetectors;
    float angleStep = (sinogram.angleEnd - sinogram.angleStart) / sinogram.numAngles;

    for (int iter = 0; iter < iterations; ++iter) {
        float totalError = 0.0f;

        for (int a = 0; a < sinogram.numAngles; ++a) {
            float theta = sinogram.angleStart + a * angleStep;
            float cosTheta = std::cos(theta);
            float sinTheta = std::sin(theta);

            for (int d = 0; d < sinogram.numDetectors; ++d) {
                float t = (d - halfDet + 0.5f) * detectorSpacing;

                // 沿射线计算当前投影值和权重
                float projVal = 0.0f;
                float rayWeight = 0.0f;
                float sMax = outputSize * std::sqrt(2.0f) / 2.0f;

                // 记录射线经过的像素
                struct RayPixel { int idx; float w; };
                std::vector<RayPixel> rayPixels;

                for (float s = -sMax; s <= sMax; s += 1.0f) {
                    float fx = t * cosTheta - s * sinTheta + halfOutput - 0.5f;
                    float fy = t * sinTheta + s * cosTheta + halfOutput - 0.5f;

                    int ix = static_cast<int>(std::floor(fx));
                    int iy = static_cast<int>(std::floor(fy));

                    if (ix >= 0 && ix < outputSize && iy >= 0 && iy < outputSize) {
                        float w = 1.0f;
                        int idx = iy * outputSize + ix;
                        projVal += result.image[idx] * w;
                        rayWeight += w * w;
                        rayPixels.push_back({idx, w});
                    }
                }

                if (rayWeight < 1e-6f) continue;

                // 残差
                float residual = sinogram.at(a, d) - projVal;
                totalError += residual * residual;

                // 沿射线更新像素
                float update = relaxation * residual / rayWeight;
                for (const auto& rp : rayPixels) {
                    result.image[rp.idx] += update * rp.w;
                    if (result.image[rp.idx] < 0.0f)
                        result.image[rp.idx] = 0.0f;
                }
            }
        }

        result.convergenceError = totalError /
            (sinogram.numAngles * sinogram.numDetectors);

        if ((iter + 1) % 5 == 0 || iter == 0) {
            std::cout << "  ART 迭代 " << (iter + 1)
                      << "/" << iterations
                      << "  MSE=" << result.convergenceError << std::endl;
        }
    }

    return result;
}

// ============================================================================
// 统一重建接口
// ============================================================================

ReconResult CTReconstructor::reconstruct(
    const Sinogram& sinogram, int outputSize,
    ReconMethod method, FilterType filter,
    int iterations, float relaxation)
{
    switch (method) {
    case ReconMethod::FBP:
        return reconstructFBP(sinogram, outputSize, filter);
    case ReconMethod::SIRT:
        return reconstructSIRT(sinogram, outputSize, iterations, relaxation);
    case ReconMethod::ART:
        return reconstructART(sinogram, outputSize, iterations, relaxation);
    default:
        return reconstructFBP(sinogram, outputSize, filter);
    }
}

// ============================================================================
// 工具函数
// ============================================================================

float CTReconstructor::bilinearSample(
    const std::vector<float>& image, int size, float x, float y)
{
    if (x < 0 || x >= size - 1 || y < 0 || y >= size - 1)
        return 0.0f;

    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    if (x1 >= size) x1 = size - 1;
    if (y1 >= size) y1 = size - 1;

    float fx = x - x0;
    float fy = y - y0;

    float v00 = image[y0 * size + x0];
    float v10 = image[y0 * size + x1];
    float v01 = image[y1 * size + x0];
    float v11 = image[y1 * size + x1];

    return v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy)
         + v01 * (1 - fx) * fy + v11 * fx * fy;
}

void CTReconstructor::addPoissonNoise(Sinogram& sinogram, float photonCount) {
    std::mt19937 rng(42);

    // 归一化正弦图
    float maxVal = *std::max_element(sinogram.data.begin(), sinogram.data.end());
    if (maxVal < 1e-10f) return;

    for (auto& v : sinogram.data) {
        // Beer-Lambert: I = I0 * exp(-integral)
        float intensity = photonCount * std::exp(-v / maxVal);
        // 泊松采样
        std::poisson_distribution<int> poisson(std::max(intensity, 0.1f));
        int detected = poisson(rng);
        // 逆变换回衰减值
        if (detected > 0) {
            v = -maxVal * std::log(static_cast<float>(detected) / photonCount);
        } else {
            v = maxVal * 5.0f; // 饱和
        }
    }
}

bool CTReconstructor::savePGM(const std::string& filename,
    const std::vector<float>& image, int width, int height)
{
    // 归一化到 [0, 255]
    float minVal = *std::min_element(image.begin(), image.end());
    float maxVal = *std::max_element(image.begin(), image.end());
    float range = maxVal - minVal;
    if (range < 1e-10f) range = 1.0f;

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    file << "P5\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        uint8_t v = static_cast<uint8_t>(
            std::clamp((image[i] - minVal) / range * 255.0f, 0.0f, 255.0f));
        file.write(reinterpret_cast<char*>(&v), 1);
    }
    file.close();
    return true;
}

bool CTReconstructor::savePGM(const std::string& filename,
    const std::vector<uint8_t>& image, int width, int height)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    file << "P5\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.data()), image.size());
    file.close();
    return true;
}
