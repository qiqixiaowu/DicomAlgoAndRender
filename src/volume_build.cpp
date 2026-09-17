#include "volume_build.hpp"
#include <algorithm>
#include <numeric>
#include <map>
#include <cmath>
#include <cstring>
#include <queue>
#include <iostream>
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/dcmdata/dcuid.h>
#include <dcmtk/dcmdata/dcvrus.h>
#include <dcmtk/dcmimgle/dcmimage.h>

// 改进的三次样条插值（Catmull-Rom立方卷积）
static uint8_t cubicInterpolateCatmullRom(float t, float p0, float p1, float p2, float p3) {
    // Catmull-Rom样条系数
    float t2 = t * t;
    float t3 = t2 * t;

    // Catmull-Rom样条公式
    return static_cast<uint8_t>((std::max)(0.0f, (std::min)(255.0f,
        0.5f * ((2.0f * p1) +
            (-p0 + p2) * t +
            (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
            (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3))));
}

// B样条插值（更平滑）
static uint8_t cubicInterpolateBSpline(float t, float p0, float p1, float p2, float p3) {
    float t2 = t * t;
    float t3 = t2 * t;

    // B样条公式
    return static_cast<uint8_t>((std::max)(0.0f, (std::min)(255.0f,
        (1.0f / 6.0f) * ((-t3 + 3 * t2 - 3 * t + 1) * p0 +
            (3 * t3 - 6 * t2 + 4) * p1 +
            (-3 * t3 + 3 * t2 + 3 * t + 1) * p2 +
            (t3)*p3))));
}

// 使用边界处理的三次样条插值
static uint8_t getVoxelWithCubicInterpolationImproved(const uint8_t* volumeData,
    int width, int height, int depth,
    int x, int y, float z) {
    // 边界检查
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return 0;
    }

    // 计算相邻切片索引
    int z0 = static_cast<int>(floor(z));
    float t = z - z0;  // 插值参数 [0, 1)

    // 获取4个相邻切片
    int z_indices[4];
    uint8_t values[4];

    for (int i = 0; i < 4; ++i) {
        z_indices[i] = z0 - 1 + i;

        // 处理边界：镜像或重复边缘
        if (z_indices[i] < 0) {
            z_indices[i] = 0;  // 重复第一个切片
        }
        else if (z_indices[i] >= depth) {
            z_indices[i] = depth - 1;  // 重复最后一个切片
        }

        // 获取该切片的值
        values[i] = volumeData[z_indices[i] * width * height + y * width + x];
    }

    // 使用Catmull-Rom插值
    return cubicInterpolateCatmullRom(t,
        static_cast<float>(values[0]),
        static_cast<float>(values[1]),
        static_cast<float>(values[2]),
        static_cast<float>(values[3]));
}

// 在Z方向使用三次样条插值获取体素值
static uint8_t getVoxelWithCubicInterpolation(const uint8_t* volumeData,
    int width, int height, int depth,
    int x, int y, float z) {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return 0;
    }

    // 计算相邻切片索引
    int z0 = static_cast<int>(floor(z));
    float t = z - z0;  // 插值参数 [0, 1)

    // 获取4个相邻切片
    int z_indices[4];
    uint8_t values[4];

    for (int i = 0; i < 4; ++i) {
        z_indices[i] = z0 - 1 + i;

        // 处理边界：镜像或重复边缘
        if (z_indices[i] < 0) {
            z_indices[i] = 0;  // 重复第一个切片
        }
        else if (z_indices[i] >= depth) {
            z_indices[i] = depth - 1;  // 重复最后一个切片
        }

        // 获取该切片的值
        values[i] = volumeData[z_indices[i] * width * height + y * width + x];
    }

    // 使用Catmull-Rom插值
    return cubicInterpolateCatmullRom(t,
        static_cast<float>(values[0]),
        static_cast<float>(values[1]),
        static_cast<float>(values[2]),
        static_cast<float>(values[3]));
}

// 对Z轴进行三次样条插值重采样
static VolumeBuildResult resampleZWithCubic(const VolumeBuildResult& input,
    float targetZSpacing,
    int targetDepth = 0) {
    VolumeBuildResult output = input;

    // 计算新的深度
    if (targetDepth == 0) {
        // 根据目标间距计算深度
        float originalThickness = (input.depth - 1) * input.spacing[2];
        output.depth = static_cast<int>(std::round(originalThickness / targetZSpacing)) + 1;
    }
    else {
        output.depth = targetDepth;
    }
    unsigned int temp = 1;
    output.depth = (std::max)(temp, output.depth);
    output.spacing[2] = targetZSpacing;

    output.buffer.resize(output.width * output.height * output.depth);

    // 进行三次样条插值重采样
    for (int z = 0; z < output.depth; ++z) {
        // 计算在原始数据中的对应Z坐标（浮点数）
        float originalZ = static_cast<float>(z) * (input.depth - 1) / (output.depth - 1);

        int sliceOffset = z * output.width * output.height;

        for (int y = 0; y < output.height; ++y) {
            for (int x = 0; x < output.width; ++x) {
                // 使用三次样条插值
                uint8_t value = getVoxelWithCubicInterpolation(
                    input.buffer.data(),
                    input.width, input.height, input.depth,
                    x, y, originalZ
                );

                output.buffer[sliceOffset + y * output.width + x] = value;
            }
        }

        // 显示进度
        if (output.depth > 10 && z % (output.depth / 10) == 0) {
            float progress = static_cast<float>(z) / output.depth * 100.0f;
            std::cout << "  进度: " << static_cast<int>(progress) << "%" << std::endl;
        }
    }

    return output;
}

// 三线性插值函数
static uint8_t trilinearInterpolation(const uint8_t* volumeData,
    int width, int height, int depth,
    float x, float y, float z) {
    // 边界检查
    x = (std::max)(0.0f, (std::min)(x, static_cast<float>(width - 1)));
    y = (std::max)(0.0f, (std::min)(y, static_cast<float>(height - 1)));
    z = (std::max)(0.0f, (std::min)(z, static_cast<float>(depth - 1)));

    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int z0 = static_cast<int>(z);
    int x1 = (std::min)(x0 + 1, width - 1);
    int y1 = (std::min)(y0 + 1, height - 1);
    int z1 = (std::min)(z0 + 1, depth - 1);

    float xd = x - x0;
    float yd = y - y0;
    float zd = z - z0;

    // 获取8个相邻体素的值
    uint8_t c000 = volumeData[z0 * width * height + y0 * width + x0];
    uint8_t c001 = volumeData[z0 * width * height + y0 * width + x1];
    uint8_t c010 = volumeData[z0 * width * height + y1 * width + x0];
    uint8_t c011 = volumeData[z0 * width * height + y1 * width + x1];
    uint8_t c100 = volumeData[z1 * width * height + y0 * width + x0];
    uint8_t c101 = volumeData[z1 * width * height + y0 * width + x1];
    uint8_t c110 = volumeData[z1 * width * height + y1 * width + x0];
    uint8_t c111 = volumeData[z1 * width * height + y1 * width + x1];

    // 三线性插值
    float c00 = c000 * (1 - xd) + c001 * xd;
    float c01 = c010 * (1 - xd) + c011 * xd;
    float c10 = c100 * (1 - xd) + c101 * xd;
    float c11 = c110 * (1 - xd) + c111 * xd;

    float c0 = c00 * (1 - yd) + c01 * yd;
    float c1 = c10 * (1 - yd) + c11 * yd;

    float result = c0 * (1 - zd) + c1 * zd;

    return static_cast<uint8_t>(result);
}

static double computeSliceSpacing(const SeriesData& series, const double N[3], double defaultSpacing) {
    if (series.slices.size() < 2) return defaultSpacing;
    std::vector<double> distances;
    for (size_t i = 1; i < series.slices.size(); ++i) {
        if (series.slices[i].hasPosition && series.slices[i - 1].hasPosition) {
            double dz = 0; for (int k = 0; k < 3; ++k) dz += (series.slices[i].imagePosition[k] - series.slices[i - 1].imagePosition[k]) * N[k];
            distances.push_back(std::fabs(dz));
        }
    }
    if (distances.empty()) return defaultSpacing;
    double sum = 0; for (double v : distances) sum += v; return sum / distances.size();
}

VolumeBuildResult buildVolume_none(const SeriesData& series,
    bool enableResampling,
    float targetZSpacing,
    InterpolationMethod method) {
    VolumeBuildResult out;
    if (series.slices.empty()) return out;

    // 读取第一张获取尺寸与像素间距
    DcmFileFormat ff;
    if (!ff.loadFile(series.slices.front().filepath.c_str()).good()) return out;
    DcmDataset* ds0 = ff.getDataset();
    if (!ds0) return out;

    Uint16 rows = 0, cols = 0;
    ds0->findAndGetUint16(DCM_Rows, rows);
    ds0->findAndGetUint16(DCM_Columns, cols);
    out.width = cols;
    out.height = rows;
    out.depth = static_cast<uint32_t>(series.slices.size());

    double spacingX = 1.0, spacingY = 1.0, sliceThickness = 1.0;
    {
        OFString pixSpacingStr;
        if (ds0->findAndGetOFString(DCM_PixelSpacing, pixSpacingStr).good()) {
            std::string s = pixSpacingStr.c_str();
            size_t pos = s.find("\\");
            if (pos != std::string::npos) {
                spacingX = std::atof(s.substr(0, pos).c_str());
                spacingY = std::atof(s.substr(pos + 1).c_str());
            }
        }
        OFString sliceThkStr;
        if (ds0->findAndGetOFString(DCM_SliceThickness, sliceThkStr).good()) {
            sliceThickness = std::atof(sliceThkStr.c_str());
        }
    }

    double N[3] = { 0,0,1 };
    if (series.slices.front().hasOrientation) {
        auto& m = series.slices.front();
        double R[3]; double C[3];
        for (int i = 0; i < 3; ++i) {
            R[i] = m.orientationRow[i];
            C[i] = m.orientationCol[i];
        }
        // 计算法向量
        N[0] = R[1] * C[2] - R[2] * C[1];
        N[1] = R[2] * C[0] - R[0] * C[2];
        N[2] = R[0] * C[1] - R[1] * C[0];
        double len = std::sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]);
        if (len > 1e-6) {
            N[0] /= len;
            N[1] /= len;
            N[2] /= len;
        }
        for (int i = 0; i < 3; ++i) {
            out.orientationRow[i] = R[i];
            out.orientationCol[i] = C[i];
        }
    }

    double sliceSpacing = computeSliceSpacing(series, N, sliceThickness);

    out.spacing[0] = spacingX;
    out.spacing[1] = spacingY;
    out.spacing[2] = sliceSpacing;

    for (int i = 0; i < 3; ++i)
        out.origin[i] = series.slices.front().imagePosition[i];

    // 分配缓冲
    out.buffer.resize(out.width * out.height * out.depth);

    // 窗宽/窗位 (如果存在) 用于归一化
    double windowCenter = 0.0, windowWidth = 0.0;
    bool hasWW = false;
    {
        OFString wcStr, wwStr;
        if (ds0->findAndGetOFString(DCM_WindowCenter, wcStr).good() &&
            ds0->findAndGetOFString(DCM_WindowWidth, wwStr).good()) {
            windowCenter = std::atof(wcStr.c_str());
            windowWidth = std::atof(wwStr.c_str());
            hasWW = (windowWidth > 0.0);
        }
    }

    for (size_t zi = 0; zi < series.slices.size(); ++zi) {
        DcmFileFormat ffslice;
        if (!ffslice.loadFile(series.slices[zi].filepath.c_str()).good())
            continue;

        DcmDataset* dss = ffslice.getDataset();
        if (!dss) continue;

        // 解码图像
        DicomImage dcmImg(dss, EXS_Unknown);
        if (dcmImg.getStatus() != EIS_Normal) continue;

        if (hasWW) {
            dcmImg.setWindow(windowCenter, windowWidth);
        }

        int depthBits = dcmImg.getDepth();
        const void* pixelData = dcmImg.getOutputData(8 /*bits per sample*/);
        if (!pixelData) continue;

        const uint8_t* src = reinterpret_cast<const uint8_t*>(pixelData);

		size_t dstOffset = zi * out.width * out.height;
		std::memcpy(out.buffer.data() + dstOffset, src, out.width * out.height);
	}

    //重建体数据的时候不做插值，渲染的时候做插值

    //if (enableResampling && targetZSpacing > 0 && out.spacing[2] > targetZSpacing) 
    //{   
    //    switch (method) 
    //    {
    //    case InterpolationMethod::CUBIC_SPLINE:
    //        out = resampleZWithCubic(out, targetZSpacing);
    //        break;
    //    case InterpolationMethod::TRILINEAR:
    //        // 三线性插值的实现
    //        // out = resampleZWithTrilinear(out, targetZSpacing);
    //        out = resampleZWithCubic(out, targetZSpacing);
    //        break;
    //    case InterpolationMethod::NEAREST_NEIGHBOR:
    //        // 最近邻插值的实现
    //        // out = resampleZWithNearest(out, targetZSpacing);
    //        out = resampleZWithCubic(out, targetZSpacing);
    //        break;
    //    }
    //}

    return out;
}

VolumeBuildResult buildVolume_none(const SeriesData& series) {
    return buildVolume_none(series, false, 0.0f, InterpolationMethod::CUBIC_SPLINE);
}

// ============================================================
//  床板伪影去除
// ============================================================
void removeBedArtifact(VolumeBuildResult& volume, uint8_t bodyThresh)
{
    const uint32_t W = volume.width;
    const uint32_t H = volume.height;
    const uint32_t D = volume.depth;
    if (W == 0 || H == 0 || D == 0) return;

    const size_t sliceSize = (size_t)W * H;

    // 预计算圆形 FOV 掩码（CT 重建圆：半径 = 短边 / 2）
    const float cx     = W * 0.5f;
    const float cy     = H * 0.5f;
    const float radius = (std::min)(W, H) * 0.5f;
    const float r2     = radius * radius;

    // 预分配所有缓冲区（循环外一次性分配，避免每片重复 new/delete）
    std::vector<uint8_t>  fgMap(sliceSize);       // 0=背景, 1=前景
    std::vector<int32_t>  labelMap(sliceSize);    // -1=背景, 0=未标记前景, >0=已标记
    std::vector<uint8_t>  trueBg(sliceSize);      // 0=未访问, 1=真实背景
    std::vector<int32_t>  compSize;               // compSize[label-1] = 像素数
    compSize.reserve(256);                         // 预留空间，避免反复 realloc
    // 用预分配数组做栈（替代 std::queue，避免 deque 的小块分配开销）
    std::vector<int32_t>  stackBuf(sliceSize);    // 存储线性索引 y*W+x

    const int ddx[4] = { 1, -1, 0,  0 };
    const int ddy[4] = { 0,  0, 1, -1 };

    uint64_t removedVoxels = 0;

    // 逐轴向切片处理
    for (uint32_t z = 0; z < D; ++z)
    {
        uint8_t* sl = volume.buffer.data() + (size_t)z * sliceSize;

        // ---- Step 1: 圆形 FOV + 阈值 → 二值前景图 ----
        bool anyFg = false;
        for (uint32_t y = 0; y < H; ++y) {
            for (uint32_t x = 0; x < W; ++x) {
                float dx = (float)x - cx;
                float dy = (float)y - cy;
                size_t idx = (size_t)y * W + x;
                if (dx * dx + dy * dy <= r2 && sl[idx] > bodyThresh) {
                    fgMap[idx] = 1;
                    anyFg = true;
                } else {
                    fgMap[idx] = 0;
                }
            }
        }
        if (!anyFg) {
            std::fill(sl, sl + sliceSize, uint8_t(0));
            removedVoxels += sliceSize;
            continue;
        }

        // ---- Step 2: 4-连通栈式标记前景连通域，找最大域 ----
        // 重置 labelMap: 前景=0, 背景=-1
        for (size_t i = 0; i < sliceSize; ++i)
            labelMap[i] = fgMap[i] ? 0 : -1;

        compSize.clear();
        int curLabel = 0;

        for (uint32_t y = 0; y < H; ++y) {
            for (uint32_t x = 0; x < W; ++x) {
                size_t seedIdx = (size_t)y * W + x;
                if (labelMap[seedIdx] != 0) continue;  // 背景或已标记
                ++curLabel;
                compSize.push_back(0);

                // 栈式 DFS（用预分配数组，无动态内存分配）
                int32_t sp = 0;
                stackBuf[sp++] = (int32_t)seedIdx;
                labelMap[seedIdx] = curLabel;
                while (sp > 0) {
                    int32_t cur = stackBuf[--sp];
                    int fy = cur / (int)W;
                    int fx = cur % (int)W;
                    compSize.back()++;
                    for (int k = 0; k < 4; ++k) {
                        int ny = fy + ddy[k];
                        int nx = fx + ddx[k];
                        if (ny < 0 || ny >= (int)H || nx < 0 || nx >= (int)W) continue;
                        int32_t nIdx = ny * (int)W + nx;
                        if (labelMap[nIdx] != 0) continue;
                        labelMap[nIdx] = curLabel;
                        stackBuf[sp++] = nIdx;
                    }
                }
            }
        }

        if (curLabel == 0) {
            std::fill(sl, sl + sliceSize, uint8_t(0));
            removedVoxels += sliceSize;
            continue;
        }

        // 最大连通域编号
        int maxLabel = 1;
        for (int i = 2; i <= curLabel; ++i)
            if (compSize[i - 1] > compSize[maxLabel - 1]) maxLabel = i;

        // ---- Step 3: 孔洞填充 ----
        // 从图像四周边界出发，栈式 DFS 穿越"非最大域"背景 → 真实背景
        std::fill(trueBg.begin(), trueBg.end(), uint8_t(0));

        int32_t sp = 0;
        auto seedBorder = [&](int y, int x) {
            int32_t idx = y * (int)W + x;
            if (labelMap[idx] != maxLabel && !trueBg[idx]) {
                trueBg[idx] = 1;
                stackBuf[sp++] = idx;
            }
        };
        for (uint32_t x = 0; x < W; ++x) { seedBorder(0,     (int)x); seedBorder((int)(H-1), (int)x); }
        for (uint32_t y = 1; y < H - 1; ++y) { seedBorder((int)y, 0); seedBorder((int)y, (int)(W-1)); }
        while (sp > 0) {
            int32_t cur = stackBuf[--sp];
            int fy = cur / (int)W;
            int fx = cur % (int)W;
            for (int k = 0; k < 4; ++k) {
                int ny = fy + ddy[k];
                int nx = fx + ddx[k];
                if (ny < 0 || ny >= (int)H || nx < 0 || nx >= (int)W) continue;
                int32_t nIdx = ny * (int)W + nx;
                if (trueBg[nIdx]) continue;
                if (labelMap[nIdx] == maxLabel) continue;  // 不穿越身体
                trueBg[nIdx] = 1;
                stackBuf[sp++] = nIdx;
            }
        }

        // ---- Step 4: 应用掩码 ----
        for (size_t i = 0; i < sliceSize; ++i) {
            bool isBody = (labelMap[i] == maxLabel) || (!trueBg[i] && labelMap[i] == -1);
            if (!isBody) {
                sl[i] = 0;
                ++removedVoxels;
            }
        }

        // 每 20 片输出一次进度
        if ((z + 1) % 20 == 0 || z + 1 == D) {
            std::cout << "  床板去除进度: " << (z + 1) << "/" << D << " 切片" << std::endl;
        }
    }

    std::cout << "[床板去除] 完成: " << W << "x" << H << "x" << D
              << "，清零体素 " << removedVoxels << " / "
              << (size_t)W * H * D << std::endl;
}