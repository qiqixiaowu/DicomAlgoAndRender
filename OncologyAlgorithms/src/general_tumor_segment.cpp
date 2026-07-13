/**
 * @file general_tumor_segment.cpp
 * @brief 通用半自动肿瘤分割实现
 *
 * 策略路由：
 *   Auto → resolveMethod() 根据 GPU/模型文件决定实际策略
 *   传统路径: K-means 初始化 + Random Walker 精化
 *   DL 路径: 调用 nnInteractive 或 SAM2 接口（需外部链接 TRT/ONNX）
 */

#include "general_tumor_segment.h"
#include "kmeans_segmentation.h"
#include "gmm_segmentation.h"
#include "random_walker_segmentation.h"
#include "sam2_segment.h"
#include <iostream>
#include <cstring>
#include <algorithm>

// ── GPU 内存查询（可选：无 CUDA 时返回 0）─────────────────
#if defined(USE_CUDA)
#include <cuda_runtime.h>
static double QueryGPUMemGB() {
    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0) return 0.0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return static_cast<double>(prop.totalGlobalMem) / (1024.0*1024.0*1024.0);
}
#else
static double QueryGPUMemGB() { return 0.0; }
#endif

namespace Onc {

// ────────────────────────────────────────
//  构造 / 析构
// ────────────────────────────────────────
GeneralTumorSegmentation::GeneralTumorSegmentation(const GeneralSegmentConfig& cfg)
    : m_cfg(cfg) {}

GeneralTumorSegmentation::~GeneralTumorSegmentation() = default;

// ────────────────────────────────────────
//  资源估算
// ────────────────────────────────────────
void GeneralTumorSegmentation::estimateResources(
    const ImageInfo& info, int& cpuMB, int& gpuMB) const
{
    // 粗估：图像体积 × sizeof(float) × 若干中间缓冲系数
    long long voxels = info.totalVoxels();
    cpuMB = static_cast<int>(voxels * 4 * 8 / (1024*1024)) + 64; // ~8 个 float 缓冲
    gpuMB = 0;
    SegmentMethod method = resolveMethod(info);
    if (method == SegmentMethod::NnInteractive) gpuMB = 8192;
    else if (method == SegmentMethod::SAM2)     gpuMB = 4096;
}

// ────────────────────────────────────────
//  方法选择
// ────────────────────────────────────────
SegmentMethod GeneralTumorSegmentation::resolveMethod(const ImageInfo& /*info*/) const {
    if (m_cfg.method != SegmentMethod::Auto) return m_cfg.method;

    double gpuGB = QueryGPUMemGB();
    double gpuMB = gpuGB * 1024.0;

    if (gpuMB >= m_cfg.minGPUMemForNnInt && !m_cfg.nninteractiveModelPath.empty()) {
        // 检测模型文件是否存在（简单 fopen）
        FILE* f = fopen(m_cfg.nninteractiveModelPath.c_str(), "rb");
        if (f) { fclose(f); return SegmentMethod::NnInteractive; }
    }
    if (gpuMB >= m_cfg.minGPUMemForSAM2 && !m_cfg.sam2ModelPath.empty()) {
        FILE* f = fopen(m_cfg.sam2ModelPath.c_str(), "rb");
        if (f) { fclose(f); return SegmentMethod::SAM2; }
    }
    return SegmentMethod::RandomWalker; // 默认 CPU 传统路径
}

// ────────────────────────────────────────
//  传统路径实现
// ────────────────────────────────────────
bool GeneralTumorSegmentation::segmentTraditional(
    const short*     data,
    const ImageInfo& info,
    const Point3i&   ldStart,
    const Point3i&   ldEnd,
    SegmentResult&   result,
    ProgressCallback progress)
{
    // Step1: K-means 粗分割，以线段中点作为种子
    Point3i seedPt = {
        (ldStart.x + ldEnd.x) / 2,
        (ldStart.y + ldEnd.y) / 2,
        (ldStart.z + ldEnd.z) / 2
    };

    KMeansSegmentation::Params kp;
    kp.numClusters = 2;
    kp.roiMargin   = static_cast<int>(
        ldStart.distanceTo(ldEnd) / 1.5f + 15.f);
    // 借用 Point3f 的 distanceTo
    float lineDx = static_cast<float>(ldEnd.x - ldStart.x)
                 * static_cast<float>(info.spacing[0]);
    float lineDy = static_cast<float>(ldEnd.y - ldStart.y)
                 * static_cast<float>(info.spacing[1]);
    float lineDz = static_cast<float>(ldEnd.z - ldStart.z)
                 * static_cast<float>(info.spacing[2]);
    float lineLen = std::sqrt(lineDx*lineDx + lineDy*lineDy + lineDz*lineDz);
    kp.roiMargin = std::max(15, static_cast<int>(lineLen / info.spacing[0]) + 10);

    KMeansSegmentation kmeans(kp);
    SegmentResult kResult;
    if (!kmeans.segment(data, info, seedPt, kResult)) {
        // K-means 失败，直接上 Random Walker
        kResult.clear();
    }
    if (progress) progress(0.3);

    // Step2: Random Walker 精化
    RandomWalkerSegmentation::Params rp;
    rp.beta       = m_cfg.randomWalkerBeta;
    rp.roiMargin  = kp.roiMargin;

    RandomWalkerSegmentation rw(rp);
    bool ok = rw.segment(data, info, ldStart, ldEnd, m_cfg.drawDir, result);
    if (progress) progress(1.0);
    return ok;
}

// ────────────────────────────────────────
//  DL 路径（占位 / 适配层）
// ────────────────────────────────────────
bool GeneralTumorSegmentation::segmentNnInteractive(
    const short*     data,
    const ImageInfo& info,
    const Point3i&   ldStart,
    const Point3i&   ldEnd,
    SegmentResult&   result,
    ProgressCallback progress)
{
    // 此处需链接 TensorRT 引擎
    // 实现步骤：
    //   1. 加载 m_cfg.nninteractiveModelPath (.engine)
    //   2. 从长轴线段计算 bounding-box prompt
    //   3. 裁剪局部 ROI patch（128³ 或 192³）
    //   4. normalize → TRT 推理 → argmax → 还原到全图坐标
    //   5. 连通域过滤
    //
    // 当外部 TensorRT 运行时不可用时，回退到传统算法
    std::cout << "[GeneralTumorSeg] NnInteractive path: "
              << "TRT engine = " << m_cfg.nninteractiveModelPath << "\n"
              << "  Falling back to traditional (TRT runtime not linked)\n";
    return segmentTraditional(data, info, ldStart, ldEnd, result, progress);
}

bool GeneralTumorSegmentation::segmentSAM2(
    const short*     data,
    const ImageInfo& info,
    const Point3i&   ldStart,
    const Point3i&   ldEnd,
    SegmentResult&   result,
    ProgressCallback progress)
{
#ifdef USE_ONNXRUNTIME
    // 延迟初始化 SAM2
    if (!m_sam2) {
        SAM2Config cfg;
        cfg.setModelDir(m_cfg.sam2ModelPath);
        cfg.useCUDA = (QueryGPUMemGB() > 0);
        m_sam2 = std::make_unique<SAM2Segmenter>();
        if (!m_sam2->initialize(cfg)) {
            std::cerr << "[GeneralTumorSeg] SAM2 init failed, falling back\n";
            m_sam2.reset();
            return segmentTraditional(data, info, ldStart, ldEnd, result, progress);
        }
    }

    bool ok = m_sam2->segment(data, info, ldStart, ldEnd, m_cfg.drawDir,
                              result, true, progress);
    if (!ok) {
        std::cerr << "[GeneralTumorSeg] SAM2 segment failed, falling back\n";
        return segmentTraditional(data, info, ldStart, ldEnd, result, progress);
    }
    return true;
#else
    std::cout << "[GeneralTumorSeg] SAM2 path: "
              << "model = " << m_cfg.sam2ModelPath << "\n"
              << "  Falling back to traditional (ONNX Runtime not linked)\n";
    return segmentTraditional(data, info, ldStart, ldEnd, result, progress);
#endif
}

// ────────────────────────────────────────
//  公开接口：short* 版本
// ────────────────────────────────────────
bool GeneralTumorSegmentation::segment(
    const short*     data,
    const ImageInfo& info,
    const Point3i&   ldStart,
    const Point3i&   ldEnd,
    SegmentResult&   result,
    ProgressCallback progress)
{
    if (!data || info.totalVoxels() == 0) return false;
    m_lastMethod = resolveMethod(info);
    std::cout << "[GeneralTumorSeg] Using method: "
              << static_cast<int>(m_lastMethod) << "\n";

    switch (m_lastMethod) {
    case SegmentMethod::NnInteractive:
        return segmentNnInteractive(data, info, ldStart, ldEnd, result, progress);
    case SegmentMethod::SAM2:
        return segmentSAM2(data, info, ldStart, ldEnd, result, progress);
    default:
        return segmentTraditional(data, info, ldStart, ldEnd, result, progress);
    }
}

// ────────────────────────────────────────
//  公开接口：unsigned short* 版本（转换后复用 short 路径）
// ────────────────────────────────────────
bool GeneralTumorSegmentation::segment(
    const unsigned short* data,
    const ImageInfo&      info,
    const Point3i&        ldStart,
    const Point3i&        ldEnd,
    SegmentResult&        result,
    ProgressCallback      progress)
{
    // 将 unsigned short 线性转换为 short（偏移 32768）
    int N = info.totalVoxels();
    std::vector<short> converted(N);
    for (int i = 0; i < N; ++i)
        converted[i] = static_cast<short>(
            static_cast<int>(data[i]) - 32768);
    return segment(converted.data(), info, ldStart, ldEnd, result, progress);
}

// ────────────────────────────────────────
//  跨时间点传播
// ────────────────────────────────────────
bool GeneralTumorSegmentation::spreadSegmentation(
    const short*         data,
    const ImageInfo&     info,
    const Point3i&       prevLdStartInCur,
    const Point3i&       prevLdEndInCur,
    const SegmentResult& /*prevSegInCur*/,
    Point3i&             newLdStart,
    Point3i&             newLdEnd,
    SegmentResult&       newSeg,
    ProgressCallback     progress)
{
    // 策略：以上一时间点线段（已配准到当前坐标）直接作为新的画线指令执行分割
    // 若新分割成功 → 返回新结果
    // 若失败 → 返回 false（前端使用上一时间点结果作为"虚拟点"显示）

    bool ok = segment(data, info, prevLdStartInCur, prevLdEndInCur, newSeg, progress);
    if (ok) {
        newLdStart = prevLdStartInCur;
        newLdEnd   = prevLdEndInCur;
    }
    return ok;
}

} // namespace Onc
