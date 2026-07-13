#pragma once
/**
 * @file general_tumor_segment.h
 * @brief 通用半自动肿瘤分割接口（传统算法 + DL 适配层）
 *
 * 分割策略决策树
 * ──────────────
 *   1. 若 GPU VRAM >= 8GB && 可找到 nnInteractive 模型文件
 *      → 使用 nnInteractive (TensorRT) 交互式深度学习分割
 *   2. 否则若 GPU VRAM >= 4GB && SAM2 模型可用
 *      → 使用 SAM2 粗分割 + Random Walker 精化
 *   3. 否则（CPU 路径）
 *      → 使用传统算法（K-means 初始化 + GMM 前景提取 + Random Walker 精化）
 *
 * 跨时间点传播
 * ────────────
 * SpreadSegmentation() 接受上一时间点分割结果（已配准到当前时间点坐标），
 * 将其作为 Random Walker 的初始前景种子继续优化。
 */

#include "oncology_types.h"
#include "sam2_segment.h"
#include <string>
#include <memory>

namespace Onc {

// ─────────────────────────────────────────────
//  分割方法枚举
// ─────────────────────────────────────────────
enum class SegmentMethod {
    Auto,          ///< 自动选择（推荐）
    KMeansGMM,     ///< 纯传统：K-means + GMM
    RandomWalker,  ///< 传统：Random Walker
    SAM2,          ///< SAM2 (需要 ONNX/TRT 环境)
    NnInteractive, ///< nnInteractive TensorRT (需要 TRT 引擎)
};

// ─────────────────────────────────────────────
//  配置结构
// ─────────────────────────────────────────────
struct GeneralSegmentConfig {
    SegmentMethod method      = SegmentMethod::Auto;
    ImageModality modality    = ImageModality::CT;
    DrawDirection drawDir     = DrawDirection::Axial;

    // DL 模型路径（Auto/SAM2/NnInteractive 时使用）
    std::string nninteractiveModelPath; ///< .engine 文件路径
    std::string sam2ModelPath;          ///< ONNX 模型路径

    // 传统算法参数
    float randomWalkerBeta  = 90.f;
    int   kmeansRoiMargin   = 20;

    // GPU 资源阈值（MB）
    int   minGPUMemForNnInt = 8192;  ///< 使用 nnInteractive 所需最低显存
    int   minGPUMemForSAM2  = 4096;  ///< 使用 SAM2 所需最低显存
};

// ─────────────────────────────────────────────
//  主接口类
// ─────────────────────────────────────────────
class GeneralTumorSegmentation {
public:
    explicit GeneralTumorSegmentation(const GeneralSegmentConfig& cfg = {});
    ~GeneralTumorSegmentation();

    /**
     * @brief 半自动分割：用户在某断面画一条长轴线
     * @param data      原始图像（short 或 unsigned short）
     * @param info      图像元信息
     * @param ldStart   长轴线段起点（图像坐标）
     * @param ldEnd     长轴线段终点
     * @param result    [out] 分割体素集合
     * @param progress  进度回调
     * @return false 表示分割失败
     */
    bool segment(const short*     data,
                 const ImageInfo& info,
                 const Point3i&   ldStart,
                 const Point3i&   ldEnd,
                 SegmentResult&   result,
                 ProgressCallback progress = nullptr);

    bool segment(const unsigned short* data,
                 const ImageInfo&      info,
                 const Point3i&        ldStart,
                 const Point3i&        ldEnd,
                 SegmentResult&        result,
                 ProgressCallback      progress = nullptr);

    /**
     * @brief 跨时间点传播：将上一时间点（已配准到当前坐标的）分割结果传播
     * @param data              当前时间点图像
     * @param info              当前时间点图像元信息
     * @param prevLdStartInCur  上一时间点长轴起点（已变换到当前坐标）
     * @param prevLdEndInCur    上一时间点长轴终点（已变换到当前坐标）
     * @param prevSegInCur      上一时间点分割结果（已变换到当前坐标）
     * @param newLdStart        [out] 新的长轴起点
     * @param newLdEnd          [out] 新的长轴终点
     * @param newSeg            [out] 新的分割结果
     * @return false 表示传播失败（前端需显示"虚拟点"并使用旧结果）
     */
    bool spreadSegmentation(
        const short*         data,
        const ImageInfo&     info,
        const Point3i&       prevLdStartInCur,
        const Point3i&       prevLdEndInCur,
        const SegmentResult& prevSegInCur,
        Point3i&             newLdStart,
        Point3i&             newLdEnd,
        SegmentResult&       newSeg,
        ProgressCallback     progress = nullptr);

    /** 查询完成上一次分割实际使用的方法 */
    SegmentMethod lastUsedMethod() const { return m_lastMethod; }

    /** 估计所需 CPU/GPU 内存（MB） */
    void estimateResources(const ImageInfo& info, int& cpuMB, int& gpuMB) const;

private:
    GeneralSegmentConfig m_cfg;
    SegmentMethod        m_lastMethod = SegmentMethod::Auto;
    std::unique_ptr<SAM2Segmenter> m_sam2;

    // 传统算法路径
    bool segmentTraditional(const short*     data,
                            const ImageInfo& info,
                            const Point3i&   ldStart,
                            const Point3i&   ldEnd,
                            SegmentResult&   result,
                            ProgressCallback progress);

    // DL 路径（占位，需链接对应运行时）
    bool segmentNnInteractive(const short*     data,
                              const ImageInfo& info,
                              const Point3i&   ldStart,
                              const Point3i&   ldEnd,
                              SegmentResult&   result,
                              ProgressCallback progress);

    bool segmentSAM2(const short*     data,
                     const ImageInfo& info,
                     const Point3i&   ldStart,
                     const Point3i&   ldEnd,
                     SegmentResult&   result,
                     ProgressCallback progress);

    // 选择实际方法
    SegmentMethod resolveMethod(const ImageInfo& info) const;
};

} // namespace Onc
