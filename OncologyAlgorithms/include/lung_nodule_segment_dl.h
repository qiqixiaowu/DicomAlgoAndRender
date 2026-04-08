#pragma once
/**
 * @file lung_nodule_segment_dl.h
 * @brief 深度学习肺结节分割接口（3D nnUNet，手写 CUDA/cuDNN 推理引擎）
 *
 * 架构说明
 * ─────────
 * 本模块实现了一个轻量级的 3D U-Net（nnUNet 风格）手写 CUDA 推理引擎，
 * 不依赖 ONNX Runtime 或 TensorRT，直接通过 cuDNN 执行卷积运算。
 *
 * 网络结构（5 层 Encoder-Decoder）
 * ─────────────────────────────────
 *  Input(1ch, 128³)
 *    │
 *  Encoder1: Conv(1→32) + IN + LReLU  × 2          ↗ skip1
 *  Encoder2: Conv(32→64, stride2) + IN + LReLU × 2  ↗ skip2
 *  Encoder3: Conv(64→128,stride2) + IN + LReLU × 2  ↗ skip3
 *  Encoder4: Conv(128→256,stride2)+ IN + LReLU × 2  ↗ skip4
 *  Bottleneck: Conv(256→320,stride2)+IN+LReLU × 2
 *    │
 *  Decoder4: DeConv(320→256) + Cat(skip4) + Conv×2
 *  Decoder3: DeConv(256→128) + Cat(skip3) + Conv×2
 *  Decoder2: DeConv(128→64)  + Cat(skip2) + Conv×2
 *  Decoder1: DeConv(64→32)   + Cat(skip1) + Conv×2
 *  Output: Conv(32→2) → Softmax → Argmax → Binary Mask
 *
 * 每个 Conv 块 = 3D Conv + Instance Normalization + LeakyReLU(α=0.01)
 *
 * 模型权重格式
 * ─────────────
 * 从 .bin 文件加载，内部格式为按层名称索引的 float32 块，
 * 文件头部为层数 (uint32) + 每层 (name_length(uint16) + name + size(uint64) + data)。
 *
 * 使用方法
 * ─────────
 * LungNoduleDLSegment seg;
 * seg.loadWeights("path/to/weights.bin");
 * seg.segment(data, info, seedPoint, result);
 */

#include "oncology_types.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace Onc {

// ─────────────────────────────────────────────
//  权重字典类型（层名 → float 权重向量）
// ─────────────────────────────────────────────
using ParamDict = std::unordered_map<std::string, std::vector<float>>;

// ─────────────────────────────────────────────
//  DL 配置
// ─────────────────────────────────────────────
struct LungNoduleDLConfig {
    std::string weightsPath;     ///< .bin 权重文件路径
    int   patchSize[3]  = {128, 128, 128}; ///< 推理 patch 尺寸
    int   inChannels    = 1;     ///< 输入通道数
    int   outChannels   = 2;     ///< 输出通道数（背景/前景）
    float huMin         = -1000.f; ///< HU 归一化下限
    float huMax         = 400.f;   ///< HU 归一化上限
    bool  useCUDA       = true;    ///< 是否使用 GPU
};

// ─────────────────────────────────────────────
//  主接口类
// ─────────────────────────────────────────────
class LungNoduleDLSegment {
public:
    explicit LungNoduleDLSegment(const LungNoduleDLConfig& cfg = {});
    ~LungNoduleDLSegment();

    /**
     * @brief 加载 .bin 格式模型权重
     * @param path  权重文件路径
     * @return false 表示文件不存在或格式错误
     */
    bool loadWeights(const std::string& path);

    /**
     * @brief 执行 DL 肺结节分割
     * @param data       CT 图像（short，HU 值）
     * @param info       图像元信息
     * @param seedPoint  种子点（用于确定 patch 中心）
     * @param result     [out] 分割结果
     * @param progress   进度回调
     * @return false 表示推理失败
     */
    bool segment(const short*      data,
                 const ImageInfo&  info,
                 const Point3i&    seedPoint,
                 SegmentResult&    result,
                 ProgressCallback  progress = nullptr);

    /** 检查 CUDA 是否可用及显存是否足够 */
    bool checkGPUAvailable(int requiredMB = 2048) const;

    /** 估计推理所需 GPU 显存（MB） */
    int estimateGPUMemMB() const;

private:
    LungNoduleDLConfig m_cfg;
    ParamDict          m_params;
    bool               m_weightsLoaded = false;

    // 从二进制文件读取权重
    bool readBinWeights(const std::string& path, ParamDict& params);

    // 图像预处理：裁剪 patch + 归一化
    void extractAndNormalizePatch(const short*     data,
                                  const ImageInfo& info,
                                  const Point3i&   center,
                                  std::vector<float>& patch,
                                  int              patchOrigin[3]);

    // CPU 推理路径（当 CUDA 不可用时的回退）
    bool inferCPU(const std::vector<float>& patch,
                  std::vector<float>&       logits);

    // GPU（cuDNN）推理路径
    bool inferGPU(const std::vector<float>& patch,
                  std::vector<float>&       logits);

    // Argmax + 还原到全图坐标
    void postProcess(const std::vector<float>& logits,
                     const ImageInfo&          info,
                     const int                 patchOrigin[3],
                     SegmentResult&            result);

    // 3D 连通域过滤（只保留含种子点的最长连通域）
    void connectedFilter(SegmentResult&    result,
                         const ImageInfo&  info,
                         const Point3i&    seed);
};

} // namespace Onc
