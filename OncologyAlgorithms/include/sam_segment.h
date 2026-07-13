#pragma once
/**
 * @file sam_segment.h
 * @brief SAM (Segment Anything Model) 医学影像交互式分割
 *
 * 架构概览
 * ────────
 *   3个 ONNX 子模型串联:
 *     Image Encoder → Prompt Encoder → Mask Decoder
 *
 *   1. Image Encoder (ViT-H):
 *        输入 [1,3,1024,1024] → 图像嵌入向量
 *   2. Prompt Encoder:
 *        将点/框提示编码为稀疏嵌入向量
 *   3. Mask Decoder (Transformer + MLP):
 *        接收提示嵌入 + 图像嵌入 → 预测mask + IoU score
 *
 *   支持双向3D传播：从用户标注层向前/向后逐层推理
 *   帧间传播采用 IoU 匹配 + 形态学引导策略
 *
 * 依赖
 * ────
 *   - ONNX Runtime C++ API (onnxruntime_cxx_api.h)
 *   - 可选 CUDA Execution Provider (GPU 加速)
 *   - 3个 ONNX 模型文件 (sam_vit_h)
 *
 * 编译条件
 * ────────
 *   USE_ONNXRUNTIME 宏控制。未定义时所有类退化为占位 stub。
 *
 * @date 2023-06
 */

#include "oncology_types.h"
#include <vector>
#include <string>
#include <array>
#include <memory>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <chrono>

#ifdef USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace Onc {

// ─────────────────────────────────────────────
//  ONNX 节点描述
// ─────────────────────────────────────────────
struct OnnxNode {
    std::vector<int64_t> dim;
    std::string name;
};

// ─────────────────────────────────────────────
//  SAM 推理参数
// ─────────────────────────────────────────────
struct SAMParams {
    int    promptType = 0;                       ///< 0=box, 1=point
    std::array<int, 4> promptBox = {0,0,0,0};    ///< x, y, width, height
    std::vector<std::pair<int,int>> promptPoints; ///< 提示点坐标
    std::vector<float> pointLabels;               ///< 点标签: 1=前景, 0=背景, -1=padding
};

// ─────────────────────────────────────────────
//  SAM 模型配置
// ─────────────────────────────────────────────
struct SAMConfig {
    std::string imgEncoderPath;    ///< sam_vit_h_encoder ONNX 路径
    std::string promptEncoderPath; ///< prompt_encoder ONNX 路径
    std::string maskDecoderPath;   ///< mask_decoder ONNX 路径

    int  encoderInputDim = 1024;   ///< encoder 输入分辨率 (ViT-H)
    bool useCUDA         = true;   ///< 是否使用 CUDA EP
    int  deviceId        = 0;      ///< GPU 设备号
    int  intraOpThreads  = 2;      ///< ONNX Runtime 内部线程数

    // 窗宽窗位（CT 默认）
    float windowCenter = 65.6f;
    float windowWidth  = 500.0f;

    /** 从目录路径自动补全3个ONNX模型路径 */
    void setModelDir(const std::string& dir, const std::string& prefix = "sam_vit_h") {
        imgEncoderPath   = dir + "/" + prefix + "_encoder.onnx";
        promptEncoderPath = dir + "/" + prefix + "_prompt_encoder.onnx";
        maskDecoderPath   = dir + "/" + prefix + "_mask_decoder.onnx";
    }
};

#ifdef USE_ONNXRUNTIME

// ─────────────────────────────────────────────
//  SAM 单方向推理引擎
// ─────────────────────────────────────────────
class SAMEngine {

public:
    SAMEngine();
    ~SAMEngine();

    SAMEngine(const SAMEngine&) = delete;
    SAMEngine& operator=(const SAMEngine&) = delete;

    /** 初始化：加载3个ONNX模型、创建Session */
    bool initialize(const SAMConfig& cfg);

    /** 设置图像信息 */
    void setImage(const short* pImage, const int iSize[3],
                  const double dSpacing[3]);

    /** 设置提示参数 */
    void setParams(const SAMParams& params);

    /** 预提取指定帧的 Image Encoder 特征（可并行） */
    bool extractFeatures(const std::vector<int>& frameIndices,
                         const short* pImage);

    /** 单帧推理（需先 extractFeatures） */
    bool inference(int frameIdx,
                   std::unordered_map<int, std::vector<float>>& maskPredMap,
                   bool firstFrame);

    /** 重置帧间记忆 */
    void reset();

    /** 查询是否已初始化 */
    bool isInitialized() const { return m_initialized; }

private:
    // ── 配置 ──
    SAMConfig m_cfg;
    int m_imageSize = 1024;
    std::vector<std::vector<size_t>> m_featSizes = {{256,256},{128,128},{64,64}};

    // ── 图像 ──
    const short* m_pImage = nullptr;
    int m_iSize[3]     = {0, 0, 0};
    double m_dSpacing[3] = {1.0, 1.0, 1.0};

    // ── 参数 ──
    SAMParams m_params;

    // ── ONNX Runtime 对象 ──
    bool m_initialized = false;
    Ort::MemoryInfo m_memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Env m_imgEncoderEnv  {ORT_LOGGING_LEVEL_WARNING, "img_encoder"};
    Ort::Env m_promptEncoderEnv{ORT_LOGGING_LEVEL_WARNING, "prompt_encoder"};
    Ort::Env m_maskDecoderEnv  {ORT_LOGGING_LEVEL_WARNING, "mask_decoder"};

    std::unique_ptr<Ort::Session> m_imgEncoderSession;
    std::unique_ptr<Ort::Session> m_promptEncoderSession;
    std::unique_ptr<Ort::Session> m_maskDecoderSession;

    Ort::SessionOptions m_imgEncoderOpts;
    Ort::SessionOptions m_promptEncoderOpts;
    Ort::SessionOptions m_maskDecoderOpts;

    // ── ONNX 节点信息 ──
    std::vector<OnnxNode> m_imgEncoderInputs, m_imgEncoderOutputs;
    std::vector<OnnxNode> m_promptEncoderInputs, m_promptEncoderOutputs;
    std::vector<OnnxNode> m_maskDecoderInputs, m_maskDecoderOutputs;

    // ── 缓存 ──
    std::vector<std::vector<Ort::Value>> m_imgEncoderOuts; // 按帧索引

    // ── 子推理步骤 ──
    bool imgEncoderInfer(const std::vector<Ort::Value>& input,
                         std::vector<Ort::Value>& output);
    bool promptEncoderInfer(const SAMParams& params,
                            std::vector<Ort::Value>& output);
    bool maskDecoderInfer(std::vector<Ort::Value>& encoderOut,
                          std::vector<Ort::Value>& promptOut,
                          std::vector<Ort::Value>& output);

    // ── 预处理 ──
    bool resizeAndNormalize(int sliceIndex, const short* pImage,
                            std::vector<float>& outData);

    // ── 后处理 ──
    bool getOutput(std::vector<Ort::Value>& decoderOut, int frameIdx,
                   std::unordered_map<int, std::vector<float>>& maskPredMap);

    // ── 工具方法 ──
    void loadSessionNodes(Ort::Session* session,
                          std::vector<OnnxNode>& inputs,
                          std::vector<OnnxNode>& outputs);
};

// ─────────────────────────────────────────────
//  SAM 3D 分割器（双向传播）
// ─────────────────────────────────────────────
class SAMSegmenter {
public:
    SAMSegmenter();
    ~SAMSegmenter();

    /** 初始化：加载模型 */
    bool initialize(const SAMConfig& cfg);

    /**
     * @brief 3D 交互式分割
     * @param data       原始图像 short 数组
     * @param info       图像元信息
     * @param ldStart    长轴线段起点
     * @param ldEnd      长轴线段终点
     * @param drawDir    绘制方向(Axial/Sagittal/Coronal)
     * @param result     [out] 分割体素集合
     * @param is3D       true=双向3D传播, false=仅标注帧2D
     * @param progress   进度回调
     */
    bool segment(const short*     data,
                 const ImageInfo& info,
                 const Point3i&   ldStart,
                 const Point3i&   ldEnd,
                 DrawDirection    drawDir,
                 SegmentResult&   result,
                 bool             is3D = true,
                 ProgressCallback progress = nullptr);

    bool isInitialized() const { return m_initialized; }

private:
    SAMConfig m_cfg;
    bool m_initialized = false;

    SAMEngine m_forward;
    SAMEngine m_backward;

    /** 体积转置（矢状/冠状→横断面对齐） */
    static void transposeVolume(const short* src, short* dst,
                                int size[3], int flag);

    /** 从长轴线段计算 Box prompt */
    static std::pair<std::pair<float,float>, std::pair<float,float>>
    getBoxFromLD(const Point3i& start, const Point3i& end,
                 int imgW, int imgH, float scale = 1.0f);

    /** 从长轴线段计算5点 prompt（前景3+背景2） */
    static void getPointsFromLD(const Point3i& start, const Point3i& end,
                                int imgW, int imgH, DrawDirection dir,
                                std::vector<std::pair<int,int>>& points,
                                std::vector<float>& labels);
};

#else // !USE_ONNXRUNTIME — 占位 Stub

class SAMEngine {
public:
    SAMEngine() = default;
    bool initialize(const SAMConfig&) {
        std::cout << "[SAMEngine] ONNX Runtime not available (USE_ONNXRUNTIME not defined)\n";
        return false;
    }
    bool isInitialized() const { return false; }
};

class SAMSegmenter {
public:
    SAMSegmenter() = default;
    bool initialize(const SAMConfig&) {
        std::cout << "[SAMSegmenter] ONNX Runtime not available (USE_ONNXRUNTIME not defined)\n";
        return false;
    }
    bool segment(const short*, const ImageInfo&, const Point3i&, const Point3i&,
                 DrawDirection, SegmentResult&, bool = true, ProgressCallback = nullptr) {
        return false;
    }
    bool isInitialized() const { return false; }
};

#endif // USE_ONNXRUNTIME

} // namespace Onc
