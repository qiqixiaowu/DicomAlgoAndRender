#pragma once
/**
 * @file sam2_segment.h
 * @brief SAM2 (Segment Anything Model 2) 医学影像交互式分割
 *
 * 架构概览
 * ────────
 *   4个 ONNX 子模型串联:
 *     Image Encoder → Memory Attention → Image Decoder → Memory Encoder
 *
 *   1. Image Encoder (Hiera ViT):
 *        输入 [1,3,512,512] → 多尺度特征金字塔
 *   2. Memory Attention (Cross-Attention):
 *        融合当前帧特征 + 历史帧记忆 → 上下文嵌入
 *   3. Image Decoder (Mask Head):
 *        接收提示(点/框) + 上下文嵌入 → 预测mask + object pointer
 *   4. Memory Encoder:
 *        把当前帧mask编码为记忆，供后续帧参考
 *
 *   支持双向3D传播：从用户标注层向前/向后逐层推理
 *
 * 依赖
 * ────
 *   - ONNX Runtime C++ API (onnxruntime_cxx_api.h)
 *   - 可选 CUDA Execution Provider (GPU 加速)
 *   - 4个 ONNX 模型文件
 *
 * 编译条件
 * ────────
 *   USE_ONNXRUNTIME 宏控制。未定义时所有类退化为占位 stub。
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
//  定长环形队列（记忆窗口）
// ─────────────────────────────────────────────
template <typename T, size_t N>
class FixedSizeQueue {
    std::vector<T> data_;
    size_t head_ = 0;
    size_t tail_ = 0;
    size_t count_ = 0;
public:
    FixedSizeQueue() = default;

    bool push(T&& value) {
        if (full()) {
            tail_ = (tail_ + 1) % N;
            data_[head_] = std::move(value);
        } else {
            if (data_.size() < N) {
                data_.push_back(std::move(value));
                count_++;
            } else {
                data_[head_] = std::move(value);
            }
        }
        head_ = (head_ + 1) % N;
        return true;
    }

    T& at(size_t idx) {
        if (idx >= count_)
            throw std::out_of_range("FixedSizeQueue: index out of range");
        idx = (tail_ + idx) % N;
        return data_[idx];
    }

    void reset() {
        data_.clear();
        head_ = 0;
        tail_ = 0;
        count_ = 0;
    }

    bool empty()  const { return count_ == 0; }
    bool full()   const { return count_ == N; }
    size_t size() const { return count_; }
};

// ─────────────────────────────────────────────
//  ONNX 节点描述
// ─────────────────────────────────────────────
struct OnnxNode {
    std::vector<int64_t> dim;
    std::string name;
};

// ─────────────────────────────────────────────
//  SAM2 推理参数
// ─────────────────────────────────────────────
struct SAM2Params {
    int    promptType = 0;                       ///< 0=box, 1=point
    std::array<int, 4> promptBox = {0,0,0,0};    ///< x, y, width, height
    std::vector<std::pair<int,int>> promptPoints; ///< 提示点坐标
    std::vector<float> pointLabels;               ///< 点标签: 1=前景, 0=背景, -1=padding
};

// ─────────────────────────────────────────────
//  SAM2 模型配置
// ─────────────────────────────────────────────
struct SAM2Config {
    std::string imgEncoderPath;    ///< image_encoder ONNX 路径
    std::string memAttentionPath;  ///< memory_attention ONNX 路径
    std::string imgDecoderPath;    ///< image_decoder ONNX 路径
    std::string memEncoderPath;    ///< memory_encoder ONNX 路径

    int  encoderInputDim = 512;    ///< encoder 输入分辨率
    bool useCUDA         = true;   ///< 是否使用 CUDA EP
    int  deviceId        = 0;      ///< GPU 设备号
    int  intraOpThreads  = 2;      ///< ONNX Runtime 内部线程数

    // 窗宽窗位（CT 默认）
    float windowCenter = 65.6f;
    float windowWidth  = 500.0f;

    /** 从目录路径自动补全4个ONNX模型路径 */
    void setModelDir(const std::string& dir, const std::string& prefix = "sam2") {
        imgEncoderPath   = dir + "/" + prefix + "_image_encoder.onnx";
        memAttentionPath = dir + "/" + prefix + "_memory_attention.onnx";
        imgDecoderPath   = dir + "/" + prefix + "_image_decoder.onnx";
        memEncoderPath   = dir + "/" + prefix + "_memory_encoder.onnx";
    }
};

#ifdef USE_ONNXRUNTIME

// ─────────────────────────────────────────────
//  SAM2 单方向推理引擎
// ─────────────────────────────────────────────
class SAM2Engine {
    static constexpr size_t OBJ_PTR_BUFFER = 15;
    static constexpr size_t MEM_BUFFER     = 7;
    static constexpr int    OBJ_PTR_DIM    = 256;
    static constexpr int    MEM_FEAT_DIM   = 64;

    // ── 记忆子状态 ──────────────────────────
    struct SubStatus {
        std::vector<Ort::Value> maskmem_features;  // [1,64,64,64]
        std::vector<Ort::Value> maskmem_pos_enc;   // [4096,1,64]
        std::vector<Ort::Value> temporal_code;     // [7,1,1,64]
    };

    // ── 推理状态（帧间记忆）─────────────────
    struct InferenceStatus {
        int32_t current_frame = 0;
        std::vector<Ort::Value> obj_ptr_first;         // 第一帧 obj_ptr [1,256]
        std::vector<SubStatus>  status_first;           // 第一帧记忆
        FixedSizeQueue<SubStatus, MEM_BUFFER>    status_recent;     // 最近7帧记忆
        FixedSizeQueue<Ort::Value, OBJ_PTR_BUFFER> obj_ptr_recent;  // 最近15帧 obj_ptr
    };

public:
    SAM2Engine();
    ~SAM2Engine();

    SAM2Engine(const SAM2Engine&) = delete;
    SAM2Engine& operator=(const SAM2Engine&) = delete;

    /** 初始化：加载4个ONNX模型、创建Session */
    bool initialize(const SAM2Config& cfg);

    /** 设置图像信息 */
    void setImage(const short* pImage, const int iSize[3],
                  const double dSpacing[3]);

    /** 设置提示参数 */
    void setParams(const SAM2Params& params);

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
    SAM2Config m_cfg;
    int m_imageSize = 512;
    std::vector<std::vector<size_t>> m_featSizes = {{128,128},{64,64},{32,32}};

    // ── 图像 ──
    const short* m_pImage = nullptr;
    int m_iSize[3]     = {0, 0, 0};
    double m_dSpacing[3] = {1.0, 1.0, 1.0};

    // ── 参数 ──
    SAM2Params m_params;

    // ── ONNX Runtime 对象 ──
    bool m_initialized = false;
    Ort::MemoryInfo m_memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Env m_imgEncoderEnv  {ORT_LOGGING_LEVEL_WARNING, "img_encoder"};
    Ort::Env m_imgDecoderEnv  {ORT_LOGGING_LEVEL_WARNING, "img_decoder"};
    Ort::Env m_memAttentionEnv{ORT_LOGGING_LEVEL_WARNING, "mem_attention"};
    Ort::Env m_memEncoderEnv  {ORT_LOGGING_LEVEL_WARNING, "mem_encoder"};

    std::unique_ptr<Ort::Session> m_imgEncoderSession;
    std::unique_ptr<Ort::Session> m_imgDecoderSession;
    std::unique_ptr<Ort::Session> m_memAttentionSession;
    std::unique_ptr<Ort::Session> m_memEncoderSession;

    Ort::SessionOptions m_imgEncoderOpts;
    Ort::SessionOptions m_imgDecoderOpts;
    Ort::SessionOptions m_memAttentionOpts;
    Ort::SessionOptions m_memEncoderOpts;

    // ── ONNX 节点信息 ──
    std::vector<OnnxNode> m_imgEncoderInputs, m_imgEncoderOutputs;
    std::vector<OnnxNode> m_imgDecoderInputs, m_imgDecoderOutputs;
    std::vector<OnnxNode> m_memAttentionInputs, m_memAttentionOutputs;
    std::vector<OnnxNode> m_memEncoderInputs, m_memEncoderOutputs;

    // ── 缓存 ──
    std::vector<std::vector<Ort::Value>> m_imgEncoderOuts; // 按帧索引
    InferenceStatus m_inferStatus;

    // ── 子推理步骤 ──
    bool imgEncoderInfer(const std::vector<Ort::Value>& input,
                         std::vector<Ort::Value>& output);
    bool memAttentionInfer(std::vector<Ort::Value>& encoderOut,
                           std::vector<Ort::Value>& output);
    bool imgDecoderInfer(std::vector<Ort::Value>& memAttOut,
                         bool firstFrame,
                         std::vector<Ort::Value>& output);
    bool memEncoderInfer(const std::vector<Ort::Value>& input,
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
//  SAM2 3D 分割器（双向传播）
// ─────────────────────────────────────────────
class SAM2Segmenter {
public:
    SAM2Segmenter();
    ~SAM2Segmenter();

    /** 初始化：加载模型 */
    bool initialize(const SAM2Config& cfg);

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
    SAM2Config m_cfg;
    bool m_initialized = false;

    SAM2Engine m_forward;
    SAM2Engine m_backward;

    /** 体积转置（矢状/冠状→横断面对齐） */
    static void transposeVolume(const short* src, short* dst,
                                int size[3], int flag);

    /** 从长轴线段计算 Box prompt（根据绘制方向提取2D坐标） */
    static std::pair<std::pair<float,float>, std::pair<float,float>>
    getBoxFromLD(const Point3i& start, const Point3i& end,
                 int imgW, int imgH, DrawDirection dir, float scale = 1.0f);

    /** 从长轴线段计算5点 prompt（前景3+背景2） */
    static void getPointsFromLD(const Point3i& start, const Point3i& end,
                                int imgW, int imgH, DrawDirection dir,
                                std::vector<std::pair<int,int>>& points,
                                std::vector<float>& labels);
};

#else // !USE_ONNXRUNTIME — 占位 Stub

class SAM2Engine {
public:
    SAM2Engine() = default;
    bool initialize(const SAM2Config&) {
        std::cout << "[SAM2Engine] ONNX Runtime not available (USE_ONNXRUNTIME not defined)\n";
        return false;
    }
    bool isInitialized() const { return false; }
};

class SAM2Segmenter {
public:
    SAM2Segmenter() = default;
    bool initialize(const SAM2Config&) {
        std::cout << "[SAM2Segmenter] ONNX Runtime not available (USE_ONNXRUNTIME not defined)\n";
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
