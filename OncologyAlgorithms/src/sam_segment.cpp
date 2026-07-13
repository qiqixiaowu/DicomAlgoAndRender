/**
 * @file sam2_segment.cpp
 * @brief SAM2 医学影像交互式分割 — 完整实现
 *
 * 实现了参考工程 McsfAlgoOncSAM2 中的完整 SAM2 推理管线：
 *   - 4 模块 ONNX 推理 (img_encoder / mem_attention / img_decoder / mem_encoder)
 *   - 帧间记忆机制 (InferenceStatus / FixedSizeQueue)
 *   - 双向 3D 传播 (forward + backward)
 *   - CT 窗宽窗位预处理 + ImageNet 归一化
 *   - Box / Point prompt 编码
 */

//#ifdef USE_ONNXRUNTIME

#include "sam2_segment.h"
#include <cmath>
#include <numeric>
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace Onc {

// ═════════════════════════════════════════════
//  SAM2Engine 实现
// ═════════════════════════════════════════════

SAM2Engine::SAM2Engine() = default;

SAM2Engine::~SAM2Engine() = default;

// ────────────────────────────────────────────
//  初始化
// ────────────────────────────────────────────
bool SAM2Engine::initialize(const SAM2Config& cfg) {
    m_cfg = cfg;
    m_imageSize = cfg.encoderInputDim;

    // 验证模型文件存在
    for (const auto& path : {cfg.imgEncoderPath, cfg.memAttentionPath,
                              cfg.imgDecoderPath, cfg.memEncoderPath}) {
        std::ifstream f(path, std::ios::binary);
        if (!f.good()) {
            std::cerr << "[SAM2] Model file not found: " << path << "\n";
            return false;
        }
    }

    // 配置 SessionOptions
    auto configOpts = [&](Ort::SessionOptions& opts) {
        opts.SetIntraOpNumThreads(cfg.intraOpThreads);
        if (cfg.useCUDA) {
            try {
                OrtCUDAProviderOptions cudaOpts;
                cudaOpts.device_id = cfg.deviceId;
                cudaOpts.arena_extend_strategy = 0;
                cudaOpts.gpu_mem_limit = SIZE_MAX;
                cudaOpts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
                cudaOpts.do_copy_in_default_stream = 1;
                opts.AppendExecutionProvider_CUDA(cudaOpts);
            } catch (const std::exception& e) {
                std::cerr << "[SAM2] CUDA EP failed, falling back to CPU: "
                          << e.what() << "\n";
            }
        }
    };

    configOpts(m_imgEncoderOpts);
    configOpts(m_imgDecoderOpts);
    configOpts(m_memAttentionOpts);
    configOpts(m_memEncoderOpts);

    // 创建 Session
    try {
#ifdef _WIN32
        auto toWide = [](const std::string& s) -> std::wstring {
            int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
            std::wstring ws(len, 0);
            MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &ws[0], len);
            return ws;
        };
        m_imgEncoderSession  = std::make_unique<Ort::Session>(m_imgEncoderEnv,
            toWide(cfg.imgEncoderPath).c_str(), m_imgEncoderOpts);
        m_memAttentionSession = std::make_unique<Ort::Session>(m_memAttentionEnv,
            toWide(cfg.memAttentionPath).c_str(), m_memAttentionOpts);
        m_imgDecoderSession  = std::make_unique<Ort::Session>(m_imgDecoderEnv,
            toWide(cfg.imgDecoderPath).c_str(), m_imgDecoderOpts);
        m_memEncoderSession  = std::make_unique<Ort::Session>(m_memEncoderEnv,
            toWide(cfg.memEncoderPath).c_str(), m_memEncoderOpts);
#else
        m_imgEncoderSession  = std::make_unique<Ort::Session>(m_imgEncoderEnv,
            cfg.imgEncoderPath.c_str(), m_imgEncoderOpts);
        m_memAttentionSession = std::make_unique<Ort::Session>(m_memAttentionEnv,
            cfg.memAttentionPath.c_str(), m_memAttentionOpts);
        m_imgDecoderSession  = std::make_unique<Ort::Session>(m_imgDecoderEnv,
            cfg.imgDecoderPath.c_str(), m_imgDecoderOpts);
        m_memEncoderSession  = std::make_unique<Ort::Session>(m_memEncoderEnv,
            cfg.memEncoderPath.c_str(), m_memEncoderOpts);
#endif
    } catch (const std::exception& e) {
        std::cerr << "[SAM2] Failed to load ONNX model: " << e.what() << "\n";
        return false;
    }

    // 读取节点信息
    loadSessionNodes(m_imgEncoderSession.get(), m_imgEncoderInputs, m_imgEncoderOutputs);
    loadSessionNodes(m_imgDecoderSession.get(), m_imgDecoderInputs, m_imgDecoderOutputs);
    loadSessionNodes(m_memAttentionSession.get(), m_memAttentionInputs, m_memAttentionOutputs);
    loadSessionNodes(m_memEncoderSession.get(), m_memEncoderInputs, m_memEncoderOutputs);

    // Warm-up: 一次空推理加速后续推理
    {
        std::vector<float> warmupData(static_cast<size_t>(m_imageSize) * m_imageSize * 3, 0.f);
        std::vector<Ort::Value> warmupInput;
        warmupInput.push_back(Ort::Value::CreateTensor<float>(
            m_memInfo, warmupData.data(), warmupData.size(),
            m_imgEncoderInputs[0].dim.data(), m_imgEncoderInputs[0].dim.size()));
        std::vector<Ort::Value> warmupOut;
        imgEncoderInfer(warmupInput, warmupOut);
    }

    m_initialized = true;
    std::cout << "[SAM2] Initialized successfully\n";
    return true;
}

void SAM2Engine::loadSessionNodes(Ort::Session* session,
                                   std::vector<OnnxNode>& inputs,
                                   std::vector<OnnxNode>& outputs) {
    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputs = session->GetInputCount();
    inputs.resize(numInputs);
    for (size_t i = 0; i < numInputs; i++) {
        auto namePtr = session->GetInputNameAllocated(i, allocator);
        inputs[i].name = namePtr.get();
        auto typeInfo = session->GetInputTypeInfo(i);
        inputs[i].dim = typeInfo.GetTensorTypeAndShapeInfo().GetShape();
    }

    size_t numOutputs = session->GetOutputCount();
    outputs.resize(numOutputs);
    for (size_t i = 0; i < numOutputs; i++) {
        auto namePtr = session->GetOutputNameAllocated(i, allocator);
        outputs[i].name = namePtr.get();
        auto typeInfo = session->GetOutputTypeInfo(i);
        outputs[i].dim = typeInfo.GetTensorTypeAndShapeInfo().GetShape();
    }
}

// ────────────────────────────────────────────
//  设置图像 / 参数
// ────────────────────────────────────────────
void SAM2Engine::setImage(const short* pImage, const int iSize[3],
                           const double dSpacing[3]) {
    m_pImage = pImage;
    for (int i = 0; i < 3; i++) {
        m_iSize[i] = iSize[i];
        m_dSpacing[i] = dSpacing[i];
    }
}

void SAM2Engine::setParams(const SAM2Params& params) {
    m_params = params;
}

void SAM2Engine::reset() {
    m_inferStatus.current_frame = 0;
    m_inferStatus.status_first.clear();
    m_inferStatus.status_recent.reset();
    m_inferStatus.obj_ptr_recent.reset();
    m_inferStatus.obj_ptr_first.clear();
}

// ────────────────────────────────────────────
//  图像预处理：窗宽窗位 + MinMax归一化 + ImageNet标准化 + Resize
// ────────────────────────────────────────────
bool SAM2Engine::resizeAndNormalize(int sliceIndex, const short* pImage,
                                     std::vector<float>& outData) {
    if (sliceIndex >= m_iSize[2]) return false;

    const int slicePixels = m_iSize[0] * m_iSize[1];
    const short* slicePtr = pImage + static_cast<int64_t>(sliceIndex) * slicePixels;

    // Step 1: 窗宽窗位变换 → [0, 255]
    float minWindow = m_cfg.windowCenter - 0.5f * m_cfg.windowWidth;
    std::vector<float> windowed(slicePixels);
    for (int i = 0; i < slicePixels; i++) {
        float val = (static_cast<float>(slicePtr[i]) - minWindow) / m_cfg.windowWidth;
        val = std::clamp(val, 0.f, 1.f);
        windowed[i] = val * 255.f;
    }

    // Step 2: MinMax 归一化 → [0, 1]
    auto [minIt, maxIt] = std::minmax_element(windowed.begin(), windowed.end());
    float minVal = *minIt, maxVal = *maxIt;
    float range = (maxVal - minVal > 1e-6f) ? (maxVal - minVal) : 1.f;
    for (auto& v : windowed) {
        v = (v - minVal) / range;
    }

    // Step 3: 复制为3通道 + ImageNet 标准化
    // mean = {0.406, 0.456, 0.485}, std = {0.225, 0.224, 0.229}
    constexpr float mean[3] = {0.406f, 0.456f, 0.485f};
    constexpr float stdv[3] = {0.225f, 0.224f, 0.229f};

    std::vector<float> normalized(static_cast<size_t>(slicePixels) * 3);
    for (int ch = 0; ch < 3; ch++) {
        float* channelPtr = normalized.data() + static_cast<size_t>(ch) * slicePixels;
        for (int i = 0; i < slicePixels; i++) {
            channelPtr[i] = (windowed[i] - mean[ch]) / stdv[ch];
        }
    }

    // Step 4: Resize 到 encoderInputDim × encoderInputDim × 3（双线性插值）
    int srcW = m_iSize[0], srcH = m_iSize[1];
    int dstW = m_imageSize, dstH = m_imageSize;
    outData.resize(static_cast<size_t>(dstW) * dstH * 3);

    for (int ch = 0; ch < 3; ch++) {
        const float* srcChannel = normalized.data() + static_cast<size_t>(ch) * slicePixels;
        float* dstChannel = outData.data() + static_cast<size_t>(ch) * dstW * dstH;

        for (int dy = 0; dy < dstH; dy++) {
            float srcY = static_cast<float>(dy) * srcH / dstH;
            int y0 = static_cast<int>(srcY);
            int y1 = std::min(y0 + 1, srcH - 1);
            float fy = srcY - y0;

            for (int dx = 0; dx < dstW; dx++) {
                float srcX = static_cast<float>(dx) * srcW / dstW;
                int x0 = static_cast<int>(srcX);
                int x1 = std::min(x0 + 1, srcW - 1);
                float fx = srcX - x0;

                float v00 = srcChannel[y0 * srcW + x0];
                float v01 = srcChannel[y0 * srcW + x1];
                float v10 = srcChannel[y1 * srcW + x0];
                float v11 = srcChannel[y1 * srcW + x1];

                dstChannel[dy * dstW + dx] =
                    v00 * (1-fx)*(1-fy) + v01 * fx*(1-fy)
                  + v10 * (1-fx)*fy     + v11 * fx*fy;
            }
        }
    }

    return true;
}

// ────────────────────────────────────────────
//  特征预提取（所有帧的 Image Encoder）
// ────────────────────────────────────────────
bool SAM2Engine::extractFeatures(const std::vector<int>& frameIndices,
                                  const short* pImage) {
    // 分配帧特征存储
    size_t numFrames = static_cast<size_t>(m_iSize[2]);
    m_imgEncoderOuts.resize(numFrames);

    int total = static_cast<int>(frameIndices.size());
#ifdef _OPENMP
    omp_set_num_threads(std::min(2, total));
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < total; i++) {
        int frameIdx = frameIndices[i];

        // 预处理
        std::vector<float> sliceData;
        resizeAndNormalize(frameIdx, pImage, sliceData);

        // 创建输入 tensor [1, 3, 512, 512]
        std::vector<Ort::Value> inputTensor;
        inputTensor.push_back(Ort::Value::CreateTensor<float>(
            m_memInfo, sliceData.data(), sliceData.size(),
            m_imgEncoderInputs[0].dim.data(),
            m_imgEncoderInputs[0].dim.size()));

        // 推理
        std::vector<Ort::Value> encoderOut;
        imgEncoderInfer(inputTensor, encoderOut);
        m_imgEncoderOuts[frameIdx] = std::move(encoderOut);
    }
    return true;
}

// ────────────────────────────────────────────
//  单帧推理
//
//  流程: encoder_out → mem_attention → img_decoder → mem_encoder → 更新记忆
// ────────────────────────────────────────────
bool SAM2Engine::inference(int frameIdx,
                            std::unordered_map<int, std::vector<float>>& maskPredMap,
                            bool firstFrame) {
    // 1. 取出预编码的特征
    std::vector<Ort::Value> encoderOut = std::move(m_imgEncoderOuts[frameIdx]);

    // 2. Memory Attention
    std::vector<Ort::Value> memAttOut;
    if (!memAttentionInfer(encoderOut, memAttOut)) {
        std::cerr << "[SAM2] mem_attention_infer failed\n";
        return false;
    }

    // 3. Image Decoder: 需要 memAttOut + high_res_feats
    memAttOut.push_back(std::move(encoderOut[1])); // high_res_feat0
    memAttOut.push_back(std::move(encoderOut[2])); // high_res_feat1

    std::vector<Ort::Value> decoderOut;
    if (!imgDecoderInfer(memAttOut, firstFrame, decoderOut)) {
        std::cerr << "[SAM2] img_decoder_infer failed\n";
        return false;
    }

    // 4. 更新记忆: obj_ptr
    if (m_inferStatus.current_frame == 0) {
        m_inferStatus.obj_ptr_first.push_back(std::move(decoderOut[0]));
    } else {
        m_inferStatus.obj_ptr_recent.push(std::move(decoderOut[0]));
    }

    // 5. Memory Encoder: mask_for_mem + pix_feat
    std::vector<Ort::Value> memEncIn;
    memEncIn.push_back(std::move(decoderOut[1]));  // mask_for_mem [1,1,1024,1024]
    memEncIn.push_back(std::move(encoderOut[0]));   // pix_feat [1,256,64,64]

    std::vector<Ort::Value> memEncOut;
    if (!memEncoderInfer(memEncIn, memEncOut)) {
        std::cerr << "[SAM2] mem_encoder_infer failed\n";
        return false;
    }

    // 6. 更新记忆: SubStatus
    SubStatus sub;
    sub.maskmem_features.push_back(std::move(memEncOut[0]));
    sub.maskmem_pos_enc.push_back(std::move(memEncOut[1]));
    sub.temporal_code.push_back(std::move(memEncOut[2]));

    if (m_inferStatus.current_frame == 0) {
        m_inferStatus.status_first.push_back(std::move(sub));
    } else {
        m_inferStatus.status_recent.push(std::move(sub));
    }

    // 7. 后处理：提取 pred_mask
    getOutput(decoderOut, frameIdx, maskPredMap);

    // 8. 后续帧使用 padding prompt
    m_params.promptPoints = {{0, 0}};
    m_params.pointLabels  = {-1.f};

    m_inferStatus.current_frame++;
    return true;
}

// ────────────────────────────────────────────
//  Image Encoder 推理
//  输入:  [1, 3, 512, 512]
//  输出:  pix_feat[1,256,64,64], high_res_feat0, high_res_feat1,
//         vision_feats[1,256,64,64], vision_pos_embed[4096,1,256]
// ────────────────────────────────────────────
bool SAM2Engine::imgEncoderInfer(const std::vector<Ort::Value>& input,
                                  std::vector<Ort::Value>& output) {
    std::vector<const char*> inNames, outNames;
    for (auto& n : m_imgEncoderInputs)  inNames.push_back(n.name.c_str());
    for (auto& n : m_imgEncoderOutputs) outNames.push_back(n.name.c_str());

    try {
        output = m_imgEncoderSession->Run(
            Ort::RunOptions{nullptr},
            inNames.data(), input.data(), input.size(),
            outNames.data(), outNames.size());
    } catch (const std::exception& e) {
        std::cerr << "[SAM2] img_encoder failed: " << e.what() << "\n";
        return false;
    }
    return true;
}

// ────────────────────────────────────────────
//  Memory Attention 推理
//  第0帧: 直接 pass-through vision_feats
//  后续帧: 构造记忆张量 → cross-attention
// ────────────────────────────────────────────
bool SAM2Engine::memAttentionInfer(std::vector<Ort::Value>& encoderOut,
                                    std::vector<Ort::Value>& output) {
    // 第一帧：无历史记忆，直接使用 vision_feats
    if (m_inferStatus.current_frame == 0) {
        output.push_back(std::move(encoderOut[3])); // vision_feats
        return true;
    }

    std::vector<const char*> inNames, outNames;
    for (auto& n : m_memAttentionInputs)  inNames.push_back(n.name.c_str());
    for (auto& n : m_memAttentionOutputs) outNames.push_back(n.name.c_str());

    std::vector<Ort::Value> inputTensor;

    // (1) current_vision_feat
    inputTensor.push_back(std::move(encoderOut[3]));
    // (2) current_vision_pos_embed
    inputTensor.push_back(std::move(encoderOut[4]));

    // (3) memory_0: obj_ptr 拼接 [first + recent]
    size_t objBufSize = 1 + m_inferStatus.obj_ptr_recent.size();
    std::vector<float> objPtrs(objBufSize * OBJ_PTR_DIM);
    {
        const float* p = m_inferStatus.obj_ptr_first[0].GetTensorData<float>();
        std::copy_n(p, OBJ_PTR_DIM, objPtrs.begin());
        for (size_t i = 0; i < m_inferStatus.obj_ptr_recent.size(); i++) {
            const float* q = m_inferStatus.obj_ptr_recent.at(i).GetTensorData<float>();
            std::copy_n(q, OBJ_PTR_DIM, objPtrs.begin() + OBJ_PTR_DIM * (i + 1));
        }
    }
    std::vector<int64_t> dim0{static_cast<int64_t>(objBufSize), OBJ_PTR_DIM};
    inputTensor.push_back(Ort::Value::CreateTensor<float>(
        m_memInfo, objPtrs.data(), objPtrs.size(), dim0.data(), dim0.size()));

    // (4) memory_1: maskmem_features 拼接 [first + recent]
    size_t featSize = 1 + m_inferStatus.status_recent.size();
    size_t singleFeatLen = MEM_FEAT_DIM * m_featSizes[2][0] * m_featSizes[2][1]; // 64*64*64
    std::vector<float> maskmemFeats(featSize * singleFeatLen);
    {
        const float* p = m_inferStatus.status_first[0].maskmem_features[0].GetTensorData<float>();
        std::copy_n(p, singleFeatLen, maskmemFeats.begin());
        for (size_t i = 0; i < m_inferStatus.status_recent.size(); i++) {
            const float* q = m_inferStatus.status_recent.at(i).maskmem_features[0].GetTensorData<float>();
            std::copy_n(q, singleFeatLen, maskmemFeats.begin() + singleFeatLen * (i + 1));
        }
    }
    std::vector<int64_t> dim1{static_cast<int64_t>(featSize), MEM_FEAT_DIM,
                               static_cast<int64_t>(m_featSizes[2][0]),
                               static_cast<int64_t>(m_featSizes[2][1])};
    inputTensor.push_back(Ort::Value::CreateTensor<float>(
        m_memInfo, maskmemFeats.data(), maskmemFeats.size(), dim1.data(), dim1.size()));

    // (5) memory_pos_embed: maskmem_pos_enc + temporal_code 融合，再补零 obj_ptr 部分
    const float* temporalCode = nullptr;
    if (m_inferStatus.current_frame > 1) {
        temporalCode = m_inferStatus.status_recent.at(featSize - 2)
                           .temporal_code[0].GetTensorData<float>();
    } else {
        temporalCode = m_inferStatus.status_first[0]
                           .temporal_code[0].GetTensorData<float>();
    }

    size_t posEncPerFrame = m_featSizes[2][0] * m_featSizes[2][1]; // 4096 (for 64x64)
    size_t totalPosEnc = (featSize * posEncPerFrame
                          + 4 * std::min(m_inferStatus.current_frame,
                                          static_cast<int32_t>(objBufSize)))
                         * MEM_FEAT_DIM;
    std::vector<float> memPosEnc(totalPosEnc, 0.f);

    // 第一部分: pos_enc + temporal_code
    for (size_t j = 0; j < featSize; j++) {
        const float* posData = nullptr;
        if (j == 0) {
            posData = m_inferStatus.status_first[0].maskmem_pos_enc[0].GetTensorData<float>();
        } else {
            posData = m_inferStatus.status_recent.at(j - 1).maskmem_pos_enc[0].GetTensorData<float>();
        }
        // 用于从尾部索引 temporal_code
        const float* tempSlice = temporalCode +
            (m_inferStatus.status_recent.size() - j) * MEM_FEAT_DIM;
        if (j == 0 && m_inferStatus.status_recent.size() > 0) {
            tempSlice = temporalCode + m_inferStatus.status_recent.size() * MEM_FEAT_DIM;
        }

        float* dst = memPosEnc.data() + j * posEncPerFrame * MEM_FEAT_DIM;
        for (size_t p = 0; p < posEncPerFrame; p++) {
            for (int k = 0; k < MEM_FEAT_DIM; k++) {
                dst[p * MEM_FEAT_DIM + k] = posData[p * MEM_FEAT_DIM + k] + tempSlice[k];
            }
        }
    }
    // 第二部分: obj_ptr 部分填零（已由初始化完成）

    std::vector<int64_t> dim3{
        static_cast<int64_t>(featSize * posEncPerFrame
            + 4 * std::min(m_inferStatus.current_frame,
                            static_cast<int32_t>(objBufSize))),
        1, MEM_FEAT_DIM};
    inputTensor.push_back(Ort::Value::CreateTensor<float>(
        m_memInfo, memPosEnc.data(), memPosEnc.size(), dim3.data(), dim3.size()));

    try {
        output = m_memAttentionSession->Run(
            Ort::RunOptions{nullptr},
            inNames.data(), inputTensor.data(), inputTensor.size(),
            outNames.data(), outNames.size());
    } catch (const std::exception& e) {
        std::cerr << "[SAM2] mem_attention failed: " << e.what() << "\n";
        return false;
    }
    return true;
}

// ────────────────────────────────────────────
//  Image Decoder 推理
//  输入: point_coords, point_labels, frame_size, image_embed, high_res_feats
//  输出: obj_ptr[1,256], mask_for_mem[1,1,1024,1024], pred_mask[1,H,W]
// ────────────────────────────────────────────
bool SAM2Engine::imgDecoderInfer(std::vector<Ort::Value>& memAttOut,
                                  bool firstFrame,
                                  std::vector<Ort::Value>& output) {
    std::vector<const char*> inNames, outNames;
    for (auto& n : m_imgDecoderInputs)  inNames.push_back(n.name.c_str());
    for (auto& n : m_imgDecoderOutputs) outNames.push_back(n.name.c_str());

    // 缩放坐标到 encoder 输入空间
    auto box = m_params.promptBox;
    auto points = m_params.promptPoints;

    box[0] = static_cast<int>(m_imageSize * (static_cast<float>(box[0]) / m_iSize[0]));
    box[1] = static_cast<int>(m_imageSize * (static_cast<float>(box[1]) / m_iSize[1]));
    box[2] = static_cast<int>(m_imageSize * (static_cast<float>(box[2]) / m_iSize[0]));
    box[3] = static_cast<int>(m_imageSize * (static_cast<float>(box[3]) / m_iSize[1]));

    for (auto& pt : points) {
        pt.first  = static_cast<int>(m_imageSize * (static_cast<float>(pt.first) / m_iSize[0]));
        pt.second = static_cast<int>(m_imageSize * (static_cast<float>(pt.second) / m_iSize[1]));
    }

    std::vector<float> pointVal, pointLabels;

    if (m_params.promptType == 0 && firstFrame) {
        // Box prompt: 左上角(label=2) + 右下角(label=3)
        pointVal = {
            static_cast<float>(box[0]),
            static_cast<float>(box[1]),
            static_cast<float>(box[0] + box[2]),
            static_cast<float>(box[1] + box[3])
        };
        pointLabels = {2.f, 3.f};
        m_imgDecoderInputs[0].dim = {1, 2, 2};
        m_imgDecoderInputs[1].dim = {1, 2};
    } else {
        // Point prompt (或后续帧 padding)
        for (const auto& pt : points) {
            pointVal.push_back(static_cast<float>(pt.first));
            pointVal.push_back(static_cast<float>(pt.second));
        }
        pointLabels = m_params.pointLabels;
        m_imgDecoderInputs[0].dim = {1, static_cast<int64_t>(points.size()), 2};
        m_imgDecoderInputs[1].dim = {1, static_cast<int64_t>(points.size())};
    }

    std::vector<int64_t> frameSize = {
        static_cast<int64_t>(m_iSize[1]),
        static_cast<int64_t>(m_iSize[0])
    };

    std::vector<Ort::Value> inputTensor;
    inputTensor.push_back(Ort::Value::CreateTensor<float>(
        m_memInfo, pointVal.data(), pointVal.size(),
        m_imgDecoderInputs[0].dim.data(), m_imgDecoderInputs[0].dim.size()));
    inputTensor.push_back(Ort::Value::CreateTensor<float>(
        m_memInfo, pointLabels.data(), pointLabels.size(),
        m_imgDecoderInputs[1].dim.data(), m_imgDecoderInputs[1].dim.size()));
    inputTensor.push_back(Ort::Value::CreateTensor<int64_t>(
        m_memInfo, frameSize.data(), frameSize.size(),
        m_imgDecoderInputs[2].dim.data(), m_imgDecoderInputs[2].dim.size()));
    inputTensor.push_back(std::move(memAttOut[0])); // image_embed
    inputTensor.push_back(std::move(memAttOut[1])); // high_res_feats_0
    inputTensor.push_back(std::move(memAttOut[2])); // high_res_feats_1

    try {
        output = m_imgDecoderSession->Run(
            Ort::RunOptions{nullptr},
            inNames.data(), inputTensor.data(), inputTensor.size(),
            outNames.data(), outNames.size());
    } catch (const std::exception& e) {
        std::cerr << "[SAM2] img_decoder failed: " << e.what() << "\n";
        return false;
    }
    return true;
}

// ────────────────────────────────────────────
//  Memory Encoder 推理
//  输入: mask_for_mem[1,1,1024,1024], pix_feat[1,256,64,64]
//  输出: maskmem_features[1,64,64,64], maskmem_pos_enc[4096,1,64], temporal_code[7,1,1,64]
// ────────────────────────────────────────────
bool SAM2Engine::memEncoderInfer(const std::vector<Ort::Value>& input,
                                  std::vector<Ort::Value>& output) {
    std::vector<const char*> inNames, outNames;
    for (auto& n : m_memEncoderInputs)  inNames.push_back(n.name.c_str());
    for (auto& n : m_memEncoderOutputs) outNames.push_back(n.name.c_str());

    try {
        output = m_memEncoderSession->Run(
            Ort::RunOptions{nullptr},
            inNames.data(), input.data(), input.size(),
            outNames.data(), outNames.size());
    } catch (const std::exception& e) {
        std::cerr << "[SAM2] mem_encoder failed: " << e.what() << "\n";
        return false;
    }
    return true;
}

// ────────────────────────────────────────────
//  后处理：提取 pred_mask → 二值化 → 存入 map
// ────────────────────────────────────────────
bool SAM2Engine::getOutput(std::vector<Ort::Value>& decoderOut, int frameIdx,
                            std::unordered_map<int, std::vector<float>>& maskPredMap) {
    const int maskSize = m_iSize[0] * m_iSize[1];
    std::vector<float> mask(maskSize);
    const float* rawMask = decoderOut[2].GetTensorData<float>();
    std::copy_n(rawMask, maskSize, mask.begin());

    // 阈值化: >0 → 1, ≤0 → 0
    for (auto& v : mask) {
        v = (v > 0.f) ? 1.f : 0.f;
    }

    maskPredMap[frameIdx] = std::move(mask);
    return true;
}

// ═════════════════════════════════════════════
//  SAM2Segmenter 实现
// ═════════════════════════════════════════════

SAM2Segmenter::SAM2Segmenter() = default;
SAM2Segmenter::~SAM2Segmenter() = default;

bool SAM2Segmenter::initialize(const SAM2Config& cfg) {
    m_cfg = cfg;
    if (!m_forward.initialize(cfg)) return false;
    if (!m_backward.initialize(cfg)) return false;
    m_initialized = true;
    return true;
}

// ────────────────────────────────────────────
//  体积转置（矢状/冠状 → 横断面对齐）
// ────────────────────────────────────────────
void SAM2Segmenter::transposeVolume(const short* src, short* dst,
                                     int size[3], int flag) {
    int x = size[0], y = size[1], z = size[2];
    if (flag == 0) {
        // 横断面: 无需转置
        std::memcpy(dst, src, sizeof(short) * static_cast<int64_t>(x) * y * z);
        return;
    }
    if (flag == 1) {
        // 矢状面: x,y,z → z,y,x
        for (int zi = 0; zi < z; zi++)
            for (int yi = 0; yi < y; yi++)
                for (int xi = 0; xi < x; xi++)
                    dst[xi * (z * y) + yi * z + zi] = src[zi * (x * y) + yi * x + xi];
        size[0] = z;
        size[2] = x;
    } else if (flag == 2) {
        // 冠状面: x,y,z → x,z,y
        for (int zi = 0; zi < z; zi++)
            for (int yi = 0; yi < y; yi++)
                for (int xi = 0; xi < x; xi++)
                    dst[yi * (x * z) + (z - 1 - zi) * x + xi] = src[zi * (x * y) + yi * x + xi];
        size[1] = z;
        size[2] = y;
    }
}

// ────────────────────────────────────────────
//  从长轴线段计算 Box prompt
//  当两点不在同一切面时，2D坐标取两点投影到起始帧上的位置
// ────────────────────────────────────────────
std::pair<std::pair<float,float>, std::pair<float,float>>
SAM2Segmenter::getBoxFromLD(const Point3i& start, const Point3i& end,
                             int imgW, int imgH, DrawDirection dir, float scale) {
    // 根据绘制方向提取2D坐标
    float x1, y1, x2, y2;
    if (dir == DrawDirection::Axial) {
        x1 = static_cast<float>(start.x); y1 = static_cast<float>(start.y);
        x2 = static_cast<float>(end.x);   y2 = static_cast<float>(end.y);
    } else if (dir == DrawDirection::Sagittal) {
        x1 = static_cast<float>(start.z); y1 = static_cast<float>(start.y);
        x2 = static_cast<float>(end.z);   y2 = static_cast<float>(end.y);
    } else { // Coronal
        x1 = static_cast<float>(start.x); y1 = static_cast<float>(start.z);
        x2 = static_cast<float>(end.x);   y2 = static_cast<float>(end.z);
    }

    float length = std::sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
    float cx = (x1 + x2) * 0.5f;
    float cy = (y1 + y2) * 0.5f;

    float halfLen = length * 0.5f * scale;

    float xmin = std::clamp(cx - halfLen, 0.f, static_cast<float>(imgW - 1));
    float xmax = std::clamp(cx + halfLen, 0.f, static_cast<float>(imgW - 1));
    float ymin = std::clamp(cy - halfLen, 0.f, static_cast<float>(imgH - 1));
    float ymax = std::clamp(cy + halfLen, 0.f, static_cast<float>(imgH - 1));

    return {{xmin, ymin}, {xmax, ymax}};
}

// ────────────────────────────────────────────
//  从长轴线段计算5点 prompt（前景3 + 背景2）
// ────────────────────────────────────────────
void SAM2Segmenter::getPointsFromLD(const Point3i& start, const Point3i& end,
                                     int imgW, int imgH, DrawDirection dir,
                                     std::vector<std::pair<int,int>>& points,
                                     std::vector<float>& labels) {
    points.clear();
    labels.clear();

    int x1, y1, x2, y2;
    if (dir == DrawDirection::Axial) {
        x1 = start.x; y1 = start.y;
        x2 = end.x;   y2 = end.y;
    } else if (dir == DrawDirection::Sagittal) {
        x1 = start.z; y1 = start.y;
        x2 = end.z;   y2 = end.y;
    } else { // Coronal
        x1 = start.x; y1 = start.z;
        x2 = end.x;   y2 = end.z;
    }

    // 前景: 长径两端 + 中点
    points.push_back({x1, y1});
    points.push_back({x2, y2});
    points.push_back({(x1 + x2) / 2, (y1 + y2) / 2});
    labels = {1.f, 1.f, 1.f};

    // 背景: 延长线两端(外扩15%)
    float dx = static_cast<float>(x2 - x1);
    float dy = static_cast<float>(y2 - y1);
    float ext = 0.15f;

    int bx1 = std::clamp(static_cast<int>(x1 - ext * dx), 0, imgW - 1);
    int by1 = std::clamp(static_cast<int>(y1 - ext * dy), 0, imgH - 1);
    int bx2 = std::clamp(static_cast<int>(x2 + ext * dx), 0, imgW - 1);
    int by2 = std::clamp(static_cast<int>(y2 + ext * dy), 0, imgH - 1);

    points.push_back({bx1, by1});
    points.push_back({bx2, by2});
    labels.push_back(0.f); // 背景
    labels.push_back(0.f); // 背景
}

// ────────────────────────────────────────────
//  3D 分割主入口
// ────────────────────────────────────────────
bool SAM2Segmenter::segment(const short*     data,
                             const ImageInfo& info,
                             const Point3i&   ldStart,
                             const Point3i&   ldEnd,
                             DrawDirection    drawDir,
                             SegmentResult&   result,
                             bool             is3D,
                             ProgressCallback progress) {
    if (!m_initialized) {
        std::cerr << "[SAM2] Not initialized\n";
        return false;
    }

    int iSize[3] = {info.dim[0], info.dim[1], info.dim[2]};
    double dSpacing[3] = {info.spacing[0], info.spacing[1], info.spacing[2]};

    // 重置两个引擎的记忆
    m_forward.reset();
    m_backward.reset();

    // ── 计算长径3D中心点 ──
    float cx = (ldStart.x + ldEnd.x) * 0.5f;
    float cy = (ldStart.y + ldEnd.y) * 0.5f;
    float cz = (ldStart.z + ldEnd.z) * 0.5f;
    int icx = static_cast<int>(cx), icy = static_cast<int>(cy), icz = static_cast<int>(cz);

    // ── 自动检测传播方向 ──
    // 当两点不在同切面时，用户绘制的线跨越多个切面，
    // 此时 drawDir 不一定代表标准的横冠矢方位，
    // 需要根据线的3D物理方向自动选择最佳传播轴
    float physDx = std::abs(ldEnd.x - ldStart.x) * static_cast<float>(dSpacing[0]);
    float physDy = std::abs(ldEnd.y - ldStart.y) * static_cast<float>(dSpacing[1]);
    float physDz = std::abs(ldEnd.z - ldStart.z) * static_cast<float>(dSpacing[2]);

    DrawDirection effectiveDir = drawDir;
    float maxDist = std::max({physDx, physDy, physDz});
    if (maxDist > 0.f) {
        // 物理距离最大的轴即为病变的主延伸方向，也是最佳传播轴
        if (physDz >= physDx && physDz >= physDy) {
            effectiveDir = DrawDirection::Axial;     // Z轴最大 → 横断面传播
        } else if (physDx >= physDy && physDx >= physDz) {
            effectiveDir = DrawDirection::Sagittal;  // X轴最大 → 矢状面传播
        } else {
            effectiveDir = DrawDirection::Coronal;   // Y轴最大 → 冠状面传播
        }
    }

    int flag = static_cast<int>(effectiveDir);

    // ── 计算传播半径（物理距离 → 像素数）──
    float dOriDis = std::sqrt(physDx*physDx + physDy*physDy + physDz*physDz);
    float dMinSpacing = static_cast<float>(std::min({dSpacing[0], dSpacing[1], dSpacing[2]}));
    int iPixelNum = static_cast<int>(dOriDis / dMinSpacing);

    // ── 根据方向确定传播范围和起始帧 ──
    // 起始帧使用中心帧，确保prompt投影偏差最小
    int iMin = 0, iMax = 0, startFrame = 0;
    auto clampRange = [](int center, int radius, int maxVal) {
        return std::make_pair(std::max(0, center - radius),
                              std::min(maxVal, center + radius));
    };

    if (flag == 0) { // Axial
        auto [lo, hi] = clampRange(icz, iPixelNum, iSize[2]);
        iMin = lo; iMax = hi; startFrame = icz;
    } else if (flag == 1) { // Sagittal
        auto [lo, hi] = clampRange(icx, iPixelNum, iSize[0]);
        iMin = lo; iMax = hi; startFrame = icx;
    } else { // Coronal
        auto [lo, hi] = clampRange(icy, iPixelNum, iSize[1]);
        iMin = lo; iMax = hi; startFrame = icy;
    }

    // ── 体积转置（如果非横断面） ──
    // 转置后，传播轴统一为Z轴，每个"切片"尺寸为 iSize[0]×iSize[1]
    std::vector<short> transposed;
    const short* volumePtr = data;
    if (flag != 0) {
        int64_t totalVox = static_cast<int64_t>(iSize[0]) * iSize[1] * iSize[2];
        transposed.resize(totalVox);
        transposeVolume(data, transposed.data(), iSize, flag);
        volumePtr = transposed.data();
    }

    // ── 计算 Prompt（在转置后的坐标系中） ──
    // 转置后体积统一为"横断面对齐"布局，prompt始终用Axial方式提取2D坐标
    // 需先将原始3D坐标映射到转置后的2D坐标系
    SAM2Params params;
    params.promptType = 0; // Box prompt

    Point3i mappedStart, mappedEnd;
    if (flag == 0) { // Axial: 无转置
        mappedStart = ldStart;
        mappedEnd   = ldEnd;
    } else if (flag == 1) { // Sagittal: x,y,z → z,y,x
        mappedStart = {ldStart.z, ldStart.y, ldStart.x};
        mappedEnd   = {ldEnd.z, ldEnd.y, ldEnd.x};
    } else { // Coronal: x,y,z → x,(z_max-z),y
        // 冠状面转置时Z轴被翻转: row = (z_max - 1 - original_z)
        mappedStart = {ldStart.x, iSize[1] - 1 - ldStart.z, ldStart.y};
        mappedEnd   = {ldEnd.x, iSize[1] - 1 - ldEnd.z, ldEnd.y};
    }

    auto [tl, br] = getBoxFromLD(mappedStart, mappedEnd, iSize[0], iSize[1],
                                  DrawDirection::Axial, 1.0f);
    params.promptBox = {
        static_cast<int>(tl.first),  static_cast<int>(tl.second),
        static_cast<int>(br.first - tl.first),
        static_cast<int>(br.second - tl.second)
    };

    // 注意：当两点跨切面时，mappedStart/mappedEnd 的 2D 坐标来自不同 z 层，
    // 在 startFrame 上病变的真实 2D 位置是其中点。
    // 对 Point prompt，将端点收缩到中点附近（保留方向信息但减小偏差）：
    //   virtualStart = midpoint - 0.5 * (end - start) * inPlaneRatio
    //   virtualEnd   = midpoint + 0.5 * (end - start) * inPlaneRatio
    // inPlaneRatio: 2D方向分量占3D总距离的比例，表示该方向的"可信度"
    {
        float mid2dx = (mappedStart.x + mappedEnd.x) * 0.5f;
        float mid2dy = (mappedStart.y + mappedEnd.y) * 0.5f;
        float half2dx = (mappedEnd.x - mappedStart.x) * 0.5f;
        float half2dy = (mappedEnd.y - mappedStart.y) * 0.5f;

        // 2D投影长度 vs 3D物理长度的比值，越小说明线越"陡直"（跨切面分量越大）
        float proj2D = std::sqrt(half2dx * half2dx + half2dy * half2dy);
        float total3D = std::sqrt(physDx * physDx + physDy * physDy + physDz * physDz) * 0.5f
                        / static_cast<float>(std::min({dSpacing[0], dSpacing[1], dSpacing[2]}));
        float inPlaneRatio = (total3D > 1e-3f) ? std::min(proj2D / total3D, 1.0f) : 1.0f;

        // 按 inPlaneRatio 缩放端点偏移，减少跨层偏差
        Point3i virtualMappedStart = {
            static_cast<int>(std::round(mid2dx - half2dx * inPlaneRatio)),
            static_cast<int>(std::round(mid2dy - half2dy * inPlaneRatio)),
            startFrame
        };
        Point3i virtualMappedEnd = {
            static_cast<int>(std::round(mid2dx + half2dx * inPlaneRatio)),
            static_cast<int>(std::round(mid2dy + half2dy * inPlaneRatio)),
            startFrame
        };
        getPointsFromLD(virtualMappedStart, virtualMappedEnd, iSize[0], iSize[1],
                        DrawDirection::Axial,
                        params.promptPoints, params.pointLabels);
    }

    // ── 执行推理 ──
    std::unordered_map<int, std::vector<float>> maskPredMap;

    if (!is3D) {
        // 仅标注帧
        std::vector<int> indices = {startFrame};
        m_forward.setImage(volumePtr, iSize, dSpacing);
        m_forward.setParams(params);
        m_forward.extractFeatures(indices, volumePtr);
        m_forward.inference(startFrame, maskPredMap, true);
    } else {
        // 构建 forward / backward 帧索引
        std::vector<int> fwdIndices, bwdIndices;
        for (int i = startFrame; i < iMax; i++) fwdIndices.push_back(i);
        for (int i = startFrame; i > iMin; i--) bwdIndices.push_back(i);

        // 设置双方向引擎
        m_forward.setImage(volumePtr, iSize, dSpacing);
        m_forward.setParams(params);
        m_backward.setImage(volumePtr, iSize, dSpacing);
        m_backward.setParams(params);

        // 预提取特征
        std::vector<int> allIndices;
        allIndices.insert(allIndices.end(), fwdIndices.begin(), fwdIndices.end());
        allIndices.insert(allIndices.end(), bwdIndices.begin(), bwdIndices.end());
        // 去重
        std::sort(allIndices.begin(), allIndices.end());
        allIndices.erase(std::unique(allIndices.begin(), allIndices.end()), allIndices.end());

        m_forward.extractFeatures(allIndices, volumePtr);
        m_backward.extractFeatures(allIndices, volumePtr);

        // Forward 传播
        for (size_t j = 0; j < fwdIndices.size(); j++) {
            m_forward.inference(fwdIndices[j], maskPredMap, j == 0);
            if (progress) {
                double p = static_cast<double>(j) / (fwdIndices.size() + bwdIndices.size());
                if (!progress(p)) return false;
            }
        }

        // Backward 传播
        for (size_t j = 0; j < bwdIndices.size(); j++) {
            m_backward.inference(bwdIndices[j], maskPredMap, j == 0);
            if (progress) {
                double p = static_cast<double>(fwdIndices.size() + j)
                         / (fwdIndices.size() + bwdIndices.size());
                if (!progress(p)) return false;
            }
        }
    }

    // ── 收集分割结果：从 mask → SegmentResult(Point3i) ──
    result.clear();
    for (auto& [z, mask] : maskPredMap) {
        for (int i = 0; i < iSize[0] * iSize[1]; i++) {
            if (mask[i] > 0.f) {
                int iy = i / iSize[0];
                int ix = i % iSize[0];
                Point3i pt;
                if (flag == 0) {
                    pt = {ix, iy, z};
                } else if (flag == 1) {
                    pt = {z, iy, ix}; // 逆转置 z,y,x → x,y,z
                } else {
                    pt = {ix, z, iSize[1] - 1 - iy}; // 逆转置 x,(z_max-z),y → x,y,z
                }
                result.push_back(pt);
            }
        }
    }

    if (progress) progress(1.0);
    return !result.empty();
}

} // namespace Onc

#endif // USE_ONNXRUNTIME
