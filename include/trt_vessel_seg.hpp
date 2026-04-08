#pragma once

/**
 * TensorRT 血管分割推理器 —— C++ 原生部署
 * =========================================
 *
 * 启用 TRT 推理的前提：
 *   1. 安装 TensorRT >= 8.6  (NvInfer.h  / nvinfer.lib)
 *   2. 安装 CUDA >= 11.8      (cuda_runtime_api.h / cudart.lib)
 *   3. 在项目属性 → C/C++ → 预处理器 中添加:  USE_TENSORRT
 *   4. 设置 TRT_ROOT 环境变量 = TensorRT 安装根目录
 *      （vcxproj 已配置 $(TRT_ROOT)\include / $(TRT_ROOT)\lib）
 *
 * 不满足时：load() 返回 false 并打印说明，程序继续运行
 *           （可回退到加载 Python 生成的 .raw 文件）。
 *
 * 完整工作流：
 *   1. Python 端：python dl_medical/export_trt.py
 *      → 生成 checkpoints/vessel_seg/vessel_seg_fp16.engine + trt_meta.json
 *
 *   2. C++ 端（命令行）：
 *      VolumeRenderOptimized.exe "E:\CT_data" "checkpoints/vessel_seg/trt_meta.json"
 *
 *   回退到 .raw 文件（Python 推理结果）：
 *      VolumeRenderOptimized.exe "E:\CT_data" "outputs/vessel_result/vessel_mask.raw"
 */

#include "region_growing.hpp"
#include "dicom_utils.hpp"
#include <memory>
#include <string>
#include <vector>
#include <cstdint>

// ---------------------------------------------------------------
// TRT 元数据（来自 export_trt.py 输出的 trt_meta.json）
// ---------------------------------------------------------------
struct TRTVesselMeta {
    std::string enginePath;    // .engine 文件路径
    std::string onnxPath;      // .onnx   文件路径（备用）
    int   patchD  = 64;        // 推理 patch 深度   (D)
    int   patchH  = 128;       // 推理 patch 高度   (H)
    int   patchW  = 128;       // 推理 patch 宽度   (W)
    float huMin   = -150.f;    // HU 归一化下限
    float huMax   =  550.f;    // HU 归一化上限
    int   outChannels = 2;     // 输出通道（背景=0, 血管=1）
};

/**
 * 从 export_trt.py 生成的 trt_meta.json 解析元数据。
 * 使用手动 key-value 解析，无需外部 JSON 库。
 */
TRTVesselMeta parseTRTVesselMeta(const std::string& jsonPath);

// ---------------------------------------------------------------
// 从 DICOM 系列提取真实 HU 值，归一化到 [0,1]
//
 // 读取每张切片的 RescaleSlope / RescaleIntercept，
// 计算 HU = pixel * slope + intercept，按 [huMin,huMax] 裁剪后归一化。
// 与 Python 训练时的预处理完全一致。
//
// 返回：float32 向量，row-major [D][H][W]，大小 = outD * outH * outW
// ---------------------------------------------------------------
std::vector<float> buildVolumeFloat(
    const SeriesData& series,
    uint32_t& outD, uint32_t& outH, uint32_t& outW,
    float huMin = -150.f,
    float huMax =  550.f);

// ---------------------------------------------------------------
// TRT 血管分割推理器
// ---------------------------------------------------------------
class TRTVesselSegmentor {
public:
    TRTVesselSegmentor();
    ~TRTVesselSegmentor();

    // 禁止复制（含 CUDA 指针）
    TRTVesselSegmentor(const TRTVesselSegmentor&)            = delete;
    TRTVesselSegmentor& operator=(const TRTVesselSegmentor&) = delete;

    /**
     * 加载 TensorRT 推理引擎。
     *
     * @param metaJsonPath  export_trt.py 输出的 trt_meta.json 路径
     * @return true  = 引擎就绪，可调用 run()
     *         false = TRT 不可用 / 引擎文件找不到 / 版本不匹配
     */
    bool load(const std::string& metaJsonPath);

    /** 引擎是否就绪（load 成功后为 true）*/
    bool isReady() const { return m_ready; }

    /**
     * 3D 滑动窗口推理。
     *
     * @param floatVolume  HU 归一化 [0,1] 体数据，row-major [D][H][W]
     * @param D/H/W        体数据尺寸
     * @param overlap      滑动窗口重叠率 0~1（越大越准，越慢）
     * @param threshold    血管概率二值化阈值（默认 0.5）
     * @return RegionGrowingResult，格式与区域生长结果相同，可直接上传 GPU 纹理
     */
    RegionGrowingResult run(
        const float* floatVolume,
        uint32_t D, uint32_t H, uint32_t W,
        float overlap   = 0.5f,
        float threshold = 0.5f
    );

    const TRTVesselMeta& meta() const { return m_meta; }

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
    TRTVesselMeta         m_meta;
    bool                  m_ready = false;
};
