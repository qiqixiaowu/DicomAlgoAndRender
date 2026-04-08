/**
 * trt_vessel_seg.cpp
 * ===================
 * TensorRT C++ 血管分割推理实现
 *
 * 编译开关
 * --------
 *   不定义 USE_TENSORRT：只提供 buildVolumeFloat / parseTRTVesselMeta，
 *                         TRTVesselSegmentor::load() 直接返回 false。
 *   定义   USE_TENSORRT：完整 TRT 推理路径（需链接 nvinfer.lib / cudart.lib）。
 *
 * TRT API 版本兼容
 * ----------------
 *   TRT 8 / 9 : getNbBindings / bindingIsInput / executeV2(bindings)
 *   TRT 10+   : getNbIOTensors / getTensorIOMode / setTensorAddress + enqueueV3
 *   通过 NV_TENSORRT_MAJOR（NvInfer.h 中定义）在编译期分派。
 */

#include "trt_vessel_seg.hpp"

// DCMTK — 用于 buildVolumeFloat 读取原始像素 + HU 变换
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/dcmimgle/dcmimage.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

// ============================================================
//  TRT / CUDA 条件编译
// ============================================================
#ifdef USE_TENSORRT
#  include <NvInfer.h>
#  include <cuda_runtime_api.h>
using namespace nvinfer1;

#  ifndef NV_TENSORRT_MAJOR         // 旧版 TRT 头未定义时的保底
#    define NV_TENSORRT_MAJOR 8
#  endif

// 最小化日志器（只输出 WARNING 及以上）
class TRTInternalLogger final : public ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING && msg && *msg)
            std::cerr << "[TRT] " << msg << "\n";
    }
};
static TRTInternalLogger g_trtLogger;

// 版本兼容析构（TRT 10 废弃 destroy()，改用 delete）
template<typename T>
static inline void trtDel(T*& p) {
    if (!p) return;
#  if NV_TENSORRT_MAJOR >= 10
    delete p;
#  else
    p->destroy();
#  endif
    p = nullptr;
}

#endif // USE_TENSORRT

// ============================================================
//  Pimpl — TRT 对象只在 USE_TENSORRT 中实例化
// ============================================================
struct TRTVesselSegmentor::Impl {
#ifdef USE_TENSORRT
    IRuntime*          runtime = nullptr;
    ICudaEngine*       engine  = nullptr;
    IExecutionContext* context = nullptr;

    void*  devIn  = nullptr;   // CUDA 设备端输入缓冲
    void*  devOut = nullptr;   // CUDA 设备端输出缓冲
    size_t inBytes  = 0;
    size_t outBytes = 0;

    std::string inpName;  // 输入张量名（TRT 10+按名索引）
    std::string outName;  // 输出张量名
    int inpIdx = 0;       // 绑定序号（TRT 8/9 按序索引）
    int outIdx = 1;

    ~Impl() {
        if (devIn)  { cudaFree(devIn);  devIn  = nullptr; }
        if (devOut) { cudaFree(devOut); devOut = nullptr; }
        trtDel(context);
        trtDel(engine);
        trtDel(runtime);
    }

    /**
     * 单 patch 推理（同步）
     * hostIn  : (1,1,pd,ph,pw) float32
     * hostOut : (1,C,pd,ph,pw) float32
     */
    bool inferPatch(const float* hostIn, float* hostOut) {
        if (cudaMemcpy(devIn, hostIn, inBytes, cudaMemcpyHostToDevice) != cudaSuccess)
            return false;

        bool ok          = false;
        cudaStream_t str = nullptr;
        cudaStreamCreate(&str);

#  if NV_TENSORRT_MAJOR >= 10
        context->setTensorAddress(inpName.c_str(), devIn);
        context->setTensorAddress(outName.c_str(), devOut);
        ok = context->enqueueV3(str);
#  else
        void* bindings[2];
        bindings[inpIdx] = devIn;
        bindings[outIdx] = devOut;
        ok = context->enqueueV2(bindings, str, nullptr);
#  endif
        cudaStreamSynchronize(str);
        cudaStreamDestroy(str);

        if (!ok) { std::cerr << "[TRT] enqueue 失败\n"; return false; }
        return cudaMemcpy(hostOut, devOut, outBytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
#endif // USE_TENSORRT
};

// ============================================================
//  构造 / 析构
// ============================================================
TRTVesselSegmentor::TRTVesselSegmentor()  : m_impl(std::make_unique<Impl>()) {}
TRTVesselSegmentor::~TRTVesselSegmentor() = default;

// ============================================================
//  parseTRTVesselMeta — 手动解析 trt_meta.json
// ============================================================
TRTVesselMeta parseTRTVesselMeta(const std::string& jsonPath) {
    TRTVesselMeta meta;
    std::ifstream f(jsonPath);
    if (!f.is_open()) {
        std::cerr << "[TRT] 无法打开 meta 文件: " << jsonPath << "\n";
        return meta;
    }

    // 提取字符串值
    auto extractStr = [](const std::string& line,
                         const std::string& key, std::string& val) -> bool {
        auto pos = line.find("\"" + key + "\"");
        if (pos == std::string::npos) return false;
        auto c = line.find(':', pos);
        if (c == std::string::npos) return false;
        auto q1 = line.find('"', c + 1);
        if (q1 == std::string::npos) return false;
        auto q2 = line.find('"', q1 + 1);
        if (q2 == std::string::npos) return false;
        val = line.substr(q1 + 1, q2 - q1 - 1);
        return true;
    };
    // 提取数值
    auto extractNum = [](const std::string& line,
                         const std::string& key, std::string& numStr) -> bool {
        auto pos = line.find("\"" + key + "\"");
        if (pos == std::string::npos) return false;
        auto c = line.find(':', pos);
        if (c == std::string::npos) return false;
        numStr = line.substr(c + 1);
        return true;
    };

    std::string line;
    bool inPatch = false;
    int  patchIdx = 0;

    while (std::getline(f, line)) {
        std::string tmp;
        if (extractStr(line, "engine_path", tmp)) meta.enginePath = tmp;
        if (extractStr(line, "onnx_path",   tmp)) meta.onnxPath   = tmp;
        if (extractNum(line, "hu_min",      tmp)) {
            try { meta.huMin = std::stof(tmp); } catch (...) {}
        }
        if (extractNum(line, "hu_max",      tmp)) {
            try { meta.huMax = std::stof(tmp); } catch (...) {}
        }
        if (extractNum(line, "out_channels",tmp)) {
            try { meta.outChannels = std::stoi(tmp); } catch (...) {}
        }
        // patch_size 是数组，可能写成单行 [64, 128, 128] 或多行
        if (line.find("\"patch_size\"") != std::string::npos) {
            inPatch  = true;
            patchIdx = 0;
            // 同行inline数组 e.g. "patch_size": [64, 128, 128]
            auto bracket = line.find('[');
            if (bracket != std::string::npos) {
                const char* p = line.c_str() + bracket + 1;
                while (*p && patchIdx < 3) {
                    while (*p && (*p < '0' || *p > '9') && *p != ']') ++p;
                    if (*p == ']' || !*p) break;
                    int n = std::atoi(p);
                    if (patchIdx == 0) meta.patchD = n;
                    else if (patchIdx == 1) meta.patchH = n;
                    else                    meta.patchW = n;
                    ++patchIdx;
                    while (*p && (*p >= '0' && *p <= '9')) ++p;
                }
                if (patchIdx >= 3) inPatch = false;
            }
        } else if (inPatch && patchIdx < 3) {
            // 多行格式：下几行每行一个数字
            const char* p = line.c_str();
            while (*p && (*p < '0' || *p > '9') && *p != ']') ++p;
            if (*p == ']') { inPatch = false; }
            else if (*p >= '0' && *p <= '9') {
                int n = std::atoi(p);
                if (patchIdx == 0) meta.patchD = n;
                else if (patchIdx == 1) meta.patchH = n;
                else                    meta.patchW = n;
                if (++patchIdx >= 3) inPatch = false;
            }
        }
    }
    return meta;
}

// ============================================================
//  buildVolumeFloat — DICOM → HU float32 [0,1]
// ============================================================
std::vector<float> buildVolumeFloat(
    const SeriesData& series,
    uint32_t& outD, uint32_t& outH, uint32_t& outW,
    float huMin, float huMax)
{
    if (series.slices.empty()) return {};
    const float range = huMax - huMin;
    if (range <= 0.f) return {};

    // 从第一张获取尺寸
    DcmFileFormat ff0;
    if (!ff0.loadFile(series.slices.front().filepath.c_str()).good()) return {};
    DcmDataset* ds0 = ff0.getDataset();
    if (!ds0) return {};

    Uint16 rows = 0, cols = 0;
    ds0->findAndGetUint16(DCM_Rows,    rows);
    ds0->findAndGetUint16(DCM_Columns, cols);
    if (!rows || !cols) return {};

    outH = rows;
    outW = cols;
    outD = static_cast<uint32_t>(series.slices.size());

    const size_t slicePx = (size_t)rows * cols;
    std::vector<float> out(outD * slicePx, 0.f);

    for (uint32_t zi = 0; zi < outD; ++zi) {
        DcmFileFormat ffz;
        if (!ffz.loadFile(series.slices[zi].filepath.c_str()).good()) continue;
        DcmDataset* dss = ffz.getDataset();
        if (!dss) continue;

        // Modality LUT：HU = pixel * slope + intercept
        Float64 slope = 1.0, intercept = 0.0;
        dss->findAndGetFloat64(DCM_RescaleSlope,     slope);
        dss->findAndGetFloat64(DCM_RescaleIntercept, intercept);

        // 像素表示：0=unsigned, 1=signed（大多数 CT 为 signed）
        Uint16 pixRep = 0;
        dss->findAndGetUint16(DCM_PixelRepresentation, pixRep);

        float* dst    = out.data() + (size_t)zi * slicePx;
        bool   filled = false;

        if (pixRep == 0) {
            const Uint16*       rawU = nullptr;
            unsigned long       cnt  = 0;
            if (dss->findAndGetUint16Array(DCM_PixelData, rawU, &cnt).good()
                && rawU && cnt >= slicePx)
            {
                for (size_t i = 0; i < slicePx; ++i) {
                    float hu = (float)rawU[i] * (float)slope + (float)intercept;
                    dst[i]   = ((std::max)(huMin, (std::min)(huMax, hu)) - huMin) / range;
                }
                filled = true;
            }
        }
        if (!filled) {
            // 有符号 16-bit（CT 主流格式）
            const Sint16*       rawS = nullptr;
            unsigned long       cnt  = 0;
            if (dss->findAndGetSint16Array(DCM_PixelData, rawS, &cnt).good()
                && rawS && cnt >= slicePx)
            {
                for (size_t i = 0; i < slicePx; ++i) {
                    float hu = (float)rawS[i] * (float)slope + (float)intercept;
                    dst[i]   = ((std::max)(huMin, (std::min)(huMax, hu)) - huMin) / range;
                }
            }
        }
    }
    return out;
}

// ============================================================
//  内部辅助
// ============================================================

// 3D 高斯权重图（中心权高，边缘权低，减少拼接伪影）
static std::vector<float> makeGaussianW(int pd, int ph, int pw) {
    std::vector<float> w(size_t(pd) * ph * pw);
    const float sigma = 0.5f;
    for (int z = 0; z < pd; ++z) {
        float fz = pd > 1 ? 2.f * z / (pd - 1) - 1.f : 0.f;
        for (int y = 0; y < ph; ++y) {
            float fy = ph > 1 ? 2.f * y / (ph - 1) - 1.f : 0.f;
            for (int x = 0; x < pw; ++x) {
                float fx = pw > 1 ? 2.f * x / (pw - 1) - 1.f : 0.f;
                w[(z * ph + y) * pw + x] =
                    std::exp(-(fz*fz + fy*fy + fx*fx) / (2.f * sigma * sigma));
            }
        }
    }
    return w;
}

// 提取 patch，越界部分 zero-pad
static void extractPatch(
    const float* vol,
    uint32_t D, uint32_t H, uint32_t W,
    int z0, int y0, int x0,
    int pd, int ph, int pw,
    float* patch)
{
    std::fill(patch, patch + size_t(pd) * ph * pw, 0.f);
    int az0 = (std::max)(0, z0), az1 = (std::min)((int)D, z0 + pd);
    int ay0 = (std::max)(0, y0), ay1 = (std::min)((int)H, y0 + ph);
    int ax0 = (std::max)(0, x0), ax1 = (std::min)((int)W, x0 + pw);
    for (int z = az0; z < az1; ++z)
        for (int y = ay0; y < ay1; ++y)
            for (int x = ax0; x < ax1; ++x) {
                int pz = z - z0, py = y - y0, px = x - x0;
                patch[(pz * ph + py) * pw + px] =
                    vol[((size_t)z * H + y) * W + x];
            }
}

// 生成滑动窗口起点列表（确保末尾完整覆盖）
static std::vector<int> getWindowStarts(int length, int patch, int stride) {
    if (length <= patch) return { 0 };
    std::vector<int> s;
    for (int i = 0; i + patch <= length; i += stride)
        s.push_back(i);
    if (s.empty() || s.back() + patch < length)
        s.push_back(length - patch);
    return s;
}

// ============================================================
//  TRTVesselSegmentor::load
// ============================================================
bool TRTVesselSegmentor::load(const std::string& metaJsonPath) {
    m_ready = false;
    m_meta  = parseTRTVesselMeta(metaJsonPath);

#ifndef USE_TENSORRT
    std::cerr << "[TRT] 当前构建未启用 USE_TENSORRT。\n"
                 "      请在 VS 项目属性 → C/C++ → 预处理器 中添加 USE_TENSORRT，\n"
                 "      并确保 $(TRT_ROOT) 环境变量已指向 TensorRT 安装目录。\n"
                 "      如 TRT 未安装，可用 Python 生成 .raw 文件后由 C++ 加载。\n";
    return false;
#else
    // 检查引擎文件
    {
        std::ifstream probe(m_meta.enginePath, std::ios::binary);
        if (!probe.is_open()) {
            std::cerr << "[TRT] 找不到引擎文件: " << m_meta.enginePath << "\n"
                         "      请先运行: python dl_medical/export_trt.py\n";
            return false;
        }
    }

    // 创建 TRT Runtime
    m_impl->runtime = createInferRuntime(g_trtLogger);
    if (!m_impl->runtime) { std::cerr << "[TRT] createInferRuntime 失败\n"; return false; }

    // 读取并反序列化引擎
    std::ifstream ef(m_meta.enginePath, std::ios::binary);
    ef.seekg(0, std::ios::end);
    size_t sz = (size_t)ef.tellg();
    ef.seekg(0, std::ios::beg);
    std::vector<char> buf(sz);
    ef.read(buf.data(), sz);

    m_impl->engine = m_impl->runtime->deserializeCudaEngine(buf.data(), sz);
    if (!m_impl->engine) {
        std::cerr << "[TRT] 引擎反序列化失败（请检查 GPU 型号/TRT 版本是否与生成时一致）\n";
        return false;
    }

    m_impl->context = m_impl->engine->createExecutionContext();
    if (!m_impl->context) { std::cerr << "[TRT] createExecutionContext 失败\n"; return false; }

    // 获取输入/输出张量信息（兼容 TRT 8/9/10）
#  if NV_TENSORRT_MAJOR >= 10
    {
        int n = m_impl->engine->getNbIOTensors();
        for (int i = 0; i < n; ++i) {
            const char* nm = m_impl->engine->getIOTensorName(i);
            auto mode = m_impl->engine->getTensorIOMode(nm);
            if (mode == TensorIOMode::kINPUT)  m_impl->inpName = nm;
            if (mode == TensorIOMode::kOUTPUT) m_impl->outName = nm;
        }
    }
#  else
    {
        int n = m_impl->engine->getNbBindings();
        for (int i = 0; i < n; ++i) {
            const char* nm = m_impl->engine->getBindingName(i);
            if (m_impl->engine->bindingIsInput(i)) {
                m_impl->inpName = nm; m_impl->inpIdx = i;
            } else {
                m_impl->outName = nm; m_impl->outIdx = i;
            }
        }
    }
#  endif

    // 分配 CUDA 设备缓冲
    const int pd = m_meta.patchD, ph = m_meta.patchH, pw = m_meta.patchW, C = m_meta.outChannels;
    m_impl->inBytes  = sizeof(float) * pd * ph * pw;          // (1,1,D,H,W)
    m_impl->outBytes = sizeof(float) * C  * pd * ph * pw;     // (1,C,D,H,W)
    if (cudaMalloc(&m_impl->devIn,  m_impl->inBytes)  != cudaSuccess ||
        cudaMalloc(&m_impl->devOut, m_impl->outBytes) != cudaSuccess) {
        std::cerr << "[TRT] CUDA 设备内存分配失败（显存不足？当前请求 "
                  << (m_impl->inBytes + m_impl->outBytes) / 1024 << " KB）\n";
        return false;
    }

    m_ready = true;
    std::cout << "[TRT] 引擎加载完成\n"
              << "       文件  : " << m_meta.enginePath << "\n"
              << "       patch : " << pd << " x " << ph << " x " << pw
              << "  out_ch=" << C << "\n"
              << "       输入  : '" << m_impl->inpName << "'\n"
              << "       输出  : '" << m_impl->outName << "'\n";
    return true;
#endif // USE_TENSORRT
}

// ============================================================
//  TRTVesselSegmentor::run
// ============================================================
RegionGrowingResult TRTVesselSegmentor::run(
    const float* floatVolume,
    uint32_t D, uint32_t H, uint32_t W,
    float overlap,
    float threshold)
{
    RegionGrowingResult result;
    result.width  = W;
    result.height = H;
    result.depth  = D;

#ifndef USE_TENSORRT
    std::cerr << "[TRT] USE_TENSORRT 未定义，run() 无效\n";
    return result;
#else
    if (!m_ready || !m_impl || !m_impl->context) {
        std::cerr << "[TRT] 推理器未就绪，请先调用 load()\n";
        return result;
    }

    const int pd = m_meta.patchD, ph = m_meta.patchH, pw = m_meta.patchW;
    const int C  = m_meta.outChannels;

    // ---- 步长 ----
    const int sz = std::max(1, (int)(pd * (1.f - overlap)));
    const int sy = std::max(1, (int)(ph * (1.f - overlap)));
    const int sx = std::max(1, (int)(pw * (1.f - overlap)));

    auto sZ = getWindowStarts((int)D, pd, sz);
    auto sY = getWindowStarts((int)H, ph, sy);
    auto sX = getWindowStarts((int)W, pw, sx);
    const int total = (int)(sZ.size() * sY.size() * sX.size());
    std::cout << "[TRT] 开始推理，共 " << total << " 个 patch\n";

    // ---- 高斯权重图 ----
    const auto gaussW = makeGaussianW(pd, ph, pw);

    // ---- 概率累加缓冲 ----
    const size_t vol = (size_t)D * H * W;
    std::vector<float> probSum  (vol, 0.f);
    std::vector<float> weightSum(vol, 0.f);

    // ---- 主机端 patch 缓冲 ----
    std::vector<float> patchBuf(size_t(pd) * ph * pw);
    std::vector<float> outBuf  (size_t(C)  * pd * ph * pw);

    const size_t patchVol = size_t(pd) * ph * pw;
    int done = 0;
    auto t0  = std::chrono::steady_clock::now();

    for (int z0 : sZ) {
        for (int y0 : sY) {
            for (int x0 : sX) {
                // 提取 patch（越界 zero-pad）
                extractPatch(floatVolume, D, H, W, z0, y0, x0, pd, ph, pw, patchBuf.data());

                // TRT 推理
                if (!m_impl->inferPatch(patchBuf.data(), outBuf.data())) {
                    std::cerr << "[TRT] patch(" << z0 << "," << y0 << "," << x0 << ") 推理失败\n";
                    ++done;
                    continue;
                }

                // Softmax channel 1 → 血管概率；outBuf layout: [c][pd][ph][pw]
                const float* log0 = outBuf.data();
                const float* log1 = outBuf.data() + patchVol;

                // 写回到有效（非 pad）区域
                int az0 = std::max(0, z0), az1 = std::min((int)D, z0 + pd);
                int ay0 = std::max(0, y0), ay1 = std::min((int)H, y0 + ph);
                int ax0 = std::max(0, x0), ax1 = std::min((int)W, x0 + pw);
                for (int z = az0; z < az1; ++z) {
                    for (int y = ay0; y < ay1; ++y) {
                        for (int x = ax0; x < ax1; ++x) {
                            int  pi  = ((z - z0) * ph + (y - y0)) * pw + (x - x0);
                            float mx  = std::max(log0[pi], log1[pi]);
                            float e0  = std::exp(log0[pi] - mx);
                            float e1  = std::exp(log1[pi] - mx);
                            float pr  = e1 / (e0 + e1 + 1e-8f);
                            float gw  = gaussW[pi];
                            size_t vi = ((size_t)z * H + y) * W + x;
                            probSum  [vi] += pr * gw;
                            weightSum[vi] += gw;
                        }
                    }
                }

                ++done;
                if (total >= 10 && done % (total / 10) == 0) {
                    auto elapsed = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count();
                    std::cout << "  [TRT] " << done << "/" << total
                              << " (" << 100 * done / total << "%)"
                              << "  已用 " << (int)elapsed << "s\n" << std::flush;
                }
            }
        }
    }

    // ---- 加权平均 → 二值化 ----
    result.mask.resize(vol, 0);
    uint64_t cnt = 0;
    for (size_t i = 0; i < vol; ++i) {
        float pr = weightSum[i] > 0.f ? probSum[i] / weightSum[i] : 0.f;
        if (pr >= threshold) { result.mask[i] = 255; ++cnt; }
    }
    result.voxelCount = cnt;

    auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::cout << "[TRT] 推理完成  耗时=" << elapsed << "s"
              << "  血管体素=" << cnt
              << " (" << 100.f * cnt / vol << "%)\n";
    return result;
#endif // USE_TENSORRT
}
