/**
 * @file main.cpp
 * @brief OncologyAlgorithms 演示入口
 *
 * 本文件演示工程中每个核心算法模块的典型用法：
 *
 *  1. 肿瘤径线计算（2D最大截面径 / 3D最长径）& 体积统计
 *  2. 通用肿瘤半自动分割（K-means / GMM / Random Walker）
 *  3. PET 病灶分割（Fixed / Percent / Adaptive 三种模式）
 *  4. CT 肺结节传统分割（Hessian + 多形态精化）
 *  5. DL 肺结节分割（3D nnUNet，CPU 路径演示）
 *
 * 所有演示均使用内存合成数据，无需任何外部文件。
 * 实际应用中请替换为真实的影像数据。
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>

// ─────────────────────────────────────────────
//  OncologyAlgorithms 模块头文件
// ─────────────────────────────────────────────
#include "oncology_types.h"
#include "tumor_diameter_calculator.h"
#include "general_tumor_segment.h"
#include "pet_lesion_segment.h"
#include "lung_nodule_segment_traditional.h"
#include "lung_nodule_segment_dl.h"

using namespace Onc;

// ═══════════════════════════════════════════════════════
//  工具函数
// ═══════════════════════════════════════════════════════

/** 打印分隔线 */
static void printSep(const char* title) {
    std::cout << "\n";
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << "  " << title << "\n";
    std::cout << "═══════════════════════════════════════════════════\n";
}

/** 进度回调：打印进度条 */
static bool progressCB(double p) {
    int barLen = 30;
    int filled = static_cast<int>(p * barLen);
    std::cout << "\r  [";
    for (int i = 0; i < barLen; ++i) std::cout << (i < filled ? '#' : '.');
    std::cout << "] " << std::fixed << std::setprecision(1) << p*100 << "%  " << std::flush;
    if (p >= 1.0) std::cout << "\n";
    return true;
}

// ═══════════════════════════════════════════════════════
//  合成测试数据生成
// ═══════════════════════════════════════════════════════

/**
 * @brief 生成一个含有球形肿瘤的合成 CT 图像
 *
 * 肺背景约 -700 HU，球形肿瘤区域约 +50 HU（实性结节），
 * 球心在图像中心，半径 r_voxels 个体素。
 */
static std::vector<short> makeSyntheticCT(
    const ImageInfo& info,
    const Point3i&   tumorCenter,
    int              tumorRadiusVoxels,
    short            tumorHU  = 50,
    short            bgHU     = -700)
{
    int N = info.totalVoxels();
    std::vector<short> img(N, bgHU);

    // 添加高斯噪声
    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.f, 30.f);

    for (int z = 0; z < info.dim[2]; ++z)
    for (int y = 0; y < info.dim[1]; ++y)
    for (int x = 0; x < info.dim[0]; ++x) {
        int idx = info.linearIndex(x, y, z);
        float dx = (x - tumorCenter.x) * (float)info.spacing[0];
        float dy = (y - tumorCenter.y) * (float)info.spacing[1];
        float dz = (z - tumorCenter.z) * (float)info.spacing[2];
        float r  = std::sqrt(dx*dx + dy*dy + dz*dz);
        float rMM = tumorRadiusVoxels * (float)info.spacing[0];
        if (r <= rMM) {
            img[idx] = static_cast<short>(tumorHU + (short)noise(rng));
        } else {
            img[idx] = static_cast<short>(bgHU + (short)noise(rng));
        }
    }
    return img;
}

/**
 * @brief 围绕肿瘤中心生成一个球形"已知分割"结果（黄金标准）
 */
static SegmentResult makeSphereSeg(
    const ImageInfo& info,
    const Point3i&   center,
    int              radiusVoxels)
{
    SegmentResult result;
    for (int z = center.z - radiusVoxels; z <= center.z + radiusVoxels; ++z)
    for (int y = center.y - radiusVoxels; y <= center.y + radiusVoxels; ++y)
    for (int x = center.x - radiusVoxels; x <= center.x + radiusVoxels; ++x) {
        if (!info.inBounds(x,y,z)) continue;
        int dx=x-center.x, dy=y-center.y, dz=z-center.z;
        if (dx*dx + dy*dy + dz*dz <= radiusVoxels*radiusVoxels)
            result.push_back({x,y,z});
    }
    return result;
}

/**
 * @brief 生成合成 PET 图像（球形病灶高摄取）
 *
 * 原始灰度值与 SUVbw 的线性关系由 SUVInfo 描述。
 * 病灶区域 SUVbw 约 8.0，背景肝脏约 2.0。
 */
static std::vector<float> makeSyntheticPET(
    const ImageInfo& info,
    const SUVInfo&   suvInfo,
    const Point3i&   lesionCenter,
    int              lesionRadiusVoxels,
    double           lesionSUV = 8.0,
    double           bgSUV     = 1.5)
{
    int N = info.totalVoxels();
    std::vector<float> img(N);
    std::mt19937 rng(123);
    std::normal_distribution<float> noise(0.f, 0.1f);

    // suvToRaw 线性插值
    auto suvToRaw = [&](double suv) -> float {
        if (!suvInfo.hasValidSUV) return (float)suv * 100.f; // 退化
        double slope = (suvInfo.rawAtSUV2 - suvInfo.rawAtSUV1) /
                       (suvInfo.suv2 - suvInfo.suv1);
        double raw = suvInfo.rawAtSUV1 + slope * (suv - suvInfo.suv1);
        return (float)raw;
    };

    for (int z = 0; z < info.dim[2]; ++z)
    for (int y = 0; y < info.dim[1]; ++y)
    for (int x = 0; x < info.dim[0]; ++x) {
        int idx = info.linearIndex(x, y, z);
        float dx=(float)(x-lesionCenter.x), dy=(float)(y-lesionCenter.y), dz=(float)(z-lesionCenter.z);
        float r = std::sqrt(dx*dx+dy*dy+dz*dz);
        bool inLesion = r <= (float)lesionRadiusVoxels;
        double suv = inLesion ? lesionSUV : bgSUV;
        img[idx] = suvToRaw(suv) + noise(rng);
    }
    return img;
}

// ═══════════════════════════════════════════════════════
//  各模块演示函数
// ═══════════════════════════════════════════════════════

// ───────────────────────────────────────────────────────
//  演示1：肿瘤径线计算 & VOI 综合统计
// ───────────────────────────────────────────────────────
static void demoTumorDiameters() {
    printSep("DEMO 1: VOI 综合统计（径线 + 灰度 + SUV + 多标签）");

    // ── 1a) 单 ROI CT 统计 ────────────────────────────
    std::cout << "\n  >> 单 ROI CT 统计\n";
    ImageInfo info;
    info.dim[0] = info.dim[1] = info.dim[2] = 64;
    info.spacing[0] = info.spacing[1] = info.spacing[2] = 1.0;

    Point3i center{32, 32, 32};
    int r = 8;
    SegmentResult seg = makeSphereSeg(info, center, r);
    auto ctData = makeSyntheticCT(info, center, r);

    VOIStatistic voiStat;
    bool ok = CalculateVOIStatistic(ctData.data(), info, seg, voiStat,
                                     nullptr, true, 10000.0);
    if (ok) {
        std::cout << "  体素数       : " << voiStat.elemCount << "\n";
        std::cout << "  体积         : " << std::fixed << std::setprecision(1)
                  << voiStat.volume_mm3 << " mm^3"
                  << "  (理论 " << (4.0/3.0*3.14159*r*r*r) << ")\n";
        std::cout << "  均值 HU      : " << std::setprecision(1) << voiStat.mean << "\n";
        std::cout << "  标准差       : " << voiStat.stdDev << "\n";
        std::cout << "  [HU range]   : [" << voiStat.minDcm << ", " << voiStat.maxDcm << "]\n";
        std::cout << "  AABB min     : (" << voiStat.iMinRange[0] << ", "
                  << voiStat.iMinRange[1] << ", " << voiStat.iMinRange[2] << ")\n";
        std::cout << "  AABB max     : (" << voiStat.iMaxRange[0] << ", "
                  << voiStat.iMaxRange[1] << ", " << voiStat.iMaxRange[2] << ")\n";
        std::cout << "  2D 长径      : " << std::setprecision(2)
                  << voiStat.longAxis_mm << " mm\n";
        std::cout << "  2D 垂直径    : " << voiStat.shortAxis_mm << " mm\n";
        std::cout << "  3D 最长径    : " << voiStat.maxDiameter_mm << " mm\n";
    }

    // ── 1b) 单 ROI PET 统计（含 SUV）────────────────
    std::cout << "\n  >> 单 ROI PET 统计（含 SUVmax/mean/peak）\n";
    ImageInfo petInfo;
    petInfo.dim[0] = petInfo.dim[1] = petInfo.dim[2] = 64;
    petInfo.spacing[0] = petInfo.spacing[1] = petInfo.spacing[2] = 4.0;

    SUVInfo suvInfo;
    suvInfo.hasValidSUV = true;
    suvInfo.rawAtSUV1   = 100.0;
    suvInfo.rawAtSUV2   = 200.0;
    suvInfo.slope       = 0.01;
    suvInfo.intercept   = 0.0;

    Point3i lesionCenter{32, 32, 32};
    auto petData = makeSyntheticPET(petInfo, suvInfo, lesionCenter, 5, 8.0, 1.5);
    SegmentResult petSeg = makeSphereSeg(petInfo, lesionCenter, 5);

    VOIStatistic petStat;
    ok = CalculateVOIStatistic(petData.data(), petInfo, petSeg, petStat,
                                &suvInfo, true, 10000.0);
    if (ok) {
        std::cout << "  体素数       : " << petStat.elemCount << "\n";
        std::cout << "  体积         : " << std::setprecision(1) << petStat.volume_mm3 << " mm^3\n";
        std::cout << "  SUVmax       : " << std::setprecision(2) << petStat.suvMax << "\n";
        std::cout << "  SUVmean      : " << petStat.suvMean << "\n";
        std::cout << "  SUVmin       : " << petStat.suvMin << "\n";
        std::cout << "  SUVpeak      : " << petStat.suvPeak << "\n";
        std::cout << "  2D 长径      : " << petStat.longAxis_mm << " mm\n";
    }

    // ── 1c) 多标签 mask 统计 ──────────────────────────
    std::cout << "\n  >> 多标签 mask 统计（label=1 肿瘤, label=2 淋巴结）\n";
    ImageInfo mInfo;
    mInfo.dim[0] = mInfo.dim[1] = mInfo.dim[2] = 64;
    mInfo.spacing[0] = mInfo.spacing[1] = mInfo.spacing[2] = 1.0;

    auto mCtData = makeSyntheticCT(mInfo, center, r);

    // 构建多标签 mask
    std::vector<uint8_t> mask(mInfo.totalVoxels(), 0);
    // label=1: 球心 (32,32,32) 半径 8
    for (int z = 24; z <= 40; ++z)
    for (int y = 24; y <= 40; ++y)
    for (int x = 24; x <= 40; ++x) {
        int dx = x-32, dy = y-32, dz = z-32;
        if (dx*dx+dy*dy+dz*dz <= 64 && mInfo.inBounds(x,y,z))
            mask[mInfo.linearIndex(x,y,z)] = 1;
    }
    // label=2: 较小球体 (50,50,50) 半径 4
    for (int z = 46; z <= 54; ++z)
    for (int y = 46; y <= 54; ++y)
    for (int x = 46; x <= 54; ++x) {
        int dx = x-50, dy = y-50, dz = z-50;
        if (dx*dx+dy*dy+dz*dz <= 16 && mInfo.inBounds(x,y,z))
            mask[mInfo.linearIndex(x,y,z)] = 2;
    }

    std::vector<uint8_t> labelList = {1, 2};
    auto multiResult = CalculateMultiLabelVOIStatistics(
        mCtData.data(), mInfo, mask.data(), labelList,
        nullptr, true, 10000.0);

    for (uint8_t lb : labelList) {
        const auto& s = multiResult[lb];
        std::cout << "  [Label " << (int)lb << "]  "
                  << "体素=" << s.elemCount
                  << "  体积=" << std::setprecision(1) << s.volume_mm3 << " mm^3"
                  << "  均值=" << std::setprecision(1) << s.mean
                  << "  2D长径=" << std::setprecision(2) << s.longAxis_mm << " mm\n";
    }
}

// ───────────────────────────────────────────────────────
//  演示2：通用肿瘤半自动分割
// ───────────────────────────────────────────────────────
static void demoGeneralTumorSegment() {
    printSep("DEMO 2: 通用肿瘤分割（KMeansGMM / RandomWalker / 跨时间点传播）");

    ImageInfo info;
    info.dim[0] = info.dim[1] = info.dim[2] = 64;
    info.spacing[0] = info.spacing[1] = info.spacing[2] = 1.5;

    Point3i center{32, 32, 32};
    auto ctData = makeSyntheticCT(info, center, 8);

    // 用户在横断位画了一条穿过肿瘤的长轴线（± 8 体素）
    Point3i ldStart{24, 32, 32};
    Point3i ldEnd  {40, 32, 32};

    // ── KMeansGMM（传统）──────────────────────────────
    GeneralSegmentConfig cfg;
    cfg.method   = SegmentMethod::KMeansGMM;
    cfg.modality = ImageModality::CT;

    GeneralTumorSegmentation seg(cfg);
    SegmentResult res;
    bool ok = seg.segment(ctData.data(), info, ldStart, ldEnd, res, progressCB);
    std::cout << "  KMeansGMM    分割体素数: " << (ok ? (int)res.size() : -1)
              << "  实际方法: " << (int)seg.lastUsedMethod() << "\n";

    // ── RandomWalker ─────────────────────────────────
    cfg.method = SegmentMethod::RandomWalker;
    GeneralTumorSegmentation seg2(cfg);
    ok = seg2.segment(ctData.data(), info, ldStart, ldEnd, res, progressCB);
    std::cout << "  RandomWalker 分割体素数: " << (ok ? (int)res.size() : -1) << "\n";

    // ── 跨时间点传播演示 ──────────────────────────────
    // 模拟随访：肿瘤略微偏移到 (33,32,31)
    Point3i center2{33, 32, 31};
    auto ctData2 = makeSyntheticCT(info, center2, 8);
    SegmentResult srcSeg = makeSphereSeg(info, center, 8);

    // 传播：将上一时间点的线段和分割结果传播到新图像
    Point3i newLdStart, newLdEnd;
    SegmentResult spreadRes;
    cfg.method = SegmentMethod::KMeansGMM;
    GeneralTumorSegmentation seg3(cfg);
    ok = seg3.spreadSegmentation(ctData2.data(), info,
                                  ldStart, ldEnd, srcSeg,
                                  newLdStart, newLdEnd, spreadRes,
                                  progressCB);
    std::cout << "  传播分割     结果体素数: " << (ok ? (int)spreadRes.size() : -1) << "\n";
}

// ───────────────────────────────────────────────────────
//  演示3：PET 病灶分割
// ───────────────────────────────────────────────────────
static void demoPETSegment() {
    printSep("DEMO 3: PET 分子影像病灶分割（Fixed / Percent / Adaptive）");

    ImageInfo info;
    info.dim[0] = info.dim[1] = info.dim[2] = 64;
    info.spacing[0] = info.spacing[1] = info.spacing[2] = 4.0; // PET 体素较大

    // 构建 SUVInfo（SUVbw=1 → raw=100，SUVbw=2 → raw=200）
    SUVInfo suvInfo;
    suvInfo.hasValidSUV  = true;
    suvInfo.rawAtSUV1    = 100.0;
    suvInfo.rawAtSUV2    = 200.0;
    suvInfo.suv1         = 1.0;
    suvInfo.suv2         = 2.0;

    Point3i lesionCenter{32, 32, 32};
    // 函数接口要求 const int[3]，将 Point3i 转为数组传入
    int lcArr[3] = {lesionCenter.x, lesionCenter.y, lesionCenter.z};
    suvInfo.ImgType = "MET_FLOAT"; // PET 合成数据为 float 类型
    // slope/intercept 由 rawAtSUV1/2 计算（SUV = slope*raw + intercept）
    suvInfo.slope     = 1.0 / 100.0;  // raw=100 → SUV=1
    suvInfo.intercept = 0.0;
    auto petData = makeSyntheticPET(info, suvInfo, lesionCenter, 5, 8.0, 1.5);

    PETSegmentResult res;

    // ── Fixed SUV = 2.5 ──────────────────────────────
    bool ok = PETSegmentFixed(suvInfo,
                               reinterpret_cast<const void*>(petData.data()),
                               info, lcArr, 2.5, res);
    std::cout << "  Fixed(SUV=2.5)  体素数: " << (ok ? (int)res.voxels.size() : -1)
              << "  阈值=" << std::setprecision(1) << res.usedThreshold << "\n";

    // ── Percent 42% SUVmax ────────────────────────────
    ok = PETSegmentPercent(suvInfo,
                            reinterpret_cast<const void*>(petData.data()),
                            info, lcArr, 42.0, res);
    std::cout << "  Percent(42%)    体素数: " << (ok ? (int)res.voxels.size() : -1)
              << "  阈值=" << res.usedThreshold << "\n";

    // ── Adaptive ─────────────────────────────────────
    double adaptWeight = 0.5;  // 初始权重，函数会自适应更新
    ok = PETSegmentAdaptive(suvInfo,
                             reinterpret_cast<const void*>(petData.data()),
                             info, lcArr, adaptWeight, res);
    std::cout << "  Adaptive        体素数: " << (ok ? (int)res.voxels.size() : -1)
              << "  阈值=" << res.usedThreshold << "  weight=" << adaptWeight << "\n";

    // ── Hover 预览（返回轮廓体素数）──────────────────
    PETSegmentResult hoverRes;
    ok = PETSegmentAdaptiveHover(suvInfo,
                                  reinterpret_cast<const void*>(petData.data()),
                                  info, lcArr, adaptWeight, hoverRes);
    std::cout << "  Hover预览       体素数: " << (ok ? (int)hoverRes.voxels.size() : -1) << "\n";
}

// ───────────────────────────────────────────────────────
//  演示4：CT 肺结节传统分割
// ───────────────────────────────────────────────────────
static void demoLungNoduleTraditional() {
    printSep("DEMO 4: CT 肺结节传统分割（Hessian + 多形态精化）");

    ImageInfo info;
    info.dim[0] = info.dim[1] = info.dim[2] = 80;
    info.spacing[0] = info.spacing[1] = info.spacing[2] = 0.5; // 高分辨率 CT

    Point3i seed{40, 40, 40};
    // 实性结节 HU ~50，肺背景 ~-700
    auto ctData = makeSyntheticCT(info, seed, 10, 50, -700);

    LungNoduleSegParams params;
    params.noduleType      = NoduleType::Auto;
    params.roiRadiusMM     = 15;
    params.hessianScale1   = 1.0f;
    params.hessianScale2   = 2.5f;
    params.enableConvexHull = true;

    LungNoduleSegmentTraditional seg(params);
    SegmentResult result;
    bool ok = seg.segment(ctData.data(), info, seed, result, progressCB);

    if (ok) {
        std::cout << "  分割体素数   : " << result.size() << "\n";

        // 计算结节直径
        TumorDiameterResult diam;
        CalculateTumorDiameters(result, info, diam);
        std::cout << "  3D 最长径    : " << std::setprecision(2)
                  << diam.longestDiameter3D_cm << " cm\n";
        std::cout << "  2D 长径      : " << diam.longestDiameter2D_cm << " cm\n";
    } else {
        std::cout << "  [WARN] 分割失败（请检查 HU 阈值配置）\n";
    }

    // 多形态类型演示
    const char* typeNames[] = {"JuxtaWall", "JuxtaVessel", "GGO"};
    NoduleType  types[]     = {NoduleType::JuxtaWall,
                                NoduleType::JuxtaVessel,
                                NoduleType::GGO};
    for (int i = 0; i < 3; ++i) {
        params.noduleType = types[i];
        LungNoduleSegmentTraditional seg2(params);
        SegmentResult r2;
        ok = seg2.segment(ctData.data(), info, seed, r2, nullptr);
        std::cout << "  类型[" << typeNames[i] << "]"
                  << "\t体素数: " << (ok ? (int)r2.size() : -1) << "\n";
    }
}

// ───────────────────────────────────────────────────────
//  演示5：DL 肺结节分割（3D nnUNet）
// ───────────────────────────────────────────────────────
static void demoLungNoduleDL() {
    printSep("DEMO 5: DL 肺结节分割（3D nnUNet，CPU 回退路径）");

    ImageInfo info;
    info.dim[0] = info.dim[1] = info.dim[2] = 96;
    info.spacing[0] = info.spacing[1] = info.spacing[2] = 0.625;

    Point3i seed{48, 48, 48};
    auto ctData = makeSyntheticCT(info, seed, 8);

    LungNoduleDLConfig cfg;
    cfg.patchSize[0] = cfg.patchSize[1] = cfg.patchSize[2] = 64;
    cfg.huMin  = -1000.f;
    cfg.huMax  = 400.f;
    cfg.useCUDA = false; // 演示中强制使用 CPU 路径

    LungNoduleDLSegment dlSeg(cfg);

    // 尝试加载权重（如果有）
    const std::string weightPath = "weights/lung_nodule_nnunet.bin";
    bool hasWeights = dlSeg.loadWeights(weightPath);
    if (!hasWeights) {
        std::cout << "  [INFO] 未找到权重文件 \"" << weightPath
                  << "\"，使用随机初始化演示推理流程...\n";
    }

    // GPU 信息
    bool gpuOk = dlSeg.checkGPUAvailable(2048);
    std::cout << "  GPU 可用     : " << (gpuOk ? "是" : "否（使用 CPU 回退）") << "\n";
    std::cout << "  预估显存需求 : " << dlSeg.estimateGPUMemMB() << " MB\n";

    SegmentResult result;
    bool ok = dlSeg.segment(ctData.data(), info, seed, result, progressCB);

    if (ok) {
        std::cout << "  分割体素数   : " << result.size() << "\n";
        TumorDiameterResult diam;
        CalculateTumorDiameters(result, info, diam);
        std::cout << "  3D 最长径    : " << std::setprecision(2)
                  << diam.longestDiameter3D_cm << " cm\n";
    } else {
        std::cout << "  [WARN] DL 推理未产生前景结果（正常：无真实权重时 logit 可能全为背景）\n";
    }
}

// ═══════════════════════════════════════════════════════
//  主函数
// ═══════════════════════════════════════════════════════
int main(int argc, char* argv[]) {
    std::cout << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║       OncologyAlgorithms 功能演示                ║\n";
    std::cout << "║  肿瘤影像分析算法库  v1.0                        ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n";

    // 解析命令行参数：可选择只运行指定演示
    int runMask = 0x1F; // 默认全运行
    if (argc >= 2) runMask = std::atoi(argv[1]);

    try {
        if (runMask & 0x01) demoTumorDiameters();
        if (runMask & 0x02) demoGeneralTumorSegment();
        if (runMask & 0x04) demoPETSegment();
        if (runMask & 0x08) demoLungNoduleTraditional();
        if (runMask & 0x10) demoLungNoduleDL();
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n✓ 所有演示完成\n";
    return 0;
}
