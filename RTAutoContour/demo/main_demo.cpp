/**
 * ============================================================
 *  RTAutoContour 综合演示程序
 * ============================================================
 *
 * 演示以下核心算法：
 *   1. 形态学操作（腐蚀/膨胀/闭运算/孔洞填充）
 *   2. 连通域分析 + 区域属性
 *   3. 轮廓 → 掩膜 → 轮廓互转
 *   4. Live-Wire 智能剪刀
 *   5. 三线性/最近邻插值重采样
 *   6. CT 切片匹配 (ResetMatch)
 *   7. 金属定位点检测 (Fiducial Detection)
 *   8. 床板检测 (BedBoard Detection)
 *
 * 构建：
 *   cd RTAutoContour/build
 *   cmake ..
 *   cmake --build . --config Release
 *   ./rtac_demo
 * ============================================================
 */

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>

#include "rt_types.hpp"
#include "morphology.hpp"
#include "connected_component.hpp"
#include "contour_mask.hpp"
#include "live_wire.hpp"
#include "interpolation.hpp"
#include "ct_reset_match.hpp"
#include "fiducial_detection.hpp"
#include "bed_board_detection.hpp"

#ifdef USE_CUDA
#include "marching_squares_cuda.h"
#include <chrono>
#endif

using namespace rtac;

// ============================================================
//  辅助：生成测试图像
// ============================================================

/// 生成带圆形的 2D 图像
Image2D<float> makeCircleImage(int W, int H, int cx, int cy, int r, float fg = 200.f) {
    Image2D<float> img(W, H, 0.f);
    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
        float dx = static_cast<float>(x - cx);
        float dy = static_cast<float>(y - cy);
        if (dx * dx + dy * dy <= r * r)
            img.at(x, y) = fg;
    }
    return img;
}

/// 生成带圆形的 2D 掩膜
Image2D<uint8_t> makeCircleMask(int W, int H, int cx, int cy, int r) {
    Image2D<uint8_t> mask(W, H, 0);
    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
        float dx = static_cast<float>(x - cx);
        float dy = static_cast<float>(y - cy);
        if (dx * dx + dy * dy <= r * r)
            mask.at(x, y) = 255;
    }
    return mask;
}

// ============================================================
//  演示1: 形态学操作
// ============================================================
void demo_morphology() {
    std::cout << "\n=== 演示1: 形态学操作 ===\n";

    // 创建带小孔和突起的掩膜
    auto mask = makeCircleMask(64, 64, 32, 32, 15);
    // 人为挖孔
    mask.at(32, 32) = 0;
    mask.at(33, 32) = 0;
    mask.at(32, 33) = 0;
    // 人为加突起
    mask.at(48, 32) = 255;

    int countBefore = 0;
    for (size_t i = 0; i < mask.size(); ++i)
        if (mask[i] == 255) ++countBefore;

    // 闭运算（填小孔）
    Closing2D(mask, 1);
    int countAfterClose = 0;
    for (size_t i = 0; i < mask.size(); ++i)
        if (mask[i] == 255) ++countAfterClose;

    // 孔洞填充
    FillHoles2D(mask);
    int countAfterFill = 0;
    for (size_t i = 0; i < mask.size(); ++i)
        if (mask[i] == 255) ++countAfterFill;

    std::cout << "  原始前景像素: " << countBefore << "\n";
    std::cout << "  闭运算后: " << countAfterClose << "\n";
    std::cout << "  孔洞填充后: " << countAfterFill << "\n";

    // 保存结果
    savePGM("morphology_result.pgm", mask);
    std::cout << "  已保存 morphology_result.pgm\n";
}

// ============================================================
//  演示2: 连通域分析
// ============================================================
void demo_connected_component() {
    std::cout << "\n=== 演示2: 连通域分析 ===\n";

    Image2D<uint8_t> mask(64, 64, 0);
    // 放置两个不相连的圆
    for (int y = 0; y < 64; ++y)
    for (int x = 0; x < 64; ++x) {
        float d1 = std::sqrt(float((x-16)*(x-16) + (y-16)*(y-16)));
        float d2 = std::sqrt(float((x-48)*(x-48) + (y-48)*(y-48)));
        if (d1 <= 8 || d2 <= 10) mask.at(x, y) = 255;
    }

    auto lr = LabelConnectedComponents2D(mask, 255, false);
    std::cout << "  检测到 " << lr.domainCount << " 个连通域\n";

    for (int i = 1; i <= lr.domainCount; ++i) {
        auto rp = ComputeRegionProps(lr.labels, i);
        std::cout << "  域 " << i << ": 面积=" << rp.area
                  << ", 质心=(" << rp.centroidX << "," << rp.centroidY << ")"
                  << ", 偏心率=" << rp.eccentricity << "\n";
    }

    // 仅保留最大连通域
    KeepLargestComponent2D(mask);
    int remaining = 0;
    for (size_t i = 0; i < mask.size(); ++i) if (mask[i]) ++remaining;
    std::cout << "  保留最大域后剩余像素: " << remaining << "\n";
}

// ============================================================
//  演示3: 轮廓 ↔ 掩膜互转
// ============================================================
void demo_contour_mask() {
    std::cout << "\n=== 演示3: 轮廓 ↔ 掩膜互转 ===\n";

    // 生成圆形轮廓
    Contour2D circle;
    int N = 64, cx = 32, cy = 32, r = 15;
    for (int i = 0; i < N; ++i) {
        float angle = 2.0f * 3.14159f * i / N;
        circle.push_back({cx + r * std::cos(angle), cy + r * std::sin(angle)});
    }

    // 轮廓 → 掩膜
    auto mask = ContourToMask(circle, 64, 64, false);
    int maskCount = 0;
    for (size_t i = 0; i < mask.size(); ++i) if (mask[i]) ++maskCount;
    std::cout << "  轮廓→掩膜: " << maskCount << " 像素 (理论: ~"
              << int(3.14159 * r * r) << ")\n";

    // 5x 超采样
    auto maskSS = ContourToMask(circle, 64, 64, false, 255, 5);
    int maskSSCount = 0;
    for (size_t i = 0; i < maskSS.size(); ++i) if (maskSS[i]) ++maskSSCount;
    std::cout << "  5x超采样: " << maskSSCount << " 像素\n";

    // 掩膜 → 轮廓
    auto contours = MaskToContours(mask);
    std::cout << "  掩膜→轮廓: 检测到 " << contours.size() << " 个轮廓";
    if (!contours.empty())
        std::cout << ", 第一个有 " << contours[0].size() << " 个点";
    std::cout << "\n";

    savePGM("contour_mask_result.pgm", mask);
    std::cout << "  已保存 contour_mask_result.pgm\n";
}

// ============================================================
//  演示4: Live-Wire 智能剪刀
// ============================================================
void demo_live_wire() {
    std::cout << "\n=== 演示4: Live-Wire 智能剪刀 ===\n";

    // 创建带边缘的图像
    auto img = makeCircleImage(64, 64, 32, 32, 20, 200.f);

    LiveWire lw;
    lw.initialize(img);

    // 模拟用户点击两个锚点（圆的左侧和右侧）
    lw.setAnchor(12, 32); // 圆左侧外
    auto wire = lw.getLiveWire(52, 32); // 到圆右侧外

    std::cout << "  锚点(12,32) → 目标(52,32): 路径长度=" << wire.size() << " 像素\n";

    // 沿圆边缘的路径
    lw.setAnchor(32, 12); // 圆顶部
    wire = lw.getLiveWire(32, 52); // 到底部
    std::cout << "  锚点(32,12) → 目标(32,52): 路径长度=" << wire.size() << " 像素\n";

    std::cout << "  Live-Wire 引擎就绪，可用于交互式勾画\n";
}

// ============================================================
//  演示5: 插值重采样
// ============================================================
void demo_interpolation() {
    std::cout << "\n=== 演示5: 插值重采样 ===\n";

    // 创建小体数据
    Image3D<short> vol(16, 16, 16, 0);
    vol.spacingX = 1.0f; vol.spacingY = 1.0f; vol.spacingZ = 2.0f;
    // 在中心放一个球
    for (int z = 0; z < 16; ++z)
    for (int y = 0; y < 16; ++y)
    for (int x = 0; x < 16; ++x) {
        float d = std::sqrt(float((x-8)*(x-8) + (y-8)*(y-8) + (z-8)*(z-8)));
        if (d < 5) vol.at(x, y, z) = 1000;
    }

    // 重采样到等间距 1mm
    auto resampled = ResampleToSpacing(vol, 1.0f, 1.0f, 1.0f);

    std::cout << "  原始: " << vol.width << "x" << vol.height << "x" << vol.depth
              << " (spacing: " << vol.spacingX << "," << vol.spacingY << "," << vol.spacingZ << ")\n";
    std::cout << "  重采样: " << resampled.width << "x" << resampled.height << "x" << resampled.depth
              << " (spacing: " << resampled.spacingX << "," << resampled.spacingY << "," << resampled.spacingZ << ")\n";

    // 测试单点三线性插值
    float val = TrilinearSample(vol, 8.5f, 8.5f, 8.0f);
    std::cout << "  三线性插值 (8.5, 8.5, 8.0) = " << val << "\n";
}

// ============================================================
//  演示6: CT 切片匹配 (ResetMatch)
// ============================================================
void demo_reset_match() {
    std::cout << "\n=== 演示6: CT 切片匹配 ===\n";

    // 创建模拟重建 CT 体数据
    int W = 32, H = 32, D = 20;
    Image3D<short> recon(W, H, D, -1000); // 背景为空气

    // 在每层放一个"骨结构"（每层不同大小，确保唯一匹配）
    for (int z = 0; z < D; ++z) {
        int cx = 16, cy = 16;
        int r = 3 + z; // 每层半径递增，确保唯一
        for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float d = std::sqrt(float((x-cx)*(x-cx) + (y-cy)*(y-cy)));
            if (d < r) recon.at(x, y, z) = 500; // 骨
        }
    }

    // 第 10 层作为"计划 CT"
    Image2D<short> plan(W, H, -1000);
    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
        plan.at(x, y) = recon.at(x, y, 10);
    }

    auto result = ResetMatch(plan, recon);
    std::cout << "  最佳匹配切片: z=" << result.bestSlice
              << " (NCC=" << result.bestNCC << ")\n";
    std::cout << "  期望: z=10" << (result.bestSlice == 10 ? " ✓" : " ✗") << "\n";
}

// ============================================================
//  演示7: 金属定位点检测（概念演示）
// ============================================================
void demo_fiducial() {
    std::cout << "\n=== 演示7: 金属定位点检测 ===\n";

    // 创建模拟 CT 体数据，嵌入几个高密度"金属点"
    int W = 32, H = 32, D = 10;
    Image3D<short> ct(W, H, D, -1000);

    // 放置 3 个金属标记（同一 Z 层面）
    auto placeMarker = [&](int mx, int my, int mz) {
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            if (ct.inBounds(mx+dx, my+dy, mz+dz))
                ct.at(mx+dx, my+dy, mz+dz) = 2000; // 高密度金属
        }
    };

    placeMarker(8,  16, 5);  // 左
    placeMarker(24, 16, 5);  // 右
    placeMarker(16, 8,  5);  // 上
    ct.spacingX = 1.0f; ct.spacingY = 1.0f; ct.spacingZ = 2.5f;

    auto groups = DetectFiducials(ct);
    std::cout << "  检测到 " << groups.size() << " 组定位点\n";
    for (size_t g = 0; g < groups.size(); ++g) {
        std::cout << "  组 " << g << ": " << groups[g].size() << " 个点\n";
        for (const auto& p : groups[g]) {
            const char* typeStr[] = {"Left", "Right", "Top", "Center"};
            std::cout << "    " << typeStr[static_cast<int>(p.type)]
                      << " (" << p.pos.x << ", " << p.pos.y << ", " << p.pos.z << ")\n";
        }
    }
}

// ============================================================
//  演示8: 床板检测
// ============================================================
void demo_bed_board() {
    std::cout << "\n=== 演示8: 床板检测 ===\n";

    // 创建模拟 CT：底部为"床板"（高密度），中间为"人体"
    int W = 64, H = 64, D = 20;
    Image3D<short> ct(W, H, D, -1000);

    // 床板区域（y > 50）
    for (int z = 0; z < D; ++z)
    for (int y = 50; y < H; ++y)
    for (int x = 10; x < 54; ++x) {
        ct.at(x, y, z) = -300; // 比空气高但比组织低
    }

    // 人体（椭圆）
    for (int z = 0; z < D; ++z)
    for (int y = 10; y < 50; ++y)
    for (int x = 16; x < 48; ++x) {
        float dx = (x - 32.f) / 16.f;
        float dy = (y - 30.f) / 20.f;
        if (dx * dx + dy * dy < 1.0f)
            ct.at(x, y, z) = 40; // 软组织
    }

    auto result = DetectBedBoard(ct);
    if (result.detected) {
        std::cout << "  床板已检测! 列中心: " << result.colPosition << "\n";
        std::cout << "  曲线系数: y = " << result.polyCoeffs[0]
                  << " + " << result.polyCoeffs[1] << "*z + "
                  << result.polyCoeffs[2] << "*z²\n";
        std::cout << "  第0层行位置: " << result.rowPositions[0]
                  << ", 第" << D-1 << "层: " << result.rowPositions[D-1] << "\n";
    } else {
        std::cout << "  未检测到床板（模拟数据可能不够典型）\n";
    }
}

// ============================================================
//  演示9: GPU Marching Squares 轮廓提取
// ============================================================
#ifdef USE_CUDA
void demo_marching_squares_cuda() {
    std::cout << "\n=== 演示9: GPU Marching Squares 轮廓提取 ===\n";

    // ── 测试1: 简单圆形 ──────────────────────────────
    const int W = 128, H = 128;
    std::vector<uint8_t> mask(W * H, 0);
    int cx = 64, cy = 64, r = 30;
    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
        float dx = static_cast<float>(x - cx);
        float dy = static_cast<float>(y - cy);
        if (dx * dx + dy * dy <= r * r)
            mask[y * W + x] = 255;
    }

    std::vector<OrderedContour> contours;
    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = MarchingSquaresCUDA(mask.data(), W, H, contours);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "  [圆形 " << W << "x" << H << "] ";
    if (ok) {
        std::cout << "提取到 " << contours.size() << " 条轮廓, 耗时 "
                  << ms << " ms\n";
        for (size_t i = 0; i < contours.size(); ++i) {
            std::cout << "    轮廓 " << i << ": " << contours[i].xs.size() << " 个点\n";
            // 打印前几个点验证
            if (contours[i].xs.size() >= 4) {
                std::cout << "    前4点: ";
                for (int j = 0; j < 4; ++j)
                    std::cout << "(" << contours[i].xs[j] << "," << contours[i].ys[j] << ") ";
                std::cout << "\n";
            }
        }
    } else {
        std::cout << "失败!\n";
    }

    // ── 测试2: 两个不相连的形状 ─────────────────────
    std::vector<uint8_t> mask2(W * H, 0);
    // 小圆 1
    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
        float d1 = std::sqrt(float((x-30)*(x-30) + (y-30)*(y-30)));
        float d2 = std::sqrt(float((x-90)*(x-90) + (y-90)*(y-90)));
        if (d1 <= 15 || d2 <= 20)
            mask2[y * W + x] = 255;
    }

    std::vector<OrderedContour> contours2;
    t0 = std::chrono::high_resolution_clock::now();
    ok = MarchingSquaresCUDA(mask2.data(), W, H, contours2);
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "  [双圆形] 提取到 " << contours2.size() << " 条轮廓, 耗时 "
              << ms << " ms\n";
    for (size_t i = 0; i < contours2.size(); ++i) {
        std::cout << "    轮廓 " << i << ": " << contours2[i].xs.size() << " 个点\n";
    }

    // ── 测试3: 大尺寸性能测试 ───────────────────────
    const int BW = 512, BH = 512;
    std::vector<uint8_t> bigMask(BW * BH, 0);
    for (int y = 0; y < BH; ++y)
    for (int x = 0; x < BW; ++x) {
        float dx2 = static_cast<float>(x - 256);
        float dy2 = static_cast<float>(y - 256);
        if (dx2 * dx2 + dy2 * dy2 <= 200 * 200)
            bigMask[y * BW + x] = 255;
    }

    std::vector<OrderedContour> bigContours;
    t0 = std::chrono::high_resolution_clock::now();
    ok = MarchingSquaresCUDA(bigMask.data(), BW, BH, bigContours);
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "  [大尺寸 " << BW << "x" << BH << "] 提取到 "
              << bigContours.size() << " 条轮廓, 耗时 " << ms << " ms\n";
    if (!bigContours.empty()) {
        std::cout << "    最大轮廓: " << bigContours[0].xs.size() << " 个点\n";
    }
}
#endif

// ============================================================
//  主函数
// ============================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║    RTAutoContour — 放疗自动勾画算法演示          ║\n";
    std::cout << "║    提炼自 McsfAlgoAutoContour 工程               ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n";

    demo_morphology();
    demo_connected_component();
    demo_contour_mask();
    demo_live_wire();
    demo_interpolation();
    demo_reset_match();
    demo_fiducial();
    demo_bed_board();

#ifdef USE_CUDA
    demo_marching_squares_cuda();
#else
    std::cout << "\n[跳过] 演示9: GPU Marching Squares (未启用 CUDA)\n";
#endif

    std::cout << "\n所有演示完成.\n";
    return 0;
}
