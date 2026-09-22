/**
 * @file    print_pipeline.cpp
 * @brief   美甲打印完整流程 实现
 */

#include "app/print_pipeline.h"
#include "reconstruction/reconstruction.h"
#include "color/color_manager.h"
#include "color/voxel_color.h"
#include "print/gcode_gen.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>

namespace NailPrint3D {

// ============================================================
// 浮雕位移映射
// ============================================================

void applyDisplacementToMesh(Scene& scene, float height) {
    if (height <= 0.001f) {
        scene.currentMesh = scene.originalMesh;
        scene.displacementApplied = false;
        scene.glMesh.upload(scene.currentMesh);
        std::cout << "[浮雕] 已恢复平面网格" << std::endl;
        return;
    }

    scene.currentMesh = scene.originalMesh;

    int w = 512, h = 512;
    std::vector<unsigned char> pixels(w * h * 4);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float u = (float)x / w;
            float v = (float)y / h;
            int idx = (y * w + x) * 4;

            float lum = 0.5f;
            unsigned char r = 128, g = 128, b = 128;

            if (scene.currentPattern == RenderPattern::Cartoon) {
                float cx = 0.5f, cy = 0.5f;
                float dx = u - cx, dy = v - cy;
                float dist = std::sqrt(dx * dx + dy * dy);
                float angle = std::atan2(dy, dx);
                float petalR = 0.3f + 0.1f * std::cos(5.0f * angle);
                if (dist < petalR) {
                    lum = 0.9f - dist * 0.5f;
                    r = 255; g = 100; b = 150;
                } else {
                    lum = 0.2f;
                    r = 255; g = 240; b = 180;
                }
            } else if (scene.currentPattern == RenderPattern::Photo) {
                float dx = (u - 0.5f) * 1.2f, dy = (v - 0.45f) * 1.5f;
                float faceDist = std::sqrt(dx * dx + dy * dy);
                if (faceDist < 0.3f) {
                    lum = 0.7f + 0.1f * (1.0f - faceDist / 0.3f);
                    r = 230; g = 195; b = 170;
                } else {
                    lum = 0.15f;
                    r = 200; g = 200; b = 210;
                }
            } else if (scene.currentPattern == RenderPattern::FlatColor) {
                float scale = 6.0f;
                float fu = u * scale - std::floor(u * scale);
                float fv = v * scale - std::floor(v * scale);
                float diamond = std::abs(fu - 0.5f) + std::abs(fv - 0.5f);
                if (diamond < 0.25f) {
                    lum = 0.8f;
                    r = 255; g = 100; b = 150;
                } else {
                    lum = 0.3f;
                    r = 200; g = 200; b = 220;
                }
            } else if (scene.currentPattern == RenderPattern::Text) {
                int gx = (int)(u * 16);
                int gy = (int)(v * 4);
                const char* letters[4] = {"NAIL", "AIL ", "IL  ", "L   "};
                char ch = letters[gy][gx % 4];
                if (ch != ' ') {
                    lum = 0.9f;
                    r = 220; g = 60; b = 100;
                } else {
                    lum = 0.1f;
                    r = 245; g = 235; b = 240;
                }
            }

            pixels[idx] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
            pixels[idx + 3] = 255;
        }
    }

    NailMeshGenerator::applyDisplacementMap(scene.currentMesh, pixels.data(), w, h, height);
    scene.glMesh.upload(scene.currentMesh);
    scene.displacementApplied = true;

    std::cout << "[浮雕] 位移高度=" << height << "mm, 网格已更新" << std::endl;
}

// ============================================================
// 装饰物放置
// ============================================================

void applyOrnamentToMesh(Scene& scene) {
    if (scene.currentOrnament < 0) {
        scene.currentMesh = scene.nailBaseMesh;
        scene.glMesh.upload(scene.currentMesh);
        std::cout << "[装饰物] 已移除，恢复纯甲片" << std::endl;
        return;
    }

    OrnamentType type = (OrnamentType)scene.currentOrnament;
    ColorRGBf ornColor = colorSchemes[scene.currentColorScheme].colors[scene.currentOrnament];
    Mesh ornament = OrnamentGenerator::generate(type, scene.ornamentSize, ornColor);

    scene.currentMesh = OrnamentGenerator::placeOnNail(
        scene.nailBaseMesh, ornament, scene.ornamentU, scene.ornamentV,
        1.0f, scene.ornamentRotation);

    scene.glMesh.upload(scene.currentMesh);
    std::cout << "[装饰物] " << ornamentNames[scene.currentOrnament]
              << " 已放置 (大小=" << scene.ornamentSize
              << ", 位置=" << scene.ornamentU << "," << scene.ornamentV
              << ", 配色=" << colorSchemes[scene.currentColorScheme].name << ")" << std::endl;
}

// ============================================================
// 完整流程
// ============================================================

void runNailPrintPipeline(Scene& scene) {
    std::cout << "\n========================================" << std::endl;
    std::cout << " NailPrint3D 完整流程" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // 步骤 1: 生成美甲网格
    std::cout << ">>> 步骤 1: 生成美甲网格..." << std::endl;
    NailMeshGenerator generator;
    PrintConfig& config = scene.printConfig;
    config.nailBedWidth = 15.0f;
    config.nailBedLength = 20.0f;
    config.nailCurvature = 1.5f;
    config.baseThickness = 0.3f;
    config.colorLayerHeight = 0.1f;
    config.topCoatThickness = 0.2f;
    config.colorCount = 4;

    scene.currentMesh = generator.generateNailPatch(config.nailBedWidth, config.nailBedLength,
                                                     config.nailCurvature, config.baseThickness);
    generator.addReliefPattern(scene.currentMesh, 0, 0.3f);

    scene.currentMesh.computeBBox();
    scene.currentMesh.computeNormals();

    std::cout << "    顶点数: " << scene.currentMesh.vertices.size() << std::endl;
    std::cout << "    三角形数: " << scene.currentMesh.triangles.size() << std::endl;

    STLLoader::saveBinary("nail_patch.stl", scene.currentMesh);
    std::cout << "    已保存 nail_patch.stl" << std::endl;

    // 步骤 2: 3D重建与网格修复
    std::cout << "\n>>> 步骤 2: 3D重建与网格修复..." << std::endl;
    MeshRepair repair;
    repair.weldVertices(scene.currentMesh, 0.01f);
    repair.fillHoles(scene.currentMesh);
    repair.unifyNormals(scene.currentMesh);
    repair.laplacianSmooth(scene.currentMesh, 1, 0.5f);
    std::cout << "    网格修复完成" << std::endl;

    // 步骤 3: RIP切片
    std::cout << "\n>>> 步骤 3: RIP切片..." << std::endl;
    NailPrintSlicer nailSlicer;
    scene.sliceLayers = nailSlicer.sliceNail(scene.currentMesh, config);

    // 步骤 4: 色彩管理
    std::cout << "\n>>> 步骤 4: 色彩管理..." << std::endl;

    scene.palette = {
        ColorRGBf{1.0f, 0.3f, 0.4f},
        ColorRGBf{0.9f, 0.6f, 0.2f},
        ColorRGBf{0.4f, 0.7f, 0.9f},
        ColorRGBf{0.6f, 0.3f, 0.8f},
    };
    config.palette = { ColorRGBA8{255,77,102,255}, ColorRGBA8{230,153,51,255},
                       ColorRGBA8{102,179,230,255}, ColorRGBA8{153,77,204,255} };

    ICCColorManager icc;
    icc.loadProfile("sRGB", "");
    icc.loadProfile("AdobeRGB", "");
    std::cout << "    ICC 配置文件已加载" << std::endl;

    LUTManager lutMgr;
    ColorLUT lut = lutMgr.generateGamma(1.0f, 16);
    std::cout << "    LUT 已生成 (size=" << lut.size << ")" << std::endl;

    ColorCalibrator calibrator;
    calibrator.addPatch(ColorRGBf{0.95f, 0.25f, 0.35f}, scene.palette[0]);
    calibrator.addPatch(ColorRGBf{0.85f, 0.55f, 0.15f}, scene.palette[1]);
    calibrator.addPatch(ColorRGBf{0.35f, 0.65f, 0.85f}, scene.palette[2]);
    calibrator.addPatch(ColorRGBf{0.55f, 0.25f, 0.75f}, scene.palette[3]);
    calibrator.calibrate();
    std::cout << "    平均 DeltaE: " << calibrator.computeAverageDeltaE() << std::endl;

    VoxelColorManager voxelColor;
    std::vector<ColorRGBA8> dummyTexture(64 * 64, ColorRGBA8{200, 150, 180, 255});
    voxelColor.fromMeshTexture(scene.currentMesh, dummyTexture, 64, 64);
    voxelColor.applyICC(icc, "sRGB", "AdobeRGB");
    voxelColor.applyLUT(lut);
    voxelColor.quantizeToPalette(scene.palette);
    voxelColor.exportTo3DTexture("nail_colors.bin");
    std::cout << "    体素颜色管理完成" << std::endl;

    // 步骤 5: G-code 生成
    std::cout << "\n>>> 步骤 5: G-code 生成..." << std::endl;
    GCodeConfig gcodeConfig;
    gcodeConfig.filamentType = "UV Resin";
    gcodeConfig.nozzleTemp = 0;
    gcodeConfig.bedTemp = 0;
    gcodeConfig.printSpeed = 30.0f;
    gcodeConfig.travelSpeed = 80.0f;
    gcodeConfig.multiColor = true;
    gcodeConfig.colorCount = config.colorCount;
    gcodeConfig.uvCurePerLayer = true;
    gcodeConfig.uvCureTime = 5.0f;
    gcodeConfig.layerHeight = config.colorLayerHeight;
    gcodeConfig.extrusionWidth = 0.4f;

    GCodeGenerator gcodeGen;
    std::string gcode = gcodeGen.generate(scene.sliceLayers, gcodeConfig);
    gcodeGen.saveToFile("nail_print.gcode", gcode);
    gcodeGen.printStats(gcode);

    std::cout << "\n========================================" << std::endl;
    std::cout << " 流程完成! 输出文件:" << std::endl;
    std::cout << "   - nail_patch.stl    (3D网格)" << std::endl;
    std::cout << "   - nail_colors.bin   (体素颜色)" << std::endl;
    std::cout << "   - nail_print.gcode  (打印代码)" << std::endl;
    std::cout << "========================================\n" << std::endl;

    printHelpText();
}

} // namespace NailPrint3D
