/**
 * @file    main_nail_print.cpp
 * @brief   3D美甲打印 — 主程序入口
 *
 * 完整流程演示:
 *   1. 生成/加载美甲网格
 *   2. 3D重建 (可选: 从点云重建)
 *   3. RIP切片
 *   4. 色彩管理 (ICC + LUT + 抖动)
 *   5. G-code 生成
 *   6. OpenGL 实时预览
 */

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include "nail_types.h"
#include "stl_loader.h"
#include "nail_reconstruction.h"
#include "nail_slicer.h"
#include "nail_color.h"
#include "nail_gcode.h"
#include "nail_renderer.h"

using namespace NailPrint3D;

// ============================================================
// 全局状态
// ============================================================

RenderCamera camera;
NailRenderer renderer;
GLNailMesh glMesh;
GLNailMesh glSliceMesh;
Mesh currentMesh;
std::vector<SliceLayer> sliceLayers;
std::vector<ColorRGBf> palette;

int currentLayer = 0;
int renderMode = 0;  // 0=Solid, 1=Wireframe, 2=Slice, 3=Color
bool mouseLeftDown = false;
bool mouseRightDown = false;
double lastMouseX = 0, lastMouseY = 0;

// ============================================================
// 回调函数
// ============================================================

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    camera.setAspect((float)width / (float)height);
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        mouseLeftDown = (action == GLFW_PRESS);
        glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        mouseRightDown = (action == GLFW_PRESS);
        glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
    }
}

void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    double dx = xpos - lastMouseX;
    double dy = ypos - lastMouseY;
    lastMouseX = xpos;
    lastMouseY = ypos;

    if (mouseLeftDown) {
        camera.orbit((float)dx, (float)-dy);
    }
    if (mouseRightDown) {
        camera.pan((float)-dx, (float)dy);
    }
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    camera.zoom((float)(-yoffset));
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

    switch (key) {
        case GLFW_KEY_1:
            renderMode = 0;
            renderer.setRenderMode(RenderMode::Solid);
            std::cout << "[模式] 实体渲染" << std::endl;
            break;
        case GLFW_KEY_2:
            renderMode = 1;
            renderer.setRenderMode(RenderMode::Wireframe);
            std::cout << "[模式] 线框渲染" << std::endl;
            break;
        case GLFW_KEY_3:
            renderMode = 2;
            renderer.setRenderMode(RenderMode::SlicePreview);
            glSliceMesh.uploadSlicePaths(sliceLayers);
            std::cout << "[模式] 切片预览" << std::endl;
            break;
        case GLFW_KEY_4:
            renderMode = 3;
            renderer.setRenderMode(RenderMode::ColorPreview);
            std::cout << "[模式] 颜色预览" << std::endl;
            break;
        case GLFW_KEY_UP:
            if (renderMode == 2) {
                currentLayer = std::min(currentLayer + 1, (int)sliceLayers.size() - 1);
                renderer.setCurrentLayer(currentLayer);
                std::cout << "[切片] 层 " << currentLayer << "/" << sliceLayers.size() << std::endl;
            }
            break;
        case GLFW_KEY_DOWN:
            if (renderMode == 2) {
                currentLayer = std::max(currentLayer - 1, 0);
                renderer.setCurrentLayer(currentLayer);
                std::cout << "[切片] 层 " << currentLayer << "/" << sliceLayers.size() << std::endl;
            }
            break;
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;
    }
}

// ============================================================
// 美甲打印完整流程
// ============================================================

void runNailPrintPipeline() {
    std::cout << "\n========================================" << std::endl;
    std::cout << " NailPrint3D 完整流程" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // -------------------------------------------------------
    // 步骤 1: 生成美甲网格
    // -------------------------------------------------------
    std::cout << ">>> 步骤 1: 生成美甲网格..." << std::endl;
    NailMeshGenerator generator;
    PrintConfig config;
    config.nailBedWidth = 15.0f;    // 美甲宽度 15mm
    config.nailBedLength = 20.0f;   // 美甲长度 20mm
    config.nailCurvature = 1.5f;    // 曲率（增加弧度，更像真实指甲）
    config.baseThickness = 0.3f;    // 底胶厚度
    config.colorLayerHeight = 0.1f; // 颜色层高
    config.topCoatThickness = 0.2f; // 封层厚度
    config.colorCount = 4;          // 4色打印

    currentMesh = generator.generateNailPatch(config.nailBedWidth, config.nailBedLength,
                                               config.nailCurvature, config.baseThickness);
    generator.addReliefPattern(currentMesh, 0, 0.3f);  // 添加花朵花纹

    // 添加法线
    currentMesh.computeBBox();
    currentMesh.computeNormals();

    std::cout << "    顶点数: " << currentMesh.vertices.size() << std::endl;
    std::cout << "    三角形数: " << currentMesh.triangles.size() << std::endl;
    std::cout << "    包围盒: (" << currentMesh.bbox.min.x << "," << currentMesh.bbox.min.y << "," << currentMesh.bbox.min.z
              << ") → (" << currentMesh.bbox.max.x << "," << currentMesh.bbox.max.y << "," << currentMesh.bbox.max.z << ")" << std::endl;

    // 保存 STL
    STLLoader::saveBinary("nail_patch.stl", currentMesh);
    std::cout << "    已保存 nail_patch.stl" << std::endl;

    // -------------------------------------------------------
    // 步骤 2: 3D重建 (演示网格修复)
    // -------------------------------------------------------
    std::cout << "\n>>> 步骤 2: 3D重建与网格修复..." << std::endl;
    MeshRepair repair;
    repair.weldVertices(currentMesh, 0.01f);
    repair.fillHoles(currentMesh);
    repair.unifyNormals(currentMesh);
    repair.laplacianSmooth(currentMesh, 1, 0.5f);
    std::cout << "    网格修复完成" << std::endl;

    // -------------------------------------------------------
    // 步骤 3: RIP切片
    // -------------------------------------------------------
    std::cout << "\n>>> 步骤 3: RIP切片..." << std::endl;
    NailPrintSlicer nailSlicer;
    sliceLayers = nailSlicer.sliceNail(currentMesh, config);

    // -------------------------------------------------------
    // 步骤 4: 色彩管理
    // -------------------------------------------------------
    std::cout << "\n>>> 步骤 4: 色彩管理..." << std::endl;

    // 4.1 调色板
    palette = {
        ColorRGBf{1.0f, 0.3f, 0.4f},    // 红色
        ColorRGBf{0.9f, 0.6f, 0.2f},    // 橙色
        ColorRGBf{0.4f, 0.7f, 0.9f},    // 蓝色
        ColorRGBf{0.6f, 0.3f, 0.8f},    // 紫色
    };
    config.palette = { ColorRGBA8{255,77,102,255}, ColorRGBA8{230,153,51,255},
                       ColorRGBA8{102,179,230,255}, ColorRGBA8{153,77,204,255} };

    // 4.2 ICC 色彩转换
    ICCColorManager icc;
    icc.loadProfile("sRGB", "");
    icc.loadProfile("AdobeRGB", "");
    std::cout << "    ICC 配置文件已加载" << std::endl;

    // 4.3 LUT 生成
    LUTManager lutMgr;
    ColorLUT lut = lutMgr.generateGamma(1.0f, 16);
    std::cout << "    LUT 已生成 (size=" << lut.size << ")" << std::endl;

    // 4.4 颜色校准
    ColorCalibrator calibrator;
    calibrator.addPatch(ColorRGBf{0.95f, 0.25f, 0.35f}, palette[0]);
    calibrator.addPatch(ColorRGBf{0.85f, 0.55f, 0.15f}, palette[1]);
    calibrator.addPatch(ColorRGBf{0.35f, 0.65f, 0.85f}, palette[2]);
    calibrator.addPatch(ColorRGBf{0.55f, 0.25f, 0.75f}, palette[3]);
    calibrator.calibrate();
    std::cout << "    平均 DeltaE: " << calibrator.computeAverageDeltaE() << std::endl;

    // 4.5 体素颜色管理
    VoxelColorManager voxelColor;
    std::vector<ColorRGBA8> dummyTexture(64 * 64, ColorRGBA8{200, 150, 180, 255});
    voxelColor.fromMeshTexture(currentMesh, dummyTexture, 64, 64);
    voxelColor.applyICC(icc, "sRGB", "AdobeRGB");
    voxelColor.applyLUT(lut);
    voxelColor.quantizeToPalette(palette);
    voxelColor.exportTo3DTexture("nail_colors.bin");
    std::cout << "    体素颜色管理完成" << std::endl;

    // -------------------------------------------------------
    // 步骤 5: G-code 生成
    // -------------------------------------------------------
    std::cout << "\n>>> 步骤 5: G-code 生成..." << std::endl;
    GCodeConfig gcodeConfig;
    gcodeConfig.filamentType = "UV Resin";
    gcodeConfig.nozzleTemp = 0;        // UV树脂不需要加热
    gcodeConfig.bedTemp = 0;
    gcodeConfig.printSpeed = 30.0f;
    gcodeConfig.travelSpeed = 80.0f;
    gcodeConfig.multiColor = true;
    gcodeConfig.colorCount = config.colorCount;
    gcodeConfig.uvCurePerLayer = true;
    gcodeConfig.uvCureTime = 5.0f;     // 每层 UV 固化 5 秒
    gcodeConfig.layerHeight = config.colorLayerHeight;
    gcodeConfig.extrusionWidth = 0.4f;

    GCodeGenerator gcodeGen;
    std::string gcode = gcodeGen.generate(sliceLayers, gcodeConfig);
    gcodeGen.saveToFile("nail_print.gcode", gcode);
    gcodeGen.printStats(gcode);

    std::cout << "\n========================================" << std::endl;
    std::cout << " 流程完成! 输出文件:" << std::endl;
    std::cout << "   - nail_patch.stl    (3D网格)" << std::endl;
    std::cout << "   - nail_colors.bin   (体素颜色)" << std::endl;
    std::cout << "   - nail_print.gcode  (打印代码)" << std::endl;
    std::cout << "========================================\n" << std::endl;

    std::cout << "操作说明:" << std::endl;
    std::cout << "  鼠标左键拖动: 旋转视角" << std::endl;
    std::cout << "  鼠标右键拖动: 平移视角" << std::endl;
    std::cout << "  鼠标滚轮: 缩放" << std::endl;
    std::cout << "  键盘 1: 实体渲染" << std::endl;
    std::cout << "  键盘 2: 线框渲染" << std::endl;
    std::cout << "  键盘 3: 切片预览 (↑↓ 切换层)" << std::endl;
    std::cout << "  键盘 4: 颜色预览" << std::endl;
    std::cout << "  ESC: 退出" << std::endl;
}

// ============================================================
// 主函数
// ============================================================

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    // 初始化 GLFW
    if (!glfwInit()) {
        std::cerr << "无法初始化 GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "NailPrint3D — 3D美甲打印系统", nullptr, nullptr);
    if (!window) {
        std::cerr << "无法创建窗口" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);

    // 初始化 GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "无法初始化 GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 运行完整流程
    runNailPrintPipeline();

    // 初始化渲染器
    std::string shaderDir = "shaders";
    // 尝试多个路径
    {
        std::ifstream test(shaderDir + "/nail_mesh.vert");
        if (!test.good()) {
            shaderDir = "NailPrint3D/shaders";
            test.open(shaderDir + "/nail_mesh.vert");
        }
        if (!test.good()) {
            shaderDir = "../NailPrint3D/shaders";
        }
    }

    if (!renderer.init(shaderDir)) {
        std::cerr << "渲染器初始化失败，着色器目录: " << shaderDir << std::endl;
        // 继续运行，仅无法渲染
    }

    // 上传网格
    glMesh.upload(currentMesh);
    camera.setAspect(1280.0f / 720.0f);

    // 根据包围盒调整相机
    Vec3 center = currentMesh.bbox.center();
    float bboxSize = std::max(
        currentMesh.bbox.max.x - currentMesh.bbox.min.x,
        std::max(currentMesh.bbox.max.y - currentMesh.bbox.min.y,
                 currentMesh.bbox.max.z - currentMesh.bbox.min.z));
    camera.setTarget(glm::vec3(center.x, center.y, center.z));
    camera.setDistance(bboxSize * 1.5f);  // 距离 = 包围盒尺寸 * 1.5

    // 渲染循环
    while (!glfwWindowShouldClose(window)) {
        renderer.render(glMesh, camera);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glMesh.destroy();
    glSliceMesh.destroy();
    glfwTerminate();
    return 0;
}
