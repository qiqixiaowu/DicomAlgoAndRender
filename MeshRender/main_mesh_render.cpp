/**
 * @file    main_mesh_render.cpp
 * @brief   Mesh 网格生成与优化渲染 Demo
 *
 * 功能：
 *  - 程序化网格生成：立方体、球体、圆环、圆柱、地形、二十面体
 *  - 网格优化：法线/切线计算、顶点焊接、Laplacian/Taubin 平滑、QEM 简化、LOD
 *  - 延迟渲染：G-Buffer、PBR 光照、阴影映射、SSAO、天空盒
 *  - 可视化：线框模式、法线可视化、深度可视化
 *  - 交互：轨道/FPS 相机、键盘控制
 *
 * 键盘：
 *   [1-6]    切换程序化网格类型
 *   [7]      DICOM 骨骼渲染（等值面阈值 ~180）
 *   [8]      DICOM 软组织渲染（等值面阈值 ~128）
 *   [9]      DICOM 皮肤渲染（等值面阈值 ~60）
 *   [D]      DICOM 自定义阈值渲染
 *   [F2]     线框模式开关
 *   [F3]     法线可视化
 *   [F4]     深度可视化
 *   [O]      优化网格（平滑/简化）
 *   [L]      LOD 切换
 *   [Tab]    切换相机模式（轨道/FPS）
 *   [Space]  自动旋转
 *   [P]      暂停
 *   [R]      重置
 *   [F1]     显示统计信息
 *   [ESC]    退出
 *
 * 鼠标：
 *   左键拖动  旋转
 *   滚轮      缩放
 *   WASD      FPS 移动
 */

#include <glad/glad.h>
#include <glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include "mesh.h"
#include "mesh_optimizer.h"
#include "gl_mesh.h"
#include "mesh_camera.h"
#include "deferred_renderer.h"
#include "dicom_mesh_loader.hpp"

using namespace MeshRender;

// ============================================================
// 全局状态
// ============================================================

static int g_windowWidth  = 1280;
static int g_windowHeight = 720;

static Camera g_camera;
static DeferredRenderer g_renderer;

static std::vector<RenderObject> g_objects;
static std::vector<Light> g_lights;

static int  g_currentMeshType = 10;  // 10=DICOM骨骼（启动即自动加载）
static bool g_autoRotate = false;
static bool g_paused = false;
static bool g_showStats = true;
static int  g_lodLevel = 0;

// DICOM 加载状态
// meshType 10=DICOM骨骼, 11=DICOM软组织, 12=DICOM皮肤, 13=DICOM自定义路径
// 默认 DICOM 数据路径 —— 自动加载，无需手动输入
static std::string g_dicomFolder = R"(E:\Data\Onco-LAZ\20150706\LIU\3.53.5CTWB)";
static float g_dicomIsovalue = 128.0f;  // 当前等值面阈值
static bool  g_dicomLoaded = false;     // 是否已加载 DICOM 数据
static bool  g_useGPU = true;           // 是否使用 GPU Marching Cubes
static std::string g_shaderDir = "shaders";  // 着色器目录（main 中初始化）

static float g_deltaTime = 0.0f;
static float g_totalTime = 0.0f;

// 网格类型名称
static const char* g_meshNames[] = {
    "Cube", "Sphere", "Torus", "Cylinder", "Terrain", "Icosahedron",
    "DICOM_Bone", "DICOM_Tissue", "DICOM_Skin", "DICOM_Custom"
};

// ============================================================
// 回调函数
// ============================================================

// 前向声明
static void rebuildScene();
static void optimizeCurrentMesh();
static void switchLOD(int level);

static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    g_windowWidth = width;
    g_windowHeight = height;
    g_camera.aspect = (height > 0) ? (float)width / (float)height : 1.0f;
    g_renderer.resize(width, height);
    glViewport(0, 0, width, height);
}

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    g_camera.processMouseButton(button, action, x, y);
}

static void cursorPosCallback(GLFWwindow* window, double x, double y) {
    g_camera.processMouseMotion(x, y);
}

static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    g_camera.processMouseScroll(yoffset);
}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
            case GLFW_KEY_TAB:
                g_camera.setMode(g_camera.mode == CameraMode::Orbit ? CameraMode::FPS : CameraMode::Orbit);
                std::cout << "相机模式: " << (g_camera.mode == CameraMode::Orbit ? "轨道" : "FPS") << std::endl;
                break;
            case GLFW_KEY_SPACE:
                if (g_camera.mode == CameraMode::Orbit)
                    g_autoRotate = !g_autoRotate;
                break;
            case GLFW_KEY_P:
                g_paused = !g_paused;
                break;
            case GLFW_KEY_R:
                g_camera.position = glm::vec3(0, 2, 5);
                g_camera.target = glm::vec3(0, 0, 0);
                g_camera.yaw = -90; g_camera.pitch = 15;
                g_camera.orbitDistance = 5;
                g_camera.setMode(g_camera.mode);  // 触发更新
                break;
            case GLFW_KEY_F1:
                g_showStats = !g_showStats;
                break;
            case GLFW_KEY_F2:
                g_renderer.enableWireframe = !g_renderer.enableWireframe;
                g_renderer.wireframeMode = 0;
                break;
            case GLFW_KEY_F3:
                g_renderer.enableWireframe = !g_renderer.enableWireframe;
                g_renderer.wireframeMode = 1;
                break;
            case GLFW_KEY_F4:
                g_renderer.enableWireframe = !g_renderer.enableWireframe;
                g_renderer.wireframeMode = 2;
                break;
            case GLFW_KEY_O:
                optimizeCurrentMesh();
                break;
            case GLFW_KEY_L:
                g_lodLevel = (g_lodLevel + 1) % 4;
                switchLOD(g_lodLevel);
                break;
            case GLFW_KEY_G: // 切换 GPU/CPU Marching Cubes
                g_useGPU = !g_useGPU;
                std::cout << "\n=== Marching Cubes: "
                          << (g_useGPU ? "GPU (Compute Shader)" : "CPU")
                          << " ===" << std::endl;
                rebuildScene();
                break;
            case GLFW_KEY_1: g_currentMeshType = 0; rebuildScene(); break;
            case GLFW_KEY_2: g_currentMeshType = 1; rebuildScene(); break;
            case GLFW_KEY_3: g_currentMeshType = 2; rebuildScene(); break;
            case GLFW_KEY_4: g_currentMeshType = 3; rebuildScene(); break;
            case GLFW_KEY_5: g_currentMeshType = 4; rebuildScene(); break;
            case GLFW_KEY_6: g_currentMeshType = 5; rebuildScene(); break;
            case GLFW_KEY_7: // DICOM 骨骼
                g_currentMeshType = 10;
                std::cout << "\n=== DICOM 骨骼模式 ===" << std::endl;
                rebuildScene();
                break;
            case GLFW_KEY_8: // DICOM 软组织
                g_currentMeshType = 11;
                std::cout << "\n=== DICOM 软组织模式 ===" << std::endl;
                rebuildScene();
                break;
            case GLFW_KEY_9: // DICOM 皮肤
                g_currentMeshType = 12;
                std::cout << "\n=== DICOM 皮肤模式 ===" << std::endl;
                rebuildScene();
                break;
            case GLFW_KEY_D: // DICOM 自定义阈值
                g_currentMeshType = 13;
                std::cout << "\n=== DICOM 自定义模式 ===" << std::endl;
                std::cout << "输入等值面阈值 (0-255, 默认 128): ";
                {
                    std::string input;
                    std::getline(std::cin, input);
                    if (!input.empty()) {
                        try {
                            g_dicomIsovalue = std::stof(input);
                            std::cout << "阈值设为: " << g_dicomIsovalue << std::endl;
                        } catch (...) {
                            g_dicomIsovalue = 128.0f;
                        }
                    }
                }
                rebuildScene();
                break;
        }
    }
    g_camera.processKey(key, action);
}

// ============================================================
// 场景构建
// ============================================================

/** 生成当前类型的网格 */
static MeshData generateCurrentMesh() {
    switch (g_currentMeshType) {
        case 0: return createCube(1.5f);
        case 1: return createSphere(0.8f, 48, 24);
        case 2: return createTorus(0.6f, 0.2f, 64, 32);
        case 3: return createCylinder(0.5f, 1.5f, 48);
        case 4: return createTerrain(4.0f, 64, 0.4f);
        case 5: return createIcosahedron(0.8f);
        case 10: // DICOM 骨骼
        case 11: // DICOM 软组织
        case 12: // DICOM 皮肤
        case 13: // DICOM 自定义路径
        {
            if (g_dicomFolder.empty()) {
                std::cout << "\n请先设置 DICOM 文件夹路径!" << std::endl;
                std::cout << "在控制台输入路径，或修改源码中的 g_dicomFolder。" << std::endl;
                std::cout << "路径: ";
                std::string path;
                std::getline(std::cin, path);
                if (!path.empty()) {
                    // 去除可能的引号
                    if (path.front() == '"' && path.back() == '"')
                        path = path.substr(1, path.size() - 2);
                    g_dicomFolder = path;
                } else {
                    std::cout << "路径为空，回退到球体。" << std::endl;
                    g_currentMeshType = 1;
                    return createSphere(0.8f, 48, 24);
                }
            }

            // 根据模式选择阈值
            float isovalue = 128.0f;
            const char* modeName = "";
            switch (g_currentMeshType) {
                case 10: isovalue = 180.0f; modeName = "骨骼"; break;
                case 11: isovalue = 128.0f; modeName = "软组织"; break;
                case 12: isovalue = 60.0f;  modeName = "皮肤"; break;
                case 13: isovalue = g_dicomIsovalue; modeName = "自定义"; break;
            }

            std::cout << "\n加载 DICOM 数据，模式: " << modeName
                      << "，阈值: " << isovalue << std::endl;

            MeshData mesh = loadDicomAsMesh(g_dicomFolder, isovalue, true, 0.01f, true,
                                             g_useGPU, g_shaderDir);
            g_dicomLoaded = !mesh.vertices.empty();

            if (g_dicomLoaded) {
                // 1. 顶点焊接 — Marching Cubes 为每个三角形生成独立顶点，
                //    共享边的顶点位置完全相同但索引不同。
                //    必须先焊接，否则 Laplacian 平滑会使每个三角形独立收缩 → 离散点
                std::cout << "  焊接前: " << mesh.vertexCount() << " 顶点, "
                          << mesh.triangleCount() << " 三角形" << std::endl;
                weldVertices(mesh, 1e-5f);
                std::cout << "  焊接后: " << mesh.vertexCount() << " 顶点, "
                          << mesh.triangleCount() << " 三角形" << std::endl;

                // 2. Laplacian 平滑（焊接后邻接表正确，平滑不会撕裂网格）
                laplacianSmooth(mesh, 2, 0.5f);

                // 3. 重新计算法线和切线
                computeNormals(mesh);
                computeTangents(mesh);
            }
            return mesh;
        }
        default: return createSphere(0.8f);
    }
}

/** 重建场景 */
static void rebuildScene() {
    // 释放旧资源
    for (auto& obj : g_objects)
        obj.mesh.release();
    g_objects.clear();

    MeshData mesh = generateCurrentMesh();
    computeNormals(mesh);
    computeTangents(mesh);

    // 计算 AABB 以自动调整相机
    AABB aabb = computeAABB(mesh);
    float diag = aabb.diagonal();
    float viewDistance = diag * 1.5f + 1.0f;

    // 主物体
    RenderObject mainObj;
    mainObj.mesh.upload(mesh);
    mainObj.name = mesh.name;

    // DICOM 网格使用不同的材质
    if (g_currentMeshType >= 10) {
        mainObj.baseColor = glm::vec3(0.9f, 0.88f, 0.85f);  // 骨骼/组织偏白
        mainObj.metallic = 0.0f;
        mainObj.roughness = 0.6f;
    } else {
        mainObj.baseColor = glm::vec3(0.7f, 0.65f, 0.6f);
        mainObj.metallic = 0.3f;
        mainObj.roughness = 0.35f;
    }
    mainObj.modelMatrix = glm::mat4(1.0f);
    g_objects.push_back(mainObj);

    // 地面（DICOM 模式下不显示地面和装饰球体）
    if (g_currentMeshType < 10) {
        MeshData planeMesh = createPlane(8.0f, 1);
        computeNormals(planeMesh);
        RenderObject plane;
        plane.mesh.upload(planeMesh);
        plane.name = "Ground";
        plane.baseColor = glm::vec3(0.3f, 0.35f, 0.4f);
        plane.metallic = 0.0f;
        plane.roughness = 0.8f;
        plane.modelMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0, -1.0f, 0));
        g_objects.push_back(plane);

        // 装饰球体
        for (int i = 0; i < 4; i++) {
            float angle = i * (3.14159f * 2.0f / 4.0f);
            float r = 2.5f;
            MeshData sphereMesh = createSphere(0.2f, 24, 12);
            computeNormals(sphereMesh);
            RenderObject sphere;
            sphere.mesh.upload(sphereMesh);
            sphere.name = "DecorSphere_" + std::to_string(i);
            sphere.baseColor = glm::vec3(0.8f, 0.2f, 0.3f);
            sphere.metallic = 0.8f;
            sphere.roughness = 0.2f;
            sphere.emissive = glm::vec3(0.2f, 0.05f, 0.05f);
            sphere.emissiveStrength = 0.5f;
            sphere.modelMatrix = glm::translate(glm::mat4(1.0f),
                glm::vec3(cos(angle) * r, -0.5f, sin(angle) * r));
            g_objects.push_back(sphere);
        }
    }

    // 自动调整相机距离
    if (g_currentMeshType >= 10) {
        g_camera.orbitDistance = viewDistance;
        g_camera.target = glm::vec3(0, 0, 0);
        g_camera.setMode(g_camera.mode);
    }

    std::cout << "场景重建: " << mesh.name
              << " | 顶点: " << mesh.vertexCount()
              << " | 三角形: " << mesh.triangleCount()
              << " | 对角线: " << diag << std::endl;
}

/** 优化当前网格 */
static void optimizeCurrentMesh() {
    if (g_objects.empty()) return;

    RenderObject& obj = g_objects[0];
    MeshData mesh;
    // 从 GPU 读回太复杂，直接重新生成
    mesh = generateCurrentMesh();
    computeNormals(mesh);
    computeTangents(mesh);

    std::cout << "\n=== 网格优化 ===" << std::endl;
    std::cout << "原始: " << mesh.vertexCount() << " 顶点, "
              << mesh.triangleCount() << " 三角形" << std::endl;

    // 1. 顶点焊接
    weldVertices(mesh, 1e-5f);
    std::cout << "焊接后: " << mesh.vertexCount() << " 顶点, "
              << mesh.triangleCount() << " 三角形" << std::endl;

    // 2. Taubin 平滑
    taubinSmooth(mesh, 5, 0.5f, -0.53f);
    std::cout << "Taubin 平滑完成" << std::endl;

    // 3. QEM 简化到 50%
    MeshData simplified = simplifyMesh(mesh, 0.5f);
    std::cout << "QEM 简化(50%): " << simplified.vertexCount() << " 顶点, "
              << simplified.triangleCount() << " 三角形" << std::endl;

    // 重新计算法线和切线
    computeNormals(simplified);
    computeTangents(simplified);

    // 上传
    obj.mesh.release();
    obj.mesh.upload(simplified);
    obj.name = simplified.name;

    std::cout << "=================\n" << std::endl;
}

/** LOD 切换 */
static void switchLOD(int level) {
    if (g_objects.empty()) return;

    MeshData baseMesh = generateCurrentMesh();
    computeNormals(baseMesh);
    computeTangents(baseMesh);

    auto lodChain = generateLODChain(baseMesh, 4);
    if (level >= (int)lodChain.size()) level = (int)lodChain.size() - 1;

    RenderObject& obj = g_objects[0];
    obj.mesh.release();
    obj.mesh.upload(lodChain[level].mesh);
    obj.name = lodChain[level].mesh.name;

    std::cout << "LOD Level " << level << ": "
              << lodChain[level].mesh.vertexCount() << " 顶点, "
              << lodChain[level].mesh.triangleCount() << " 三角形" << std::endl;
}

/** 设置光源 */
static void setupLights() {
    g_lights.clear();

    // 主方向光（太阳）
    Light dirLight;
    dirLight.type = LightType::Directional;
    dirLight.direction = glm::normalize(glm::vec3(-0.5f, -0.8f, -0.3f));
    dirLight.color = glm::vec3(1.0f, 0.95f, 0.85f);
    dirLight.intensity = 4.0f;
    g_lights.push_back(dirLight);

    // 点光 1（红色）
    Light pointLight1;
    pointLight1.type = LightType::Point;
    pointLight1.position = glm::vec3(3, 2, 3);
    pointLight1.color = glm::vec3(1.0f, 0.3f, 0.2f);
    pointLight1.intensity = 5.0f;
    pointLight1.range = 10.0f;
    g_lights.push_back(pointLight1);

    // 点光 2（蓝色）
    Light pointLight2;
    pointLight2.type = LightType::Point;
    pointLight2.position = glm::vec3(-3, 2, -3);
    pointLight2.color = glm::vec3(0.2f, 0.4f, 1.0f);
    pointLight2.intensity = 5.0f;
    pointLight2.range = 10.0f;
    g_lights.push_back(pointLight2);

    // 聚光灯
    Light spotLight;
    spotLight.type = LightType::Spot;
    spotLight.position = glm::vec3(0, 5, 0);
    spotLight.direction = glm::normalize(glm::vec3(0, -1, 0));
    spotLight.color = glm::vec3(0.9f, 0.9f, 1.0f);
    spotLight.intensity = 8.0f;
    spotLight.range = 15.0f;
    spotLight.innerCone = 0.95f;
    spotLight.outerCone = 0.85f;
    g_lights.push_back(spotLight);
}

// ============================================================
// 统计信息
// ============================================================

static void printStats() {
    if (!g_showStats) return;

    static double lastPrint = 0;
    double now = g_totalTime;
    if (now - lastPrint < 0.5f) return;
    lastPrint = now;

    int totalVerts = 0, totalTris = 0;
    for (const auto& obj : g_objects) {
        totalVerts += obj.mesh.vertexCount;
        totalTris  += obj.mesh.indexCount / 3;
    }

    std::cout << "\r[统计] FPS: " << std::fixed << std::setprecision(1) << (1.0f / g_deltaTime)
              << " | 物体: " << g_objects.size()
              << " | 顶点: " << totalVerts
              << " | 三角形: " << totalTris
              << " | 光源: " << g_lights.size()
              << " | 阴影: " << (g_renderer.enableShadow ? "ON" : "OFF")
              << " | SSAO: " << (g_renderer.enableSSAO ? "ON" : "OFF")
              << " | 线框: " << (g_renderer.enableWireframe ? "ON" : "OFF")
              << "        " << std::flush;
}

// ============================================================
// 主函数
// ============================================================

int main() {
    // 设置控制台为 UTF-8，避免中文乱码
    // SetConsoleOutputCP 在 VS Code ConPTY 中可能不生效，
    // system("chcp 65001 > nul") 通过子进程修改控制台代码页更可靠
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    system("chcp 65001 > nul");
#endif

    std::cout << "========================================" << std::endl;
    std::cout << "  Mesh 网格生成与优化渲染 Demo" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "键盘控制:" << std::endl;
    std::cout << "  [1-6] 切换网格类型" << std::endl;
    std::cout << "  [F2]  线框模式" << std::endl;
    std::cout << "  [F3]  法线可视化" << std::endl;
    std::cout << "  [F4]  深度可视化" << std::endl;
    std::cout << "  [O]   优化网格(焊接+平滑+简化)" << std::endl;
    std::cout << "  [L]   LOD 切换" << std::endl;
    std::cout << "  [G]   切换 GPU/CPU Marching Cubes" << std::endl;
    std::cout << "  [Tab] 切换相机模式" << std::endl;
    std::cout << "  [Space] 自动旋转" << std::endl;
    std::cout << "  [P]   暂停" << std::endl;
    std::cout << "  [R]   重置相机" << std::endl;
    std::cout << "  [F1]  统计信息" << std::endl;
    std::cout << "  [7]   DICOM 骨骼" << std::endl;
    std::cout << "  [8]   DICOM 软组织" << std::endl;
    std::cout << "  [9]   DICOM 皮肤" << std::endl;
    std::cout << "  [D]   DICOM 自定义阈值" << std::endl;
    std::cout << "  [ESC] 退出" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "默认 DICOM 路径: " << g_dicomFolder << std::endl;
    std::cout << "启动自动加载 DICOM 骨骼模式..." << std::endl;

    // GLFW 初始化
    if (!glfwInit()) {
        std::cerr << "GLFW 初始化失败!" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);  // MSAA

    GLFWwindow* window = glfwCreateWindow(g_windowWidth, g_windowHeight,
                                          "Mesh Render - 网格生成与优化渲染", nullptr, nullptr);
    if (!window) {
        std::cerr << "窗口创建失败!" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);

    // GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "GLAD 初始化失败!" << std::endl;
        return -1;
    }

    std::cout << "OpenGL 版本: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GPU: " << glGetString(GL_RENDERER) << std::endl;

    // 确定着色器目录
    // 尝试多个可能的路径（相对于工作目录和可执行文件目录）
    std::string shaderDir;
    FILE* test = nullptr;

    // 路径候选列表
    const char* paths[] = {
        "shaders",
        "MeshRender/shaders",
        "../shaders",
        "../MeshRender/shaders",
        "../../MeshRender/shaders"
    };
    bool found = false;
    for (int i = 0; i < 5; i++) {
        std::string p = std::string(paths[i]) + "/mesh_geometry_pass.vert";
        fopen_s(&test, p.c_str(), "rb");
        if (test) {
            fclose(test);
            shaderDir = paths[i];
            found = true;
            break;
        }
    }

    // 如果相对路径都找不到，尝试可执行文件所在目录
    if (!found) {
        // 获取可执行文件路径
        char exePath[MAX_PATH];
        GetModuleFileNameA(nullptr, exePath, MAX_PATH);
        std::string exeDir(exePath);
        size_t lastSlash = exeDir.find_last_of("\\/");
        if (lastSlash != std::string::npos) {
            exeDir = exeDir.substr(0, lastSlash);
        }
        // 尝试 exeDir/shaders 和 exeDir/../shaders
        std::string candidates[] = {
            exeDir + "/shaders",
            exeDir + "/../shaders",
            exeDir + "/../../MeshRender/shaders"
        };
        for (int i = 0; i < 3; i++) {
            std::string p = candidates[i] + "/mesh_geometry_pass.vert";
            fopen_s(&test, p.c_str(), "rb");
            if (test) {
                fclose(test);
                shaderDir = candidates[i];
                found = true;
                break;
            }
        }
    }

    if (!found) {
        shaderDir = "shaders"; // 回退默认值
    }

    std::cout << "着色器目录: " << shaderDir << std::endl;

    // 保存着色器目录供 GPU Marching Cubes 使用
    g_shaderDir = shaderDir;

    // 渲染器初始化
    if (!g_renderer.init(g_windowWidth, g_windowHeight, shaderDir)) {
        std::cerr << "渲染器初始化失败! 着色器目录: " << shaderDir << std::endl;
        return -1;
    }

    // 相机初始化
    g_camera.target = glm::vec3(0, 0, 0);
    g_camera.orbitDistance = 5.0f;
    g_camera.yaw = -90;
    g_camera.pitch = 15;
    g_camera.aspect = (float)g_windowWidth / g_windowHeight;
    g_camera.setMode(CameraMode::Orbit);  // 设置模式并触发 updateOrbit

    // 场景
    setupLights();
    rebuildScene();

    // OpenGL 状态
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glEnable(GL_MULTISAMPLE);

    // 主循环
    auto lastTime = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window)) {
        auto now = std::chrono::high_resolution_clock::now();
        g_deltaTime = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;

        if (!g_paused) {
            g_totalTime += g_deltaTime;

            // 自动旋转
            if (g_autoRotate && !g_objects.empty()) {
                g_objects[0].modelMatrix = glm::rotate(
                    glm::mat4(1.0f), g_totalTime * 0.5f, glm::vec3(0, 1, 0));
            }

            // 动态光源
            if (g_lights.size() >= 3) {
                float t = g_totalTime;
                g_lights[1].position = glm::vec3(
                    cos(t * 0.7f) * 3.0f, 2.0f + sin(t * 0.5f) * 0.5f, sin(t * 0.7f) * 3.0f);
                g_lights[2].position = glm::vec3(
                    cos(t * 0.7f + 3.14159f) * 3.0f, 2.0f + sin(t * 0.5f + 3.14159f) * 0.5f,
                    sin(t * 0.7f + 3.14159f) * 3.0f);
            }

            // 相机更新
            g_camera.update(g_deltaTime);
        }

        // 渲染
        g_renderer.render(g_objects, g_lights, g_camera);

        // 统计
        printStats();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // 清理
    for (auto& obj : g_objects)
        obj.mesh.release();
    g_renderer.destroy();

    glfwTerminate();

    std::cout << "\n程序退出。" << std::endl;
    return 0;
}
