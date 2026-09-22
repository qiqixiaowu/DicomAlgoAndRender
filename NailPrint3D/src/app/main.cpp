/**
 * @file    main.cpp
 * @brief   NailPrint3D 主程序入口
 *
 * 完整流程:
 *   1. 生成/加载美甲网格
 *   2. 3D重建 (网格修复)
 *   3. RIP切片
 *   4. 色彩管理 (ICC + LUT + 抖动)
 *   5. G-code 生成
 *   6. OpenGL 实时预览
 */

#include "app/scene.h"
#include "app/input_manager.h"
#include "app/print_pipeline.h"
#include "app/ui_help.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <iostream>
#include <fstream>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

using namespace NailPrint3D;

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
    glfwWindowHint(GLFW_SAMPLES, 4);  // 4x MSAA 抗锯齿

    GLFWwindow* window = glfwCreateWindow(1280, 720, "NailPrint3D", nullptr, nullptr);
    if (!window) {
        std::cerr << "无法创建窗口" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    // 初始化 GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "无法初始化 GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);  // 启用 MSAA 抗锯齿
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 创建场景
    Scene scene;

    // 注册回调
    registerCallbacks(window, scene);

    // 运行完整流程
    runNailPrintPipeline(scene);

    // 初始化渲染器
    std::string shaderDir = "shaders";
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

    if (!scene.renderer.init(shaderDir)) {
        std::cerr << "渲染器初始化失败，着色器目录: " << shaderDir << std::endl;
    }

    // 上传网格
    scene.originalMesh = scene.currentMesh;
    scene.nailBaseMesh = scene.currentMesh;
    scene.glMesh.upload(scene.currentMesh);
    scene.camera.setAspect(1280.0f / 720.0f);

    // 根据包围盒调整相机
    Vec3 center = scene.currentMesh.bbox.center();
    float bboxSize = std::max(
        scene.currentMesh.bbox.max.x - scene.currentMesh.bbox.min.x,
        std::max(scene.currentMesh.bbox.max.y - scene.currentMesh.bbox.min.y,
                 scene.currentMesh.bbox.max.z - scene.currentMesh.bbox.min.z));
    scene.camera.setTarget(glm::vec3(center.x, center.y, center.z));
    scene.camera.setDistance(bboxSize * 1.5f);

    // 生成测试纹理
    std::cout << "\n>>> 生成测试纹理..." << std::endl;
    scene.texCartoon.generateTestPattern(0);
    scene.texPortrait.generateTestPattern(1);
    scene.texGeometric.generateTestPattern(2);
    scene.texText.generateTestPattern(3);
    std::cout << "    4个测试纹理已生成" << std::endl;

    // 设置默认纹理变换
    TextureTransform texXform;
    texXform.scale = 1.0f;
    texXform.opacity = 0.85f;
    texXform.blendMode = 0;
    scene.renderer.setTextureTransform(texXform);

    // 渲染循环
    while (!glfwWindowShouldClose(window)) {
        scene.renderer.render(scene.glMesh, scene.camera);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // 清理
    scene.glMesh.destroy();
    scene.glSliceMesh.destroy();
    scene.glHandMesh.destroy();
    scene.texCartoon.destroy();
    scene.texPortrait.destroy();
    scene.texGeometric.destroy();
    scene.texText.destroy();
    scene.texUserPhoto.destroy();
    glfwTerminate();
    return 0;
}
