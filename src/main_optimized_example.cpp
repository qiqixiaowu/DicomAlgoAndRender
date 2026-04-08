/**
 * ==========================================
 * 优化的Main.cpp - 支持高级体绘制特性
 * ==========================================
 * 
 * 新增功能：
 * 1. 多种渲染模式切换（VolRen, MIP, MinIP, Average）
 * 2. 环境光遮蔽（Ambient Occlusion）
 * 3. 边缘增强（Edge Enhancement）
 * 4. 裁剪平面（Clipping Plane）
 * 5. 改进的光照模型（Blinn-Phong + Rim Light）
 * 6. 蓝噪声抖动（Blue Noise Jittering）
 * 
 * 键盘控制：
 * - ESC: 退出程序
 * - 1/2/3/4: 切换渲染模式（VolRen/MIP/MinIP/Average）
 * - Q/A: 增加/减少采样率
 * - W/S: 增加/减少密度缩放
 * - E/D: 增加/减少亮度
 * - R/F: 增加/减少抖动强度
 * - T/G: 启用/禁用环境光遮蔽
 * - Y/H: 增加/减少AO强度
 * - U/J: 启用/禁用边缘增强
 * - I/K: 增加/减少边缘强度
 * - O/L: 启用/禁用裁剪平面
 * - 上/下: 调整窗位
 * - 左/右: 调整窗宽
 * - P: 打印当前参数
 * 
 * MPR控制：
 * - M: 切换MPR四视口模式开关
 * - Z/X: Axial切面 上/下移动
 * - C/V: Sagittal切面 左/右移动
 * - B/N: Coronal切面 前/后移动
 * - 5: 切换MPR灰度/传递函数着色
 * - 6: 切换3D视图中MPR线叠加显示
 * - 7: 切换十字线显示
 */

#include "transferFunction.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#define NOMINMAX
#include <windows.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include "camera.h"
#include "dicom_utils.hpp"
#include "volume_build.hpp"
#include "post_Processor.h"
#include "mpr_renderer.h"
#include "region_growing.hpp"
#include "trt_vessel_seg.hpp"  // TensorRT C++ 部署推理
#include <chrono>

// ==========================================
// 全局状态结构体
// ==========================================
struct RenderState {
    // 渲染模式
    int compositeMode = 0;        // 0=VolRen, 1=MIP, 2=MinIP, 3=Average
    
    // 采样参数
    float sampleRate = 1.0f;
    float densityScale = 1.2f;
    float stepSize = 0.0015f;
    
    // 窗宽窗位
    float windowLevel = 0.35f;
    float windowWidth = 0.1f;
    
    // 光照参数
    float brightness = 1.5f;
    float ambient = 0.3f;
    float diffuse = 0.7f;
    float specular = 0.3f;
    float shininess = 32.0f;
    glm::vec3 lightPos = glm::vec3(5.0f, 5.0f, 5.0f);
    glm::vec3 lightColor = glm::vec3(1.0f, 1.0f, 0.9f);
    
    // 抖动参数
    float jitterStrength = 0.5f;
    float noiseScale = 2.0f;
    
    // 环境光遮蔽
    bool enableAO = false;
    float aoStrength = 0.5f;
    int aoSamples = 4;
    
    // 边缘增强
    bool enableEdgeEnhance = false;
    float edgeStrength = 0.5f;
    
    // 裁剪平面
    bool enableClipping = false;
    glm::vec4 clipPlane = glm::vec4(0.0f, 1.0f, 0.0f, 0.5f); // (normal.xyz, distance)
    
    // 体数据信息
    glm::vec3 volumeSize = glm::vec3(512.0f, 512.0f, 200.0f);
    glm::vec3 spacing = glm::vec3(1.0f, 1.0f, 1.0f);
    
    // MPR参数
    bool enableMPR = false;              // 是否启用四视口MPR布局
    bool enableMPROverlay = false;       // 是否在3D视图中显示MPR切面线
    float mprLineWidth = 0.005f;         // MPR线宽（纹理空间）
    float mprLineAlpha = 0.7f;           // MPR线透明度

    // 区域生长分割参数
    bool  enableSegmentation = false;    // 是否显示分割结果叠加
    float segTolerance = 20.0f;          // 强度容差（uint8量纲，0~255）
    glm::vec3 segColor = glm::vec3(1.0f, 0.5f, 0.1f); // 分割高亮颜色（橙色）
    float segOpacity = 0.8f;             // 分割叠加不透明度
    float segBorderWidth = 0.003f;       // 边界线宽（纹理空间，0关闭边界高亮）
    bool  seedPointSet = false;          // 是否已设置种子点
    glm::ivec3 seedVoxel = glm::ivec3(0);
};

RenderState renderState;
MPRRenderer mprRenderer;

// ==========================================
// 区域生长全局状态
// ==========================================
RegionGrowingResult g_segResult;    // 最近一次分割结果
GLuint g_segMaskTexture = 0;        // 分割掩码3D纹理
const VolumeBuildResult* g_volumePtr = nullptr; // 指向体数据（main初始化后设置）

// 用于右键拾取种子点的全局状态
struct PickRequest {
    bool pending = false;
    double clickX = 0.0, clickY = 0.0;
    glm::mat4 projection = glm::mat4(1.0f);
    glm::mat4 view       = glm::mat4(1.0f);
    int winW = 1, winH = 1;
} g_pickRequest;

// ==========================================
// 相机控制
// ==========================================
OrbitCamera orbitCamera;
bool mousePressed = false;
float lastX = 400, lastY = 300;
bool firstMouse = true;

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            orbitCamera.processMouseButton(button, action, xpos, ypos);
            mousePressed = true;
        } else if (action == GLFW_RELEASE) {
            orbitCamera.processMouseButton(button, action, 0, 0);
            mousePressed = false;
        }
    }

    // 右键点击：设置区域生长种子点
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        g_pickRequest.pending = true;
        g_pickRequest.clickX  = xpos;
        g_pickRequest.clickY  = ypos;
        g_pickRequest.winW    = w;
        g_pickRequest.winH    = h;
        // projection/view 在主循环中每帧更新，这里在下一帧处理时读取
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (mousePressed) {
        orbitCamera.processMouseMotion(xpos, ypos);
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    orbitCamera.processMouseScroll(yoffset);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

// ==========================================
// 键盘输入处理
// ==========================================
void printRenderState() {
    system("cls");
   /* std::cout << "========================================\n";
    std::cout << "        体绘制参数状态\n";
    std::cout << "========================================\n";
    std::cout << "渲染模式: ";
    switch(renderState.compositeMode) {
        case 0: std::cout << "体绘制 (VolRen)\n"; break;
        case 1: std::cout << "最大密度投影 (MIP)\n"; break;
        case 2: std::cout << "最小密度投影 (MinIP)\n"; break;
        case 3: std::cout << "平均值投影 (Average)\n"; break;
    }
    std::cout << "----------------------------------------\n";
    std::cout << "采样率: " << renderState.sampleRate << "\n";
    std::cout << "密度缩放: " << renderState.densityScale << "\n";
    std::cout << "步长: " << renderState.stepSize << "\n";
    std::cout << "----------------------------------------\n";
    std::cout << "窗位: " << renderState.windowLevel << "\n";
    std::cout << "窗宽: " << renderState.windowWidth << "\n";
    std::cout << "----------------------------------------\n";
    std::cout << "亮度: " << renderState.brightness << "\n";
    std::cout << "环境光: " << renderState.ambient << "\n";
    std::cout << "漫反射: " << renderState.diffuse << "\n";
    std::cout << "镜面反射: " << renderState.specular << "\n";
    std::cout << "----------------------------------------\n";
    std::cout << "抖动强度: " << renderState.jitterStrength << "\n";
    std::cout << "噪声缩放: " << renderState.noiseScale << "\n";
    std::cout << "----------------------------------------\n";
    std::cout << "环境光遮蔽: " << (renderState.enableAO ? "启用" : "禁用") << "\n";
    std::cout << "AO强度: " << renderState.aoStrength << "\n";
    std::cout << "AO采样数: " << renderState.aoSamples << "\n";
    std::cout << "----------------------------------------\n";
    std::cout << "边缘增强: " << (renderState.enableEdgeEnhance ? "启用" : "禁用") << "\n";
    std::cout << "边缘强度: " << renderState.edgeStrength << "\n";
    std::cout << "----------------------------------------\n";
    std::cout << "裁剪平面: " << (renderState.enableClipping ? "启用" : "禁用") << "\n";
    std::cout << "----------------------------------------\n";
    std::cout << "MPR四视口: " << (renderState.enableMPR ? "启用" : "禁用") << "\n";
    std::cout << "MPR 3D叠加: " << (renderState.enableMPROverlay ? "启用" : "禁用") << "\n";
    std::cout << "Axial切面: " << mprRenderer.state.axialPos;
    if (renderState.volumeSize.z > 0)
        std::cout << " (层" << mprRenderer.getSliceIndex(MPRAxis::Axial, renderState.volumeSize) << "/" << (int)renderState.volumeSize.z << ")";
    std::cout << "\n";
    std::cout << "Sagittal切面: " << mprRenderer.state.sagittalPos;
    if (renderState.volumeSize.x > 0)
        std::cout << " (层" << mprRenderer.getSliceIndex(MPRAxis::Sagittal, renderState.volumeSize) << "/" << (int)renderState.volumeSize.x << ")";
    std::cout << "\n";
    std::cout << "Coronal切面: " << mprRenderer.state.coronalPos;
    if (renderState.volumeSize.y > 0)
        std::cout << " (层" << mprRenderer.getSliceIndex(MPRAxis::Coronal, renderState.volumeSize) << "/" << (int)renderState.volumeSize.y << ")";
    std::cout << "\n";
    std::cout << "MPR着色: " << (mprRenderer.state.useTransferFunc ? "传递函数" : "灰度") << "\n";
    std::cout << "----------------------------------------\n";
    std::cout << "MPR窗位: " << mprRenderer.state.mprWindowLevel << "\n";
    std::cout << "MPR窗宽: " << mprRenderer.state.mprWindowWidth << "\n";
    std::cout << "MPR亮度: " << mprRenderer.state.mprBrightness << "\n";
    std::cout << "========================================\n";*/
}

// 前向声明（定义在后面的辅助函数，processInput 中会用到）
void runRegionGrowing(const VolumeBuildResult& volumeData, int seedX, int seedY, int seedZ);

void processInput(GLFWwindow* window, Shader* shader) {
    static bool keyPressed[GLFW_KEY_LAST] = {false};
    bool stateChanged = false;
    
    // ESC退出
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
    
    // 渲染模式切换
    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS && !keyPressed[GLFW_KEY_1]) {
        renderState.compositeMode = 0;
        stateChanged = true;
        keyPressed[GLFW_KEY_1] = true;
    }
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS && !keyPressed[GLFW_KEY_2]) {
        renderState.compositeMode = 1;
        stateChanged = true;
        keyPressed[GLFW_KEY_2] = true;
    }
    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS && !keyPressed[GLFW_KEY_3]) {
        renderState.compositeMode = 2;
        stateChanged = true;
        keyPressed[GLFW_KEY_3] = true;
    }
    if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS && !keyPressed[GLFW_KEY_4]) {
        renderState.compositeMode = 3;
        stateChanged = true;
        keyPressed[GLFW_KEY_4] = true;
    }
    
    // 采样率调整
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        renderState.sampleRate = glm::clamp(renderState.sampleRate + 0.01f, 0.1f, 5.0f);
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        renderState.sampleRate = glm::clamp(renderState.sampleRate - 0.01f, 0.1f, 5.0f);
    }
    
    // 密度缩放
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        renderState.densityScale = glm::clamp(renderState.densityScale + 0.01f, 0.1f, 5.0f);
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        renderState.densityScale = glm::clamp(renderState.densityScale - 0.01f, 0.1f, 5.0f);
    }
    
    // 亮度调整
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        renderState.brightness = glm::clamp(renderState.brightness + 0.01f, 0.1f, 5.0f);
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        renderState.brightness = glm::clamp(renderState.brightness - 0.01f, 0.1f, 5.0f);
    }
    
    // 抖动强度
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        renderState.jitterStrength = glm::clamp(renderState.jitterStrength + 0.01f, 0.0f, 2.0f);
    }
    if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) {
        renderState.jitterStrength = glm::clamp(renderState.jitterStrength - 0.01f, 0.0f, 2.0f);
    }
    
    // 环境光遮蔽开关
    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS && !keyPressed[GLFW_KEY_T]) {
        renderState.enableAO = true;
        stateChanged = true;
        keyPressed[GLFW_KEY_T] = true;
    }
    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS && !keyPressed[GLFW_KEY_G]) {
        renderState.enableAO = false;
        stateChanged = true;
        keyPressed[GLFW_KEY_G] = true;
    }
    
    // AO强度
    if (glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS) {
        renderState.aoStrength = glm::clamp(renderState.aoStrength + 0.01f, 0.0f, 2.0f);
    }
    if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS) {
        renderState.aoStrength = glm::clamp(renderState.aoStrength - 0.01f, 0.0f, 2.0f);
    }
    
    // 边缘增强开关
    if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS && !keyPressed[GLFW_KEY_U]) {
        renderState.enableEdgeEnhance = true;
        stateChanged = true;
        keyPressed[GLFW_KEY_U] = true;
    }
    if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS && !keyPressed[GLFW_KEY_J]) {
        renderState.enableEdgeEnhance = false;
        stateChanged = true;
        keyPressed[GLFW_KEY_J] = true;
    }
    
    // 边缘强度
    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) {
        renderState.edgeStrength = glm::clamp(renderState.edgeStrength + 0.01f, 0.0f, 2.0f);
    }
    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) {
        renderState.edgeStrength = glm::clamp(renderState.edgeStrength - 0.01f, 0.0f, 2.0f);
    }
    
    // 裁剪平面开关
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS && !keyPressed[GLFW_KEY_O]) {
        renderState.enableClipping = true;
        stateChanged = true;
        keyPressed[GLFW_KEY_O] = true;
    }
    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS && !keyPressed[GLFW_KEY_L]) {
        renderState.enableClipping = false;
        stateChanged = true;
        keyPressed[GLFW_KEY_L] = true;
    }
    
    // ---- MPR控制 ----
    
    // M键：切换MPR四视口模式
    if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS && !keyPressed[GLFW_KEY_M]) {
        renderState.enableMPR = !renderState.enableMPR;
        stateChanged = true;
        keyPressed[GLFW_KEY_M] = true;
    }
    
    // Z/X键：Axial切面上下移动
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) {
        mprRenderer.adjustSlice(MPRAxis::Axial, mprRenderer.state.sliceStep);
    }
    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS) {
        mprRenderer.adjustSlice(MPRAxis::Axial, -mprRenderer.state.sliceStep);
    }
    
    // C/V键：Sagittal切面左右移动
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
        mprRenderer.adjustSlice(MPRAxis::Sagittal, mprRenderer.state.sliceStep);
    }
    if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS) {
        mprRenderer.adjustSlice(MPRAxis::Sagittal, -mprRenderer.state.sliceStep);
    }
    
    // B/N键：Coronal切面前后移动
    if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS) {
        mprRenderer.adjustSlice(MPRAxis::Coronal, mprRenderer.state.sliceStep);
    }
    if (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS) {
        mprRenderer.adjustSlice(MPRAxis::Coronal, -mprRenderer.state.sliceStep);
    }
    
    // 5键：切换MPR灰度/传递函数着色
    if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS && !keyPressed[GLFW_KEY_5]) {
        mprRenderer.state.useTransferFunc = !mprRenderer.state.useTransferFunc;
        stateChanged = true;
        keyPressed[GLFW_KEY_5] = true;
    }
    
    // 6键：切换3D视图中MPR线叠加
    if (glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS && !keyPressed[GLFW_KEY_6]) {
        renderState.enableMPROverlay = !renderState.enableMPROverlay;
        stateChanged = true;
        keyPressed[GLFW_KEY_6] = true;
    }
    
    // 7键：切换十字线显示
    if (glfwGetKey(window, GLFW_KEY_7) == GLFW_PRESS && !keyPressed[GLFW_KEY_7]) {
        mprRenderer.state.showCrosshair = !mprRenderer.state.showCrosshair;
        stateChanged = true;
        keyPressed[GLFW_KEY_7] = true;
    }
    
    // 窗宽窗位调整（3D体绘制）
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        renderState.windowLevel = glm::clamp(renderState.windowLevel + 0.003f, 0.01f, 1.0f);
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        renderState.windowLevel = glm::clamp(renderState.windowLevel - 0.003f, 0.01f, 1.0f);
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        renderState.windowWidth = glm::clamp(renderState.windowWidth + 0.003f, 0.001f, 2.0f);
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        renderState.windowWidth = glm::clamp(renderState.windowWidth - 0.003f, 0.001f, 2.0f);
    }
    
    // MPR独立窗宽窗位调整（8/9调窗位，0/-调窗宽，=调亮度）
    if (glfwGetKey(window, GLFW_KEY_8) == GLFW_PRESS) {
        mprRenderer.state.mprWindowLevel = glm::clamp(mprRenderer.state.mprWindowLevel + 0.005f, 0.0f, 1.0f);
    }
    if (glfwGetKey(window, GLFW_KEY_9) == GLFW_PRESS) {
        mprRenderer.state.mprWindowLevel = glm::clamp(mprRenderer.state.mprWindowLevel - 0.005f, 0.0f, 1.0f);
    }
    if (glfwGetKey(window, GLFW_KEY_0) == GLFW_PRESS) {
        mprRenderer.state.mprWindowWidth = glm::clamp(mprRenderer.state.mprWindowWidth + 0.005f, 0.01f, 2.0f);
    }
    if (glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS) {
        mprRenderer.state.mprWindowWidth = glm::clamp(mprRenderer.state.mprWindowWidth - 0.005f, 0.01f, 2.0f);
    }
    if (glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS) {
        mprRenderer.state.mprBrightness = glm::clamp(mprRenderer.state.mprBrightness + 0.02f, 0.1f, 5.0f);
    }
    if (glfwGetKey(window, GLFW_KEY_BACKSPACE) == GLFW_PRESS) {
        mprRenderer.state.mprBrightness = glm::clamp(mprRenderer.state.mprBrightness - 0.02f, 0.1f, 5.0f);
    }
    
    // 打印当前状态
    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS && !keyPressed[GLFW_KEY_P]) {
        printRenderState();
        keyPressed[GLFW_KEY_P] = true;
    }

    // ---- 区域生长分割控制 ----

    // F1: 开/关分割叠加显示
    if (glfwGetKey(window, GLFW_KEY_F1) == GLFW_PRESS && !keyPressed[GLFW_KEY_F1]) {
        renderState.enableSegmentation = !renderState.enableSegmentation;
        keyPressed[GLFW_KEY_F1] = true;
    }

    // F2: 增大容差（+5，更宽松，区域更大）
    if (glfwGetKey(window, GLFW_KEY_F2) == GLFW_PRESS && !keyPressed[GLFW_KEY_F2]) {
        renderState.segTolerance = glm::clamp(renderState.segTolerance + 5.0f, 1.0f, 255.0f);
        keyPressed[GLFW_KEY_F2] = true;
    }

    // F3: 减小容差（-5，更严格，区域更小）
    if (glfwGetKey(window, GLFW_KEY_F3) == GLFW_PRESS && !keyPressed[GLFW_KEY_F3]) {
        renderState.segTolerance = glm::clamp(renderState.segTolerance - 5.0f, 1.0f, 255.0f);
        keyPressed[GLFW_KEY_F3] = true;
    }

    // F4: 清除分割结果
    if (glfwGetKey(window, GLFW_KEY_F4) == GLFW_PRESS && !keyPressed[GLFW_KEY_F4]) {
        renderState.enableSegmentation = false;
        renderState.seedPointSet = false;
        if (g_segMaskTexture != 0) {
            glDeleteTextures(1, &g_segMaskTexture);
            g_segMaskTexture = 0;
        }
        g_segResult = RegionGrowingResult{};
        keyPressed[GLFW_KEY_F4] = true;
    }

    // F5: 切换边界高亮开关
    if (glfwGetKey(window, GLFW_KEY_F5) == GLFW_PRESS && !keyPressed[GLFW_KEY_F5]) {
        renderState.segBorderWidth = (renderState.segBorderWidth > 0.0f) ? 0.0f : 0.003f;
        keyPressed[GLFW_KEY_F5] = true;
    }

    // F6: 如果已有种子点，以当前容差重新生长
    if (glfwGetKey(window, GLFW_KEY_F6) == GLFW_PRESS && !keyPressed[GLFW_KEY_F6]) {
        if (g_volumePtr && renderState.seedPointSet) {
            runRegionGrowing(*g_volumePtr,
                renderState.seedVoxel.x,
                renderState.seedVoxel.y,
                renderState.seedVoxel.z);
        } else {
        }
        keyPressed[GLFW_KEY_F6] = true;
    }

    // 重置按键状态
    for (int key = 0; key < GLFW_KEY_LAST; key++) {
        if (glfwGetKey(window, key) == GLFW_RELEASE) {
            keyPressed[key] = false;
        }
    }
    
    // 如果状态改变，打印信息
    if (stateChanged) {
        printRenderState();
    }
}

// ==========================================
// 立方体几何体
// ==========================================
struct CubeGeometry {
    GLuint VAO, VBO, EBO;
    GLsizei indexCount;
};

CubeGeometry createCubeGeometry() {
    float vertices[] = {
        0.0f, 0.0f, 0.0f,  // v0
        1.0f, 0.0f, 0.0f,  // v1
        1.0f, 1.0f, 0.0f,  // v2
        0.0f, 1.0f, 0.0f,  // v3
        0.0f, 0.0f, 1.0f,  // v4
        1.0f, 0.0f, 1.0f,  // v5
        1.0f, 1.0f, 1.0f,  // v6
        0.0f, 1.0f, 1.0f   // v7
    };

    unsigned int indices[] = {
        // 前面
        0, 1, 2,  0, 2, 3,
        // 右面
        1, 5, 6,  1, 6, 2,
        // 后面
        5, 4, 7,  5, 7, 6,
        // 左面
        4, 0, 3,  4, 3, 7,
        // 上面
        3, 2, 6,  3, 6, 7,
        // 下面
        4, 5, 1,  4, 1, 0
    };

    CubeGeometry cube;
    glGenVertexArrays(1, &cube.VAO);
    glGenBuffers(1, &cube.VBO);
    glGenBuffers(1, &cube.EBO);

    glBindVertexArray(cube.VAO);

    glBindBuffer(GL_ARRAY_BUFFER, cube.VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cube.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    cube.indexCount = sizeof(indices) / sizeof(indices[0]);
    return cube;
}

// ==========================================
// 3D纹理创建
// ==========================================
GLuint create3DTextureFromVolume(const VolumeBuildResult& volumeData) {
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_3D, textureID);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // 转换为浮点数据
    std::vector<float> floatData(volumeData.buffer.size());
    for (size_t i = 0; i < volumeData.buffer.size(); ++i) {
        floatData[i] = static_cast<float>(volumeData.buffer[i]) / 255.0f;
    }

    // 上传3D纹理数据
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F,
        volumeData.width, volumeData.height, volumeData.depth,
        0, GL_RED, GL_FLOAT, floatData.data());

    glBindTexture(GL_TEXTURE_3D, 0);
    
    // 更新体数据尺寸
    renderState.volumeSize = glm::vec3(volumeData.width, volumeData.height, volumeData.depth);
    
    return textureID;
}

// ==========================================
// 分割掩码3D纹理创建/更新
// ==========================================
/**
 * 将区域生长结果的二值掩码上传为3D纹理。
 * 8-bit单通道（GL_R8）：0=背景，255=分割区域（着色器中归一化为0.0/1.0）
 * 会自动释放旧纹理。
 */
GLuint createOrUpdateSegMaskTexture(const RegionGrowingResult& segResult, GLuint oldTexture)
{
    if (oldTexture != 0) {
        glDeleteTextures(1, &oldTexture);
    }

    GLuint texID;
    glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_3D, texID);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    // 使用 NEAREST 过滤，避免掩码边界模糊
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage3D(GL_TEXTURE_3D, 0, GL_R8,
        segResult.width, segResult.height, segResult.depth,
        0, GL_RED, GL_UNSIGNED_BYTE, segResult.mask.data());

    glBindTexture(GL_TEXTURE_3D, 0);
    return texID;
}

// ==========================================
// 深度学习分割掩码加载（从 Python 推理导出的 .raw 文件）
// ==========================================
/**
 * 从 infer_vessel_seg.py 导出的 .raw + _meta.json 文件读取血管分割掩码。
 *
 * .raw 格式：uint8，连续存储，布局 [D][H][W]（C 行主序），
 *           值 0=背景, 255=血管，与 RegionGrowingResult.mask 完全一致。
 *
 * 用法:
 *   auto dlMask = loadDLVesselMask("vessel_mask.raw", "vessel_mask_meta.json");
 *   if (!dlMask.mask.empty()) {
 *       g_segMaskTexture = createOrUpdateSegMaskTexture(dlMask, g_segMaskTexture);
 *       renderState.enableSegmentation = true;
 *       renderState.segColor = glm::vec3(1.0f, 0.15f, 0.1f); // 血管红色
 *   }
 */
RegionGrowingResult loadDLVesselMask(const std::string& rawPath,
                                     const std::string& metaPath)
{
    RegionGrowingResult result;

    // --- 1. 读取元数据 JSON（手动解析，避免依赖第三方库）---
    uint32_t W = 0, H = 0, D = 0;
    {
        std::ifstream mf(metaPath);
        if (!mf.is_open()) {
            std::cerr << "[DLMask] 无法打开元数据文件: " << metaPath << std::endl;
            return result;
        }
        std::string line;
        while (std::getline(mf, line)) {
            // 简单 key-value 解析："width": 512
            auto extract = [&line](const std::string& key, uint32_t& val) {
                auto pos = line.find("\"" + key + "\"");
                if (pos != std::string::npos) {
                    auto colon = line.find(':', pos);
                    if (colon != std::string::npos) {
                        val = static_cast<uint32_t>(std::stoul(line.substr(colon + 1)));
                        return true;
                    }
                }
                return false;
            };
            extract("width",  W);
            extract("height", H);
            extract("depth",  D);
        }
    }

    if (W == 0 || H == 0 || D == 0) {
        std::cerr << "[DLMask] 元数据解析失败（width/height/depth 为 0）: "
                  << metaPath << std::endl;
        return result;
    }

    // --- 2. 读取 .raw 二值掩码 ---
    size_t expected = static_cast<size_t>(W) * H * D;
    {
        std::ifstream rf(rawPath, std::ios::binary);
        if (!rf.is_open()) {
            std::cerr << "[DLMask] 无法打开掩码文件: " << rawPath << std::endl;
            return result;
        }
        result.mask.resize(expected);
        rf.read(reinterpret_cast<char*>(result.mask.data()),
                static_cast<std::streamsize>(expected));
        if (!rf) {
            std::cerr << "[DLMask] 读取不完整，文件可能已损坏: " << rawPath << std::endl;
            result.mask.clear();
            return result;
        }
    }

    result.width  = W;
    result.height = H;
    result.depth  = D;
    result.voxelCount = static_cast<uint64_t>(
        std::count_if(result.mask.begin(), result.mask.end(),
                      [](uint8_t v) { return v > 0; }));

    std::cout << "[DLMask] 加载成功: " << rawPath << "\n"
              << "         尺寸=" << W << "x" << H << "x" << D
              << " 血管体素=" << result.voxelCount << "\n";
    return result;
}

// ==========================================
// 区域生长入口函数
// ==========================================
/**
 * 在体数据中以给定体素坐标为种子点执行区域生长，
 * 并将结果上传为 GPU 分割掩码纹理。
 *
 * @param volumeData  原始体数据
 * @param seedX/Y/Z   种子点体素坐标
 */
void runRegionGrowing(const VolumeBuildResult& volumeData, int seedX, int seedY, int seedZ)
{
    std::cout << "[RegionGrow] 开始区域生长，种子点=("
              << seedX << ", " << seedY << ", " << seedZ
              << "), 容差=" << (int)renderState.segTolerance << "\n";

    // 执行区域生长（6连通，无体素数量限制）
    g_segResult = regionGrow3D(
        volumeData.buffer,
        volumeData.width,
        volumeData.height,
        volumeData.depth,
        seedX, seedY, seedZ,
        (uint8_t)glm::clamp((int)renderState.segTolerance, 0, 255),
        Connectivity::CONNECT_6,
        0  // 不限制最大体素数
    );

    if (g_segResult.empty()) {
        std::cout << "[RegionGrow] 警告: 未分割到任何体素，请检查种子点或调整容差\n";
        return;
    }

    // 上传掩码纹理到 GPU
    g_segMaskTexture = createOrUpdateSegMaskTexture(g_segResult, g_segMaskTexture);

    // 自动启用分割显示
    renderState.enableSegmentation = true;
    renderState.seedPointSet = true;

    // 打印统计信息
    float x0, y0, z0, x1, y1, z1;
    g_segResult.getBoundingBox(x0, y0, z0, x1, y1, z1);
}

// ==========================================
// 射线与 AABB 求交（CPU侧，用于拾取种子点）
// ==========================================
static bool rayCastAABB(const glm::vec3& ro, const glm::vec3& rd,
                        float& tEnter, float& tExit)
{
    glm::vec3 invDir = 1.0f / (rd + glm::vec3(1e-9f));
    glm::vec3 t1 = (glm::vec3(0.0f) - ro) * invDir;
    glm::vec3 t2 = (glm::vec3(1.0f) - ro) * invDir;
    glm::vec3 tMin = (glm::min)(t1, t2);
    glm::vec3 tMax = (glm::max)(t1, t2);
    tEnter = glm::max(glm::max(tMin.x, tMin.y), tMin.z);
    tExit  = glm::min(glm::min(tMax.x, tMax.y), tMax.z);
    return tExit >= (glm::max)(tEnter, 0.0f);
}

/**
 * 通过屏幕点击位置选择种子体素。
 * 射线从相机出发，穿过单位包围盒 [0,1]^3，
 * 沿射线方向找到第一个密度值超过 windowLevel 阈值的体素作为种子点。
 *
 * @param clickX/Y     窗口坐标（像素）
 * @param winW/winH    窗口尺寸
 * @param projection   投影矩阵
 * @param view         观察矩阵
 * @param volumeData   体数据（用于读取体素强度）
 * @param outSeedX/Y/Z 输出的种子体素坐标
 * @return true 表示找到有效种子点
 */
bool pickSeedPointFromClick(
    double clickX, double clickY,
    int winW, int winH,
    const glm::mat4& projection,
    const glm::mat4& view,
    const VolumeBuildResult& volumeData,
    int& outSeedX, int& outSeedY, int& outSeedZ)
{
    // 屏幕坐标 → NDC
    float ndcX = (float)(2.0 * clickX / winW - 1.0);
    float ndcY = (float)(1.0 - 2.0 * clickY / winH); // Y轴翻转

    // 反投影：NDC → 视空间
    glm::mat4 invProj = glm::inverse(projection);
    glm::vec4 clipPos  = glm::vec4(ndcX, ndcY, 1.0f, 1.0f);
    glm::vec4 viewPos  = invProj * clipPos;
    viewPos /= viewPos.w;

    // 视空间 → 世界空间，得到射线方向
    glm::mat4 invView     = glm::inverse(view);
    glm::vec3 worldTarget = glm::vec3(invView * viewPos);
    glm::vec3 rayOrigin   = orbitCamera.getPosition();
    glm::vec3 rayDir      = glm::normalize(worldTarget - rayOrigin);

    // 射线与体包围盒 [0,1]^3 求交
    float tEnter, tExit;
    if (!rayCastAABB(rayOrigin, rayDir, tEnter, tExit)) {
        std::cout << "[RegionGrow] 点击位置未命中体包围盒\n";
        return false;
    }
    tEnter = (glm::max)(tEnter, 0.0f);

    // 沿射线步进，找第一个密度 > windowLevel 的体素
    const float pickStep = 0.002f; // 步长（纹理空间）
    int numSteps = (int)((tExit - tEnter) / pickStep) + 1;

    uint32_t w = volumeData.width;
    uint32_t h = volumeData.height;
    uint32_t d = volumeData.depth;

    for (int i = 0; i < numSteps; ++i) {
        float t = tEnter + pickStep * i;
        glm::vec3 pos = rayOrigin + rayDir * t;

        // 边界裁剪
        if (glm::any(glm::lessThan(pos, glm::vec3(0.0f))) ||
            glm::any(glm::greaterThan(pos, glm::vec3(1.0f)))) continue;

        // 纹理坐标 → 体素坐标
        int vx = (int)(pos.x * (float)(w - 1) + 0.5f);
        int vy = (int)(pos.y * (float)(h - 1) + 0.5f);
        int vz = (int)(pos.z * (float)(d - 1) + 0.5f);
        vx = glm::clamp(vx, 0, (int)w - 1);
        vy = glm::clamp(vy, 0, (int)h - 1);
        vz = glm::clamp(vz, 0, (int)d - 1);

        float density = (float)volumeData.buffer[(size_t)vz * h * w + (size_t)vy * w + vx] / 255.0f;

        // 以当前 windowLevel 作为判断有效体素的最低密度
        if (density >= renderState.windowLevel) {
            outSeedX = vx;
            outSeedY = vy;
            outSeedZ = vz;
            std::cout << "[RegionGrow] 拾取到种子点: ("
                      << vx << ", " << vy << ", " << vz
                      << "), 密度=" << density << "\n";
            return true;
        }
    }

    std::cout << "[RegionGrow] 射线未命中有效体素（密度均低于窗位），请调整窗位后重试\n";
    return false;
}
void updateShaderUniforms(Shader* shader, int width, int height) {
    shader->use();
    
    // 基本渲染参数
    shader->setInt("uCompositeMode", renderState.compositeMode);
    shader->setFloat("stepSize", renderState.stepSize);
    shader->setFloat("uSampleRate", renderState.sampleRate);
    shader->setFloat("uDensityScale", renderState.densityScale);
    
    // 窗宽窗位
    shader->setFloat("windowLevel", renderState.windowLevel);
    shader->setFloat("windowWidth", renderState.windowWidth);
    
    // 光照参数
    shader->setFloat("brightness", renderState.brightness);
    shader->setFloat("ambient", renderState.ambient);
    shader->setFloat("diffuse", renderState.diffuse);
    shader->setFloat("specular", renderState.specular);
    shader->setFloat("shininess", renderState.shininess);
    shader->setVec3("lightPos", renderState.lightPos);
    shader->setVec3("lightColor", renderState.lightColor);
    
    // 抖动参数
    shader->setFloat("jitterStrength", renderState.jitterStrength);
    shader->setFloat("noiseScale", renderState.noiseScale);
    shader->setFloat("time", (float)glfwGetTime());
    shader->setVec2("screenSize", glm::vec2((float)width, (float)height));
    
    // 环境光遮蔽
    shader->setBool("enableAO", renderState.enableAO);
    shader->setFloat("aoStrength", renderState.aoStrength);
    shader->setInt("aoSamples", renderState.aoSamples);
    
    // 边缘增强
    shader->setBool("enableEdgeEnhance", renderState.enableEdgeEnhance);
    shader->setFloat("edgeStrength", renderState.edgeStrength);
    
    // 裁剪平面
    shader->setBool("enableClipping", renderState.enableClipping);
    shader->setVec4("clipPlane", renderState.clipPlane);
    
    // 体数据信息
    shader->setVec3("uVolumeSize", renderState.volumeSize);
    shader->setVec3("uSpacing", renderState.spacing);
    
    // MPR叠加参数（3D视图中的切面线）
    shader->setBool("enableMPROverlay", renderState.enableMPROverlay);
    shader->setFloat("mprAxialPos", mprRenderer.state.axialPos);
    shader->setFloat("mprSagittalPos", mprRenderer.state.sagittalPos);
    shader->setFloat("mprCoronalPos", mprRenderer.state.coronalPos);
    shader->setFloat("mprLineWidth", renderState.mprLineWidth);
    shader->setFloat("mprLineAlpha", renderState.mprLineAlpha);

    // 区域生长分割参数
    shader->setBool("enableSegmentation", renderState.enableSegmentation && g_segMaskTexture != 0);
    shader->setVec3("segColor", renderState.segColor);
    shader->setFloat("segOpacity", renderState.segOpacity);
    shader->setFloat("segBorderWidth", renderState.segBorderWidth);
    // 分割掩码绑定到纹理单元 3（0=volume, 1=transferFunc, 2=保留, 3=segMask）
    shader->setInt("segMask", 3);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_3D, g_segMaskTexture != 0 ? g_segMaskTexture : 0);
}

// ==========================================
// 主函数
// ==========================================
int main(int argc, char** argv) {
    // 隐藏控制台窗口（可选）
    #ifdef _WIN32
    // HWND hWnd = GetConsoleWindow();
    // ShowWindow(hWnd, SW_HIDE);
    #endif

    if (argc < 2) {
        std::cerr << "用法:\n"
                  << "  TRT 推理模式（直接在 C++ 中调用 TensorRT）:\n"
                  << "    " << argv[0] << " <DICOM目录> <trt_meta.json>\n"
                  << "    步骤1: python dl_medical/export_trt.py\n"
                  << "    步骤2: " << argv[0] << " \"E:\\CT\" checkpoints/vessel_seg/trt_meta.json\n\n"
                  << "  加载 Python 预生成掩码模式：\n"
                  << "    " << argv[0] << " <DICOM目录> <vessel_mask.raw>\n"
                  << "    步骤1: python dl_medical/infer_vessel_seg.py --dicom_dir <DICOM目录>\n"
                  << "    步骤2: " << argv[0] << " \"E:\\CT\" outputs/vessel_result/vessel_mask.raw\n";
        return 1;
    }

    // 解析命令行参数
    // argv[2] 以 .json 结尾 → TRT 推理模式；以 .raw 结尾 → 加载 Python 掩码
    std::string dicomDir    = argv[1];
    std::string trtMetaPath;   // TRT 推理模式：trt_meta.json
    std::string dlRawPath;     // 加载模式：vessel_mask.raw
    std::string dlMetaPath;    // 加载模式：vessel_mask_meta.json

    if (argc >= 3) {
        std::string a2 = argv[2];
        // 检测扩展名：.json → TRT 模式
        if (a2.size() > 5 && a2.compare(a2.size() - 5, 5, ".json") == 0)
            trtMetaPath = a2;
        else
            dlRawPath = a2;
    }
    if (argc >= 4 && trtMetaPath.empty())
        dlMetaPath = argv[3];

    // 自动推导 meta 路径（raw 同目录，后缀替换）
    if (!dlRawPath.empty() && dlMetaPath.empty()) {
        dlMetaPath = dlRawPath;
        auto pos = dlMetaPath.rfind(".raw");
        if (pos != std::string::npos)
            dlMetaPath.replace(pos, 4, "_meta.json");
        else
            dlMetaPath += "_meta.json";
    }

    // 加载DICOM数据
    SeriesData series = collectSeries(dicomDir.c_str());
    VolumeBuildResult volumeData = buildVolume_none(series);
    if (volumeData.buffer.empty()) {
        return 1;
    }

    // 床板伪影去除（圆形FOV裁剪 + 最大连通域 + 孔洞填充）
    std::cout << "\n预处理: 自动去除床板伪影..." << std::endl;
    removeBedArtifact(volumeData);

    // 设置体数据全局指针（供区域生长使用）
    g_volumePtr = &volumeData;

    // ================================================================
    // TRT 推理（在 OpenGL 初始化前完成，推理结果存入 g_segResult）
    // 需在项目预处理器中定义 USE_TENSORRT，并设置 TRT_ROOT 环境变量
    // ================================================================
#ifdef USE_TENSORRT
    if (!trtMetaPath.empty()) {
        std::cout << "\n[TRT] 正在初始化 TensorRT 推理器...\n";
        TRTVesselSegmentor trtSeg;
        if (trtSeg.load(trtMetaPath)) {
            uint32_t fD = 0, fH = 0, fW = 0;
            std::cout << "[TRT] 重新读取 DICOM HU 浮点数据（精确 HU 值）...\n";
            auto floatVol = buildVolumeFloat(
                series, fD, fH, fW,
                trtSeg.meta().huMin,
                trtSeg.meta().huMax
            );
            if (!floatVol.empty()) {
                auto t0 = std::chrono::steady_clock::now();
                g_segResult = trtSeg.run(floatVol.data(), fD, fH, fW);
                double sec  = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - t0).count();
                std::cout << "[TRT] 推理完成  耗时=" << sec << "s"
                          << "  血管体素=" << g_segResult.voxelCount << "\n";
            } else {
                std::cerr << "[TRT] 警告: HU 数据加载失败\n";
            }
        }
    }
#endif

    // 初始化GLFW
    if (!glfwInit()) {
        std::cerr << "无法初始化GLFW" << std::endl;
        return -1;
    }

    // 设置相机
    orbitCamera.setTarget(glm::vec3(0.5f, 0.5f, 0.5f));
    orbitCamera.setDistance(3.0f);
    orbitCamera.setPitchLimits(-89.0f, 89.0f);
    orbitCamera.setDistanceLimits(1.0f, 10.0f);

    // 创建窗口
    glfwWindowHint(GLFW_SAMPLES, 4);  
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
 
    GLFWwindow* window = glfwCreateWindow(1280, 720, "DicomViewer", NULL, NULL);
    if (!window) {
        std::cerr << "无法创建GLFW窗口" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwSetWindowPos(window, 100, 100);
    glfwMakeContextCurrent(window);

    // 设置回调
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // 初始化GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "无法初始化GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    // OpenGL设置
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_MULTISAMPLE);

    // 加载优化的着色器
    std::unique_ptr<Shader> shaderProgram = std::make_unique<Shader>(
        "shader/volume_vert_optimized.vert", 
        "shader/volume_frag_optimized.frag"
    );
    if (!shaderProgram) {
        std::cerr << "无法加载着色器程序" << std::endl;
        glfwTerminate();
        return -1;
    }

    // 创建3D纹理
    GLuint volumeTexture = create3DTextureFromVolume(volumeData);

    // --- 上传分割掩码 GPU 纹理：TRT 推理结果 OR Python .raw 文件 ---
    if (!g_segResult.mask.empty()) {
        // 来源：TRT 推理（g_segResult 已在 OpenGL 初始化前写好）
        g_segMaskTexture = createOrUpdateSegMaskTexture(g_segResult, g_segMaskTexture);
        renderState.enableSegmentation = true;
        renderState.segColor           = glm::vec3(1.0f, 0.15f, 0.1f); // 血管红色
        renderState.segOpacity         = 0.85f;
        renderState.segBorderWidth     = 0.002f;
        std::cout << "[TRT] 血管掩码已上传 GPU，F1 切换显示\n";
    } else if (!dlRawPath.empty()) {
        // 来源：Python infer_vessel_seg.py / infer_vessel_trt.py 生成的 .raw
        std::cout << "[DLMask] 正在加载深度学习分割掩码...\n";
        RegionGrowingResult dlMask = loadDLVesselMask(dlRawPath, dlMetaPath);
        if (!dlMask.mask.empty()) {
            g_segResult      = std::move(dlMask);
            g_segMaskTexture = createOrUpdateSegMaskTexture(g_segResult, g_segMaskTexture);
            renderState.enableSegmentation = true;
            renderState.segColor           = glm::vec3(1.0f, 0.15f, 0.1f);
            renderState.segOpacity         = 0.85f;
            renderState.segBorderWidth     = 0.002f;
            std::cout << "[DLMask] 血管分割掩码已就绪，F1 切换显示\n";
        }
    }

    // 创建传递函数
    TransferFunction transferFunction;
    transferFunction.update(renderState.windowLevel, renderState.windowWidth);

    // 创建立方体几何
    CubeGeometry cube = createCubeGeometry();

    // 初始化投影矩阵
    glm::mat4 projection = glm::perspective(
        glm::radians(45.0f),
        (float)width / (float)height,
        0.1f,
        100.0f
    );
    glm::mat4 model = glm::mat4(1.0f);

    // 打印初始状态
    printRenderState();
    
    // 初始化MPR渲染器
    if (!mprRenderer.init()) {
        std::cerr << "警告: MPR渲染器初始化失败（MPR功能不可用）" << std::endl;
    }
    
    std::cout << "\n控制说明：\n";
    std::cout << "1/2/3/4: 切换渲染模式\n";
    std::cout << "Q/A: 采样率  W/S: 密度缩放  E/D: 亮度\n";
    std::cout << "R/F: 抖动强度  T/G: AO开关  Y/H: AO强度\n";
    std::cout << "U/J: 边缘增强开关  I/K: 边缘强度\n";
    std::cout << "O/L: 裁剪平面开关\n";
    std::cout << "方向键: 窗宽窗位  P: 打印参数\n";
    std::cout << "鼠标左键拖动: 旋转  滚轮: 缩放\n";
    std::cout << "\nMPR控制：\n";
    std::cout << "M: MPR四视口开关  6: 3D叠加开关\n";
    std::cout << "Z/X: Axial移动  C/V: Sagittal移动  B/N: Coronal移动\n";
    std::cout << "8/9: MPR窗位+/-  0/-: MPR窗宽+/-  =/退格: MPR亮度+/-\n";
    std::cout << "5: MPR灰度/传递函数切换  7: 十字线开关\n";
    std::cout << "\n区域生长分割控制：\n";
    std::cout << "鼠标右键点击: 在体上设置种子点并执行区域生长\n";
    std::cout << "F1: 开/关分割叠加显示\n";
    std::cout << "F2/F3: 增大/减小容差（当前=" << (int)renderState.segTolerance << "）\n";
    std::cout << "F4: 清除分割结果\n";
    std::cout << "F5: 开/关边界高亮\n";
    std::cout << "F6: 以当前容差重新分割（需已有种子点）\n";

    // 主渲染循环
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        
        // 处理键盘输入
        processInput(window, shaderProgram.get());

        // 获取最新窗口尺寸
        glfwGetFramebufferSize(window, &width, &height);

        // ---- 处理右键点击的种子点拾取请求 ----
        if (g_pickRequest.pending && g_volumePtr) {
            g_pickRequest.pending = false;

            // 重新计算当前帧的投影和观察矩阵
            glm::mat4 curProj = glm::perspective(
                glm::radians(45.0f),
                (float)width / (float)(std::max)(height, 1),
                0.1f, 100.0f);
            glm::mat4 curView = orbitCamera.getViewMatrix();

            int sx, sy, sz;
            if (pickSeedPointFromClick(
                    g_pickRequest.clickX, g_pickRequest.clickY,
                    g_pickRequest.winW, g_pickRequest.winH,
                    curProj, curView,
                    *g_volumePtr, sx, sy, sz))
            {
                renderState.seedVoxel = glm::ivec3(sx, sy, sz);
                runRegionGrowing(*g_volumePtr, sx, sy, sz);
            }
        }
        
        // 清屏
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // 更新传递函数
        transferFunction.update(renderState.windowLevel, renderState.windowWidth);
        
        // ====== MPR模式：四视口布局 ======
        if (renderState.enableMPR) {
            // 更新投影矩阵（半视口宽高比）
            int halfW = width / 2;
            int halfH = height / 2;
            glm::mat4 mprProjection = glm::perspective(
                glm::radians(45.0f),
                (float)halfW / (float)halfH,
                0.1f,
                100.0f
            );
            
            // ---- 左上视口：3D体绘制 ----
            mprRenderer.set3DViewport(width, height);
            
            shaderProgram->use();
            glm::mat4 view = orbitCamera.getViewMatrix();
            glm::mat4 modelview = view * model;
            shaderProgram->setMat4("projection", mprProjection);
            shaderProgram->setMat4("modelview", modelview);
            updateShaderUniforms(shaderProgram.get(), halfW, halfH);
            
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_3D, volumeTexture);
            shaderProgram->setInt("volume", 0);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_1D, transferFunction.getTextureID());
            shaderProgram->setInt("transferFunc", 1);
            
            glBindVertexArray(cube.VAO);
            glDrawElements(GL_TRIANGLES, cube.indexCount, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
            
            // ---- 右上/左下/右下：MPR切面视图（使用MPR独立窗宽窗位） ----
            mprRenderer.renderAllMPRViews(
                width, height,
                volumeTexture, transferFunction.getTextureID()
            );
            
            // 恢复全视口（为下一帧做准备）
            glViewport(0, 0, width, height);
        
        } else {
            // ====== 普通模式：全视口3D体绘制 ======
            glViewport(0, 0, width, height);
            
            shaderProgram->use();
            glm::mat4 view = orbitCamera.getViewMatrix();
            glm::mat4 modelview = view * model;
            shaderProgram->setMat4("projection", projection);
            shaderProgram->setMat4("modelview", modelview);
            updateShaderUniforms(shaderProgram.get(), width, height);
            
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_3D, volumeTexture);
            shaderProgram->setInt("volume", 0);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_1D, transferFunction.getTextureID());
            shaderProgram->setInt("transferFunc", 1);
            
            glBindVertexArray(cube.VAO);
            glDrawElements(GL_TRIANGLES, cube.indexCount, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }

        glfwSwapBuffers(window);
    }

    // 清理资源
    mprRenderer.cleanup();
    glDeleteProgram(shaderProgram->ID);
    glDeleteTextures(1, &volumeTexture);
    if (g_segMaskTexture != 0) glDeleteTextures(1, &g_segMaskTexture);
    glDeleteVertexArrays(1, &cube.VAO);
    glDeleteBuffers(1, &cube.VBO);
    glDeleteBuffers(1, &cube.EBO);

    glfwTerminate();
    return 0;
}
