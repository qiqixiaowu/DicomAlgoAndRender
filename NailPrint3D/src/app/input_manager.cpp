/**
 * @file    input_manager.cpp
 * @brief   输入管理 实现
 */

#include "app/input_manager.h"
#include "app/print_pipeline.h"
#include "app/ui_help.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include <algorithm>

namespace NailPrint3D {

// ============================================================
// GLFW 回调
// ============================================================

static Scene* g_scene = nullptr;

static void framebufferSizeCallback(GLFWwindow* /*window*/, int width, int height) {
    glViewport(0, 0, width, height);
    if (g_scene) g_scene->camera.setAspect((float)width / (float)height);
}

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int /*mods*/) {
    if (!g_scene) return;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        g_scene->mouseLeftDown = (action == GLFW_PRESS);
        glfwGetCursorPos(window, &g_scene->lastMouseX, &g_scene->lastMouseY);
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        g_scene->mouseRightDown = (action == GLFW_PRESS);
        glfwGetCursorPos(window, &g_scene->lastMouseX, &g_scene->lastMouseY);
    }
}

static void cursorPosCallback(GLFWwindow* /*window*/, double xpos, double ypos) {
    if (!g_scene) return;
    double dx = xpos - g_scene->lastMouseX;
    double dy = ypos - g_scene->lastMouseY;
    g_scene->lastMouseX = xpos;
    g_scene->lastMouseY = ypos;

    if (g_scene->mouseLeftDown) {
        g_scene->camera.orbit((float)dx, (float)-dy);
    }
    if (g_scene->mouseRightDown) {
        g_scene->camera.pan((float)-dx, (float)dy);
    }
}

static void scrollCallback(GLFWwindow* /*window*/, double /*xoffset*/, double yoffset) {
    if (g_scene) g_scene->camera.zoom((float)(-yoffset));
}

static void keyCallback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    if (g_scene) handleKey(window, key, action, *g_scene);
}

// ============================================================
// 键盘处理
// ============================================================

void handleKey(GLFWwindow* window, int key, int /*action*/, Scene& s) {
    switch (key) {
        // --- 渲染模式 ---
        case GLFW_KEY_1:
            s.renderMode = RM_Solid;
            s.renderer.setRenderMode(RenderMode::Solid);
            std::cout << "[模式] 实体渲染" << std::endl;
            break;
        case GLFW_KEY_2:
            s.renderMode = RM_Wireframe;
            s.renderer.setRenderMode(RenderMode::Wireframe);
            std::cout << "[模式] 线框渲染" << std::endl;
            break;
        case GLFW_KEY_3:
            s.renderMode = RM_Slice;
            s.renderer.setRenderMode(RenderMode::SlicePreview);
            s.uploadSlicePaths();
            std::cout << "[模式] 切片预览" << std::endl;
            break;
        case GLFW_KEY_4:
            s.renderMode = RM_Color;
            s.renderer.setRenderMode(RenderMode::ColorPreview);
            std::cout << "[模式] 颜色预览" << std::endl;
            break;
        case GLFW_KEY_5:
            s.renderMode = RM_Pattern;
            s.currentPattern = RenderPattern::Procedural;
            s.renderer.setRenderMode(RenderMode::PatternPreview);
            s.renderer.setPattern(s.currentPattern);
            if (s.reliefHeight > 0.0f) applyDisplacementToMesh(s, s.reliefHeight);
            std::cout << "[模式] 图案预览 — 程序化纹理" << std::endl;
            break;
        case GLFW_KEY_6:
            s.renderMode = RM_Pattern;
            s.currentPattern = RenderPattern::Photo;
            s.renderer.setRenderMode(RenderMode::PatternPreview);
            s.renderer.setPattern(s.currentPattern);
            s.renderer.setPatternTexture(s.userPhotoLoaded ? s.texUserPhoto : s.texPortrait);
            if (s.reliefHeight > 0.0f) applyDisplacementToMesh(s, s.reliefHeight);
            std::cout << "[模式] 图案预览 — 照片/头像"
                      << (s.userPhotoLoaded ? "（用户图片）" : "（内置测试）") << std::endl;
            break;
        case GLFW_KEY_7:
            s.renderMode = RM_Pattern;
            s.currentPattern = RenderPattern::Cartoon;
            s.renderer.setRenderMode(RenderMode::PatternPreview);
            s.renderer.setPattern(s.currentPattern);
            s.renderer.setPatternTexture(s.texCartoon);
            if (s.reliefHeight > 0.0f) applyDisplacementToMesh(s, s.reliefHeight);
            std::cout << "[模式] 图案预览 — 卡通风格" << std::endl;
            break;
        case GLFW_KEY_8:
            s.renderMode = RM_Pattern;
            s.currentPattern = RenderPattern::FlatColor;
            s.renderer.setRenderMode(RenderMode::PatternPreview);
            s.renderer.setPattern(s.currentPattern);
            s.renderer.setPatternTexture(s.texGeometric);
            if (s.reliefHeight > 0.0f) applyDisplacementToMesh(s, s.reliefHeight);
            std::cout << "[模式] 图案预览 — 纯色块" << std::endl;
            break;
        case GLFW_KEY_9:
            s.renderMode = RM_Pattern;
            s.currentPattern = RenderPattern::Text;
            s.renderer.setRenderMode(RenderMode::PatternPreview);
            s.renderer.setPattern(s.currentPattern);
            s.renderer.setPatternTexture(s.texText);
            if (s.reliefHeight > 0.0f) applyDisplacementToMesh(s, s.reliefHeight);
            std::cout << "[模式] 图案预览 — 文字图案" << std::endl;
            break;

        // --- 浮雕高度 ---
        case GLFW_KEY_R:
            s.reliefHeight = std::min(s.reliefHeight + 0.2f, 3.0f);
            applyDisplacementToMesh(s, s.reliefHeight);
            {
                TextureTransform t = s.renderer.getTexTransform();
                t.reliefHeight = s.reliefHeight * 0.05f;
                s.renderer.setTextureTransform(t);
            }
            std::cout << "[浮雕] 高度 = " << s.reliefHeight << "mm" << std::endl;
            break;
        case GLFW_KEY_F:
            s.reliefHeight = std::max(s.reliefHeight - 0.2f, 0.0f);
            applyDisplacementToMesh(s, s.reliefHeight);
            {
                TextureTransform t = s.renderer.getTexTransform();
                t.reliefHeight = s.reliefHeight * 0.05f;
                s.renderer.setTextureTransform(t);
            }
            std::cout << "[浮雕] 高度 = " << s.reliefHeight << "mm" << std::endl;
            break;
        case GLFW_KEY_0:
            s.reliefHeight = 0.0f;
            applyDisplacementToMesh(s, 0.0f);
            {
                TextureTransform t = s.renderer.getTexTransform();
                t.reliefHeight = 0.0f;
                s.renderer.setTextureTransform(t);
            }
            std::cout << "[浮雕] 已重置为平面" << std::endl;
            break;

        // --- UV 校正 ---
        case GLFW_KEY_U: {
            TextureTransform t = s.renderer.getTexTransform();
            t.uvCorrectMode = (t.uvCorrectMode + 1) % 3;
            if (t.uvCorrectMode == 1) {
                t.uvAspect = 0.75f;
                std::cout << "[UV校正] 模式1: 纵横比校正 (aspect=" << t.uvAspect << ")" << std::endl;
            } else if (t.uvCorrectMode == 2) {
                t.uvAspect = 0.85f;
                std::cout << "[UV校正] 模式2: 宽边校正（补偿指尖收窄）" << std::endl;
            } else {
                t.uvAspect = 1.0f;
                std::cout << "[UV校正] 模式0: 不校正" << std::endl;
            }
            s.renderer.setTextureTransform(t);
            break;
        }

        // --- 3D装饰物 F1~F8 ---
        case GLFW_KEY_F1:
            s.currentOrnament = 0; applyOrnamentToMesh(s);
            std::cout << "[3D装饰] 小蜘蛛" << std::endl; break;
        case GLFW_KEY_F2:
            s.currentOrnament = 1; applyOrnamentToMesh(s);
            std::cout << "[3D装饰] 立体花朵" << std::endl; break;
        case GLFW_KEY_F3:
            s.currentOrnament = 2; applyOrnamentToMesh(s);
            std::cout << "[3D装饰] 五角星" << std::endl; break;
        case GLFW_KEY_F4:
            s.currentOrnament = 3; applyOrnamentToMesh(s);
            std::cout << "[3D装饰] 蝴蝶" << std::endl; break;
        case GLFW_KEY_F5:
            s.currentOrnament = 4; applyOrnamentToMesh(s);
            std::cout << "[3D装饰] 爱心" << std::endl; break;
        case GLFW_KEY_F6:
            s.currentOrnament = 5; applyOrnamentToMesh(s);
            std::cout << "[3D装饰] 蝴蝶结" << std::endl; break;
        case GLFW_KEY_F7:
            s.currentOrnament = 6; applyOrnamentToMesh(s);
            std::cout << "[3D装饰] 小皇冠" << std::endl; break;
        case GLFW_KEY_F8:
            s.currentOrnament = 7; applyOrnamentToMesh(s);
            std::cout << "[3D装饰] 宝石" << std::endl; break;
        case GLFW_KEY_F9:
            s.currentOrnament = -1; applyOrnamentToMesh(s);
            std::cout << "[3D装饰] 已移除" << std::endl; break;
        case GLFW_KEY_F10:
            s.renderMode = RM_Solid;
            s.renderer.setRenderMode(RenderMode::Solid);
            std::cout << "[模式] 实体渲染（查看3D装饰物）" << std::endl;
            break;

        // --- 装饰物调整 ---
        case GLFW_KEY_LEFT_BRACKET:
            s.ornamentSize = std::max(s.ornamentSize - 2.0f, 5.0f);
            if (s.currentOrnament >= 0) applyOrnamentToMesh(s);
            std::cout << "[3D装饰] 大小 = " << s.ornamentSize << std::endl;
            break;
        case GLFW_KEY_RIGHT_BRACKET:
            s.ornamentSize = std::min(s.ornamentSize + 2.0f, 40.0f);
            if (s.currentOrnament >= 0) applyOrnamentToMesh(s);
            std::cout << "[3D装饰] 大小 = " << s.ornamentSize << std::endl;
            break;
        case GLFW_KEY_I:
            s.ornamentV = std::min(s.ornamentV + 0.1f, 0.9f);
            if (s.currentOrnament >= 0) applyOrnamentToMesh(s);
            std::cout << "[3D装饰] 位置 V=" << s.ornamentV << std::endl;
            break;
        case GLFW_KEY_K:
            s.ornamentV = std::max(s.ornamentV - 0.1f, 0.1f);
            if (s.currentOrnament >= 0) applyOrnamentToMesh(s);
            std::cout << "[3D装饰] 位置 V=" << s.ornamentV << std::endl;
            break;
        case GLFW_KEY_J:
            s.ornamentU = std::max(s.ornamentU - 0.1f, 0.1f);
            if (s.currentOrnament >= 0) applyOrnamentToMesh(s);
            std::cout << "[3D装饰] 位置 U=" << s.ornamentU << std::endl;
            break;
        case GLFW_KEY_L:
            s.ornamentU = std::min(s.ornamentU + 0.1f, 0.9f);
            if (s.currentOrnament >= 0) applyOrnamentToMesh(s);
            std::cout << "[3D装饰] 位置 U=" << s.ornamentU << std::endl;
            break;
        case GLFW_KEY_O:
            s.ornamentRotation += 0.3f;
            if (s.currentOrnament >= 0) applyOrnamentToMesh(s);
            std::cout << "[3D装饰] 旋转 = " << s.ornamentRotation << std::endl;
            break;
        case GLFW_KEY_C:
            s.currentColorScheme = (s.currentColorScheme + 1) % numColorSchemes;
            if (s.currentOrnament >= 0) applyOrnamentToMesh(s);
            std::cout << "[配色] " << colorSchemes[s.currentColorScheme].name << std::endl;
            break;

        // --- 流光溢彩 ---
        case GLFW_KEY_T:
            s.renderMode = RM_Pattern;
            s.currentPattern = RenderPattern::Iridescent;
            s.renderer.setRenderMode(RenderMode::PatternPreview);
            s.renderer.setPattern(s.currentPattern);
            if (s.userPhotoLoaded) s.renderer.setPatternTexture(s.texUserPhoto);
            std::cout << "[模式] 流光溢彩（虹彩/珠光效果）" << std::endl;
            break;

        // --- 导入图片 ---
        case GLFW_KEY_P: {
            std::cout << "\n[导入图片] 请输入图片路径（支持 PNG/JPG/BMP/TGA）：" << std::endl;
            std::cout << "  > ";
            std::string imgPath;
            std::cin >> imgPath;
            if (!imgPath.empty() && imgPath.front() == '"') imgPath.erase(0, 1);
            if (!imgPath.empty() && imgPath.back() == '"') imgPath.pop_back();
            if (s.texUserPhoto.loadFromFile(imgPath)) {
                s.userPhotoLoaded = true;
                s.renderMode = RM_Pattern;
                s.currentPattern = RenderPattern::Photo;
                s.renderer.setRenderMode(RenderMode::PatternPreview);
                s.renderer.setPattern(s.currentPattern);
                s.renderer.setPatternTexture(s.texUserPhoto);
                std::cout << "[导入图片] 成功！已切换到照片模式显示" << std::endl;
            } else {
                std::cout << "[导入图片] 失败！请检查路径是否正确" << std::endl;
            }
            break;
        }

        // --- 切片层切换 ---
        case GLFW_KEY_UP:
            if (s.renderMode == RM_Slice) {
                s.currentLayer = std::min(s.currentLayer + 1, (int)s.sliceLayers.size() - 1);
                s.renderer.setCurrentLayer(s.currentLayer);
                std::cout << "[切片] 层 " << s.currentLayer << "/" << s.sliceLayers.size() << std::endl;
            }
            break;
        case GLFW_KEY_DOWN:
            if (s.renderMode == RM_Slice) {
                s.currentLayer = std::max(s.currentLayer - 1, 0);
                s.renderer.setCurrentLayer(s.currentLayer);
                std::cout << "[切片] 层 " << s.currentLayer << "/" << s.sliceLayers.size() << std::endl;
            }
            break;

        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;
    }
}

// ============================================================
// 注册回调
// ============================================================

void registerCallbacks(GLFWwindow* window, Scene& scene) {
    g_scene = &scene;
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);
}

} // namespace NailPrint3D
