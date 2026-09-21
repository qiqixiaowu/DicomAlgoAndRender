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
int renderMode = 0;  // 0=Solid, 1=Wireframe, 2=Slice, 3=Color, 4=Pattern
bool mouseLeftDown = false;
bool mouseRightDown = false;
double lastMouseX = 0, lastMouseY = 0;

// 多图案纹理
GLTexture texCartoon;    // 卡通图案
GLTexture texPortrait;   // 头像照片
GLTexture texGeometric;  // 几何图案
GLTexture texText;       // 文字图案
GLTexture texUserPhoto;  // 用户导入的图片
bool userPhotoLoaded = false;  // 是否已导入图片
RenderPattern currentPattern = RenderPattern::Procedural;

// 3D浮雕控制
float reliefHeight = 0.0f;       // 浮雕高度（mm），0=平面
bool  displacementApplied = false; // 是否已对网格做位移映射
Mesh  originalMesh;              // 保存原始网格（位移前），用于重新位移

// 3D立体装饰物控制
int  currentOrnament = -1;       // -1=无装饰物, 0=蜘蛛, 1=花, 2=星, 3=蝴蝶, 4=爱心, 5=蝴蝶结, 6=皇冠, 7=宝石
float ornamentSize = 15.0f;      // 装饰物大小
float ornamentU = 0.5f;          // 装饰物U位置 (0~1)
float ornamentV = 0.5f;          // 装饰物V位置 (0~1)
float ornamentRotation = 0.0f;   // 装饰物旋转
Mesh  nailBaseMesh;              // 纯甲片网格（无装饰物），用于重新放置装饰物

// ============================================================
// 回调函数
// ============================================================

/// 将当前纹理的像素数据应用到网格做3D位移映射
void applyDisplacementToMesh(float height) {
    if (height <= 0.001f) {
        // 恢复原始网格
        currentMesh = originalMesh;
        displacementApplied = false;
        glMesh.upload(currentMesh);
        std::cout << "[浮雕] 已恢复平面网格" << std::endl;
        return;
    }

    // 从当前纹理获取像素数据
    // 使用stb_image重新读取已生成的测试纹理数据
    // 这里我们直接用generateTestPattern的内部逻辑重新生成像素数据
    // 更好的方式是从GPU读回纹理，但为简化，我们用stb_image加载文件或重新生成

    // 恢复原始网格再做位移
    currentMesh = originalMesh;

    // 根据当前图案类型生成对应的像素数据
    int w = 512, h = 512;
    std::vector<unsigned char> pixels(w * h * 4);

    // 重新生成当前图案的像素数据
    // 简化方案：直接用程序化方式生成亮度图
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float u = (float)x / w;
            float v = (float)y / h;
            int idx = (y * w + x) * 4;

            // 根据当前图案类型生成亮度
            float lum = 0.5f;
            unsigned char r = 128, g = 128, b = 128;

            if (currentPattern == RenderPattern::Cartoon) {
                // 卡通花朵：花瓣亮，背景暗
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
            } else if (currentPattern == RenderPattern::Photo) {
                // 头像：脸部亮，背景暗
                float dx = (u - 0.5f) * 1.2f, dy = (v - 0.45f) * 1.5f;
                float faceDist = std::sqrt(dx * dx + dy * dy);
                if (faceDist < 0.3f) {
                    lum = 0.7f + 0.1f * (1.0f - faceDist / 0.3f);
                    r = 230; g = 195; b = 170;
                } else {
                    lum = 0.15f;
                    r = 200; g = 200; b = 210;
                }
            } else if (currentPattern == RenderPattern::FlatColor) {
                // 几何菱格
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
            } else if (currentPattern == RenderPattern::Text) {
                // 文字：字母亮，背景暗
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

    // 应用位移映射
    NailMeshGenerator::applyDisplacementMap(currentMesh, pixels.data(), w, h, height);
    glMesh.upload(currentMesh);
    displacementApplied = true;

    std::cout << "[浮雕] 位移高度=" << height << "mm, 网格已更新" << std::endl;
}

/// 装饰物类型名称
const char* ornamentNames[] = {
    "小蜘蛛", "立体花朵", "五角星", "蝴蝶", "爱心", "蝴蝶结", "小皇冠", "宝石"
};

/// 装饰物颜色方案（每套8种装饰物各一个主色）
struct ColorScheme {
    const char* name;
    ColorRGBf colors[8];
};
static ColorScheme colorSchemes[] = {
    { "鲜艳",
      { ColorRGBf(0.15f, 0.08f, 0.08f),  // 蜘蛛：黑色
        ColorRGBf(0.9f, 0.3f, 0.5f),   // 花朵：玫粉
        ColorRGBf(1.0f, 0.8f, 0.2f),   // 星星：金色
        ColorRGBf(0.6f, 0.3f, 0.8f),   // 蝴蝶：紫色
        ColorRGBf(0.9f, 0.2f, 0.3f),   // 爱心：红色
        ColorRGBf(0.8f, 0.2f, 0.4f),   // 蝴蝶结：玫红
        ColorRGBf(1.0f, 0.85f, 0.3f),  // 皇冠：金色
        ColorRGBf(0.3f, 0.6f, 0.9f) } }, // 宝石：蓝色
    { "马卡龙",
      { ColorRGBf(0.3f, 0.3f, 0.35f),   // 蜘蛛：深灰
        ColorRGBf(0.95f, 0.6f, 0.7f),  // 花朵：粉色
        ColorRGBf(0.9f, 0.75f, 0.3f),  // 星星：香槟金
        ColorRGBf(0.6f, 0.8f, 0.9f),   // 蝴蝶：天蓝
        ColorRGBf(0.9f, 0.5f, 0.6f),   // 爱心：粉红
        ColorRGBf(0.8f, 0.6f, 0.9f),   // 蝴蝶结：淡紫
        ColorRGBf(0.95f, 0.8f, 0.4f),  // 皇冠：浅金
        ColorRGBf(0.5f, 0.8f, 0.7f) } }, // 宝石：薄荷绿
    { "宝石",
      { ColorRGBf(0.1f, 0.1f, 0.15f),   // 蜘蛛：黑
        ColorRGBf(0.8f, 0.1f, 0.2f),   // 花朵：宝石红
        ColorRGBf(0.9f, 0.7f, 0.1f),   // 星星：宝石黄
        ColorRGBf(0.1f, 0.4f, 0.8f),   // 蝴蝶：蓝宝石
        ColorRGBf(0.9f, 0.15f, 0.3f),  // 爱心：红宝石
        ColorRGBf(0.2f, 0.6f, 0.4f),   // 蝴蝶结：祖母绿
        ColorRGBf(0.85f, 0.7f, 0.2f),  // 皇冠：金色
        ColorRGBf(0.3f, 0.5f, 0.9f) } }, // 宝石：蓝宝石
    { "糖果",
      { ColorRGBf(0.25f, 0.15f, 0.1f),  // 蜘蛛：棕黑
        ColorRGBf(1.0f, 0.4f, 0.6f),   // 花朵：糖果粉
        ColorRGBf(1.0f, 0.9f, 0.3f),   // 星星：柠檬黄
        ColorRGBf(0.5f, 0.2f, 0.9f),   // 蝴蝶：葡萄紫
        ColorRGBf(1.0f, 0.3f, 0.4f),   // 爱心：草莓红
        ColorRGBf(0.9f, 0.3f, 0.5f),   // 蝴蝶结：覆盆子
        ColorRGBf(1.0f, 0.8f, 0.2f),   // 皇冠：金色
        ColorRGBf(0.2f, 0.7f, 0.9f) } }, // 宝石：海蓝
};
static int currentColorScheme = 0;
static const int numColorSchemes = sizeof(colorSchemes) / sizeof(colorSchemes[0]);

/// 在甲片上放置/移除3D装饰物
void applyOrnamentToMesh() {
    if (currentOrnament < 0) {
        // 移除装饰物，恢复纯甲片
        currentMesh = nailBaseMesh;
        glMesh.upload(currentMesh);
        std::cout << "[装饰物] 已移除，恢复纯甲片" << std::endl;
        return;
    }

    // 生成装饰物
    OrnamentType type = (OrnamentType)currentOrnament;
    ColorRGBf ornColor = colorSchemes[currentColorScheme].colors[currentOrnament];
    Mesh ornament = OrnamentGenerator::generate(type, ornamentSize, ornColor);

    // 放置到甲片上
    currentMesh = OrnamentGenerator::placeOnNail(
        nailBaseMesh, ornament, ornamentU, ornamentV, 1.0f, ornamentRotation);

    glMesh.upload(currentMesh);
    std::cout << "[装饰物] " << ornamentNames[currentOrnament]
              << " 已放置 (大小=" << ornamentSize
              << ", 位置=" << ornamentU << "," << ornamentV
              << ", 配色=" << colorSchemes[currentColorScheme].name << ")" << std::endl;
}

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
        case GLFW_KEY_5:
            renderMode = 4;
            currentPattern = RenderPattern::Procedural;
            renderer.setRenderMode(RenderMode::PatternPreview);
            renderer.setPattern(currentPattern);
            if (reliefHeight > 0.0f) applyDisplacementToMesh(reliefHeight);
            std::cout << "[模式] 图案预览 — 程序化纹理" << std::endl;
            break;
        case GLFW_KEY_6:
            renderMode = 4;
            currentPattern = RenderPattern::Photo;
            renderer.setRenderMode(RenderMode::PatternPreview);
            renderer.setPattern(currentPattern);
            // 优先使用用户导入的图片，否则用内置测试头像
            renderer.setPatternTexture(userPhotoLoaded ? texUserPhoto : texPortrait);
            if (reliefHeight > 0.0f) applyDisplacementToMesh(reliefHeight);
            std::cout << "[模式] 图案预览 — 照片/头像"
                      << (userPhotoLoaded ? "（用户图片）" : "（内置测试）") << std::endl;
            break;
        case GLFW_KEY_7:
            renderMode = 4;
            currentPattern = RenderPattern::Cartoon;
            renderer.setRenderMode(RenderMode::PatternPreview);
            renderer.setPattern(currentPattern);
            renderer.setPatternTexture(texCartoon);
            if (reliefHeight > 0.0f) applyDisplacementToMesh(reliefHeight);
            std::cout << "[模式] 图案预览 — 卡通风格" << std::endl;
            break;
        case GLFW_KEY_8:
            renderMode = 4;
            currentPattern = RenderPattern::FlatColor;
            renderer.setRenderMode(RenderMode::PatternPreview);
            renderer.setPattern(currentPattern);
            renderer.setPatternTexture(texGeometric);
            if (reliefHeight > 0.0f) applyDisplacementToMesh(reliefHeight);
            std::cout << "[模式] 图案预览 — 纯色块" << std::endl;
            break;
        case GLFW_KEY_9:
            renderMode = 4;
            currentPattern = RenderPattern::Text;
            renderer.setRenderMode(RenderMode::PatternPreview);
            renderer.setPattern(currentPattern);
            renderer.setPatternTexture(texText);
            if (reliefHeight > 0.0f) applyDisplacementToMesh(reliefHeight);
            std::cout << "[模式] 图案预览 — 文字图案" << std::endl;
            break;
        case GLFW_KEY_R:
            // 增加浮雕高度
            reliefHeight = std::min(reliefHeight + 0.2f, 3.0f);
            applyDisplacementToMesh(reliefHeight);
            {
                TextureTransform t = renderer.getTexTransform();
                t.reliefHeight = reliefHeight * 0.05f;  // 视差映射系数
                renderer.setTextureTransform(t);
            }
            std::cout << "[浮雕] 高度 = " << reliefHeight << "mm" << std::endl;
            break;
        case GLFW_KEY_F:
            // 减少浮雕高度
            reliefHeight = std::max(reliefHeight - 0.2f, 0.0f);
            applyDisplacementToMesh(reliefHeight);
            {
                TextureTransform t2 = renderer.getTexTransform();
                t2.reliefHeight = reliefHeight * 0.05f;
                renderer.setTextureTransform(t2);
            }
            std::cout << "[浮雕] 高度 = " << reliefHeight << "mm" << std::endl;
            break;
        case GLFW_KEY_0:
            // 重置浮雕
            reliefHeight = 0.0f;
            applyDisplacementToMesh(0.0f);
            {
                TextureTransform t3 = renderer.getTexTransform();
                t3.reliefHeight = 0.0f;
                renderer.setTextureTransform(t3);
            }
            std::cout << "[浮雕] 已重置为平面" << std::endl;
            break;
        // === 3D立体装饰物 ===
        // F1~F8 选择装饰物类型
        case GLFW_KEY_F1:
            currentOrnament = 0;
            applyOrnamentToMesh();
            std::cout << "[3D装饰] 小蜘蛛" << std::endl;
            break;
        case GLFW_KEY_F2:
            currentOrnament = 1;
            applyOrnamentToMesh();
            std::cout << "[3D装饰] 立体花朵" << std::endl;
            break;
        case GLFW_KEY_F3:
            currentOrnament = 2;
            applyOrnamentToMesh();
            std::cout << "[3D装饰] 五角星" << std::endl;
            break;
        case GLFW_KEY_F4:
            currentOrnament = 3;
            applyOrnamentToMesh();
            std::cout << "[3D装饰] 蝴蝶" << std::endl;
            break;
        case GLFW_KEY_F5:
            currentOrnament = 4;
            applyOrnamentToMesh();
            std::cout << "[3D装饰] 爱心" << std::endl;
            break;
        case GLFW_KEY_F6:
            currentOrnament = 5;
            applyOrnamentToMesh();
            std::cout << "[3D装饰] 蝴蝶结" << std::endl;
            break;
        case GLFW_KEY_F7:
            currentOrnament = 6;
            applyOrnamentToMesh();
            std::cout << "[3D装饰] 小皇冠" << std::endl;
            break;
        case GLFW_KEY_F8:
            currentOrnament = 7;
            applyOrnamentToMesh();
            std::cout << "[3D装饰] 宝石" << std::endl;
            break;
        case GLFW_KEY_F9:
            // 移除装饰物
            currentOrnament = -1;
            applyOrnamentToMesh();
            std::cout << "[3D装饰] 已移除" << std::endl;
            break;
        case GLFW_KEY_F10:
            // 切换到实体渲染模式查看3D装饰物
            renderMode = 0;
            renderer.setRenderMode(RenderMode::Solid);
            std::cout << "[模式] 实体渲染（查看3D装饰物）" << std::endl;
            break;
        // 装饰物调整
        case GLFW_KEY_LEFT_BRACKET:  // [
            ornamentSize = std::max(ornamentSize - 2.0f, 5.0f);
            if (currentOrnament >= 0) applyOrnamentToMesh();
            std::cout << "[3D装饰] 大小 = " << ornamentSize << std::endl;
            break;
        case GLFW_KEY_RIGHT_BRACKET: // ]
            ornamentSize = std::min(ornamentSize + 2.0f, 40.0f);
            if (currentOrnament >= 0) applyOrnamentToMesh();
            std::cout << "[3D装饰] 大小 = " << ornamentSize << std::endl;
            break;
        case GLFW_KEY_I:
            // 装饰物向上移动
            ornamentV = std::min(ornamentV + 0.1f, 0.9f);
            if (currentOrnament >= 0) applyOrnamentToMesh();
            std::cout << "[3D装饰] 位置 V=" << ornamentV << std::endl;
            break;
        case GLFW_KEY_K:
            // 装饰物向下移动
            ornamentV = std::max(ornamentV - 0.1f, 0.1f);
            if (currentOrnament >= 0) applyOrnamentToMesh();
            std::cout << "[3D装饰] 位置 V=" << ornamentV << std::endl;
            break;
        case GLFW_KEY_J:
            // 装饰物向左移动
            ornamentU = std::max(ornamentU - 0.1f, 0.1f);
            if (currentOrnament >= 0) applyOrnamentToMesh();
            std::cout << "[3D装饰] 位置 U=" << ornamentU << std::endl;
            break;
        case GLFW_KEY_L:
            // 装饰物向右移动
            ornamentU = std::min(ornamentU + 0.1f, 0.9f);
            if (currentOrnament >= 0) applyOrnamentToMesh();
            std::cout << "[3D装饰] 位置 U=" << ornamentU << std::endl;
            break;
        case GLFW_KEY_O:
            // 旋转装饰物
            ornamentRotation += 0.3f;
            if (currentOrnament >= 0) applyOrnamentToMesh();
            std::cout << "[3D装饰] 旋转 = " << ornamentRotation << std::endl;
            break;
        case GLFW_KEY_C:
            // 切换配色方案
            currentColorScheme = (currentColorScheme + 1) % numColorSchemes;
            if (currentOrnament >= 0) applyOrnamentToMesh();
            std::cout << "[配色] " << colorSchemes[currentColorScheme].name << std::endl;
            break;
        case GLFW_KEY_T:
            // 流光溢彩模式
            renderMode = 4;
            currentPattern = RenderPattern::Iridescent;
            renderer.setRenderMode(RenderMode::PatternPreview);
            renderer.setPattern(currentPattern);
            if (userPhotoLoaded) renderer.setPatternTexture(texUserPhoto);
            std::cout << "[模式] 流光溢彩（虹彩/珠光效果）" << std::endl;
            break;
        case GLFW_KEY_P:
            // 导入外部图片
            {
                std::cout << "\n[导入图片] 请输入图片路径（支持 PNG/JPG/BMP/TGA）：" << std::endl;
                std::cout << "  > ";
                std::string imgPath;
                std::cin >> imgPath;
                // 去除可能的引号
                if (!imgPath.empty() && imgPath.front() == '"') imgPath.erase(0, 1);
                if (!imgPath.empty() && imgPath.back() == '"') imgPath.pop_back();
                if (texUserPhoto.loadFromFile(imgPath)) {
                    userPhotoLoaded = true;
                    // 自动切换到照片模式显示导入的图片
                    renderMode = 4;
                    currentPattern = RenderPattern::Photo;
                    renderer.setRenderMode(RenderMode::PatternPreview);
                    renderer.setPattern(currentPattern);
                    renderer.setPatternTexture(texUserPhoto);
                    std::cout << "[导入图片] 成功！已切换到照片模式显示" << std::endl;
                    std::cout << "  按 T 切换流光溢彩叠加效果" << std::endl;
                    std::cout << "  按 6 切换回内置测试头像" << std::endl;
                } else {
                    std::cout << "[导入图片] 失败！请检查路径是否正确" << std::endl;
                }
            }
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
    std::cout << "  键盘 5: 图案预览 — 程序化纹理" << std::endl;
    std::cout << "  键盘 6: 图案预览 — 照片/头像" << std::endl;
    std::cout << "  键盘 7: 图案预览 — 卡通风格" << std::endl;
    std::cout << "  键盘 8: 图案预览 — 纯色块" << std::endl;
    std::cout << "  键盘 9: 图案预览 — 文字图案" << std::endl;
    std::cout << "  键盘 T: 流光溢彩（虹彩/珠光效果）" << std::endl;
    std::cout << "  键盘 P: 导入外部图片（PNG/JPG/BMP）" << std::endl;
    std::cout << "  键盘 R: 增加浮雕高度 (+0.2mm)" << std::endl;
    std::cout << "  键盘 F: 减少浮雕高度 (-0.2mm)" << std::endl;
    std::cout << "  键盘 0: 重置浮雕（平面）" << std::endl;
    std::cout << "  --- 3D立体装饰物 ---" << std::endl;
    std::cout << "  F1: 小蜘蛛    F2: 立体花朵  F3: 五角星" << std::endl;
    std::cout << "  F4: 蝴蝶      F5: 爱心      F6: 蝴蝶结" << std::endl;
    std::cout << "  F7: 小皇冠    F8: 宝石      F9: 移除装饰物" << std::endl;
    std::cout << "  F10: 实体渲染模式（查看3D装饰物）" << std::endl;
    std::cout << "  [ ]: 调整装饰物大小  I/J/K/L: 移动位置  O: 旋转" << std::endl;
    std::cout << "  C: 切换配色方案（鲜艳/马卡龙/宝石/糖果）" << std::endl;
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
    originalMesh = currentMesh;  // 保存原始网格用于位移映射
    nailBaseMesh = currentMesh;  // 保存纯甲片网格用于装饰物放置
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

    // 生成测试纹理
    std::cout << "\n>>> 生成测试纹理..." << std::endl;
    texCartoon.generateTestPattern(0);     // 卡通花朵
    texPortrait.generateTestPattern(1);    // 头像占位
    texGeometric.generateTestPattern(2);   // 几何菱格
    texText.generateTestPattern(3);        // 文字图案
    std::cout << "    4个测试纹理已生成" << std::endl;

    // 设置默认纹理变换
    TextureTransform texXform;
    texXform.scale = 1.0f;
    texXform.opacity = 0.85f;
    texXform.blendMode = 0;  // 正常混合
    renderer.setTextureTransform(texXform);

    // 渲染循环
    while (!glfwWindowShouldClose(window)) {
        renderer.render(glMesh, camera);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glMesh.destroy();
    glSliceMesh.destroy();
    texCartoon.destroy();
    texPortrait.destroy();
    texGeometric.destroy();
    texText.destroy();
    texUserPhoto.destroy();
    glfwTerminate();
    return 0;
}
