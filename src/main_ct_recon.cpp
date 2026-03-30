// ============================================================================
// CT 重建演示程序
//
// 功能:
//   1. 生成 Shepp-Logan 体模 (经典 CT 测试图像)
//   2. 模拟平行束 CT 扫描 -> 正弦图
//   3. 使用 FBP / SIRT / ART 重建
//   4. OpenGL 实时可视化: 体模 | 正弦图 | 重建结果 三栏对比
//   5. 交互: 切换滤波器/算法, 调节窗宽窗位, 添加噪声
//
// 按键说明:
//   1-5     切换滤波器 (Ram-Lak / Shepp-Logan / Cosine / Hamming / Hann)
//   F/S/A   切换重建算法 (FBP / SIRT / ART)
//   N       添加泊松噪声并重新重建
//   R       重置 (无噪声)
//   C       切换颜色模式 (灰度/热力图/反色)
//   +/-     调节窗宽
//   上/下   调节窗位
//   ESC     退出
// ============================================================================

#include <glad/glad.h>
#include <glfw3.h>

#include "ct_reconstruction.hpp"

#include <iostream>
#include <string>
#include <chrono>

// ---- 全局状态 ----
static int   g_phantomSize    = 256;
static int   g_numAngles      = 360;
static int   g_reconSize      = 256;
static FilterType g_filter    = FilterType::RAM_LAK;
static ReconMethod g_method   = ReconMethod::FBP;
static int   g_colorMode      = 0;
static float g_windowCenter   = 0.5f;
static float g_windowWidth    = 1.0f;
static bool  g_needRecon      = true;
static bool  g_noisy          = false;

static std::vector<float> g_phantom;
static Sinogram g_sinogram;
static Sinogram g_sinogramClean;
static ReconResult g_recon;

// OpenGL 资源
static GLuint g_phantomTex   = 0;
static GLuint g_sinogramTex  = 0;
static GLuint g_reconTex     = 0;
static GLuint g_shaderProg   = 0;
static GLuint g_vao          = 0;
static GLuint g_vbo          = 0;

// ---- 着色器编译工具 ----
static GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        std::cerr << "Shader compile error: " << log << std::endl;
    }
    return shader;
}

static GLuint createDisplayShader() {
    // 内嵌着色器 (也可以从文件加载)
    const char* vertSrc = R"(
        #version 430 core
        layout(location = 0) in vec2 aPos;
        layout(location = 1) in vec2 aTexCoord;
        out vec2 TexCoord;
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
            TexCoord = aTexCoord;
        }
    )";

    const char* fragSrc = R"(
        #version 430 core
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform sampler2D displayTex;
        uniform float windowCenter;
        uniform float windowWidth;
        uniform int colorMode;

        vec3 heatmap(float t) {
            t = clamp(t, 0.0, 1.0);
            if (t < 0.25) return vec3(0.0, 4.0*t, 1.0);
            if (t < 0.5)  return vec3(0.0, 1.0, 1.0-4.0*(t-0.25));
            if (t < 0.75) return vec3(4.0*(t-0.5), 1.0, 0.0);
            return vec3(1.0, 1.0-4.0*(t-0.75), 0.0);
        }

        void main() {
            float val = texture(displayTex, TexCoord).r;
            float lower = windowCenter - windowWidth * 0.5;
            float upper = windowCenter + windowWidth * 0.5;
            float mapped = clamp((val - lower) / (upper - lower), 0.0, 1.0);
            vec3 color;
            if (colorMode == 0) color = vec3(mapped);
            else if (colorMode == 1) color = heatmap(mapped);
            else color = vec3(1.0 - mapped);
            FragColor = vec4(color, 1.0);
        }
    )";

    GLuint vs = compileShader(GL_VERTEX_SHADER, vertSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    int success;
    glGetProgramiv(prog, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(prog, 512, nullptr, log);
        std::cerr << "Shader link error: " << log << std::endl;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

// ---- 纹理上传 ----
static GLuint uploadFloatTexture(const std::vector<float>& data,
    int width, int height)
{
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0,
                 GL_RED, GL_FLOAT, data.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return tex;
}

static void updateFloatTexture(GLuint tex, const std::vector<float>& data,
    int width, int height)
{
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                    GL_RED, GL_FLOAT, data.data());
}

// ---- 归一化 (针对纹理显示) ----
static std::vector<float> normalizeData(const std::vector<float>& data) {
    std::vector<float> result = data;
    float minVal = *std::min_element(result.begin(), result.end());
    float maxVal = *std::max_element(result.begin(), result.end());
    float range = maxVal - minVal;
    if (range < 1e-10f) range = 1.0f;
    for (auto& v : result) v = (v - minVal) / range;
    return result;
}

// ---- 全屏四边形 (分三栏) ----
static void initQuadGeometry() {
    glGenVertexArrays(1, &g_vao);
    glGenBuffers(1, &g_vbo);

    // 全屏四边形 (会在绘制时通过 viewport 分栏)
    float vertices[] = {
        // pos        texcoord
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
    };

    glBindVertexArray(g_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

// ---- 绘制单个纹理到指定视口 ----
static void drawTexture(GLuint tex, int x, int y, int w, int h) {
    glViewport(x, y, w, h);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);

    glUseProgram(g_shaderProg);
    glUniform1i(glGetUniformLocation(g_shaderProg, "displayTex"), 0);
    glUniform1f(glGetUniformLocation(g_shaderProg, "windowCenter"), g_windowCenter);
    glUniform1f(glGetUniformLocation(g_shaderProg, "windowWidth"), g_windowWidth);
    glUniform1i(glGetUniformLocation(g_shaderProg, "colorMode"), g_colorMode);

    glBindVertexArray(g_vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

// ---- 执行重建 ----
static void doReconstruction() {
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "\n===== CT 重建 =====" << std::endl;
    std::cout << "算法: ";
    switch (g_method) {
    case ReconMethod::FBP:  std::cout << "FBP (滤波反投影)"; break;
    case ReconMethod::SIRT: std::cout << "SIRT (同时迭代)"; break;
    case ReconMethod::ART:  std::cout << "ART (代数重建)"; break;
    }
    std::cout << std::endl;

    if (g_method == ReconMethod::FBP) {
        std::cout << "滤波器: ";
        switch (g_filter) {
        case FilterType::RAM_LAK:     std::cout << "Ram-Lak"; break;
        case FilterType::SHEPP_LOGAN: std::cout << "Shepp-Logan"; break;
        case FilterType::COSINE:      std::cout << "Cosine"; break;
        case FilterType::HAMMING:     std::cout << "Hamming"; break;
        case FilterType::HANN:        std::cout << "Hann"; break;
        }
        std::cout << std::endl;
    }

    g_recon = CTReconstructor::reconstruct(
        g_sinogram, g_reconSize, g_method, g_filter);
    g_recon.normalize();

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "重建完成, 耗时: " << ms << " ms" << std::endl;

    // 更新纹理
    updateFloatTexture(g_reconTex, g_recon.image, g_reconSize, g_reconSize);

    // 同时保存 PGM 文件供查看
    CTReconstructor::savePGM("ct_recon_result.pgm",
        g_recon.image, g_reconSize, g_reconSize);
}

// ---- 键盘回调 ----
static void keyCallback(GLFWwindow* window, int key, int /*scancode*/,
    int action, int /*mods*/)
{
    if (action != GLFW_PRESS) return;

    switch (key) {
    case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        break;

    // 滤波器选择
    case GLFW_KEY_1: g_filter = FilterType::RAM_LAK;     g_needRecon = true; break;
    case GLFW_KEY_2: g_filter = FilterType::SHEPP_LOGAN; g_needRecon = true; break;
    case GLFW_KEY_3: g_filter = FilterType::COSINE;      g_needRecon = true; break;
    case GLFW_KEY_4: g_filter = FilterType::HAMMING;     g_needRecon = true; break;
    case GLFW_KEY_5: g_filter = FilterType::HANN;        g_needRecon = true; break;

    // 重建算法
    case GLFW_KEY_F: g_method = ReconMethod::FBP;  g_needRecon = true; break;
    case GLFW_KEY_S: g_method = ReconMethod::SIRT; g_needRecon = true; break;
    case GLFW_KEY_A: g_method = ReconMethod::ART;  g_needRecon = true; break;

    // 噪声
    case GLFW_KEY_N:
        g_sinogram = g_sinogramClean;
        CTReconstructor::addPoissonNoise(g_sinogram, 1e5f);
        g_noisy = true;
        // 更新正弦图纹理
        updateFloatTexture(g_sinogramTex,
            normalizeData(g_sinogram.data),
            g_sinogram.numDetectors, g_sinogram.numAngles);
        g_needRecon = true;
        
        break;

    // 重置
    case GLFW_KEY_R:
        g_sinogram = g_sinogramClean;
        g_noisy = false;
        updateFloatTexture(g_sinogramTex,
            normalizeData(g_sinogram.data),
            g_sinogram.numDetectors, g_sinogram.numAngles);
        g_needRecon = true;

        break;

    // 颜色模式
    case GLFW_KEY_C:
        g_colorMode = (g_colorMode + 1) % 3;
        break;

    // 窗宽
    case GLFW_KEY_EQUAL:      // +
    case GLFW_KEY_KP_ADD:
        g_windowWidth = std::min(g_windowWidth + 0.05f, 3.0f);
        break;
    case GLFW_KEY_MINUS:
    case GLFW_KEY_KP_SUBTRACT:
        g_windowWidth = std::max(g_windowWidth - 0.05f, 0.05f);
        break;

    // 窗位
    case GLFW_KEY_UP:
        g_windowCenter = std::min(g_windowCenter + 0.02f, 2.0f);
        break;
    case GLFW_KEY_DOWN:
        g_windowCenter = std::max(g_windowCenter - 0.02f, -1.0f);
        break;
    }
}

// ---- 绘制文字标题 (简单方法: 在控制台输出) ----
static void printHelp() {
    //std::cout << "========================================" << std::endl;
    //std::cout << "  CT 重建演示程序" << std::endl;
    //std::cout << "========================================" << std::endl;
    //std::cout << "窗口布局: [原始体模] | [正弦图] | [重建结果]" << std::endl;
    //std::cout << std::endl;
    //std::cout << "按键操作:" << std::endl;
    //std::cout << "  1-5   切换滤波器" << std::endl;
    //std::cout << "        1=Ram-Lak  2=Shepp-Logan  3=Cosine" << std::endl;
    //std::cout << "        4=Hamming  5=Hann" << std::endl;
    //std::cout << "  F     FBP 滤波反投影" << std::endl;
    //std::cout << "  S     SIRT 同时迭代重建" << std::endl;
    //std::cout << "  A     ART 代数重建" << std::endl;
    //std::cout << "  N     添加泊松噪声" << std::endl;
    //std::cout << "  R     重置 (清除噪声)" << std::endl;
    //std::cout << "  C     颜色模式: 灰度/热力图/反色" << std::endl;
    //std::cout << "  +/-   窗宽调节" << std::endl;
    //std::cout << "  ↑↓    窗位调节" << std::endl;
    //std::cout << "  ESC   退出" << std::endl;
    //std::cout << "========================================" << std::endl;
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    printHelp();

    // ---- 1. 生成 Shepp-Logan 体模 ----
    std::cout << "\n[1/3] 生成 Shepp-Logan 体模 (" << g_phantomSize
              << "x" << g_phantomSize << ")..." << std::endl;
    g_phantom = CTReconstructor::generateSheppLoganPhantom(g_phantomSize);
    CTReconstructor::savePGM("ct_phantom.pgm", g_phantom,
        g_phantomSize, g_phantomSize);

    // ---- 2. 正向投影 (模拟CT扫描) ----
    std::cout << "[2/3] 正向投影, " << g_numAngles << " 个角度..." << std::endl;
    {
        auto start = std::chrono::high_resolution_clock::now();
        g_sinogram = CTReconstructor::forwardProject(
            g_phantom, g_phantomSize, g_numAngles);
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "正向投影完成, 耗时: " << ms << " ms" << std::endl;
    }
    g_sinogramClean = g_sinogram; // 保存干净版本
    CTReconstructor::savePGM("ct_sinogram.pgm", g_sinogram.data,
        g_sinogram.numDetectors, g_sinogram.numAngles);

    // ---- 3. 初始化 OpenGL ----
    //std::cout << "[3/3] 初始化 OpenGL 窗口..." << std::endl;

    if (!glfwInit()) {
        //std::cerr << "GLFW 初始化失败" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    int winWidth = 1200, winHeight = 420;
    GLFWwindow* window = glfwCreateWindow(winWidth, winHeight,
        "CT Reconstruction Demo", nullptr, nullptr);
    if (!window) {
        std::cerr << "窗口创建失败" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "GLAD 加载失败" << std::endl;
        return -1;
    }

    std::cout << "OpenGL: " << glGetString(GL_VERSION) << std::endl;

    // ---- 初始化渲染资源 ----
    g_shaderProg = createDisplayShader();
    initQuadGeometry();

    // 上传体模纹理 (归一化)
    auto phantomNorm = normalizeData(g_phantom);
    g_phantomTex = uploadFloatTexture(phantomNorm, g_phantomSize, g_phantomSize);

    // 上传正弦图纹理 (归一化)
    auto sinoNorm = normalizeData(g_sinogram.data);
    g_sinogramTex = uploadFloatTexture(sinoNorm,
        g_sinogram.numDetectors, g_sinogram.numAngles);

    // 创建重建纹理 (初始黑色)
    std::vector<float> blank(g_reconSize * g_reconSize, 0.0f);
    g_reconTex = uploadFloatTexture(blank, g_reconSize, g_reconSize);

    // ---- 渲染主循环 ----
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // 需要时执行重建
        if (g_needRecon) {
            doReconstruction();
            g_needRecon = false;
        }

        // 获取窗口尺寸
        int fbW, fbH;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        int panelW = fbW / 3;

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // 左栏: 体模
        drawTexture(g_phantomTex, 0, 0, panelW, fbH);
        // 中栏: 正弦图
        drawTexture(g_sinogramTex, panelW, 0, panelW, fbH);
        // 右栏: 重建结果
        drawTexture(g_reconTex, panelW * 2, 0, fbW - panelW * 2, fbH);

        glfwSwapBuffers(window);
    }

    // ---- 清理 ----
    glDeleteTextures(1, &g_phantomTex);
    glDeleteTextures(1, &g_sinogramTex);
    glDeleteTextures(1, &g_reconTex);
    glDeleteProgram(g_shaderProg);
    glDeleteVertexArrays(1, &g_vao);
    glDeleteBuffers(1, &g_vbo);
    glfwDestroyWindow(window);
    glfwTerminate();

    std::cout << "程序结束." << std::endl;
    return 0;
}
