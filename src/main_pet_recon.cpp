// ============================================================================
// PET 重建演示程序
//
// 功能:
//   1. 生成测试体模 (热球体模 / Derenzo 体模)
//   2. 模拟 PET 环形探测器扫描 -> 正弦图
//   3. 使用 MLEM / OSEM / FBP 重建
//   4. OpenGL 实时三栏可视化
//
// 按键:
//   M       MLEM 重建
//   O       OSEM 重建
//   F       FBP 重建
//   1/2     切换体模 (热球 / Derenzo)
//   N       添加泊松噪声
//   R       重置
//   +/-     迭代次数 +/- 5
//   C       颜色模式
//   ↑↓      窗位
//   ESC     退出
// ============================================================================

#include <glad/glad.h>
#include <glfw3.h>

#include "pet_reconstruction.hpp"

#include <iostream>
#include <string>
#include <chrono>

// ---- 全局状态 ----
static int g_phantomSize      = 128;
static int g_numAngles        = 180;
static int g_reconSize        = 128;
static PETReconMethod g_method = PETReconMethod::OSEM;
static int g_iterations       = 10;
static int g_numSubsets       = 12;
static int g_colorMode        = 0;
static float g_windowCenter   = 0.5f;
static float g_windowWidth    = 1.0f;
static bool g_needRecon       = true;
static int g_phantomType      = 0; // 0=热球, 1=Derenzo
static int g_numSlices        = 16;
static int g_currentSlice     = 8;

static std::vector<float> g_phantom;        // 3D 体积 [numSlices * size * size]
static PETSinogram3D g_sinogram;
static PETSinogram3D g_sinogramClean;
static PETReconResult3D g_recon;

// OpenGL
static GLuint g_phantomTex  = 0;
static GLuint g_sinogramTex = 0;
static GLuint g_reconTex    = 0;
static GLuint g_shaderProg  = 0;
static GLuint g_vao = 0, g_vbo = 0;

// ---- 着色器 ----
static GLuint compileShader(GLenum type, const char* source) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &source, nullptr);
    glCompileShader(s);
    int ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[512]; glGetShaderInfoLog(s, 512, nullptr, log);
        std::cerr << "Shader error: " << log << std::endl; }
    return s;
}

static GLuint createDisplayShader() {
    const char* vs = R"(
        #version 430 core
        layout(location=0) in vec2 aPos;
        layout(location=1) in vec2 aTex;
        out vec2 TexCoord;
        void main() { gl_Position = vec4(aPos,0,1); TexCoord = aTex; }
    )";
    const char* fs = R"(
        #version 430 core
        in vec2 TexCoord; out vec4 FragColor;
        uniform sampler2D displayTex;
        uniform float windowCenter, windowWidth;
        uniform int colorMode;
        vec3 hotMetal(float t) {
            t = clamp(t,0,1);
            return vec3(clamp(3.0*t, 0, 1),
                        clamp(3.0*t - 1.0, 0, 1),
                        clamp(3.0*t - 2.0, 0, 1));
        }
        void main() {
            float v = texture(displayTex, TexCoord).r;
            float lo = windowCenter - windowWidth*0.5;
            float hi = windowCenter + windowWidth*0.5;
            float m = clamp((v - lo)/(hi - lo), 0, 1);
            vec3 c;
            if (colorMode == 0) c = vec3(m);
            else if (colorMode == 1) c = hotMetal(m);
            else c = vec3(1.0-m);
            FragColor = vec4(c, 1);
        }
    )";
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

// ---- 纹理 ----
static GLuint uploadTex(const std::vector<float>& d, int w, int h) {
    GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, d.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return t;
}
static void updateTex(GLuint t, const std::vector<float>& d, int w, int h) {
    glBindTexture(GL_TEXTURE_2D, t);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_FLOAT, d.data());
}
static std::vector<float> norm01(const std::vector<float>& d) {
    auto r = d;
    float mn = *std::min_element(r.begin(), r.end());
    float mx = *std::max_element(r.begin(), r.end());
    float rg = mx - mn; if (rg < 1e-10f) rg = 1.0f;
    for (auto& v : r) v = (v - mn) / rg;
    return r;
}

static void initQuad() {
    glGenVertexArrays(1, &g_vao); glGenBuffers(1, &g_vbo);
    float verts[] = {
        -1,-1, 0,0,  1,-1, 1,0,  1,1, 1,1,
        -1,-1, 0,0,  1,1, 1,1,  -1,1, 0,1
    };
    glBindVertexArray(g_vao); glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, (void*)8);
    glEnableVertexAttribArray(1);
}

static void drawTex(GLuint tex, int x, int y, int w, int h) {
    glViewport(x, y, w, h);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex);
    glUseProgram(g_shaderProg);
    glUniform1i(glGetUniformLocation(g_shaderProg, "displayTex"), 0);
    glUniform1f(glGetUniformLocation(g_shaderProg, "windowCenter"), g_windowCenter);
    glUniform1f(glGetUniformLocation(g_shaderProg, "windowWidth"), g_windowWidth);
    glUniform1i(glGetUniformLocation(g_shaderProg, "colorMode"), g_colorMode);
    glBindVertexArray(g_vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

// ---- 辅助: 提取 3D 体积中的某个切片 ----
static std::vector<float> getVolumeSlice(const std::vector<float>& vol, int size, int z) {
    std::vector<float> s(size * size);
    std::copy(vol.begin() + z * size * size,
              vol.begin() + (z + 1) * size * size, s.begin());
    return s;
}

// ---- 辅助: 将当前切片上传到三个显示纹理 ----
static void updateSliceTextures() {
    if (g_phantom.empty() || g_sinogram.slices.empty()) return;
    updateTex(g_phantomTex,
        norm01(getVolumeSlice(g_phantom, g_phantomSize, g_currentSlice)),
        g_phantomSize, g_phantomSize);
    const auto& sSlice = g_sinogram.slices[g_currentSlice];
    updateTex(g_sinogramTex,
        norm01(sSlice.data), sSlice.numRadialBins, sSlice.numAngles);
    if (!g_recon.volume.empty())
        updateTex(g_reconTex,
            g_recon.getSlice(g_currentSlice), g_reconSize, g_reconSize);
}

// ---- 生成体模 + 正弦图 ----
static void generateData() {
    if (g_phantomType == 0)
        g_phantom = PETReconstructor::generate3DHotColdPhantom(g_phantomSize, g_numSlices);
    else
        g_phantom = PETReconstructor::generate3DDerenzoPhantom(g_phantomSize, g_numSlices);

    auto start = std::chrono::high_resolution_clock::now();
    g_sinogram = PETReconstructor::forwardProject3D(
        g_phantom, g_phantomSize, g_numSlices, g_numAngles);
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "正弦图生成完成: " << ms << " ms  ("
              << g_numSlices << " 切片)" << std::endl;
    g_sinogramClean = g_sinogram;
}

// ---- 重建 ----
static void doRecon() {
    switch (g_method) {
    case PETReconMethod::MLEM:    std::cout << "MLEM"; break;
    case PETReconMethod::OSEM:    std::cout << "OSEM (" << g_numSubsets << " subsets)"; break;
    case PETReconMethod::FBP_PET: std::cout << "FBP"; break;
    }
    std::cout << "  迭代: " << g_iterations
              << "  切片: " << g_numSlices << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    g_recon = PETReconstructor::reconstruct3D(
        g_sinogram, g_reconSize, g_method, g_iterations, g_numSubsets);
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "3D 重建完成: " << ms << " ms" << std::endl;

    updateTex(g_reconTex, g_recon.getSlice(g_currentSlice), g_reconSize, g_reconSize);
    PETReconstructor::savePGM("pet_recon_result.pgm",
        g_recon.getSlice(g_numSlices / 2), g_reconSize, g_reconSize);
}

// ---- 按键 ----
static void keyCallback(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    switch (key) {
    case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(w, GLFW_TRUE); break;
    case GLFW_KEY_M: g_method = PETReconMethod::MLEM;    g_needRecon = true; break;
    case GLFW_KEY_O: g_method = PETReconMethod::OSEM;    g_needRecon = true; break;
    case GLFW_KEY_F: g_method = PETReconMethod::FBP_PET; g_needRecon = true; break;
    case GLFW_KEY_1: case GLFW_KEY_2:
        g_phantomType = key - GLFW_KEY_1;
        generateData();
        g_recon = {};  // 清除旧重建结果
        updateSliceTextures();
        g_needRecon = true;
        break;
    case GLFW_KEY_N:
        g_sinogram = g_sinogramClean;
        PETReconstructor::addPoissonNoise3D(g_sinogram, 100.0f);
        updateSliceTextures();
        g_needRecon = true;
        break;
    case GLFW_KEY_R:
        g_sinogram = g_sinogramClean;
        updateSliceTextures();
        g_needRecon = true;
        break;
    case GLFW_KEY_C: g_colorMode = (g_colorMode + 1) % 3; break;
    case GLFW_KEY_EQUAL: case GLFW_KEY_KP_ADD:
        g_iterations = std::min(g_iterations + 5, 200);
        g_needRecon = true;
        //std::cout << "迭代次数: " << g_iterations << std::endl;
        break;
    case GLFW_KEY_MINUS: case GLFW_KEY_KP_SUBTRACT:
        g_iterations = std::max(g_iterations - 5, 1);
        g_needRecon = true;
        //std::cout << "迭代次数: " << g_iterations << std::endl;
        break;
    case GLFW_KEY_UP:
        g_windowCenter = std::min(g_windowCenter + 0.02f, 2.0f); break;
    case GLFW_KEY_DOWN:
        g_windowCenter = std::max(g_windowCenter - 0.02f, -1.0f); break;
    case GLFW_KEY_PAGE_UP:
        g_currentSlice = std::min(g_currentSlice + 1, g_numSlices - 1);
        updateSliceTextures(); break;
    case GLFW_KEY_PAGE_DOWN:
        g_currentSlice = std::max(g_currentSlice - 1, 0);
        updateSliceTextures(); break;
    }
}

static void printHelp() {
    //std::cout << "========================================" << std::endl;
    //std::cout << "  PET 重建演示程序" << std::endl;
    //std::cout << "========================================" << std::endl;
    //std::cout << "布局: [体模] | [正弦图] | [重建结果]" << std::endl;
    //std::cout << "  M     MLEM 重建" << std::endl;
    //std::cout << "  O     OSEM 重建 (默认)" << std::endl;
    //std::cout << "  F     FBP 重建" << std::endl;
    //std::cout << "  1/2   体模: 热球 / Derenzo" << std::endl;
    //std::cout << "  N     泊松噪声" << std::endl;
    //std::cout << "  R     重置" << std::endl;
    //std::cout << "  +/-   迭代次数" << std::endl;
    //std::cout << "  C     颜色: 灰度/热金属/反色" << std::endl;
    //std::cout << "  ↑↓    窗位" << std::endl;
    //std::cout << "  ESC   退出" << std::endl;
    //std::cout << "========================================" << std::endl;
}

// ============================================================================
int main() {
    printHelp();

    // 生成体模和正弦图
    generateData();
    PETReconstructor::savePGM("pet_phantom.pgm",
        getVolumeSlice(g_phantom, g_phantomSize, g_numSlices / 2),
        g_phantomSize, g_phantomSize);
    {
        const auto& midSlice = g_sinogram.slices[g_numSlices / 2];
        PETReconstructor::savePGM("pet_sinogram.pgm", midSlice.data,
            midSlice.numRadialBins, midSlice.numAngles);
    }

    // OpenGL 初始化
    if (!glfwInit()) { std::cerr << "GLFW 失败" << std::endl; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1200, 420,
        "PET 3D Reconstruction Demo", nullptr, nullptr);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return -1;
    std::cout << "OpenGL: " << glGetString(GL_VERSION) << std::endl;

    g_shaderProg = createDisplayShader();
    initQuad();

    g_phantomTex  = uploadTex(
        norm01(getVolumeSlice(g_phantom, g_phantomSize, g_currentSlice)),
        g_phantomSize, g_phantomSize);
    {
        const auto& sSlice = g_sinogram.slices[g_currentSlice];
        g_sinogramTex = uploadTex(norm01(sSlice.data),
            sSlice.numRadialBins, sSlice.numAngles);
    }
    std::vector<float> blank(g_reconSize * g_reconSize, 0.0f);
    g_reconTex = uploadTex(blank, g_reconSize, g_reconSize);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (g_needRecon) { doRecon(); g_needRecon = false; }

        int fbW, fbH; glfwGetFramebufferSize(window, &fbW, &fbH);
        int pw = fbW / 3;
        {
            std::string title = "PET 3D Recon  ["
                + std::to_string(g_currentSlice + 1) + "/" + std::to_string(g_numSlices)
                + "]  " + (g_method == PETReconMethod::MLEM ? "MLEM" :
                           g_method == PETReconMethod::OSEM ? "OSEM" : "FBP");
            glfwSetWindowTitle(window, title.c_str());
        }
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        drawTex(g_phantomTex, 0, 0, pw, fbH);
        drawTex(g_sinogramTex, pw, 0, pw, fbH);
        drawTex(g_reconTex, pw * 2, 0, fbW - pw * 2, fbH);

        glfwSwapBuffers(window);
    }

    glDeleteTextures(1, &g_phantomTex);
    glDeleteTextures(1, &g_sinogramTex);
    glDeleteTextures(1, &g_reconTex);
    glDeleteProgram(g_shaderProg);
    glDeleteVertexArrays(1, &g_vao);
    glDeleteBuffers(1, &g_vbo);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
