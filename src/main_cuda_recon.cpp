// ============================================================================
// CUDA 加速重建演示程序
//
// 对比 CPU 和 GPU 的 CT/PET 重建性能
//
// 功能:
//   1. CT FBP GPU vs CPU 对比
//   2. CT SIRT GPU 迭代
//   3. PET MLEM GPU
//   4. PET OSEM GPU
//   5. OpenGL 可视化 (CPU结果 | GPU结果 | 差值图)
//
// 按键:
//   1   CT FBP (GPU)
//   2   CT SIRT (GPU)
//   3   PET MLEM (GPU)
//   4   PET OSEM (GPU)
//   C   颜色模式
//   ESC 退出
// ============================================================================

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "ct_reconstruction.hpp"
#include "pet_reconstruction.hpp"
#include "cuda_reconstruction.cuh"

#include <iostream>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>

// ---- 全局状态 ----
static int g_phantomSize  = 256;
static int g_reconSize    = 256;
static int g_mode         = 1;  // 1=CT FBP, 2=CT SIRT, 3=PET MLEM, 4=PET OSEM
static int g_colorMode    = 0;
static float g_windowCenter = 0.5f;
static float g_windowWidth  = 1.0f;
static bool g_needRecon    = true;

static std::vector<float> g_cpuImage;   // CPU 重建结果
static std::vector<float> g_gpuImage;   // GPU 重建结果
static std::vector<float> g_diffImage;  // 差值

static GLuint g_cpuTex = 0, g_gpuTex = 0, g_diffTex = 0;
static GLuint g_shaderProg = 0, g_vao = 0, g_vbo = 0;

// ---- 着色器 ----
static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    int ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[512]; glGetShaderInfoLog(s, 512, nullptr, log);
        std::cerr << "Shader: " << log << std::endl; }
    return s;
}

static GLuint createShader() {
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
        uniform sampler2D tex;
        uniform float wc, ww;
        uniform int cm;
        vec3 hot(float t) {
            return vec3(clamp(3.0*t,0,1), clamp(3.0*t-1.0,0,1), clamp(3.0*t-2.0,0,1));
        }
        void main() {
            float v = texture(tex, TexCoord).r;
            float lo = wc - ww*0.5, hi = wc + ww*0.5;
            float m = clamp((v-lo)/(hi-lo), 0, 1);
            vec3 c;
            if (cm==0) c = vec3(m);
            else if (cm==1) c = hot(m);
            else c = vec3(1.0-m);
            FragColor = vec4(c,1);
        }
    )";
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p,v); glAttachShader(p,f);
    glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

static GLuint uploadTex(const std::vector<float>& d, int w, int h) {
    GLuint t; glGenTextures(1,&t); glBindTexture(GL_TEXTURE_2D,t);
    glTexImage2D(GL_TEXTURE_2D,0,GL_R32F,w,h,0,GL_RED,GL_FLOAT,d.data());
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    return t;
}
static void updateTex(GLuint t, const std::vector<float>& d, int w, int h) {
    glBindTexture(GL_TEXTURE_2D,t);
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,w,h,GL_RED,GL_FLOAT,d.data());
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
    glGenVertexArrays(1,&g_vao); glGenBuffers(1,&g_vbo);
    float v[] = { -1,-1,0,0, 1,-1,1,0, 1,1,1,1, -1,-1,0,0, 1,1,1,1, -1,1,0,1 };
    glBindVertexArray(g_vao); glBindBuffer(GL_ARRAY_BUFFER,g_vbo);
    glBufferData(GL_ARRAY_BUFFER,sizeof(v),v,GL_STATIC_DRAW);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,16,(void*)0); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,16,(void*)8); glEnableVertexAttribArray(1);
}

static void drawTex(GLuint tex, int x, int y, int w, int h) {
    glViewport(x,y,w,h);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,tex);
    glUseProgram(g_shaderProg);
    glUniform1i(glGetUniformLocation(g_shaderProg,"tex"),0);
    glUniform1f(glGetUniformLocation(g_shaderProg,"wc"),g_windowCenter);
    glUniform1f(glGetUniformLocation(g_shaderProg,"ww"),g_windowWidth);
    glUniform1i(glGetUniformLocation(g_shaderProg,"cm"),g_colorMode);
    glBindVertexArray(g_vao);
    glDrawArrays(GL_TRIANGLES,0,6);
}

// ---- 重建逻辑 ----
static void doRecon() {
    std::cout << "\n===== CUDA 重建对比 (模式 " << g_mode << ") =====" << std::endl;

    if (g_mode <= 2) {
        // CT
        auto phantom = CTReconstructor::generateSheppLoganPhantom(g_phantomSize);
        auto sinogram = CTReconstructor::forwardProject(phantom, g_phantomSize, 360);

        // CPU
        auto t0 = std::chrono::high_resolution_clock::now();
        ReconResult cpuResult;
        if (g_mode == 1) {
            cpuResult = CTReconstructor::reconstructFBP(sinogram, g_reconSize);
        } else {
            cpuResult = CTReconstructor::reconstructSIRT(sinogram, g_reconSize, 30);
        }
        cpuResult.normalize();
        auto t1 = std::chrono::high_resolution_clock::now();
        float cpuMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
        std::cout << "CPU: " << cpuMs << " ms" << std::endl;

        // GPU
        CudaCTParams params;
        params.numAngles = sinogram.numAngles;
        params.numDetectors = sinogram.numDetectors;
        params.outputSize = g_reconSize;
        params.angleStart = sinogram.angleStart;
        params.angleEnd = sinogram.angleEnd;

        CudaReconResult gpuResult;
        if (g_mode == 1) {
            gpuResult = CudaReconstructor::ctFBP(sinogram.data.data(), params);
        } else {
            params.iterations = 30;
            gpuResult = CudaReconstructor::ctSIRT(sinogram.data.data(), params);
        }

        // 归一化
        auto gpuImg = gpuResult.image;
        float mn = *std::min_element(gpuImg.begin(), gpuImg.end());
        float mx = *std::max_element(gpuImg.begin(), gpuImg.end());
        float rg = mx - mn; if (rg < 1e-10f) rg = 1.0f;
        for (auto& v : gpuImg) v = (v - mn) / rg;

        g_cpuImage = cpuResult.image;
        g_gpuImage = gpuImg;

        std::cout << "加速比: " << cpuMs / gpuResult.elapsedMs << "x" << std::endl;

    } else {
        // PET
        int petSize = 128;
        auto phantom = PETReconstructor::generateHotColdPhantom(petSize);
        auto sinogram = PETReconstructor::forwardProject(phantom, petSize, 180);

        // CPU
        auto t0 = std::chrono::high_resolution_clock::now();
        PETReconResult cpuResult;
        if (g_mode == 3) {
            cpuResult = PETReconstructor::reconstructMLEM(sinogram, petSize, 20);
        } else {
            cpuResult = PETReconstructor::reconstructOSEM(sinogram, petSize, 5, 12);
        }
        cpuResult.normalize();
        auto t1 = std::chrono::high_resolution_clock::now();
        float cpuMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
        std::cout << "CPU: " << cpuMs << " ms" << std::endl;

        // GPU
        CudaPETParams params;
        params.numAngles = sinogram.numAngles;
        params.numRadialBins = sinogram.numRadialBins;
        params.outputSize = petSize;

        CudaReconResult gpuResult;
        if (g_mode == 3) {
            params.iterations = 20;
            gpuResult = CudaReconstructor::petMLEM(sinogram.data.data(), params);
        } else {
            params.iterations = 5;
            params.numSubsets = 12;
            gpuResult = CudaReconstructor::petOSEM(sinogram.data.data(), params);
        }

        auto gpuImg = gpuResult.image;
        float mn = *std::min_element(gpuImg.begin(), gpuImg.end());
        float mx = *std::max_element(gpuImg.begin(), gpuImg.end());
        float rg = mx - mn; if (rg < 1e-10f) rg = 1.0f;
        for (auto& v : gpuImg) v = (v - mn) / rg;

        g_cpuImage = cpuResult.image;
        g_gpuImage = gpuImg;
        g_reconSize = petSize;

        std::cout << "加速比: " << cpuMs / gpuResult.elapsedMs << "x" << std::endl;
    }

    // 差值图
    g_diffImage.resize(g_cpuImage.size());
    for (size_t i = 0; i < g_cpuImage.size(); ++i) {
        g_diffImage[i] = std::abs(g_cpuImage[i] - g_gpuImage[i]) * 10.0f; // 放大10x
    }

    updateTex(g_cpuTex, g_cpuImage, g_reconSize, g_reconSize);
    updateTex(g_gpuTex, g_gpuImage, g_reconSize, g_reconSize);
    updateTex(g_diffTex, g_diffImage, g_reconSize, g_reconSize);
}

// ---- 按键 ----
static void keyCallback(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    switch (key) {
    case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(w, GLFW_TRUE); break;
    case GLFW_KEY_1: g_mode = 1; g_reconSize = 256; g_needRecon = true; break;
    case GLFW_KEY_2: g_mode = 2; g_reconSize = 256; g_needRecon = true; break;
    case GLFW_KEY_3: g_mode = 3; g_reconSize = 128; g_needRecon = true; break;
    case GLFW_KEY_4: g_mode = 4; g_reconSize = 128; g_needRecon = true; break;
    case GLFW_KEY_C: g_colorMode = (g_colorMode + 1) % 3; break;
    case GLFW_KEY_UP: g_windowCenter += 0.02f; break;
    case GLFW_KEY_DOWN: g_windowCenter -= 0.02f; break;
    case GLFW_KEY_EQUAL: g_windowWidth = std::min(g_windowWidth + 0.05f, 3.0f); break;
    case GLFW_KEY_MINUS: g_windowWidth = std::max(g_windowWidth - 0.05f, 0.05f); break;
    }
}

// ============================================================================
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CUDA 加速重建演示" << std::endl;
    std::cout << "========================================" << std::endl;

    // 检测 GPU
    if (!CudaReconstructor::isAvailable()) {
        std::cerr << "未检测到 CUDA 设备! 请确认安装了 NVIDIA 驱动." << std::endl;
        return -1;
    }
    auto dev = CudaReconstructor::queryDevice();
    std::cout << "GPU: " << dev.name
              << " (SM " << dev.major << "." << dev.minor
              << ", " << dev.totalMemMB << " MB, "
              << dev.smCount << " SMs)" << std::endl;

    std::cout << "\n按键:" << std::endl;
    std::cout << "  1  CT FBP (GPU)      2  CT SIRT (GPU)" << std::endl;
    std::cout << "  3  PET MLEM (GPU)    4  PET OSEM (GPU)" << std::endl;
    std::cout << "  C  颜色    +/- 窗宽  ↑↓ 窗位    ESC 退出" << std::endl;
    std::cout << "布局: [CPU 结果] | [GPU 结果] | [差值×10]" << std::endl;
    std::cout << "========================================" << std::endl;

    // OpenGL
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1200, 420,
        "CUDA Reconstruction Demo", nullptr, nullptr);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return -1;

    g_shaderProg = createShader();
    initQuad();

    std::vector<float> blank(g_reconSize * g_reconSize, 0.0f);
    g_cpuTex  = uploadTex(blank, g_reconSize, g_reconSize);
    g_gpuTex  = uploadTex(blank, g_reconSize, g_reconSize);
    g_diffTex = uploadTex(blank, g_reconSize, g_reconSize);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (g_needRecon) {
            // 纹理大小可能变化, 重新创建
            glDeleteTextures(1, &g_cpuTex);
            glDeleteTextures(1, &g_gpuTex);
            glDeleteTextures(1, &g_diffTex);
            std::vector<float> b(g_reconSize * g_reconSize, 0.0f);
            g_cpuTex  = uploadTex(b, g_reconSize, g_reconSize);
            g_gpuTex  = uploadTex(b, g_reconSize, g_reconSize);
            g_diffTex = uploadTex(b, g_reconSize, g_reconSize);

            doRecon();
            g_needRecon = false;
        }

        int fbW, fbH; glfwGetFramebufferSize(window, &fbW, &fbH);
        int pw = fbW / 3;
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        drawTex(g_cpuTex, 0, 0, pw, fbH);
        drawTex(g_gpuTex, pw, 0, pw, fbH);
        drawTex(g_diffTex, pw*2, 0, fbW - pw*2, fbH);

        glfwSwapBuffers(window);
    }

    glDeleteTextures(1, &g_cpuTex);
    glDeleteTextures(1, &g_gpuTex);
    glDeleteTextures(1, &g_diffTex);
    glDeleteProgram(g_shaderProg);
    glDeleteVertexArrays(1, &g_vao);
    glDeleteBuffers(1, &g_vbo);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
