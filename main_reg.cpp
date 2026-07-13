/**
 * @file    main_ct_pet_registration.cpp
 * @brief   CT-PET 刚体配准 OpenGL 可视化 Demo
 *
 * 体数据：64×64×40 的多模态 CT/PET 头部体模（自动生成，无需外部数据）
 *
 * 窗口布局（6面板 2行×3列）：
 *   ┌────────────┬────────────┬────────────┐
 *   │  CT (Ref)  │ PET (Mov)  │ Registered │
 *   ├────────────┼────────────┼────────────┤
 *   │ Checker    │ NMI Curve  │ Info Panel │
 *   └────────────┴────────────┴────────────┘
 *
 * 键盘：
 *   [Space] 运行 CT-PET 配准
 *   [↑/↓]   切换显示切片（沿Z轴）
 *   [C]     棋盘融合/普通 切换
 *   [R]     重置
 *   [Q/ESC] 退出
 */

#include <glad/glad.h>
#include <glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <cmath>

#include "ct_pet_registration.hpp"

 // ============================================================
 //  配置
 // ============================================================
static constexpr int WIN_W = 1440;
static constexpr int WIN_H = 810;
static constexpr int VOL_W = 64;
static constexpr int VOL_H = 64;
static constexpr int VOL_D = 40;
static constexpr int COLS = 3;
static constexpr int ROWS = 2;

// 已知施加到 PET 的变换（用于验证配准结果）
static constexpr float GT_TX = 4.0f, GT_TY = 3.0f, GT_TZ = 2.0f;
static constexpr float GT_RX = 0.05f, GT_RY = 0.04f, GT_RZ = 0.06f;

// ============================================================
//  Shader（内嵌）
// ============================================================
static const char* VS = R"GLSL(
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
uniform mat4 uT;
void main(){ gl_Position=uT*vec4(aPos,0,1); vUV=aUV; }
)GLSL";

// 灰度显示 shader（支持棋盘融合 + 伪彩色）
static const char* FS_GRAY = R"GLSL(
#version 330 core
in  vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
uniform sampler2D uRef;
uniform int   uMode;     // 0=灰度, 1=棋盘, 2=伪彩色(热图)
uniform float uChk;
uniform float uMin;
uniform float uMax;
void main(){
    float raw = texture(uTex, vUV).r;
    float v = clamp((raw - uMin) / (uMax - uMin + 1e-8), 0.0, 1.0);
    if(uMode==1){
        float cx=floor(vUV.x/uChk), cy=floor(vUV.y/uChk);
        float useRef = mod(cx+cy,2.0)<1.0 ? 1.0:0.0;
        float rv = texture(uRef, vUV).r;
        rv = clamp((rv - uMin) / (uMax - uMin + 1e-8), 0.0, 1.0);
        v = mix(v, rv, useRef);
        FragColor = vec4(v, v, v, 1.0);
    } else if(uMode==2){
        // 热图伪彩色（PET 高代谢区域显示为红/黄）
        vec3 col;
        if(v < 0.25)      col = vec3(0.0, v*4.0, 0.0);
        else if(v < 0.5)  col = vec3(0.0, 1.0, (v-0.25)*4.0);
        else if(v < 0.75) col = vec3((v-0.5)*4.0, 1.0, 1.0-(v-0.5)*4.0);
        else              col = vec3(1.0, 1.0-(v-0.75)*4.0, 0.0);
        FragColor = vec4(col, 1.0);
    } else {
        FragColor = vec4(v, v, v, 1.0);
    }
}
)GLSL";

// NMI 曲线绘制 shader
static const char* FS_CURVE = R"GLSL(
#version 330 core
in  vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;  // 1D texture: NMI history
uniform int uCount;
uniform float uMinVal;
uniform float uMaxVal;
void main(){
    float x = vUV.x;
    int idx = int(x * float(uCount - 1));
    idx = clamp(idx, 0, uCount - 1);
    float val = texelFetch(uTex, ivec2(idx, 0), 0).r;
    float y = (val - uMinVal) / (uMaxVal - uMinVal + 1e-8);
    y = clamp(y, 0.0, 1.0);
    float lineW = 0.003;
    if(abs(vUV.y - y) < lineW)
        FragColor = vec4(0.2, 0.8, 1.0, 1.0);
    else if(vUV.y < y)
        FragColor = vec4(0.1, 0.3, 0.5, 0.4);
    else
        FragColor = vec4(0.05, 0.05, 0.08, 1.0);
}
)GLSL";

// 文字渲染（简单点阵）
static const char* FS_TEXT = R"GLSL(
#version 330 core
in  vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
uniform vec3 uColor;
void main(){
    float a = texture(uTex, vUV).r;
    FragColor = vec4(uColor * a, a);
}
)GLSL";

// ============================================================
//  OpenGL 工具
// ============================================================
static GLuint compileShader(GLenum t, const char* src) {
    GLuint s = glCreateShader(t);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char b[512]; glGetShaderInfoLog(s, 512, nullptr, b); std::cerr << b << "\n"; }
    return s;
}

static GLuint createProgram(const char* vs, const char* fs) {
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

static GLuint makeTextureR32F(int w, int h, const float* data = nullptr) {
    GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return t;
}

static GLuint makeTextureR32F_1D(int w, const float* data = nullptr) {
    GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, 1, 0, GL_RED, GL_FLOAT, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    return t;
}

static void updateTextureR32F(GLuint t, int w, int h, const float* data) {
    glBindTexture(GL_TEXTURE_2D, t);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_FLOAT, data);
}

static GLuint g_quadVAO, g_quadVBO;
static void createQuad() {
    float verts[] = {
        -1,-1, 0,1,  1,-1, 1,1,  -1,1, 0,0,
         1,-1, 1,1,   1,1, 1,0,  -1,1, 0,0
    };
    glGenVertexArrays(1, &g_quadVAO);
    glGenBuffers(1, &g_quadVBO);
    glBindVertexArray(g_quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, g_quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, (void*)8);
}

// ============================================================
//  面板
// ============================================================
struct Panel {
    int col, row;
    std::string title;
    GLuint tex = 0, texRef = 0;
    bool ready = false;
    int displayMode = 0;  // 0=灰度, 1=棋盘, 2=伪彩色
    float texMin = 0, texMax = 1;
    CTPETReg::VolumeData vol;
};

// ============================================================
//  全局状态
// ============================================================
static CTPETReg::VolumeData g_ct, g_pet, g_petMoved, g_petRegistered;
static CTPETReg::RegistrationResult g_regResult;
static std::vector<Panel> g_panels(COLS* ROWS);
static GLuint g_progGray = 0, g_progCurve = 0;
static GLuint g_nmiCurveTex = 0;
static bool g_checkerMode = false;
static int g_currentSlice = VOL_D / 2;
static std::atomic<bool> g_busy{ false };
static std::atomic<bool> g_regDone{ false };
static GLFWwindow* g_win = nullptr;

static double nowMs() {
    using namespace std::chrono;
    return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count() / 1000.0;
}

// ============================================================
//  初始化数据
// ============================================================
static void initData() {
    // 生成 CT 和 PET 体模
    g_ct = CTPETReg::createCTPhantom(VOL_W, VOL_H, VOL_D);
    g_pet = CTPETReg::createPETPhantom(VOL_W, VOL_H, VOL_D);

    // 施加已知刚体变换到 PET（模拟未对齐）
    CTPETReg::VersorTransform gt =
        CTPETReg::VersorTransform::fromEuler(GT_RX, GT_RY, GT_RZ, GT_TX, GT_TY, GT_TZ);
    g_petMoved = CTPETReg::resample(g_pet, g_ct, gt);

    g_regDone = false;
    g_currentSlice = VOL_D / 2;

    // 创建面板
    auto makePanel = [&](int col, int row, const std::string& title,
        const CTPETReg::VolumeData& vol, bool ready, int mode) -> Panel {
            Panel p;
            p.col = col; p.row = row; p.title = title; p.ready = ready;
            p.displayMode = mode; p.vol = vol;
            auto slice = vol.getAxialSlice(g_currentSlice);
            p.tex = makeTextureR32F(vol.width, vol.height, slice.data());
            auto refSlice = g_ct.getAxialSlice(g_currentSlice);
            p.texRef = makeTextureR32F(g_ct.width, g_ct.height, refSlice.data());

            // 计算显示范围
            float mn = *std::min_element(vol.data.begin(), vol.data.end());
            float mx = *std::max_element(vol.data.begin(), vol.data.end());
            p.texMin = mn; p.texMax = mx;
            return p;
        };

    CTPETReg::VolumeData blank; blank.resize(VOL_W, VOL_H, VOL_D);
    std::fill(blank.data.begin(), blank.data.end(), 0.0f);

    g_panels[0] = makePanel(0, 0, "CT (Reference)", g_ct, true, 0);
    g_panels[1] = makePanel(1, 0, "PET (Moving)", g_petMoved, true, 2);
    g_panels[2] = makePanel(2, 0, "Registered PET", blank, false, 2);
    g_panels[3] = makePanel(0, 1, "Checker Overlay", blank, false, 1);
    g_panels[4] = makePanel(1, 1, "NMI Convergence", blank, false, 0);
    g_panels[5] = makePanel(2, 1, "Info & Results", blank, false, 0);

    // NMI 曲线纹理
    g_nmiCurveTex = makeTextureR32F_1D(1024, nullptr);
}

static void refreshPanelSlice(Panel& pan, int z) {
    if (pan.vol.empty()) return;
    auto slice = pan.vol.getAxialSlice(z);
    updateTextureR32F(pan.tex, pan.vol.width, pan.vol.height, slice.data());
}

static void refreshAllSlices(int z) {
    for (auto& p : g_panels) refreshPanelSlice(p, z);
}

// ============================================================
//  运行配准
// ============================================================
static void runRegistration() {
    std::cout << "\n========================================\n";
    std::cout << "  CT-PET Rigid Registration\n";
    std::cout << "  Volume: " << VOL_W << "x" << VOL_H << "x" << VOL_D << "\n";
    std::cout << "  Ground truth: tx=" << GT_TX << " ty=" << GT_TY << " tz=" << GT_TZ
        << " rx=" << GT_RX << "r ry=" << GT_RY << "r rz=" << GT_RZ << "r\n";
    std::cout << "========================================\n\n";

    CTPETReg::RegConfig cfg;
    cfg.numBins = 32;
    cfg.numSamples = 2000;
    cfg.maxIterations = 150;
    cfg.maxStep = 3.0f;
    cfg.minStep = 0.01f;
    cfg.relaxFactor = 0.9f;
    cfg.magnitudeTol = 0.001f;
    cfg.transScale = 0.01f;
    cfg.rotScale = 1000.0f;
    cfg.numLevels = 2;
    cfg.useCenterAlign = false;  // 已知有偏移，不使用中心对齐

    g_regResult = CTPETReg::registerCTPET(g_ct, g_petMoved, cfg);
    g_petRegistered = g_regResult.registeredPET;
    g_regDone = true;

    // 更新面板
    g_panels[2].vol = g_petRegistered;
    g_panels[2].ready = true;
    g_panels[2].displayMode = 2;
    float mn = *std::min_element(g_petRegistered.data.begin(), g_petRegistered.data.end());
    float mx = *std::max_element(g_petRegistered.data.begin(), g_petRegistered.data.end());
    g_panels[2].texMin = mn; g_panels[2].texMax = mx;

    // 棋盘面板
    g_panels[3].vol = g_petRegistered;
    g_panels[3].ready = true;
    g_panels[3].displayMode = 1;
    g_panels[3].texMin = mn; g_panels[3].texMax = mx;

    refreshAllSlices(g_currentSlice);

    // 更新 NMI 曲线纹理
    if (!g_regResult.nmiHistory.empty()) {
        int curveLen = std::min(1024, (int)g_regResult.nmiHistory.size());
        std::vector<float> curveData(1024, 0.0f);
        for (int i = 0; i < curveLen; i++)
            curveData[i] = g_regResult.nmiHistory[i];
        glBindTexture(GL_TEXTURE_2D, g_nmiCurveTex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1024, 1, GL_RED, GL_FLOAT, curveData.data());
    }

    // 打印结果
    float rx, ry, rz;
    g_regResult.transform.toEuler(rx, ry, rz);

    std::cout << "\n--- Registration Results ---\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Estimated: tx=" << g_regResult.transform.tx
        << " ty=" << g_regResult.transform.ty
        << " tz=" << g_regResult.transform.tz << "\n";
    std::cout << "  Estimated: rx=" << rx << "r ry=" << ry << "r rz=" << rz << "r\n";
    std::cout << "  Ground truth: tx=" << -GT_TX << " ty=" << -GT_TY << " tz=" << -GT_TZ << "\n";
    std::cout << "  Ground truth: rx=" << -GT_RX << "r ry=" << -GT_RY << "r rz=" << -GT_RZ << "r\n";
    std::cout << "  Initial NMI: " << g_regResult.initialNMI << "\n";
    std::cout << "  Final NMI:   " << g_regResult.finalNMI << "\n";
    std::cout << "  Iterations:  " << g_regResult.nmiHistory.size() << "\n";
    std::cout << "  Time:        " << (int)g_regResult.timeMs << " ms\n";
    std::cout << "  CT bed voxels removed: " << g_regResult.ctPre.bedVoxelsRemoved << "\n";
    std::cout << "  PET clip range: [" << g_regResult.petPre.clipLow
        << ", " << g_regResult.petPre.clipUp << "]\n";
    std::cout << "  PET background ratio: " << g_regResult.petPre.backRatio << "\n";
    std::cout << "------------------------------\n\n";
}

// ============================================================
//  渲染
// ============================================================
static void drawGrayPanel(const Panel& pan, int winW, int winH) {
    int cW = winW / COLS, cH = winH / ROWS;
    int x = pan.col * cW, y = (ROWS - 1 - pan.row) * cH;
    glViewport(x, y, cW, cH);

    glUseProgram(g_progGray);
    glm::mat4 I(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(g_progGray, "uT"), 1, GL_FALSE, glm::value_ptr(I));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, pan.tex);
    glUniform1i(glGetUniformLocation(g_progGray, "uTex"), 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, pan.texRef);
    glUniform1i(glGetUniformLocation(g_progGray, "uRef"), 1);

    int mode = pan.displayMode;
    if (g_checkerMode && pan.col + pan.row > 0 && pan.ready) mode = 1;
    glUniform1i(glGetUniformLocation(g_progGray, "uMode"), mode);
    glUniform1f(glGetUniformLocation(g_progGray, "uChk"), 0.08f);
    glUniform1f(glGetUniformLocation(g_progGray, "uMin"), pan.texMin);
    glUniform1f(glGetUniformLocation(g_progGray, "uMax"), pan.texMax);

    glBindVertexArray(g_quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

static void drawCurvePanel(int winW, int winH) {
    int cW = winW / COLS, cH = winH / ROWS;
    int x = 1 * cW, y = 0;  // row=1, col=1
    glViewport(x, y, cW, cH);

    glUseProgram(g_progCurve);
    glm::mat4 I(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(g_progCurve, "uT"), 1, GL_FALSE, glm::value_ptr(I));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_nmiCurveTex);
    glUniform1i(glGetUniformLocation(g_progCurve, "uTex"), 0);

    int count = g_regDone ? std::min(1024, (int)g_regResult.nmiHistory.size()) : 2;
    glUniform1i(glGetUniformLocation(g_progCurve, "uCount"), count);

    float minVal = 0, maxVal = 1;
    if (g_regDone && !g_regResult.nmiHistory.empty()) {
        minVal = *std::min_element(g_regResult.nmiHistory.begin(), g_regResult.nmiHistory.end());
        maxVal = *std::max_element(g_regResult.nmiHistory.begin(), g_regResult.nmiHistory.end());
        if (maxVal - minVal < 1e-6f) { minVal -= 0.1f; maxVal += 0.1f; }
    }
    glUniform1f(glGetUniformLocation(g_progCurve, "uMinVal"), minVal);
    glUniform1f(glGetUniformLocation(g_progCurve, "uMaxVal"), maxVal);

    glBindVertexArray(g_quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

static void drawInfoPanel(int winW, int winH) {
    // Info 面板用纯色背景 + 文字（这里用简单方式：渲染一个灰色面板）
    int cW = winW / COLS, cH = winH / ROWS;
    int x = 2 * cW, y = 0;  // row=1, col=2
    glViewport(x, y, cW, cH);

    glUseProgram(g_progGray);
    glm::mat4 I(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(g_progGray, "uT"), 1, GL_FALSE, glm::value_ptr(I));

    // 用 CT 切片作为背景
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_panels[0].tex);
    glUniform1i(glGetUniformLocation(g_progGray, "uTex"), 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, g_panels[0].texRef);
    glUniform1i(glGetUniformLocation(g_progGray, "uRef"), 1);
    glUniform1i(glGetUniformLocation(g_progGray, "uMode"), 0);
    glUniform1f(glGetUniformLocation(g_progGray, "uMin"), g_panels[0].texMin);
    glUniform1f(glGetUniformLocation(g_progGray, "uMax"), g_panels[0].texMax);

    glBindVertexArray(g_quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

// ============================================================
//  键盘回调
// ============================================================
static void onKey(GLFWwindow* win, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q) {
        glfwSetWindowShouldClose(win, 1);
        return;
    }
    if (key == GLFW_KEY_C) {
        g_checkerMode = !g_checkerMode;
        std::cout << "[Checker] " << (g_checkerMode ? "ON" : "OFF") << "\n";
        return;
    }
    if (key == GLFW_KEY_R) {
        initData();
        std::cout << "[Reset] Data reinitialized.\n";
        return;
    }
    if (key == GLFW_KEY_UP) {
        g_currentSlice = std::min(g_currentSlice + 1, VOL_D - 1);
        refreshAllSlices(g_currentSlice);
        std::cout << "[Slice] Z=" << g_currentSlice << "/" << VOL_D - 1 << "\n";
        return;
    }
    if (key == GLFW_KEY_DOWN) {
        g_currentSlice = std::max(g_currentSlice - 1, 0);
        refreshAllSlices(g_currentSlice);
        std::cout << "[Slice] Z=" << g_currentSlice << "/" << VOL_D - 1 << "\n";
        return;
    }
    if (key == GLFW_KEY_SPACE) {
        if (g_busy) {
            std::cout << "[Busy] Registration in progress...\n";
            return;
        }
        g_busy = true;
        runRegistration();
        g_busy = false;
        return;
    }
}

// ============================================================
//  Main
// ============================================================
int main() {
    if (!glfwInit()) {
        std::cerr << "GLFW init fail\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    std::ostringstream title;
    title << "CT-PET Rigid Registration Demo  [" << VOL_W << "x" << VOL_H << "x" << VOL_D
        << "]  —  [Space] Run | [↑/↓] Slice | [C] Checker";

    g_win = glfwCreateWindow(WIN_W, WIN_H, title.str().c_str(), nullptr, nullptr);
    if (!g_win) {
        std::cerr << "Window creation fail\n";
        return -1;
    }
    glfwMakeContextCurrent(g_win);
    glfwSwapInterval(1);
    glfwSetKeyCallback(g_win, onKey);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "GLAD init fail\n";
        return -1;
    }
    std::cout << "[GL] " << glGetString(GL_RENDERER) << " — " << glGetString(GL_VERSION) << "\n";

    g_progGray = createProgram(VS, FS_GRAY);
    g_progCurve = createProgram(VS, FS_CURVE);
    createQuad();
    initData();

    std::cout <<
        "\n==========================================================\n"
        "  CT-PET Rigid Registration Demo\n"
        "  Volume: " << VOL_W << "x" << VOL_H << "x" << VOL_D << " voxels\n"
        "  Modality: CT (HU) + PET (SUV) — Multi-modal NMI\n"
        "  Ground truth transform:\n"
        "    tx=" << GT_TX << " ty=" << GT_TY << " tz=" << GT_TZ << "\n"
        "    rx=" << GT_RX << "r ry=" << GT_RY << "r rz=" << GT_RZ << "r\n"
        "----------------------------------------------------------\n"
        "  [Space] Run CT-PET registration\n"
        "  [↑/↓]   Navigate slices\n"
        "  [C]     Toggle checker overlay\n"
        "  [R]     Reset    [Q/ESC] Quit\n"
        "==========================================================\n\n";

    std::cout << "Press [Space] to start registration...\n";

    glClearColor(0.05f, 0.05f, 0.08f, 1.0f);

    while (!glfwWindowShouldClose(g_win)) {
        glfwPollEvents();
        int ww, wh;
        glfwGetFramebufferSize(g_win, &ww, &wh);
        glClear(GL_COLOR_BUFFER_BIT);

        // 绘制 6 个面板
        drawGrayPanel(g_panels[0], ww, wh);  // CT
        drawGrayPanel(g_panels[1], ww, wh);  // PET Moving
        drawGrayPanel(g_panels[2], ww, wh);  // Registered
        drawGrayPanel(g_panels[3], ww, wh);  // Checker
        drawCurvePanel(ww, wh);               // NMI Curve
        drawInfoPanel(ww, wh);                // Info

        glfwSwapBuffers(g_win);
    }

    // 清理
    glDeleteProgram(g_progGray);
    glDeleteProgram(g_progCurve);
    glDeleteVertexArrays(1, &g_quadVAO);
    glDeleteBuffers(1, &g_quadVBO);
    for (auto& p : g_panels) {
        if (p.tex)    glDeleteTextures(1, &p.tex);
        if (p.texRef) glDeleteTextures(1, &p.texRef);
    }
    if (g_nmiCurveTex) glDeleteTextures(1, &g_nmiCurveTex);
    glfwDestroyWindow(g_win);
    glfwTerminate();
    return 0;
}
