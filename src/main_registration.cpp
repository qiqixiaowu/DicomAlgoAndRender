/**
 * @file    main_registration.cpp
 * @brief   3D 医疗影像配准算法 OpenGL 可视化 Demo
 *
 * 体数据：从真实 DICOM 文件加载 CT 和 PET 体数据
 *   CT  : E:\Data\CT-Oncology\LIU^AI\CT
 *   PET : E:\Data\CT-Oncology\LIU^AI\PET
 * 加载后自动降采样到 VOL_W×VOL_H×VOL_D 进行配准。
 *
 * 窗口布局（6面板 2行×3列）：
 *   ┌──────────┬──────────┬──────────┐
 *   │ Reference│  Moving  │  Rigid   │
 *   ├──────────┼──────────┼──────────┤
 *   │  Demons  │MG-Demons │LCC/FFD   │
 *   └──────────┴──────────┴──────────┘
 *
 * 每个面板显示 3D 体积的当前轴位（Axial）切片。
 *
 * 键盘：
 *   [1] Rigid (MSE)         [2] Rigid (NMI)
 *   [3] Demons              [4] Multi-Grid Demons
 *   [5] LCC Demons          [6] B-Spline FFD
 *   [A] 运行所有算法并打印对比表格
 *   [↑/↓] 切换显示切片（沿Z轴）
 *   [C] 棋盘融合/普通 切换
 *   [R] 重置    [Q/ESC] 退出
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

#include "registration.hpp"
#include "dicom_utils.hpp"
#include "volume_build.hpp"
#include <functional>

// ============================================================
//  配置
// ============================================================
static constexpr int WIN_W  = 1440;
static constexpr int WIN_H  = 720;
static constexpr int VOL_W  = 96;   // 配准用降采样目标尺寸
static constexpr int VOL_H  = 96;
static constexpr int VOL_D  = 96;
static constexpr int COLS   = 3;
static constexpr int ROWS   = 2;

// DICOM 数据路径
static const std::string CT_FOLDER  = R"(E:\Data\CT-Oncology\LIU^AI\CT)";
static const std::string PET_FOLDER = R"(E:\Data\CT-Oncology\LIU^AI\PET)";

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

static const char* FS = R"GLSL(
#version 330 core
in  vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
uniform sampler2D uRef;      // 棋盘融合时的参考纹理
uniform int   uMode;         // 0=普通灰度, 1=棋盘融合
uniform float uChk;          // 棋盘格 UV 大小
uniform vec3  uTint;
void main(){
    float v = texture(uTex, vUV).r;
    if(uMode==1){
        float cx=floor(vUV.x/uChk), cy=floor(vUV.y/uChk);
        float useRef = mod(cx+cy,2.0)<1.0 ? 1.0:0.0;
        float rv = texture(uRef, vUV).r;
        v = mix(v, rv, useRef);
    }
    FragColor = vec4(v*uTint, 1.0);
}
)GLSL";

// ============================================================
//  OpenGL 工具
// ============================================================
static GLuint compileShader(GLenum t, const char* src) {
    GLuint s = glCreateShader(t);
    glShaderSource(s,1,&src,nullptr); glCompileShader(s);
    GLint ok; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if (!ok) { char b[512]; glGetShaderInfoLog(s,512,nullptr,b); std::cerr<<b<<"\n"; }
    return s;
}
static GLuint createProgram(const char* vs, const char* fs) {
    GLuint v=compileShader(GL_VERTEX_SHADER,vs), f=compileShader(GL_FRAGMENT_SHADER,fs);
    GLuint p=glCreateProgram(); glAttachShader(p,v); glAttachShader(p,f); glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f); return p;
}

/** 将切片数据（W×H float）上传为 GL_R32F 纹理 */
static GLuint makeTexture(int w, int h, const float* data=nullptr) {
    GLuint t; glGenTextures(1,&t); glBindTexture(GL_TEXTURE_2D,t);
    glTexImage2D(GL_TEXTURE_2D,0,GL_R32F,w,h,0,GL_RED,GL_FLOAT,data);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    return t;
}
static void updateTexture(GLuint t, int w, int h, const float* data) {
    glBindTexture(GL_TEXTURE_2D,t);
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,w,h,GL_RED,GL_FLOAT,data);
}

static GLuint g_quadVAO, g_quadVBO;
static void createQuad() {
    float verts[]={ -1,-1, 0,1,  1,-1, 1,1,  -1,1, 0,0,
                     1,-1, 1,1,   1,1, 1,0,  -1,1, 0,0 };
    glGenVertexArrays(1,&g_quadVAO); glGenBuffers(1,&g_quadVBO);
    glBindVertexArray(g_quadVAO); glBindBuffer(GL_ARRAY_BUFFER,g_quadVBO);
    glBufferData(GL_ARRAY_BUFFER,sizeof(verts),verts,GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,4*4,(void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,4*4,(void*)8);
}

// ============================================================
//  面板
// ============================================================
struct Panel {
    int col, row;
    std::string title;
    GLuint tex=0, texRef=0;   // texRef 用于棋盘融合
    bool   ready=false;
    float  mse=-1, nmi=-1;
    double timeMs=0;
    std::string info;
    MedReg::Image3D vol;      // 完整 3D 体积（用于切换切片）
};

// ============================================================
//  全局状态
// ============================================================
static MedReg::Image3D g_ref, g_mov;
static std::vector<Panel> g_panels(COLS*ROWS);
static GLuint  g_prog=0;
static bool    g_checkerMode=false;
static int     g_currentSlice = VOL_D / 2;  // 当前显示的切片（轴位 Z）
static std::atomic<bool> g_busy{false};
static GLFWwindow* g_win=nullptr;

static double nowMs() {
    using namespace std::chrono;
    return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count()/1000.0;
}

// ============================================================
//  辅助：从 3D 体积更新面板纹理（显示第 z 层轴位切片）
// ============================================================
static void refreshPanelSlice(Panel& pan, int z) {
    if (!pan.ready || pan.vol.empty()) return;
    auto slice = pan.vol.getAxialSlice(z);
    updateTexture(pan.tex, pan.vol.width, pan.vol.height, slice.data());
}

static void refreshAllSlices(int z) {
    for (auto& p : g_panels) refreshPanelSlice(p, z);
}

// ============================================================
//  DICOM 加载：从文件夹读取序列 → 8-bit 体数据 → Image3D
// ============================================================

/**
 * @brief 将 VolumeBuildResult (8-bit) 转换为 MedReg::Image3D (float)
 *
 * 8-bit 像素值 (0-255) → float (0-1)，并记录物理间距
 */
static MedReg::Image3D volumeBuildToImage3D(const VolumeBuildResult& vb) {
    MedReg::Image3D img;
    img.resize(vb.width, vb.height, vb.depth);
    img.spacingX = static_cast<float>(vb.spacing[0]);
    img.spacingY = static_cast<float>(vb.spacing[1]);
    img.spacingZ = static_cast<float>(vb.spacing[2]);
    for (size_t i = 0; i < vb.buffer.size(); ++i)
        img.data[i] = static_cast<float>(vb.buffer[i]) / 255.0f;
    return img;
}

/**
 * @brief 从 DICOM 文件夹加载体数据并降采样到目标尺寸
 *
 * 流程：
 *   1. collectSeries() — 扫描文件夹，按 ImagePositionPatient 排序切片
 *   2. buildVolume_none() — 用 DCMTK DicomImage 解码每张切片，拼接为 3D 体数据
 *   3. volumeBuildToImage3D() — 转为 float 归一化
 *   4. resizeVolume() — 三线性插值降采样到 targetW×targetH×targetD
 *   5. normalize() — 归一化到 [0,1]
 *
 * @param folder       DICOM 文件夹路径
 * @param targetW/H/D  降采样目标尺寸
 * @param label        用于日志输出的标签 ("CT" / "PET")
 * @return Image3D     加载失败时返回 empty volume
 */
static MedReg::Image3D loadDICOMVolume(const std::string& folder,
                                        int targetW, int targetH, int targetD,
                                        const std::string& label) {
    std::cout << "[" << label << "] Scanning DICOM folder: " << folder << "\n";

    // Step 1: 收集并排序切片
    SeriesData series = collectSeries(folder);
    if (series.slices.empty()) {
        std::cerr << "[" << label << "] ERROR: No DICOM slices found in " << folder << "\n";
        return {};
    }
    std::cout << "[" << label << "] Found " << series.slices.size()
              << " slices (Series UID: " << series.seriesUID << ")\n";

    // Step 2: 构建 8-bit 体数据
    VolumeBuildResult vol = buildVolume_none(series);
    if (vol.buffer.empty()) {
        std::cerr << "[" << label << "] ERROR: Failed to build volume from DICOM\n";
        return {};
    }
    std::cout << "[" << label << "] Original volume: " << vol.width << "x" << vol.height
              << "x" << vol.depth << "  spacing=(" << vol.spacing[0] << ","
              << vol.spacing[1] << "," << vol.spacing[2] << ") mm\n";

    // Step 3: 转为 float Image3D
    MedReg::Image3D img = volumeBuildToImage3D(vol);

    // Step 4: 降采样到目标尺寸（如果原始尺寸与目标不同）
    if (img.width != targetW || img.height != targetH || img.depth != targetD) {
        std::cout << "[" << label << "] Resampling " << img.width << "x" << img.height
                  << "x" << img.depth << " -> " << targetW << "x" << targetH
                  << "x" << targetD << "...\n";
        img = MedReg::resizeVolume(img, targetW, targetH, targetD);
    }

    // Step 5: 归一化到 [0,1]
    img.normalize();
    std::cout << "[" << label << "] Loaded & normalized: " << img.width << "x"
              << img.height << "x" << img.depth << "\n";

    return img;
}

// ============================================================
//  初始化
// ============================================================
static bool initData() {
    // 加载 CT（作为 Fixed/Reference）
    g_ref = loadDICOMVolume(CT_FOLDER, VOL_W, VOL_H, VOL_D, "CT");
    if (g_ref.empty()) {
        std::cerr << "Failed to load CT data. Aborting.\n";
        return false;
    }

    // 加载 PET（作为 Moving）
    g_mov = loadDICOMVolume(PET_FOLDER, VOL_W, VOL_H, VOL_D, "PET");
    if (g_mov.empty()) {
        std::cerr << "Failed to load PET data. Aborting.\n";
        return false;
    }

    auto makePanel = [&](int col, int row, const std::string& title,
                          const MedReg::Image3D& vol, bool ready) -> Panel {
        Panel p; p.col=col; p.row=row; p.title=title; p.ready=ready; p.vol=vol;
        auto slice = vol.getAxialSlice(g_currentSlice);
        p.tex = makeTexture(vol.width, vol.height, slice.data());
        auto refSlice = g_ref.getAxialSlice(g_currentSlice);
        p.texRef = makeTexture(g_ref.width, g_ref.height, refSlice.data());
        return p;
    };

    // 占位 3D 体积（灰色）
    MedReg::Image3D blank; blank.resize(VOL_W,VOL_H,VOL_D);
    std::fill(blank.data.begin(), blank.data.end(), 0.12f);

    g_panels[0] = makePanel(0,0,"CT (Fixed)",           g_ref, true);
    g_panels[1] = makePanel(1,0,"PET (Before Reg)",     g_mov, true);
    g_panels[2] = makePanel(2,0,"Rigid — press [1]/[2]",blank, false);
    g_panels[3] = makePanel(0,1,"Demons — press [3]",   blank, false);
    g_panels[4] = makePanel(1,1,"MG-Demons — press [4]",blank, false);
    g_panels[5] = makePanel(2,1,"LCC/FFD — press [5/6]",blank, false);

    // 计算 PET 初始度量（配准前）
    float mse = MedReg::computeMSE3D(g_ref, g_mov);
    float nmi = MedReg::computeNMI3D(g_ref, g_mov);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4)
       << "CT vs PET (before reg)\n"
       << "MSE="<<mse<<"  NMI="<<nmi;
    g_panels[1].info=ss.str(); g_panels[1].mse=mse; g_panels[1].nmi=nmi;

    return true;
}

// ============================================================
//  配准任务
// ============================================================
static void finishTask(Panel& pan, MedReg::Image3D&& result,
                       double elapsed, const std::string& info,
                       float mse, float nmi) {
    pan.vol   = std::move(result);
    pan.mse   = mse; pan.nmi = nmi;
    pan.timeMs = elapsed;
    pan.info   = info;
    pan.ready  = true;
    refreshPanelSlice(pan, g_currentSlice);
    std::cout << "[" << pan.title << "] " << info << "\n";
}

static void runRigid(int panelIdx, bool useNMI) {
    std::cout << "[Rigid-" << (useNMI?"NMI":"MSE") << "] Running on 3D volume "
              << VOL_W<<"x"<<VOL_H<<"x"<<VOL_D<<" ...\n";
    double t0 = nowMs();

    MedReg::RigidRegConfig3D cfg;
    cfg.useNMI=useNMI; cfg.numLevels=2; cfg.maxIterations=120;
    cfg.maxStep=3.f; cfg.minStep=0.008f; cfg.relaxFactor=0.95f;

    auto p = MedReg::rigidRegistration3D(g_ref, g_mov, cfg);
    auto result = MedReg::applyRigid3D(g_mov, g_ref, p);
    double elapsed = nowMs()-t0;

    float mse=MedReg::computeMSE3D(g_ref,result);
    float nmi=MedReg::computeNMI3D(g_ref,result);
    std::ostringstream ss;
    ss<<std::fixed<<std::setprecision(3)
      <<"tx="<<p.tx<<" ty="<<p.ty<<" tz="<<p.tz<<"\n"
      <<"rx="<<p.rx<<"r ry="<<p.ry<<"r rz="<<p.rz<<"r\n"
      <<"MSE="<<mse<<"  NMI="<<nmi<<"\nTime:"<<(int)elapsed<<"ms";

    g_panels[panelIdx].title = std::string("Rigid (") + (useNMI?"NMI":"MSE") + ")";
    finishTask(g_panels[panelIdx], std::move(result), elapsed, ss.str(), mse, nmi);
}

static void runDemons(int panelIdx) {
    std::cout << "[Demons] Running 3D...\n";
    double t0=nowMs();

    MedReg::DemonsConfig3D cfg;
    cfg.alpha=1.f; cfg.sigma=2.f; cfg.step=0.4f; cfg.maxIterations=100;

    auto disp   = MedReg::demonsRegistration3D(g_ref, g_mov, cfg);
    auto result = MedReg::applyDeform3D(g_mov, disp);
    double elapsed=nowMs()-t0;

    float mse=MedReg::computeMSE3D(g_ref,result), nmi=MedReg::computeNMI3D(g_ref,result);
    std::ostringstream ss;
    ss<<std::fixed<<std::setprecision(4)
      <<"alpha="<<cfg.alpha<<" sigma="<<cfg.sigma<<" step="<<cfg.step<<"\n"
      <<"MSE="<<mse<<"  NMI="<<nmi<<"\nTime:"<<(int)elapsed<<"ms";

    g_panels[panelIdx].title = "Demons (Thirion 3D)";
    finishTask(g_panels[panelIdx], std::move(result), elapsed, ss.str(), mse, nmi);
}

static void runMultiGridDemons(int panelIdx) {
    std::cout << "[MG-Demons] Running 3D...\n";
    double t0=nowMs();

    MedReg::DemonsConfig3D cfg;
    cfg.sigma=2.f; cfg.step=0.4f; cfg.maxIterations=60;

    auto disp   = MedReg::multiGridDemons3D(g_ref, g_mov, 3, cfg);
    auto result = MedReg::applyDeform3D(g_mov, disp);
    double elapsed=nowMs()-t0;

    float mse=MedReg::computeMSE3D(g_ref,result), nmi=MedReg::computeNMI3D(g_ref,result);
    std::ostringstream ss;
    ss<<std::fixed<<std::setprecision(4)
      <<"depth=3 sigma="<<cfg.sigma<<"\n"
      <<"MSE="<<mse<<"  NMI="<<nmi<<"\nTime:"<<(int)elapsed<<"ms";

    g_panels[panelIdx].title = "Multi-Grid Demons 3D";
    finishTask(g_panels[panelIdx], std::move(result), elapsed, ss.str(), mse, nmi);
}

static void runLCCDemons(int panelIdx) {
    std::cout << "[LCC-Demons] Running 3D...\n";
    double t0=nowMs();

    MedReg::LCCConfig3D cfg;
    cfg.sigma=3.f; cfg.smoothSigma=2.f; cfg.step=0.06f; cfg.maxIterations=50; cfg.relax=true;

    auto disp   = MedReg::lccDemonsRegistration3D(g_ref, g_mov, cfg);
    auto result = MedReg::applyDeform3D(g_mov, disp);
    double elapsed=nowMs()-t0;

    float mse=MedReg::computeMSE3D(g_ref,result), nmi=MedReg::computeNMI3D(g_ref,result);
    std::ostringstream ss;
    ss<<std::fixed<<std::setprecision(4)
      <<"sigma="<<cfg.sigma<<" step="<<cfg.step<<" relax=Y\n"
      <<"MSE="<<mse<<"  NMI="<<nmi<<"\nTime:"<<(int)elapsed<<"ms";

    g_panels[panelIdx].title = "LCC Demons 3D";
    finishTask(g_panels[panelIdx], std::move(result), elapsed, ss.str(), mse, nmi);
}

static void runFFD(int panelIdx) {
    std::cout << "[B-Spline FFD] Running 3D (rigid init + FFD)...\n";
    double t0=nowMs();

    // 刚性预对齐
    MedReg::RigidRegConfig3D rcfg; rcfg.numLevels=2; rcfg.maxIterations=80;
    auto rp = MedReg::rigidRegistration3D(g_ref, g_mov, rcfg);
    auto preAligned = MedReg::applyRigid3D(g_mov, g_ref, rp);

    // B-Spline FFD 精化
    MedReg::FFDConfig3D cfg;
    cfg.gridSpacing = (std::max)(4, VOL_W/6);
    cfg.stepSize=0.5f; cfg.maxIterations=35;

    auto disp   = MedReg::bsplineFFD3D(g_ref, preAligned, cfg);
    auto result = MedReg::applyDeform3D(preAligned, disp);
    double elapsed=nowMs()-t0;

    float mse=MedReg::computeMSE3D(g_ref,result), nmi=MedReg::computeNMI3D(g_ref,result);
    std::ostringstream ss;
    ss<<std::fixed<<std::setprecision(4)
      <<"grid="<<cfg.gridSpacing<<"vox step="<<cfg.stepSize<<"\n"
      <<"MSE="<<mse<<"  NMI="<<nmi<<"\nTime:"<<(int)elapsed<<"ms";

    g_panels[panelIdx].title = "B-Spline FFD 3D";
    finishTask(g_panels[panelIdx], std::move(result), elapsed, ss.str(), mse, nmi);
}

// ============================================================
//  渲染
// ============================================================
static void drawPanel(const Panel& pan, int winW, int winH) {
    int cW=winW/COLS, cH=winH/ROWS;
    int x=pan.col*cW, y=(ROWS-1-pan.row)*cH;
    glViewport(x, y, cW, cH);
    glUseProgram(g_prog);

    glm::mat4 I(1.f);
    glUniformMatrix4fv(glGetUniformLocation(g_prog,"uT"),1,GL_FALSE,glm::value_ptr(I));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, pan.tex);
    glUniform1i(glGetUniformLocation(g_prog,"uTex"),0);

    int mode = (g_checkerMode && pan.texRef && pan.ready && pan.col+pan.row>0) ? 1 : 0;
    glUniform1i(glGetUniformLocation(g_prog,"uMode"),mode);
    if (mode==1) {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, pan.texRef);
        glUniform1i(glGetUniformLocation(g_prog,"uRef"),1);
        glUniform1f(glGetUniformLocation(g_prog,"uChk"),0.10f);
    }
    glm::vec3 tint = pan.ready ? glm::vec3(1.f) : glm::vec3(0.5f,0.45f,0.25f);
    glUniform3fv(glGetUniformLocation(g_prog,"uTint"),1,glm::value_ptr(tint));

    glBindVertexArray(g_quadVAO);
    glDrawArrays(GL_TRIANGLES,0,6);
}

// ============================================================
//  键盘回调
// ============================================================
static void onKey(GLFWwindow* win, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    if (key==GLFW_KEY_ESCAPE || key==GLFW_KEY_Q) { glfwSetWindowShouldClose(win,1); return; }
    if (key==GLFW_KEY_C) { g_checkerMode=!g_checkerMode; return; }
    if (key==GLFW_KEY_R) { initData(); g_currentSlice=VOL_D/2; refreshAllSlices(g_currentSlice); return; }
    if (key==GLFW_KEY_UP)   {
        g_currentSlice = (std::min)(g_currentSlice+1, VOL_D-1);
        refreshAllSlices(g_currentSlice);
        std::cout << "[Slice] Z=" << g_currentSlice << "\n"; return;
    }
    if (key==GLFW_KEY_DOWN) {
        g_currentSlice = (std::max)(g_currentSlice-1, 0);
        refreshAllSlices(g_currentSlice);
        std::cout << "[Slice] Z=" << g_currentSlice << "\n"; return;
    }

    if (g_busy) { std::cout << "[Busy] Please wait for current task to finish.\n"; return; }

    auto run = [&](std::function<void()> fn) {
        g_busy=true; fn(); g_busy=false;
        refreshAllSlices(g_currentSlice);
    };

    switch(key) {
        case GLFW_KEY_1: run([](){ runRigid(2,false); }); break;
        case GLFW_KEY_2: run([](){ runRigid(2,true);  }); break;
        case GLFW_KEY_3: run([](){ runDemons(3);       }); break;
        case GLFW_KEY_4: run([](){ runMultiGridDemons(4); }); break;
        case GLFW_KEY_5: run([](){ runLCCDemons(5);    }); break;
        case GLFW_KEY_6: run([](){ runFFD(5);          }); break;
        case GLFW_KEY_A:
            run([](){
                std::cout << "\n=== Running ALL 3D registration algorithms ===\n";
                runRigid(2,false);
                runDemons(3);
                runMultiGridDemons(4);
                runLCCDemons(5);

                //std::cout << "\n┌─────────────────────────┬────────────┬────────────┬─────────────┐\n";
                //std::cout <<   "│ Algorithm               │    MSE     │    NMI     │  Time (ms)  │\n";
                //std::cout <<   "├─────────────────────────┼────────────┼────────────┼─────────────┤\n";
                //std::cout << "│ Moving (before 3D reg)  │"
                //           <<std::setw(12)<<std::fixed<<std::setprecision(5)<<g_panels[1].mse
                //           <<"│"<<std::setw(12)<<g_panels[1].nmi<<"│      —      │\n";
                /*for (int i=2;i<COLS*ROWS;i++) {
                    auto& p=g_panels[i];
                    if (p.ready && p.mse>=0) {
                        std::cout<<"│ "<<std::left<<std::setw(23)<<p.title.substr(0,23)
                                  <<"│"<<std::setw(12)<<std::right<<std::fixed<<std::setprecision(5)<<p.mse
                                  <<"│"<<std::setw(12)<<p.nmi
                                  <<"│"<<std::setw(13)<<(int)p.timeMs<<"│\n";
                    }
                }*/
                std::cout << "└─────────────────────────┴────────────┴────────────┴─────────────┘\n";
            });
            break;
        default: break;
    }
}

// ============================================================
//  Main
// ============================================================
int main() {
    if (!glfwInit()) { std::cerr<<"GLFW init fail\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);

    std::ostringstream title;
    title << "3D Medical Image Registration Demo  ["<<VOL_W<<"x"<<VOL_H<<"x"<<VOL_D
          << "]  —  Press [A] to run all | [↑/↓] change slice";
    g_win = glfwCreateWindow(WIN_W, WIN_H, title.str().c_str(), nullptr, nullptr);
    if (!g_win) { std::cerr<<"Window fail\n"; return -1; }
    glfwMakeContextCurrent(g_win);
    glfwSwapInterval(1);
    glfwSetKeyCallback(g_win, onKey);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { std::cerr<<"GLAD fail\n"; return -1; }
    std::cout << "[GL] " << glGetString(GL_RENDERER) << " — " << glGetString(GL_VERSION) << "\n";

    g_prog = createProgram(VS, FS);
    createQuad();
    if (!initData()) {
        std::cerr << "Press [R] to retry after fixing data paths, or [Q] to quit.\n";
    }

    std::cout <<
        "\n==========================================================\n"
        "  3D Medical Image Registration Demo (CT-PET)\n"
        "  Volume: " << VOL_W<<"x"<<VOL_H<<"x"<<VOL_D << " voxels (resampled)\n"
        "  CT  : " << CT_FOLDER << "\n"
        "  PET : " << PET_FOLDER << "\n"
        "----------------------------------------------------------\n"
        "  [1] Rigid (MSE)          [2] Rigid (NMI)\n"
        "  [3] Demons 3D            [4] Multi-Grid Demons 3D\n"
        "  [5] LCC Demons 3D        [6] B-Spline FFD 3D\n"
        "  [A] Run ALL algorithms   [C] Checker overlay\n"
        "  [↑/↓] Navigate slices   [R] Reset  [Q] Quit\n"
        "==========================================================\n\n";

    glClearColor(0.07f, 0.07f, 0.1f, 1.f);

    while (!glfwWindowShouldClose(g_win)) {
        glfwPollEvents();
        int ww, wh; glfwGetFramebufferSize(g_win, &ww, &wh);
        glClear(GL_COLOR_BUFFER_BIT);
        for (const auto& p : g_panels) drawPanel(p, ww, wh);
        glViewport(0,0,ww,wh);
        glfwSwapBuffers(g_win);
    }

    glDeleteProgram(g_prog);
    glDeleteVertexArrays(1,&g_quadVAO);
    glDeleteBuffers(1,&g_quadVBO);
    for (auto& p : g_panels) {
        if (p.tex)    glDeleteTextures(1,&p.tex);
        if (p.texRef) glDeleteTextures(1,&p.texRef);
    }
    glfwDestroyWindow(g_win);
    glfwTerminate();
    return 0;
}
