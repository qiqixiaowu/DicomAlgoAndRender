#pragma once

/**
 * ==========================================
 * MPR渲染器 (Multi-Planar Reconstruction)
 * ==========================================
 * 
 * 提供三个正交切面视图：
 * - Axial   (横断面): XY平面，沿Z轴切片
 * - Sagittal(矢状面): YZ平面，沿X轴切片
 * - Coronal (冠状面): XZ平面，沿Y轴切片
 * 
 * 支持功能：
 * - 四视口布局（3D + 三个MPR视图）
 * - 交互式切片位置调整
 * - 十字线联动显示
 * - 窗宽窗位调整
 * - 灰度/传递函数着色切换
 */

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"
#include <memory>
#include <iostream>

// ==========================================
// 切面轴向枚举
// ==========================================
enum class MPRAxis {
    Axial    = 0,  // 横断面 (XY平面，沿Z轴)
    Sagittal = 1,  // 矢状面 (YZ平面，沿X轴)
    Coronal  = 2   // 冠状面 (XZ平面，沿Y轴)
};

// ==========================================
// MPR切面状态
// ==========================================
struct MPRSliceState {
    float axialPos    = 0.5f;   // Axial切面位置 [0, 1]
    float sagittalPos = 0.5f;   // Sagittal切面位置 [0, 1]
    float coronalPos  = 0.5f;   // Coronal切面位置 [0, 1]
    
    // 切面调整步长
    float sliceStep = 0.005f;
    
    // MPR独立窗宽窗位（与3D体绘制分离）
    float mprWindowLevel = 0.5f;     // MPR窗位 [0, 1]（默认居中，显示全范围）
    float mprWindowWidth = 1.0f;     // MPR窗宽 [0, 2]（默认1.0，覆盖完整密度区间）
    float mprBrightness  = 1.0f;     // MPR亮度
    
    // 显示选项
    bool useTransferFunc   = false;  // 是否使用传递函数着色
    bool showCrosshair     = true;   // 显示十字线
    bool showBorder        = true;   // 显示边框
    float crosshairThick   = 1.5f;   // 十字线粗细
    float borderThick      = 2.0f;   // 边框粗细
    
    // 各视图的边框/十字线颜色
    // Axial:    红色边框 | 十字线: 绿(水平/Coronal位置) + 蓝(垂直/Sagittal位置)
    // Sagittal: 蓝色边框 | 十字线: 红(水平/Axial位置)  + 绿(垂直/Coronal位置)
    // Coronal:  绿色边框 | 十字线: 红(水平/Axial位置)  + 蓝(垂直/Sagittal位置)
    glm::vec3 axialBorderColor    = glm::vec3(1.0f, 0.3f, 0.3f);  // 红
    glm::vec3 sagittalBorderColor = glm::vec3(0.3f, 0.3f, 1.0f);  // 蓝
    glm::vec3 coronalBorderColor  = glm::vec3(0.3f, 1.0f, 0.3f);  // 绿
};

// ==========================================
// MPR渲染器类
// ==========================================
class MPRRenderer {
public:
    MPRSliceState state;
    
    MPRRenderer() : m_quadVAO(0), m_quadVBO(0), m_initialized(false) {}
    
    ~MPRRenderer() {
        cleanup();
    }
    
    /**
     * 初始化MPR渲染器
     * 加载着色器、创建全屏quad几何
     */
    bool init() {
        // 加载MPR着色器
        try {
            m_shader = std::make_unique<Shader>(
                "shader/mpr_vert.vert",
                "shader/mpr_frag.frag"
            );
        } catch (...) {
            std::cerr << "MPR: 无法加载着色器" << std::endl;
            return false;
        }
        
        // 创建全屏quad
        createQuad();
        
        m_initialized = true;
        return true;
    }
    
    /**
     * 渲染一个MPR切面到当前视口
     * 使用 state 中的独立窗宽窗位参数（与3D体绘制分离）
     * @param axis          切面轴向
     * @param volumeTex     3D体数据纹理ID
     * @param transferTex   1D传递函数纹理ID
     * @param viewportW     当前视口宽度（像素）
     * @param viewportH     当前视口高度（像素）
     */
    void renderSlice(MPRAxis axis, GLuint volumeTex, GLuint transferTex,
                     int viewportW, int viewportH)
    {
        if (!m_initialized) return;
        
        m_shader->use();
        
        // 设置切面参数
        int axisInt = static_cast<int>(axis);
        m_shader->setInt("uSliceAxis", axisInt);
        
        // 切面位置
        float slicePos = getSlicePosition(axis);
        m_shader->setFloat("uSlicePosition", slicePos);
        
        // 使用MPR独立的窗宽窗位（不共用3D体绘制的参数）
        m_shader->setFloat("windowLevel", state.mprWindowLevel);
        m_shader->setFloat("windowWidth", state.mprWindowWidth);
        m_shader->setFloat("brightness", state.mprBrightness);
        m_shader->setBool("useTransferFunc", state.useTransferFunc);
        
        // 十字线参数
        m_shader->setBool("showCrosshair", state.showCrosshair);
        m_shader->setFloat("crosshairThickness", state.crosshairThick);
        m_shader->setVec2("viewportSize", glm::vec2((float)viewportW, (float)viewportH));
        
        // 根据轴向设置十字线位置和颜色
        setCrosshairParams(axis);
        
        // 边框参数
        m_shader->setBool("showBorder", state.showBorder);
        m_shader->setFloat("borderThickness", state.borderThick);
        setBorderColor(axis);
        
        // 绑定纹理
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, volumeTex);
        m_shader->setInt("volume", 0);
        
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_1D, transferTex);
        m_shader->setInt("transferFunc", 1);
        
        // 渲染全屏quad
        glBindVertexArray(m_quadVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);
    }
    
    /**
     * 在四视口布局中渲染所有MPR视图
     * 布局：
     * ┌──────────┬──────────┐
     * │  3D体绘制 │  Axial   │
     * │ (外部渲染)│  横断面   │
     * ├──────────┼──────────┤
     * │ Sagittal │ Coronal  │
     * │  矢状面   │  冠状面   │
     * └──────────┴──────────┘
     * 
     * @param windowW       窗口总宽度
     * @param windowH       窗口总高度
     * @param volumeTex     3D体数据纹理ID
     * @param transferTex   1D传递函数纹理ID
     */
    void renderAllMPRViews(int windowW, int windowH,
                           GLuint volumeTex, GLuint transferTex)
    {
        int halfW = windowW / 2;
        int halfH = windowH / 2;
        
        // 禁用深度测试（MPR是2D渲染）
        glDisable(GL_DEPTH_TEST);
        
        // Axial视图 - 右上角
        glViewport(halfW, halfH, halfW, halfH);
        renderSlice(MPRAxis::Axial, volumeTex, transferTex, halfW, halfH);
        
        // Sagittal视图 - 左下角
        glViewport(0, 0, halfW, halfH);
        renderSlice(MPRAxis::Sagittal, volumeTex, transferTex, halfW, halfH);
        
        // Coronal视图 - 右下角
        glViewport(halfW, 0, halfW, halfH);
        renderSlice(MPRAxis::Coronal, volumeTex, transferTex, halfW, halfH);
        
        // 恢复深度测试
        glEnable(GL_DEPTH_TEST);
    }
    
    /**
     * 获取3D体绘制的视口区域（四视口模式下左上角）
     */
    void set3DViewport(int windowW, int windowH) {
        glViewport(0, windowH / 2, windowW / 2, windowH / 2);
    }
    
    /**
     * 获取指定轴向的切面位置
     */
    float getSlicePosition(MPRAxis axis) const {
        switch (axis) {
            case MPRAxis::Axial:    return state.axialPos;
            case MPRAxis::Sagittal: return state.sagittalPos;
            case MPRAxis::Coronal:  return state.coronalPos;
            default:                return 0.5f;
        }
    }
    
    /**
     * 设置指定轴向的切面位置
     */
    void setSlicePosition(MPRAxis axis, float pos) {
        pos = glm::clamp(pos, 0.0f, 1.0f);
        switch (axis) {
            case MPRAxis::Axial:    state.axialPos = pos; break;
            case MPRAxis::Sagittal: state.sagittalPos = pos; break;
            case MPRAxis::Coronal:  state.coronalPos = pos; break;
        }
    }
    
    /**
     * 调整指定轴向的切面位置（增量）
     */
    void adjustSlice(MPRAxis axis, float delta) {
        float current = getSlicePosition(axis);
        setSlicePosition(axis, current + delta);
    }
    
    /**
     * 获取切面位置对应的实际层数索引
     */
    int getSliceIndex(MPRAxis axis, const glm::vec3& volumeSize) const {
        float pos = getSlicePosition(axis);
        float maxIdx;
        switch (axis) {
            case MPRAxis::Axial:    maxIdx = volumeSize.z - 1.0f; break;
            case MPRAxis::Sagittal: maxIdx = volumeSize.x - 1.0f; break;
            case MPRAxis::Coronal:  maxIdx = volumeSize.y - 1.0f; break;
            default:                maxIdx = 0.0f;
        }
        return static_cast<int>(pos * maxIdx + 0.5f);
    }
    
    /**
     * 获取MPR着色器（供外部额外配置）
     */
    Shader* getShader() { return m_shader.get(); }
    
    /**
     * 清理GPU资源
     */
    void cleanup() {
        if (m_quadVAO) {
            glDeleteVertexArrays(1, &m_quadVAO);
            m_quadVAO = 0;
        }
        if (m_quadVBO) {
            glDeleteBuffers(1, &m_quadVBO);
            m_quadVBO = 0;
        }
        m_initialized = false;
    }

private:
    std::unique_ptr<Shader> m_shader;
    GLuint m_quadVAO;
    GLuint m_quadVBO;
    bool m_initialized;
    
    /**
     * 创建全屏quad几何体
     * 顶点格式：position(2) + texcoord(2)
     */
    void createQuad() {
        float quadVertices[] = {
            // position    // texcoord
            -1.0f, -1.0f,  0.0f, 0.0f,
             1.0f, -1.0f,  1.0f, 0.0f,
            -1.0f,  1.0f,  0.0f, 1.0f,
             1.0f,  1.0f,  1.0f, 1.0f,
        };
        
        glGenVertexArrays(1, &m_quadVAO);
        glGenBuffers(1, &m_quadVBO);
        
        glBindVertexArray(m_quadVAO);
        
        glBindBuffer(GL_ARRAY_BUFFER, m_quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
        
        // position
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        // texcoord
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
        
        glBindVertexArray(0);
    }
    
    /**
     * 设置十字线参数（位置和颜色根据视图联动）
     */
    void setCrosshairParams(MPRAxis axis) {
        glm::vec2 crossPos;
        glm::vec3 colorH, colorV;
        
        switch (axis) {
            case MPRAxis::Axial:
                // Axial视图：水平线=Coronal位置(绿)，垂直线=Sagittal位置(蓝)
                crossPos = glm::vec2(state.sagittalPos, state.coronalPos);
                colorH = state.coronalBorderColor;   // 绿
                colorV = state.sagittalBorderColor;   // 蓝
                break;
                
            case MPRAxis::Sagittal:
                // Sagittal视图：水平线=Axial位置(红)，垂直线=Coronal位置(绿)
                crossPos = glm::vec2(state.coronalPos, state.axialPos);
                colorH = state.axialBorderColor;      // 红
                colorV = state.coronalBorderColor;    // 绿
                break;
                
            case MPRAxis::Coronal:
                // Coronal视图：水平线=Axial位置(红)，垂直线=Sagittal位置(蓝)
                crossPos = glm::vec2(state.sagittalPos, state.axialPos);
                colorH = state.axialBorderColor;      // 红
                colorV = state.sagittalBorderColor;   // 蓝
                break;
        }
        
        m_shader->setVec2("crosshairPos", crossPos);
        m_shader->setVec3("crosshairColorH", colorH);
        m_shader->setVec3("crosshairColorV", colorV);
    }
    
    /**
     * 设置边框颜色
     */
    void setBorderColor(MPRAxis axis) {
        glm::vec3 color;
        switch (axis) {
            case MPRAxis::Axial:    color = state.axialBorderColor; break;
            case MPRAxis::Sagittal: color = state.sagittalBorderColor; break;
            case MPRAxis::Coronal:  color = state.coronalBorderColor; break;
        }
        m_shader->setVec3("borderColor", color);
    }
};
