#pragma once

#include "shader.h"
 
#include <glfw3.h>
#include <vector>
#include <glm/fwd.hpp>
#include <glm/common.hpp>
#include <glm/ext/vector_float4.hpp>

class TransferFunction {
private:
    GLuint textureID;
    int textureSize;
    std::vector<glm::vec4> tfData;

public:
    TransferFunction(int size = 256) : textureSize(size) {
        tfData.resize(textureSize);
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_1D, textureID);

        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // 默认窗宽窗位
        update(0.5f, 1.0f);
    }

    ~TransferFunction() {
        glDeleteTextures(1, &textureID);
    }

    void update(float windowLevel, float windowWidth) {
        // 确保窗宽大于0
        windowWidth = std::max(windowWidth, 0.001f);

        // 计算窗口范围
        float windowMin = windowLevel - windowWidth / 2.0f;
        float windowMax = windowLevel + windowWidth / 2.0f;

        for (int i = 0; i < textureSize; ++i) {
            float density = float(i) / float(textureSize - 1);

            // 应用窗宽窗位
            float normalizedDensity = 0.0f;
            if (windowWidth > 0) {
                normalizedDensity = (density - windowMin) / windowWidth;
                normalizedDensity = glm::clamp(normalizedDensity, 0.0f, 1.0f);
            }

            // 1. CT扫描的典型灰度显示
            glm::vec4 color = ctGrayTransferFunction(normalizedDensity);

            // 2. 伪彩色增强特定组织
            //glm::vec4 color = pseudoColorTransferFunction(normalizedDensity);

            // 3. 医学常用的颜色映射
            //glm::vec4 color = medicalTransferFunction(density, windowLevel, windowWidth);

            tfData[i] = color;
        }

        glBindTexture(GL_TEXTURE_1D, textureID);
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, textureSize, 0, GL_RGBA, GL_FLOAT, tfData.data());
        glBindTexture(GL_TEXTURE_1D, 0);
    }

    glm::vec4 ctGrayTransferFunction(float t) {
        if (t <= 0.0f) {
            return glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        else if (t >= 1.0f) {
            return glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
        }
        else {
            // 使用平滑的sigmoid函数
            float smooth_t = t * t * (3.0f - 2.0f * t); // 三次平滑插值
            // 或者使用平滑步函数
            // float smooth_t = t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); // 五次平滑步

            // 透明度使用更平滑的曲线
            float alpha = 1.0f - (1.0f - smooth_t) * (1.0f - smooth_t); // 平滑的alpha过渡

            return glm::vec4(smooth_t, smooth_t, smooth_t, alpha);
        }
    }

    // 伪彩色传输函数（增强特定组织对比度）
    glm::vec4 pseudoColorTransferFunction(float t) {
        if (t < 0.1f) {
            // 空气/肺组织：蓝色，低透明度
            float alpha = t / 0.1f;
            return glm::vec4(0.0f, 0.0f, 0.7f, alpha * 0.1f);
        }
        else if (t < 0.3f) {
            // 脂肪组织：黄色
            float alpha = (t - 0.1f) / 0.2f;
            return glm::vec4(1.0f, 1.0f, 0.0f, 0.2f + alpha * 0.3f);
        }
        else if (t < 0.5f) {
            // 软组织：红色到橙色
            float alpha = (t - 0.3f) / 0.2f;
            return glm::vec4(1.0f, 0.5f - alpha * 0.5f, 0.0f, 0.5f + alpha * 0.3f);
        }
        else if (t < 0.7f) {
            // 骨骼：白色，高对比度
            float alpha = (t - 0.5f) / 0.2f;
            return glm::vec4(1.0f, 1.0f, 1.0f, 0.8f + alpha * 0.2f);
        }
        else {
            // 高密度骨骼/钙化：品红色
            float alpha = (t - 0.7f) / 0.3f;
            return glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
        }
    }

    glm::vec4 medicalTransferFunction(float originalDensity, float windowLevel, float windowWidth) {
        // 根据组织类型定义不同颜色
        // 假设我们知道不同组织的HU值范围

        // 转换为HU值（假设数据已标准化到0-1，需要反算）
        float huValue = originalDensity * 4000 - 1000; // 示例转换

        // 根据HU值判断组织类型
        if (huValue < -900) {
            // 空气：黑色透明
            return glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        else if (huValue < -100) {
            // 肺组织：深蓝色
            float alpha = (huValue + 900) / 800.0f;
            return glm::vec4(0.0f, 0.0f, 0.5f + alpha * 0.5f, alpha * 0.3f);
        }
        else if (huValue < 40) {
            // 脂肪：黄色
            float alpha = (huValue + 100) / 140.0f;
            return glm::vec4(1.0f, 1.0f, 0.0f, 0.3f + alpha * 0.2f);
        }
        else if (huValue < 80) {
            // 水/脑脊液：蓝色
            float alpha = (huValue - 40) / 40.0f;
            return glm::vec4(0.0f, 0.0f, 1.0f, 0.5f + alpha * 0.2f);
        }
        else if (huValue < 300) {
            // 软组织：红色
            float alpha = (huValue - 80) / 220.0f;
            return glm::vec4(1.0f, 0.0f, 0.0f, 0.7f + alpha * 0.2f);
        }
        else {
            // 骨骼：白色
            float alpha = glm::min(1.0f, (huValue - 300) / 1700.0f);
            return glm::vec4(1.0f, 1.0f, 1.0f, 0.9f + alpha * 0.1f);
        }
    }

    GLuint getTextureID() const { return textureID; }

    void bind(GLuint textureUnit) {
        glActiveTexture(GL_TEXTURE0 + textureUnit);
        glBindTexture(GL_TEXTURE_1D, textureID);
    }
};
