/**
 * @file    texture.cpp
 * @brief   OpenGL 纹理封装 实现
 *          STB_IMAGE_IMPLEMENTATION 定义在此文件中（全项目唯一）
 */

#include "render/texture.h"
#include <iostream>
#include <cmath>
#include <vector>

// stb_image — 图片加载（全项目唯一 IMPLEMENTATION）
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace NailPrint3D {

GLTexture::GLTexture() {}

GLTexture::~GLTexture() {
    destroy();
}

bool GLTexture::loadFromFile(const std::string& path) {
    destroy();

    unsigned char* data = stbi_load(path.c_str(), &width_, &height_, &channels_, 0);
    if (!data) {
        std::cerr << "[Texture] 无法加载: " << path << " — " << stbi_failure_reason() << std::endl;
        return false;
    }

    std::cout << "[Texture] 加载: " << path << " (" << width_ << "x" << height_
              << ", " << channels_ << " channels)" << std::endl;

    bool ok = loadFromMemory(data, width_, height_, channels_);
    stbi_image_free(data);
    return ok;
}

bool GLTexture::loadFromMemory(const unsigned char* data, int w, int h, int channels) {
    destroy();
    width_ = w;
    height_ = h;
    channels_ = channels;

    glGenTextures(1, &id_);
    glBindTexture(GL_TEXTURE_2D, id_);

    GLenum format = GL_RGBA;
    if (channels == 1) format = GL_RED;
    else if (channels == 2) format = GL_RG;
    else if (channels == 3) format = GL_RGB;

    if (channels != 4) {
        std::vector<unsigned char> rgba(w * h * 4, 255);
        for (int i = 0; i < w * h; i++) {
            for (int c = 0; c < channels && c < 3; c++)
                rgba[i * 4 + c] = data[i * channels + c];
            if (channels < 4) rgba[i * 4 + 3] = 255;
        }
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    } else {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    }

    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindTexture(GL_TEXTURE_2D, 0);
    return true;
}

bool GLTexture::generateTestPattern(int patternType, int w, int h) {
    std::vector<unsigned char> data(w * h * 4);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float u = (float)x / w;
            float v = (float)y / h;
            int idx = (y * w + x) * 4;

            unsigned char r = 255, g = 255, b = 255, a = 255;

            if (patternType == 0) {
                float cx = 0.5f, cy = 0.5f;
                float dx = u - cx, dy = v - cy;
                float dist = std::sqrt(dx * dx + dy * dy);
                float angle = std::atan2(dy, dx);
                float petalR = 0.3f + 0.1f * std::cos(5.0f * angle);
                if (dist < petalR) {
                    float t = dist / petalR;
                    r = (unsigned char)(255 * (1.0f - t * 0.3f));
                    g = (unsigned char)(100 + t * 100);
                    b = (unsigned char)(150 + t * 80);
                } else {
                    r = 255; g = 240; b = 180;
                }
                float edge = std::abs(dist - petalR);
                if (edge < 0.02f) { r = 60; g = 40; b = 20; }

            } else if (patternType == 1) {
                float cx = 0.5f, cy = 0.45f;
                float dx = (u - cx) * 1.2f, dy = (v - cy) * 1.5f;
                float faceDist = std::sqrt(dx * dx + dy * dy);
                if (faceDist < 0.3f) {
                    r = 230; g = 195; b = 170;
                    float eyeY = cy - 0.05f;
                    float leyeD = std::sqrt((u - 0.4f) * (u - 0.4f) + (v - eyeY) * (v - eyeY));
                    float reyeD = std::sqrt((u - 0.6f) * (u - 0.6f) + (v - eyeY) * (v - eyeY));
                    if (leyeD < 0.03f || reyeD < 0.03f) { r = 40; g = 30; b = 20; }
                    float mouthD = std::sqrt((u - 0.5f) * (u - 0.5f) + (v - 0.58f) * (v - 0.58f));
                    if (mouthD < 0.04f && v > 0.55f) { r = 180; g = 80; b = 80; }
                    float blushL = std::sqrt((u - 0.35f) * (u - 0.35f) + (v - 0.5f) * (v - 0.5f));
                    float blushR = std::sqrt((u - 0.65f) * (u - 0.65f) + (v - 0.5f) * (v - 0.5f));
                    if (blushL < 0.06f || blushR < 0.06f) { r = 255; g = 180; b = 170; }
                } else {
                    r = 200; g = 200; b = 210;
                }

            } else if (patternType == 2) {
                float scale = 6.0f;
                float fu = u * scale - std::floor(u * scale);
                float fv = v * scale - std::floor(v * scale);
                float diamond = std::abs(fu - 0.5f) + std::abs(fv - 0.5f);
                int checker = ((int)std::floor(u * scale) + (int)std::floor(v * scale)) % 2;
                if (diamond < 0.25f) {
                    if (checker) { r = 255; g = 100; b = 150; }
                    else { r = 100; g = 180; b = 255; }
                } else {
                    r = (unsigned char)(255 * (0.8f + 0.2f * u));
                    g = (unsigned char)(255 * (0.7f + 0.2f * v));
                    b = (unsigned char)(255 * (0.75f + 0.15f * (1.0f - u)));
                }

            } else if (patternType == 3) {
                r = 245; g = 235; b = 240;
                int gx = (int)(u * 16);
                int gy = (int)(v * 4);
                const char* letters[4] = {"NAIL", "AIL ", "IL  ", "L   "};
                char ch = letters[gy][gx % 4];
                if (ch != ' ') {
                    float cx = ((gx % 4) + 0.5f) / 4.0f;
                    float cy = (gy + 0.5f) / 4.0f;
                    float cd = std::sqrt((u - cx) * (u - cx) * 4.0f + (v - cy) * (v - cy) * 16.0f);
                    if (cd < 0.3f) { r = 220; g = 60; b = 100; }
                }
            }

            data[idx] = r;
            data[idx + 1] = g;
            data[idx + 2] = b;
            data[idx + 3] = 255;
        }
    }

    return loadFromMemory(data.data(), w, h, 4);
}

void GLTexture::bind(int unit) const {
    if (!id_) return;
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, id_);
}

void GLTexture::destroy() {
    if (id_) { glDeleteTextures(1, &id_); id_ = 0; }
    width_ = height_ = channels_ = 0;
}

} // namespace NailPrint3D
