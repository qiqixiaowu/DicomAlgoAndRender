#pragma once
/**
 * @file    texture.h
 * @brief   OpenGL 纹理封装（支持 stb_image 加载图片）
 *
 * 依赖：GLAD, stb_image
 */

#include <glad/glad.h>
#include <string>

namespace NailPrint3D {

/** @brief OpenGL纹理封装（支持stb_image加载图片） */
struct GLTexture {
    GLuint id_ = 0;
    int width_ = 0;
    int height_ = 0;
    int channels_ = 0;

    GLTexture();
    ~GLTexture();

    bool loadFromFile(const std::string& path);
    bool loadFromMemory(const unsigned char* data, int w, int h, int channels);
    bool generateTestPattern(int patternType, int w = 512, int h = 512);
    void bind(int unit = 0) const;
    void destroy();
};

} // namespace NailPrint3D
