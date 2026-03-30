#pragma once

#include "transferFunction.h"
#include <glfw3.h>
#include <iostream>
#include <vector>

class PostProcessor {
public:
    GLuint fbo;           // 帧缓冲区对象
    GLuint colorTexture;  // 颜色纹理
    GLuint rboDepth;      // 深度渲染缓冲区（可选）
    int width, height;

    // 全屏四边形几何
    GLuint quadVAO, quadVBO;

    GLuint fxaaProgram;
    GLuint screenTextureProgram;

    PostProcessor(int w, int h) : width(w), height(h) {
        createFramebuffer();
        setupQuad();
        setupShaders();
    }

    ~PostProcessor() {
        cleanup();
    }

    void resize(int w, int h) {
        width = w;
        height = h;

        // 重新创建纹理和渲染缓冲区
        glBindTexture(GL_TEXTURE_2D, colorTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

        if (rboDepth) {
            glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
        }
    }

    void beginRender() {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    void endRender() {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, width, height);
    }

    void applyFXAA() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(fxaaProgram);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, colorTexture);
        glUniform1i(glGetUniformLocation(fxaaProgram, "screenTexture"), 0);

        glUniform2f(glGetUniformLocation(fxaaProgram, "resolution"), (float)width, (float)height);

        // 渲染全屏四边形
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
    }


private:
    // 创建帧缓冲对象
    void createFramebuffer() {
        // 创建帧缓冲
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        // 创建颜色纹理附件
        glGenTextures(1, &colorTexture);
        glBindTexture(GL_TEXTURE_2D, colorTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);

        // 创建深度和模板渲染缓冲区（可选，体绘制通常不需要）
        glGenRenderbuffers(1, &rboDepth);
        glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rboDepth);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::cerr << "帧缓冲区不完整!" << std::endl;
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void setupQuad() {
        // 全屏四边形顶点数据
        float quadVertices[] = {
            // 位置        // 纹理坐标
            -1.0f,  1.0f,  0.0f, 1.0f,
            -1.0f, -1.0f,  0.0f, 0.0f,
             1.0f, -1.0f,  1.0f, 0.0f,

            -1.0f,  1.0f,  0.0f, 1.0f,
             1.0f, -1.0f,  1.0f, 0.0f,
             1.0f,  1.0f,  1.0f, 1.0f
        };

        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);

        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

        // 位置属性
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);

        // 纹理坐标属性
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

        glBindVertexArray(0);
    }

    GLuint createShaderProgram(const char* vertSource, const char* fragSource) {
        GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertShader, 1, &vertSource, nullptr);
        glCompileShader(vertShader);

        GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragShader, 1, &fragSource, nullptr);
        glCompileShader(fragShader);

        GLuint program = glCreateProgram();
        glAttachShader(program, vertShader);
        glAttachShader(program, fragShader);
        glLinkProgram(program);

        glDeleteShader(vertShader);
        glDeleteShader(fragShader);

        return program;
    }

    void cleanup() {
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &colorTexture);
        glDeleteRenderbuffers(1, &rboDepth);
        glDeleteVertexArrays(1, &quadVAO);
        glDeleteBuffers(1, &quadVBO);
        glDeleteProgram(fxaaProgram);
        glDeleteProgram(screenTextureProgram);
    }

    void setupShaders() {
        // FXAA着色器
        const char* fxaaVert = R"(#version 430 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTexCoords;

        out vec2 v_texCoord;

        void main() {
            gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
            v_texCoord = aTexCoords;
        })";

        const char* fxaaFrag = R"(#version 430 core
        uniform sampler2D screenTexture;
        uniform vec2 resolution;

        in vec2 v_texCoord;
        out vec4 FragColor;

        #ifndef FXAA_REDUCE_MIN
            #define FXAA_REDUCE_MIN   (1.0/128.0)
        #endif
        #ifndef FXAA_REDUCE_MUL
            #define FXAA_REDUCE_MUL   (1.0/8.0)
        #endif
        #ifndef FXAA_SPAN_MAX
            #define FXAA_SPAN_MAX     8.0
        #endif

    void main() {
        vec2 uv = v_texCoord;
    
        vec3 rgbNW = textureOffset(screenTexture, uv, ivec2(-1, -1)).rgb;
        vec3 rgbNE = textureOffset(screenTexture, uv, ivec2(1, -1)).rgb;
        vec3 rgbSW = textureOffset(screenTexture, uv, ivec2(-1, 1)).rgb;
        vec3 rgbSE = textureOffset(screenTexture, uv, ivec2(1, 1)).rgb;
        vec3 rgbM  = texture(screenTexture, uv).rgb;
    
        vec3 luma = vec3(0.299, 0.587, 0.114);
        float lumaNW = dot(rgbNW, luma);
        float lumaNE = dot(rgbNE, luma);
        float lumaSW = dot(rgbSW, luma);
        float lumaSE = dot(rgbSE, luma);
        float lumaM  = dot(rgbM,  luma);
    
        float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
        float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
    
        vec2 dir;
        dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
        dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));
    
        float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);
        float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    
        dir = min(vec2(FXAA_SPAN_MAX, FXAA_SPAN_MAX), 
                  max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX), dir * rcpDirMin)) / resolution;
    
        vec3 rgbA = 0.5 * (
            texture(screenTexture, uv + dir * (1.0 / 3.0 - 0.5)).rgb +
            texture(screenTexture, uv + dir * (2.0 / 3.0 - 0.5)).rgb);
    
        vec3 rgbB = rgbA * 0.5 + 0.25 * (
            texture(screenTexture, uv + dir * -0.5).rgb +
            texture(screenTexture, uv + dir * 0.5).rgb);
    
        float lumaB = dot(rgbB, luma);
    
        if ((lumaB < lumaMin) || (lumaB > lumaMax)) {
            FragColor = vec4(rgbA, 1.0);
        } else {
            FragColor = vec4(rgbB, 1.0);
        }
    })";

        fxaaProgram = createShaderProgram(fxaaVert, fxaaFrag);
    }
};
