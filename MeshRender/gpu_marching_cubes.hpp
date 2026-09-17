#pragma once
/**
 * @file    gpu_marching_cubes.hpp
 * @brief   GPU Marching Cubes — Compute Shader 实现
 *
 * 用 OpenGL 4.3 Compute Shader 并行提取等值面，比 CPU 版快 10-50×。
 *
 * 算法：
 *   1. 上传体数据为 3D 纹理 (r8ui)
 *   2. 预分配 SSBO（顶点 + 法线 + 原子计数器）
 *   3. Dispatch compute shader（每个线程处理一个体素）
 *   4. 读回结果，构建 MeshData
 *
 * 注意：必须在 OpenGL 上下文创建后调用。
 *       需要 OpenGL 4.3+ (GL_COMPUTE_SHADER)。
 */

#include "mesh.h"
#include "marching_cubes.hpp"  // 复用 edgeTable / triTable

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <cstdint>
#include <cmath>

namespace MeshRender {

// ============================================================
// 内部工具
// ============================================================

namespace detail {

/** 读取文件内容 */
inline std::string readFileContent(const std::string& path) {
    FILE* f = nullptr;
    fopen_s(&f, path.c_str(), "rb");
    if (!f) {
        std::cerr << "[GPU MC] 无法打开文件: " << path << std::endl;
        return "";
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string content(len, '\0');
    fread(&content[0], 1, len, f);
    fclose(f);
    return content;
}

/**
 * 生成 edgeTable 数据（用于上传到 SSBO）
 */
inline std::vector<int> getEdgeTableData() {
    return std::vector<int>(edgeTable, edgeTable + 256);
}

/**
 * 生成 triTable 展平数据（用于上传到 SSBO）
 */
inline std::vector<int> getTriTableData() {
    std::vector<int> data(4096);
    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 16; j++)
            data[i * 16 + j] = triTable[i][j];
    return data;
}

/**
 * 编译 compute shader 并链接为 program
 * @param shaderDir 着色器目录（包含 marching_cubes.comp）
 * @return program ID, 0 表示失败
 */
inline GLuint compileComputeShader(const std::string& shaderDir) {
    // 读取 compute shader 源码
    std::string compPath = shaderDir + "/marching_cubes.comp";
    std::string compSrc = readFileContent(compPath);
    if (compSrc.empty()) {
        std::cerr << "[GPU MC] 无法读取 compute shader: " << compPath << std::endl;
        return 0;
    }

    // 前置 #version（查找表通过 SSBO 上传，不再内联到 GLSL）
    std::string prefix =
        "#version 430 core\n"
        "#extension GL_ARB_compute_shader : enable\n"
        "#extension GL_ARB_shader_storage_buffer_object : enable\n"
        "#extension GL_ARB_shader_image_load_store : enable\n";

    std::string fullSrc = prefix + compSrc;

    // 编译
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    const char* src = fullSrc.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(shader, 4096, nullptr, log);
        std::cerr << "[GPU MC] Compute shader 编译失败:\n" << log << std::endl;
        glDeleteShader(shader);
        return 0;
    }

    // 链接
    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glDeleteShader(shader);

    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(program, 4096, nullptr, log);
        std::cerr << "[GPU MC] Program 链接失败:\n" << log << std::endl;
        glDeleteProgram(program);
        return 0;
    }

    return program;
}

} // namespace detail

// ============================================================
// GPU Marching Cubes 主函数
// ============================================================

/**
 * @brief GPU Marching Cubes 等值面提取
 *
 * @param buffer       体数据 (uint8_t)
 * @param w,h,d        体数据尺寸
 * @param spacing      物理间距 [3]
 * @param origin       原点 [3]
 * @param isovalue     等值面阈值 (8-bit)
 * @param spacingScale 间距缩放因子
 * @param shaderDir    着色器目录路径
 * @return MeshData    提取的网格（非索引，每个三角形 3 个独立顶点）
 */
inline MeshData extractIsosurfaceGPU(const std::vector<uint8_t>& buffer,
                                      int w, int h, int d,
                                      const double spacing[3],
                                      const double origin[3],
                                      float isovalue,
                                      float spacingScale,
                                      const std::string& shaderDir) {
    MeshData mesh;
    mesh.name = "Isosurface_GPU";

    if (buffer.empty() || w <= 0 || h <= 0 || d <= 0) {
        std::cerr << "[GPU MC] 体数据为空" << std::endl;
        return mesh;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // --- 1. 编译 compute shader ---
    static GLuint s_program = 0;
    static std::string s_lastDir;
    if (s_program == 0 || s_lastDir != shaderDir) {
        s_program = detail::compileComputeShader(shaderDir);
        s_lastDir = shaderDir;
        if (s_program == 0) {
            std::cerr << "[GPU MC] 着色器编译失败，回退到 CPU" << std::endl;
            return mesh;
        }
    }

    // --- 2. 创建 3D 纹理 ---
    GLuint volumeTex = 0;
    glGenTextures(1, &volumeTex);
    glBindTexture(GL_TEXTURE_3D, volumeTex);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R8UI, w, h, d, 0,
                 GL_RED_INTEGER, GL_UNSIGNED_BYTE, buffer.data());
    glBindTexture(GL_TEXTURE_3D, 0);

    // --- 3. 预分配 SSBO ---
    // 估算最大顶点数：每个体素最多 5 个三角形 = 15 个顶点
    // 但实际只有 ~10% 体素产生三角形，所以 15× 是安全上界
    size_t numVoxels = (size_t)(w - 1) * (h - 1) * (d - 1);
    size_t maxVerts = numVoxels * 15;
    // 限制最大分配量（避免显存爆炸）
    const size_t MAX_ALLOC = 12000000;  // 1200 万顶点上限
    if (maxVerts > MAX_ALLOC) maxVerts = MAX_ALLOC;

    GLuint vboSSBO = 0, nboSSBO = 0, counterSSBO = 0;
    GLuint edgeTableSSBO = 0, triTableSSBO = 0;

    // 顶点缓冲 (vec4 = 16 bytes)
    glGenBuffers(1, &vboSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vboSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, maxVerts * sizeof(glm::vec4),
                 nullptr, GL_DYNAMIC_COPY);

    // 法线缓冲 (vec4 = 16 bytes)
    glGenBuffers(1, &nboSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, nboSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, maxVerts * sizeof(glm::vec4),
                 nullptr, GL_DYNAMIC_COPY);

    // 原子计数器
    glGenBuffers(1, &counterSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, counterSSBO);
    uint32_t zero = 0;
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(uint32_t), &zero, GL_DYNAMIC_COPY);

    // 查找表 SSBO（binding 4 = edgeTable, binding 5 = triTableFlat）
    auto edgeData = detail::getEdgeTableData();
    glGenBuffers(1, &edgeTableSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, edgeTableSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, edgeData.size() * sizeof(int),
                 edgeData.data(), GL_STATIC_READ);

    auto triData = detail::getTriTableData();
    glGenBuffers(1, &triTableSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, triTableSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, triData.size() * sizeof(int),
                 triData.data(), GL_STATIC_READ);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    auto t1 = std::chrono::high_resolution_clock::now();

    // --- 4. Dispatch ---
    glUseProgram(s_program);

    // 绑定 3D 纹理到 image unit 0
    glBindImageTexture(0, volumeTex, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R8UI);

    // 绑定 SSBO
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vboSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, nboSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, counterSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, edgeTableSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, triTableSSBO);

    // 设置 uniforms
    GLint loc;
    loc = glGetUniformLocation(s_program, "volumeW");
    if (loc >= 0) glUniform1i(loc, w);
    loc = glGetUniformLocation(s_program, "volumeH");
    if (loc >= 0) glUniform1i(loc, h);
    loc = glGetUniformLocation(s_program, "volumeD");
    if (loc >= 0) glUniform1i(loc, d);
    loc = glGetUniformLocation(s_program, "uIsovalue");
    if (loc >= 0) glUniform1f(loc, isovalue);
    loc = glGetUniformLocation(s_program, "uSx");
    if (loc >= 0) glUniform1f(loc, (float)spacing[0] * spacingScale);
    loc = glGetUniformLocation(s_program, "uSy");
    if (loc >= 0) glUniform1f(loc, (float)spacing[1] * spacingScale);
    loc = glGetUniformLocation(s_program, "uSz");
    if (loc >= 0) glUniform1f(loc, (float)spacing[2] * spacingScale);
    loc = glGetUniformLocation(s_program, "uOx");
    if (loc >= 0) glUniform1f(loc, (float)origin[0]);
    loc = glGetUniformLocation(s_program, "uOy");
    if (loc >= 0) glUniform1f(loc, (float)origin[1]);
    loc = glGetUniformLocation(s_program, "uOz");
    if (loc >= 0) glUniform1f(loc, (float)origin[2]);
    loc = glGetUniformLocation(s_program, "uMaxVertices");
    if (loc >= 0) glUniform1ui(loc, (uint32_t)maxVerts);

    // 计算工作组数
    // local_size = 8×8×8, 所以 dispatch = ceil((W-1)/8) × ceil((H-1)/8) × ceil((D-1)/8)
    int groupsX = (w + 6) / 8;  // (w-1+7)/8 = (w+6)/8
    int groupsY = (h + 6) / 8;
    int groupsZ = (d + 6) / 8;

    glDispatchCompute(groupsX, groupsY, groupsZ);

    // 等待完成
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT);
    glFinish();

    auto t2 = std::chrono::high_resolution_clock::now();

    // --- 5. 读回结果 ---
    // 读取顶点计数
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, counterSSBO);
    uint32_t actualVerts = 0;
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &actualVerts);

    if (actualVerts == 0) {
        std::cerr << "[GPU MC] 未提取到任何顶点！" << std::endl;
        // 清理
        glDeleteTextures(1, &volumeTex);
        glDeleteBuffers(1, &vboSSBO);
        glDeleteBuffers(1, &nboSSBO);
        glDeleteBuffers(1, &counterSSBO);
        glDeleteBuffers(1, &edgeTableSSBO);
        glDeleteBuffers(1, &triTableSSBO);
        return mesh;
    }

    if (actualVerts > maxVerts) {
        std::cerr << "[GPU MC] 警告: 顶点数 " << actualVerts << " 超过缓冲区 " << maxVerts
                  << "，结果可能不完整" << std::endl;
        actualVerts = (uint32_t)maxVerts;
    }

    // 读回顶点数据
    std::vector<glm::vec4> vertData(actualVerts);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vboSSBO);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                       actualVerts * sizeof(glm::vec4), vertData.data());

    // 读回法线数据
    std::vector<glm::vec4> normData(actualVerts);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, nboSSBO);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                       actualVerts * sizeof(glm::vec4), normData.data());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    auto t3 = std::chrono::high_resolution_clock::now();

    // --- 6. 构建 MeshData ---
    mesh.vertices.resize(actualVerts);
    mesh.indices.resize(actualVerts);

    // 计算纹理坐标归一化用
    float sx = (float)spacing[0] * spacingScale;
    float sy = (float)spacing[1] * spacingScale;
    float sz = (float)spacing[2] * spacingScale;
    float totalW = w * sx + 1e-6f;
    float totalH = h * sy + 1e-6f;

    for (uint32_t i = 0; i < actualVerts; i++) {
        Vertex& v = mesh.vertices[i];
        v.position[0] = vertData[i].x;
        v.position[1] = vertData[i].y;
        v.position[2] = vertData[i].z;

        v.normal[0] = normData[i].x;
        v.normal[1] = normData[i].y;
        v.normal[2] = normData[i].z;

        // 纹理坐标用归一化的物理坐标
        v.texcoord[0] = vertData[i].x / totalW;
        v.texcoord[1] = vertData[i].y / totalH;

        // 切线/副切线留空，后续 computeTangents 会计算
        v.tangent[0] = v.tangent[1] = v.tangent[2] = 0;
        v.bitangent[0] = v.bitangent[1] = v.bitangent[2] = 0;

        mesh.indices[i] = i;  // 非索引：顺序索引
    }

    // --- 7. 清理 GPU 资源 ---
    glDeleteTextures(1, &volumeTex);
    glDeleteBuffers(1, &vboSSBO);
    glDeleteBuffers(1, &nboSSBO);
    glDeleteBuffers(1, &counterSSBO);
    glDeleteBuffers(1, &edgeTableSSBO);
    glDeleteBuffers(1, &triTableSSBO);

    // --- 8. 打印计时信息 ---
    auto msSetup    = std::chrono::duration<double, std::milli>(t1 - t0).count();
    auto msCompute  = std::chrono::duration<double, std::milli>(t2 - t1).count();
    auto msReadback = std::chrono::duration<double, std::milli>(t3 - t2).count();
    auto msTotal    = std::chrono::duration<double, std::milli>(t3 - t0).count();

    std::cout << "  [GPU MC] 提取完成" << std::endl;
    std::cout << "    顶点数: " << actualVerts << std::endl;
    std::cout << "    三角形数: " << actualVerts / 3 << std::endl;
    std::cout << "    耗时: 准备=" << msSetup << "ms"
              << " | 计算=" << msCompute << "ms"
              << " | 读回=" << msReadback << "ms"
              << " | 总计=" << msTotal << "ms" << std::endl;

    return mesh;
}

} // namespace MeshRender
