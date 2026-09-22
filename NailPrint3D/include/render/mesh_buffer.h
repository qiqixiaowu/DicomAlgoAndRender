#pragma once
/**
 * @file    mesh_buffer.h
 * @brief   GPU 网格资源（顶点缓冲 + 索引缓冲）
 *
 * 依赖：GLAD, core/geometry.h
 * 注意：不依赖 slicing/ 或 reconstruction/ — 使用 LineSegment3D 代替 SliceLayer
 */

#include "core/geometry.h"
#include <glad/glad.h>
#include <vector>

namespace NailPrint3D {

/** @brief 3D 线段（用于切片路径上传，解耦 slicing 依赖） */
struct LineSegment3D {
    Vec3 start;
    Vec3 end;
};

/** @brief GPU网格资源（顶点缓冲 + 索引缓冲） */
struct GLNailMesh {
    GLuint vao_ = 0, vbo_ = 0, ebo_ = 0;
    GLsizei indexCount_ = 0;

    GLNailMesh();
    ~GLNailMesh();

    /** @brief 上传三角网格 */
    void upload(const Mesh& mesh);

    /** @brief 上传线段路径（替代原 uploadSlicePaths，解耦 slicing 依赖） */
    void uploadLines(const std::vector<LineSegment3D>& segments);

    /** @brief 绘制线段 */
    void draw() const;

    /** @brief 绘制三角形 */
    void drawTriangles() const;

    /** @brief 释放GPU资源 */
    void destroy();
};

} // namespace NailPrint3D
