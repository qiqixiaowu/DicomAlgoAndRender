#pragma once
/**
 * @file    gl_mesh.h
 * @brief   OpenGL 网格资源封装（VAO/VBO/EBO 管理）
 */

#include "mesh.h"
#include <glad/glad.h>

namespace MeshRender {

/** @brief GPU 端网格资源 */
struct GLMesh {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLsizei indexCount = 0;
    GLsizei vertexCount = 0;
    AABB aabb;
    std::string name;

    /** 上传到 GPU */
    void upload(const MeshData& mesh) {
        name = mesh.name;
        indexCount = (GLsizei)mesh.indexCount();
        vertexCount = (GLsizei)mesh.vertexCount();
        aabb = computeAABB(mesh);

        if (vao) { glDeleteVertexArrays(1, &vao); glDeleteBuffers(1, &vbo); glDeleteBuffers(1, &ebo); }

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ebo);

        glBindVertexArray(vao);

        // VBO
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, mesh.vertices.size() * sizeof(Vertex),
                     mesh.vertices.data(), GL_STATIC_DRAW);

        // EBO
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size() * sizeof(uint32_t),
                     mesh.indices.data(), GL_STATIC_DRAW);

        // 顶点属性布局
        // position = location 0, offset 0
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position));

        // normal = location 1, offset 12
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));

        // texcoord = location 2, offset 24
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texcoord));

        // tangent = location 3, offset 32
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, tangent));

        // bitangent = location 4, offset 44
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, bitangent));

        glBindVertexArray(0);
    }

    /** 绘制 */
    void draw() const {
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

    /** 绘制线框 */
    void drawWireframe() const {
        glBindVertexArray(vao);
        glDrawElements(GL_LINES, indexCount, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

    /** 释放 */
    void release() {
        if (vao) { glDeleteVertexArrays(1, &vao); vao = 0; }
        if (vbo) { glDeleteBuffers(1, &vbo); vbo = 0; }
        if (ebo) { glDeleteBuffers(1, &ebo); ebo = 0; }
        indexCount = vertexCount = 0;
    }
};

} // namespace MeshRender
