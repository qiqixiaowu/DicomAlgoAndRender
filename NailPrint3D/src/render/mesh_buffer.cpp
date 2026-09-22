/**
 * @file    mesh_buffer.cpp
 * @brief   GPU 网格资源 实现
 */

#include "render/mesh_buffer.h"

namespace NailPrint3D {

GLNailMesh::GLNailMesh() : vao_(0), vbo_(0), ebo_(0), indexCount_(0) {}

GLNailMesh::~GLNailMesh() { destroy(); }

void GLNailMesh::upload(const Mesh& mesh) {
    destroy();

    std::vector<float> vertices;
    vertices.reserve(mesh.vertices.size() * 11);
    for (size_t i = 0; i < mesh.vertices.size(); i++) {
        vertices.push_back(mesh.vertices[i].x);
        vertices.push_back(mesh.vertices[i].y);
        vertices.push_back(mesh.vertices[i].z);
        if (i < mesh.normals.size()) {
            vertices.push_back(mesh.normals[i].x);
            vertices.push_back(mesh.normals[i].y);
            vertices.push_back(mesh.normals[i].z);
        } else {
            vertices.push_back(0); vertices.push_back(0); vertices.push_back(1);
        }
        if (i < mesh.colors.size()) {
            vertices.push_back(mesh.colors[i].r);
            vertices.push_back(mesh.colors[i].g);
            vertices.push_back(mesh.colors[i].b);
        } else {
            vertices.push_back(0.92f); vertices.push_back(0.82f); vertices.push_back(0.78f);
        }
        if (i < mesh.uvs.size()) {
            vertices.push_back(mesh.uvs[i].u);
            vertices.push_back(mesh.uvs[i].v);
        } else {
            vertices.push_back(0); vertices.push_back(0);
        }
    }

    std::vector<unsigned int> indices;
    indices.reserve(mesh.triangles.size() * 3);
    for (const auto& tri : mesh.triangles) {
        indices.push_back(tri.idx[0]);
        indices.push_back(tri.idx[1]);
        indices.push_back(tri.idx[2]);
    }
    indexCount_ = (GLsizei)indices.size();

    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);

    glGenBuffers(1, &vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &ebo_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    int stride = 11 * sizeof(float);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride, (void*)(9 * sizeof(float)));

    glBindVertexArray(0);
}

void GLNailMesh::uploadLines(const std::vector<LineSegment3D>& segments) {
    destroy();

    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    unsigned int vIdx = 0;

    for (const auto& seg : segments) {
        float color[3] = { 1.0f, 0.5f, 0.2f };
        vertices.insert(vertices.end(), {
            seg.start.x, seg.start.y, seg.start.z, 0, 0, 1, color[0], color[1], color[2], 0, 0
        });
        vertices.insert(vertices.end(), {
            seg.end.x, seg.end.y, seg.end.z, 0, 0, 1, color[0], color[1], color[2], 0, 0
        });
        indices.push_back(vIdx);
        indices.push_back(vIdx + 1);
        vIdx += 2;
    }

    indexCount_ = (GLsizei)indices.size();

    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);

    glGenBuffers(1, &vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &ebo_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    int stride = 11 * sizeof(float);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride, (void*)(9 * sizeof(float)));

    glBindVertexArray(0);
}

void GLNailMesh::draw() const {
    if (!vao_ || indexCount_ == 0) return;
    glBindVertexArray(vao_);
    glDrawElements(GL_LINES, indexCount_, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void GLNailMesh::drawTriangles() const {
    if (!vao_ || indexCount_ == 0) return;
    glBindVertexArray(vao_);
    glDrawElements(GL_TRIANGLES, indexCount_, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void GLNailMesh::destroy() {
    if (vao_) { glDeleteVertexArrays(1, &vao_); vao_ = 0; }
    if (vbo_) { glDeleteBuffers(1, &vbo_); vbo_ = 0; }
    if (ebo_) { glDeleteBuffers(1, &ebo_); ebo_ = 0; }
    indexCount_ = 0;
}

} // namespace NailPrint3D
