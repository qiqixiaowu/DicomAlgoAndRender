/**
 * @file    nail_renderer.cpp
 * @brief   3D美甲打印 — OpenGL 渲染器 实现
 */

#include "nail_renderer.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

namespace NailPrint3D {

// ============================================================
// RenderCamera
// ============================================================

RenderCamera::RenderCamera()
    : target_(0, 0, 0)
    , distance_(5.0f)
    , angleX_(-45.0f)   // 俯仰角：从斜上方看
    , angleY_(0.0f)     // 方位角：正面
    , fov_(45.0f)
    , aspect_(1.0f)
    , near_(0.1f)
    , far_(100.0f) {
    updatePosition();
}

void RenderCamera::orbit(float dx, float dy) {
    angleY_ += dx * 0.5f;   // 水平拖动 = 方位角
    angleX_ += dy * 0.5f;   // 垂直拖动 = 俯仰角
    angleX_ = std::clamp(angleX_, -89.0f, 89.0f);
    updatePosition();
}

void RenderCamera::pan(float dx, float dy) {
    glm::vec3 right = glm::normalize(glm::cross(front_, up_));
    glm::vec3 up = glm::normalize(up_);
    target_ -= right * dx * 0.01f * distance_;
    target_ -= up * dy * 0.01f * distance_;  // Z 轴向上，鼠标向下 = 视角向下
    updatePosition();
}

void RenderCamera::zoom(float delta) {
    distance_ *= (1.0f + delta * 0.1f);
    distance_ = std::clamp(distance_, 0.5f, 50.0f);
    updatePosition();
}

void RenderCamera::setAspect(float aspect) {
    aspect_ = aspect;
}

void RenderCamera::setTarget(const glm::vec3& target) {
    target_ = target;
    updatePosition();
}

void RenderCamera::setDistance(float dist) {
    distance_ = std::clamp(dist, 0.5f, 50.0f);
    updatePosition();
}

glm::mat4 RenderCamera::getViewMatrix() const {
    return glm::lookAt(position_, target_, up_);
}

glm::mat4 RenderCamera::getProjectionMatrix() const {
    return glm::perspective(glm::radians(fov_), aspect_, near_, far_);
}

void RenderCamera::updatePosition() {
    // 美甲网格在 XY 平面上，Z 是弧度方向（法线）
    // 使用球面坐标，up = Z 轴
    float radX = glm::radians(angleX_);  // 俯仰角
    float radY = glm::radians(angleY_);  // 方位角
    // position = target + distance * (cos(pitch)*cos(yaw), cos(pitch)*sin(yaw), sin(pitch))
    position_.x = target_.x + distance_ * std::cos(radX) * std::cos(radY);
    position_.y = target_.y + distance_ * std::cos(radX) * std::sin(radY);
    position_.z = target_.z + distance_ * std::sin(radX);
    front_ = glm::normalize(target_ - position_);
    up_ = glm::vec3(0, 0, 1);  // Z 轴为 up
}

// ============================================================
// NailShader
// ============================================================

NailShader::NailShader() : program_(0) {}

NailShader::~NailShader() {
    if (program_) glDeleteProgram(program_);
}

bool NailShader::loadFromFiles(const std::string& vertPath, const std::string& fragPath) {
    std::string vertSrc, fragSrc;
    if (!readFile(vertPath, vertSrc)) return false;
    if (!readFile(fragPath, fragSrc)) return false;
    return loadFromSource(vertSrc, fragSrc);
}

bool NailShader::loadFromSource(const std::string& vertSrc, const std::string& fragSrc) {
    GLuint vert = compileShader(GL_VERTEX_SHADER, vertSrc);
    if (!vert) return false;
    GLuint frag = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    if (!frag) { glDeleteShader(vert); return false; }

    program_ = glCreateProgram();
    glAttachShader(program_, vert);
    glAttachShader(program_, frag);
    glLinkProgram(program_);

    GLint success;
    glGetProgramiv(program_, GL_LINK_STATUS, &success);
    if (!success) {
        char log[1024];
        glGetProgramInfoLog(program_, sizeof(log), nullptr, log);
        std::cerr << "[Shader] 链接失败:\n" << log << std::endl;
        glDeleteShader(vert);
        glDeleteShader(frag);
        glDeleteProgram(program_);
        program_ = 0;
        return false;
    }

    glDeleteShader(vert);
    glDeleteShader(frag);
    return true;
}

void NailShader::use() const { glUseProgram(program_); }

void NailShader::setMat4(const std::string& name, const glm::mat4& m) const {
    GLint loc = glGetUniformLocation(program_, name.c_str());
    glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(m));
}

void NailShader::setMat3(const std::string& name, const glm::mat3& m) const {
    GLint loc = glGetUniformLocation(program_, name.c_str());
    glUniformMatrix3fv(loc, 1, GL_FALSE, glm::value_ptr(m));
}

void NailShader::setVec3(const std::string& name, const glm::vec3& v) const {
    GLint loc = glGetUniformLocation(program_, name.c_str());
    glUniform3fv(loc, 1, glm::value_ptr(v));
}

void NailShader::setVec4(const std::string& name, const glm::vec4& v) const {
    GLint loc = glGetUniformLocation(program_, name.c_str());
    glUniform4fv(loc, 1, glm::value_ptr(v));
}

void NailShader::setFloat(const std::string& name, float v) const {
    GLint loc = glGetUniformLocation(program_, name.c_str());
    glUniform1f(loc, v);
}

void NailShader::setInt(const std::string& name, int v) const {
    GLint loc = glGetUniformLocation(program_, name.c_str());
    glUniform1i(loc, v);
}

void NailShader::setColor(const std::string& name, const ColorRGBf& c) const {
    setVec4(name, glm::vec4(c.r, c.g, c.b, 1.0f));
}

GLuint NailShader::compileShader(GLenum type, const std::string& src) {
    GLuint shader = glCreateShader(type);
    const char* srcPtr = src.c_str();
    glShaderSource(shader, 1, &srcPtr, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[1024];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::cerr << "[Shader] 编译失败:\n" << log << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

bool NailShader::readFile(const std::string& path, std::string& content) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[Shader] 无法打开文件: " << path << std::endl;
        return false;
    }
    std::stringstream ss;
    ss << file.rdbuf();
    content = ss.str();
    return true;
}

// ============================================================
// GLNailMesh
// ============================================================

GLNailMesh::GLNailMesh() : vao_(0), vbo_(0), ebo_(0), indexCount_(0) {}

GLNailMesh::~GLNailMesh() { destroy(); }

void GLNailMesh::upload(const Mesh& mesh) {
    destroy();

    // 构建顶点数据: pos(3) + normal(3) + color(3) + uv(2) = 11 floats per vertex
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
        // 顶点颜色
        if (i < mesh.colors.size()) {
            vertices.push_back(mesh.colors[i].r);
            vertices.push_back(mesh.colors[i].g);
            vertices.push_back(mesh.colors[i].b);
        } else {
            vertices.push_back(0.9f); vertices.push_back(0.75f); vertices.push_back(0.8f);
        }
        // UV 坐标
        if (i < mesh.uvs.size()) {
            vertices.push_back(mesh.uvs[i].u);
            vertices.push_back(mesh.uvs[i].v);
        } else {
            vertices.push_back(0); vertices.push_back(0);
        }
    }

    // 索引
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
    // 位置 (location = 0)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
    // 法线 (location = 1)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
    // 颜色 (location = 2)
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, (void*)(6 * sizeof(float)));
    // UV (location = 3)
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride, (void*)(9 * sizeof(float)));

    glBindVertexArray(0);
}

void GLNailMesh::uploadSlicePaths(const std::vector<SliceLayer>& layers) {
    destroy();

    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    unsigned int vIdx = 0;

    for (const auto& layer : layers) {
        float z = layer.zHeight;
        float color[3] = { 1.0f, 0.5f, 0.2f };  // 切片路径颜色

        auto addLine = [&](const Vec2& s, const Vec2& e) {
            vertices.insert(vertices.end(), { s.x, s.y, z, 0, 0, 1, color[0], color[1], color[2], 0, 0 });
            vertices.insert(vertices.end(), { e.x, e.y, z, 0, 0, 1, color[0], color[1], color[2], 0, 0 });
            indices.push_back(vIdx);
            indices.push_back(vIdx + 1);
            vIdx += 2;
        };

        for (const auto& seg : layer.perimeters) {
            if (!seg.travel) addLine(seg.start, seg.end);
        }
        for (const auto& seg : layer.infill) {
            if (!seg.travel) addLine(seg.start, seg.end);
        }
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

// ============================================================
// NailMeshRenderer
// ============================================================

NailMeshRenderer::NailMeshRenderer() : wireframe_(false) {}

NailMeshRenderer::~NailMeshRenderer() {}

bool NailMeshRenderer::init(const std::string& shaderDir) {
    bool ok = shader_.loadFromFiles(shaderDir + "/nail_mesh.vert",
                                     shaderDir + "/nail_mesh.frag");
    if (!ok) std::cerr << "[Renderer] 无法加载网格着色器" << std::endl;
    return ok;
}

void NailMeshRenderer::render(const GLNailMesh& mesh, const RenderCamera& camera,
                                const glm::vec3& lightDir) {
    shader_.use();
    shader_.setMat4("uModel", glm::mat4(1.0f));
    shader_.setMat4("uView", camera.getViewMatrix());
    shader_.setMat4("uProjection", camera.getProjectionMatrix());
    shader_.setVec3("uLightDir", glm::normalize(lightDir));
    shader_.setVec3("uViewPos", camera.getPosition());
    shader_.setVec4("uBaseColor", glm::vec4(0.85f, 0.75f, 0.8f, 1.0f));  // 美甲粉色调

    if (wireframe_)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    else
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    mesh.drawTriangles();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

// ============================================================
// SlicePreviewRenderer
// ============================================================

SlicePreviewRenderer::SlicePreviewRenderer() : currentLayer_(0) {}

SlicePreviewRenderer::~SlicePreviewRenderer() {}

bool SlicePreviewRenderer::init(const std::string& shaderDir) {
    bool ok = shader_.loadFromFiles(shaderDir + "/nail_slice_line.vert",
                                     shaderDir + "/nail_slice_line.frag");
    if (!ok) std::cerr << "[Renderer] 无法加载切片预览着色器" << std::endl;
    return ok;
}

void SlicePreviewRenderer::render(const GLNailMesh& pathMesh, const RenderCamera& camera,
                                    int currentLayer) {
    currentLayer_ = currentLayer;
    shader_.use();
    shader_.setMat4("uModel", glm::mat4(1.0f));
    shader_.setMat4("uView", camera.getViewMatrix());
    shader_.setMat4("uProjection", camera.getProjectionMatrix());
    shader_.setFloat("uCurrentLayer", (float)currentLayer);
    pathMesh.draw();
}

// ============================================================
// ColorPreviewRenderer
// ============================================================

ColorPreviewRenderer::ColorPreviewRenderer() {}

ColorPreviewRenderer::~ColorPreviewRenderer() {}

bool ColorPreviewRenderer::init(const std::string& shaderDir) {
    bool ok = shader_.loadFromFiles(shaderDir + "/nail_palette.vert",
                                     shaderDir + "/nail_palette.frag");
    if (!ok) std::cerr << "[Renderer] 无法加载颜色预览着色器" << std::endl;
    return ok;
}

void ColorPreviewRenderer::render(const GLNailMesh& mesh, const RenderCamera& camera,
                                    const std::vector<ColorRGBf>& palette) {
    shader_.use();
    shader_.setMat4("uModel", glm::mat4(1.0f));
    shader_.setMat4("uView", camera.getViewMatrix());
    shader_.setMat4("uProjection", camera.getProjectionMatrix());

    // 上传调色板
    for (int i = 0; i < (int)palette.size() && i < 16; i++) {
        shader_.setVec4("uPalette[" + std::to_string(i) + "]",
                         glm::vec4(palette[i].r, palette[i].g, palette[i].b, 1.0f));
    }
    shader_.setInt("uPaletteSize", (int)palette.size());

    mesh.drawTriangles();
}

// ============================================================
// NailRenderer (集成渲染器)
// ============================================================

NailRenderer::NailRenderer()
    : mode_(RenderMode::Solid)
    , shaderDir_("./shaders")
    , currentLayer_(0)
    , lightDir_(0.5f, 1.0f, 0.3f) {}

NailRenderer::~NailRenderer() {}

bool NailRenderer::init(const std::string& shaderDir) {
    shaderDir_ = shaderDir;
    bool ok1 = meshRenderer_.init(shaderDir);
    bool ok2 = sliceRenderer_.init(shaderDir);
    bool ok3 = colorRenderer_.init(shaderDir);
    return ok1 && ok2 && ok3;
}

void NailRenderer::render(const GLNailMesh& mesh, const RenderCamera& camera) {
    glClearColor(0.15f, 0.15f, 0.18f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    switch (mode_) {
        case RenderMode::Solid:
            meshRenderer_.render(mesh, camera, lightDir_);
            break;
        case RenderMode::Wireframe:
            meshRenderer_.setWireframe(true);
            meshRenderer_.render(mesh, camera, lightDir_);
            meshRenderer_.setWireframe(false);
            break;
        case RenderMode::Normal:
            meshRenderer_.render(mesh, camera, lightDir_);
            break;
        case RenderMode::SlicePreview:
            sliceRenderer_.render(mesh, camera, currentLayer_);
            break;
        case RenderMode::ColorPreview:
            colorRenderer_.render(mesh, camera, palette_);
            break;
    }
}

void NailRenderer::setRenderMode(RenderMode mode) { mode_ = mode; }
void NailRenderer::setPalette(const std::vector<ColorRGBf>& palette) { palette_ = palette; }
void NailRenderer::setCurrentLayer(int layer) { currentLayer_ = layer; }
void NailRenderer::setLightDir(const glm::vec3& dir) { lightDir_ = dir; }

} // namespace NailPrint3D
