#pragma once
/**
 * @file    nail_renderer.h
 * @brief   3D美甲打印 — OpenGL渲染与预览模块
 *
 * 功能：
 *  - 网格渲染（Blinn-Phong光照 + 菲涅尔光泽）
 *  - 切片预览（逐层路径可视化）
 *  - 颜色预览（调色板 + 纹理映射）
 *  - 相机控制（轨道 + 平移 + 缩放）
 */

#include "nail_types.h"
#include "nail_reconstruction.h"
#include "nail_slicer.h"
#include "nail_color.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <glad/glad.h>
#include <glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <string>
#include <vector>

namespace NailPrint3D {

// ============================================================
// 轨道相机
// ============================================================

/** @brief 轨道相机（鼠标旋转/平移/缩放） */
class RenderCamera {
public:
    RenderCamera();

    void orbit(float dx, float dy);
    void pan(float dx, float dy);
    void zoom(float delta);
    void setAspect(float aspect);
    void setTarget(const glm::vec3& target);
    void setDistance(float dist);

    glm::vec3 getPosition() const { return position_; }
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;

private:
    void updatePosition();

    glm::vec3 target_   = glm::vec3(0.0f);
    glm::vec3 position_ = glm::vec3(0.0f, 0.0f, 5.0f);
    glm::vec3 front_    = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 up_       = glm::vec3(0.0f, 1.0f, 0.0f);

    float distance_ = 5.0f;
    float angleX_   = -30.0f;
    float angleY_   = 45.0f;
    float fov_      = 45.0f;
    float aspect_   = 1.0f;
    float near_     = 0.1f;
    float far_      = 100.0f;
};

// ============================================================
// 着色器程序
// ============================================================

/** @brief GLSL着色器封装 */
class NailShader {
public:
    NailShader();
    ~NailShader();

    bool loadFromFiles(const std::string& vertPath, const std::string& fragPath);
    bool loadFromSource(const std::string& vertSrc, const std::string& fragSrc);

    void use() const;

    void setMat4(const std::string& name, const glm::mat4& m) const;
    void setMat3(const std::string& name, const glm::mat3& m) const;
    void setVec3(const std::string& name, const glm::vec3& v) const;
    void setVec4(const std::string& name, const glm::vec4& v) const;
    void setFloat(const std::string& name, float v) const;
    void setInt(const std::string& name, int v) const;
    void setColor(const std::string& name, const ColorRGBf& c) const;

private:
    GLuint program_ = 0;

    GLuint compileShader(GLenum type, const std::string& src);
    static bool readFile(const std::string& path, std::string& content);
};

// ============================================================
// GPU网格资源
// ============================================================

/** @brief GPU网格资源（顶点缓冲 + 索引缓冲） */
struct GLNailMesh {
    GLuint vao_ = 0, vbo_ = 0, ebo_ = 0;
    GLsizei indexCount_ = 0;

    GLNailMesh();
    ~GLNailMesh();

    /** @brief 上传三角网格 */
    void upload(const Mesh& mesh);

    /** @brief 上传切片路径（线段渲染） */
    void uploadSlicePaths(const std::vector<SliceLayer>& layers);

    /** @brief 绘制线段 */
    void draw() const;

    /** @brief 绘制三角形 */
    void drawTriangles() const;

    /** @brief 释放GPU资源 */
    void destroy();
};

// ============================================================
// 渲染模式
// ============================================================

enum class RenderMode {
    Solid,          ///< 实体渲染
    Wireframe,      ///< 线框
    Normal,         ///< 法线可视化
    SlicePreview,   ///< 切片预览
    ColorPreview    ///< 颜色预览
};

// ============================================================
// 网格渲染器
// ============================================================

/** @brief 网格渲染器（Blinn-Phong光照） */
class NailMeshRenderer {
public:
    NailMeshRenderer();
    ~NailMeshRenderer();

    bool init(const std::string& shaderDir);

    void render(const GLNailMesh& mesh, const RenderCamera& camera,
                const glm::vec3& lightDir);

    void setWireframe(bool wireframe) { wireframe_ = wireframe; }

private:
    NailShader shader_;
    bool wireframe_ = false;
};

// ============================================================
// 切片预览渲染器
// ============================================================

/** @brief 切片预览渲染器 */
class SlicePreviewRenderer {
public:
    SlicePreviewRenderer();
    ~SlicePreviewRenderer();

    bool init(const std::string& shaderDir);

    void render(const GLNailMesh& pathMesh, const RenderCamera& camera,
                int currentLayer);

private:
    NailShader shader_;
    int currentLayer_ = 0;
};

// ============================================================
// 颜色预览渲染器
// ============================================================

/** @brief 颜色预览渲染器（调色板映射） */
class ColorPreviewRenderer {
public:
    ColorPreviewRenderer();
    ~ColorPreviewRenderer();

    bool init(const std::string& shaderDir);

    void render(const GLNailMesh& mesh, const RenderCamera& camera,
                const std::vector<ColorRGBf>& palette);

private:
    NailShader shader_;
};

// ============================================================
// 主渲染器（集成）
// ============================================================

/** @brief 主渲染器（整合所有渲染功能） */
class NailRenderer {
public:
    NailRenderer();
    ~NailRenderer();

    bool init(const std::string& shaderDir);

    void render(const GLNailMesh& mesh, const RenderCamera& camera);

    void setRenderMode(RenderMode mode);
    void setPalette(const std::vector<ColorRGBf>& palette);
    void setCurrentLayer(int layer);
    void setLightDir(const glm::vec3& dir);

private:
    NailMeshRenderer meshRenderer_;
    SlicePreviewRenderer sliceRenderer_;
    ColorPreviewRenderer colorRenderer_;

    RenderMode mode_ = RenderMode::Solid;
    std::string shaderDir_;
    int currentLayer_ = 0;
    glm::vec3 lightDir_ = glm::vec3(0.5f, 1.0f, 0.3f);
    std::vector<ColorRGBf> palette_;
};

} // namespace NailPrint3D
