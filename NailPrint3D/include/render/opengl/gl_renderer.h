#pragma once
/**
 * @file    gl_renderer.h
 * @brief   OpenGL 渲染器集成（网格/切片/颜色/图案预览）
 *
 * 依赖：render/ (camera, shader, texture, mesh_buffer), core/ (render_types, color_types)
 *       GLAD, GLFW, GLM
 * 注意：不依赖 slicing/ 或 reconstruction/
 */

#include "render/camera.h"
#include "render/shader.h"
#include "render/texture.h"
#include "render/mesh_buffer.h"
#include "core/render_types.h"
#include "core/color_types.h"

#include <glad/glad.h>
#include <glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <string>
#include <vector>

namespace NailPrint3D {

// ============================================================
// 渲染模式 — 使用 core/render_types.h 中的 RenderMode，不重复定义
// ============================================================

// ============================================================
// 网格渲染器
// ============================================================

/** @brief 网格渲染器（Blinn-Phong光照 + 多模式纹理） */
class NailMeshRenderer {
public:
    NailMeshRenderer();
    ~NailMeshRenderer();

    bool init(const std::string& shaderDir);

    void render(const GLNailMesh& mesh, const RenderCamera& camera,
                const glm::vec3& lightDir);

    void renderWithTexture(const GLNailMesh& mesh, const RenderCamera& camera,
                           const glm::vec3& lightDir, const GLTexture& texture,
                           RenderPattern pattern, const TextureTransform& texXform);

    void setWireframe(bool wireframe) { wireframe_ = wireframe; }

private:
    NailShader shader_;
    bool wireframe_ = false;
};

// ============================================================
// 切片预览渲染器
// ============================================================

/** @brief 切片预览渲染器（使用通用线段，不依赖 SliceLayer） */
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

    void setPatternTexture(const GLTexture& tex) { patternTexture_ = tex; }
    void setPattern(RenderPattern p) { pattern_ = p; }
    void setTextureTransform(const TextureTransform& t) { texXform_ = t; }
    TextureTransform getTexTransform() const { return texXform_; }

    RenderPattern getPattern() const { return pattern_; }
    RenderMode getRenderMode() const { return mode_; }

private:
    NailMeshRenderer meshRenderer_;
    SlicePreviewRenderer sliceRenderer_;
    ColorPreviewRenderer colorRenderer_;

    RenderMode mode_ = RenderMode::Solid;
    std::string shaderDir_;
    int currentLayer_ = 0;
    glm::vec3 lightDir_ = glm::vec3(0.5f, 1.0f, 0.3f);
    std::vector<ColorRGBf> palette_;

    GLTexture patternTexture_;
    RenderPattern pattern_ = RenderPattern::Procedural;
    TextureTransform texXform_;
};

} // namespace NailPrint3D
