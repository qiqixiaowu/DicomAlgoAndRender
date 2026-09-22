/**
 * @file    gl_renderer.cpp
 * @brief   OpenGL 渲染器集成 实现
 */

#include "render/opengl/gl_renderer.h"
#include <iostream>

namespace NailPrint3D {

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
    shader_.setVec4("uBaseColor", glm::vec4(0.92f, 0.82f, 0.78f, 1.0f));

    shader_.setInt("uPatternMode", 0);
    shader_.setInt("uTextureEnabled", 0);
    shader_.setFloat("uOpacity", 1.0f);
    shader_.setFloat("uReliefHeight", 0.0f);
    shader_.setFloat("uIridescenceIntensity", 0.0f);
    shader_.setFloat("uTime", (float)glfwGetTime());
    shader_.setFloat("uUVAspect", 1.0f);
    shader_.setInt("uUVCorrectMode", 0);

    if (wireframe_)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    else
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    mesh.drawTriangles();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void NailMeshRenderer::renderWithTexture(const GLNailMesh& mesh, const RenderCamera& camera,
                                           const glm::vec3& lightDir, const GLTexture& texture,
                                           RenderPattern pattern, const TextureTransform& texXform) {
    shader_.use();
    shader_.setMat4("uModel", glm::mat4(1.0f));
    shader_.setMat4("uView", camera.getViewMatrix());
    shader_.setMat4("uProjection", camera.getProjectionMatrix());
    shader_.setVec3("uLightDir", glm::normalize(lightDir));
    shader_.setVec3("uViewPos", camera.getPosition());
    shader_.setVec4("uBaseColor", glm::vec4(0.92f, 0.82f, 0.78f, 1.0f));

    texture.bind(0);
    shader_.setInt("uTexture", 0);
    shader_.setInt("uTextureEnabled", 1);
    shader_.setInt("uPatternMode", (int)pattern);

    shader_.setFloat("uTexOffsetX", texXform.offsetX);
    shader_.setFloat("uTexOffsetY", texXform.offsetY);
    shader_.setFloat("uTexScale", texXform.scale);
    shader_.setFloat("uTexRotation", texXform.rotation);
    shader_.setFloat("uOpacity", texXform.opacity);
    shader_.setInt("uBlendMode", texXform.blendMode);
    shader_.setFloat("uReliefHeight", texXform.reliefHeight);
    shader_.setFloat("uIridescenceIntensity", (int)pattern == 5 ? 1.0f : 0.0f);
    shader_.setFloat("uTime", (float)glfwGetTime());
    shader_.setFloat("uUVAspect", texXform.uvAspect);
    shader_.setInt("uUVCorrectMode", texXform.uvCorrectMode);

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
    // 打印配置默认值（与 PrintConfig 默认一致）
    shader_.setFloat("uLayerHeight", 0.08f);
    shader_.setFloat("uBaseThickness", 0.3f);
    shader_.setInt("uColorCount", 4);
    // 默认调色板
    glm::vec3 defaultPalette[4] = {
        glm::vec3(1.0f, 0.3f, 0.4f),
        glm::vec3(0.9f, 0.6f, 0.2f),
        glm::vec3(0.4f, 0.7f, 0.9f),
        glm::vec3(0.6f, 0.3f, 0.8f)
    };
    for (int i = 0; i < 4; i++) {
        shader_.setVec3("uPalette[" + std::to_string(i) + "]", defaultPalette[i]);
    }
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

    for (int i = 0; i < (int)palette.size() && i < 16; i++) {
        shader_.setVec4("uPalette[" + std::to_string(i) + "]",
                         glm::vec4(palette[i].r, palette[i].g, palette[i].b, 1.0f));
    }
    shader_.setInt("uPaletteSize", (int)palette.size());

    // 打印配置 — 统一色彩管理
    shader_.setFloat("uLayerHeight", 0.08f);
    shader_.setFloat("uBaseThickness", 0.3f);
    shader_.setInt("uColorCount", (int)palette.size());

    mesh.drawTriangles();
}

// ============================================================
// NormalRenderer
// ============================================================

NormalRenderer::NormalRenderer() {}

NormalRenderer::~NormalRenderer() {}

bool NormalRenderer::init(const std::string& shaderDir) {
    bool ok = shader_.loadFromFiles(shaderDir + "/nail_normal.vert",
                                     shaderDir + "/nail_normal.frag");
    if (!ok) std::cerr << "[Renderer] 无法加载法线着色器" << std::endl;
    return ok;
}

void NormalRenderer::render(const GLNailMesh& mesh, const RenderCamera& camera) {
    shader_.use();
    shader_.setMat4("uModel", glm::mat4(1.0f));
    shader_.setMat4("uView", camera.getViewMatrix());
    shader_.setMat4("uProjection", camera.getProjectionMatrix());
    mesh.drawTriangles();
}

// ============================================================
// PrintPreviewRenderer
// ============================================================

PrintPreviewRenderer::PrintPreviewRenderer() {}

PrintPreviewRenderer::~PrintPreviewRenderer() {}

bool PrintPreviewRenderer::init(const std::string& shaderDir) {
    bool ok = shader_.loadFromFiles(shaderDir + "/nail_print_preview.vert",
                                     shaderDir + "/nail_print_preview.frag");
    if (!ok) std::cerr << "[Renderer] 无法加载打印预览着色器" << std::endl;
    return ok;
}

void PrintPreviewRenderer::render(const GLNailMesh& mesh, const RenderCamera& camera,
                                    const glm::vec3& lightDir,
                                    const std::vector<ColorRGBf>& palette,
                                    float layerHeight, float baseThickness,
                                    int colorCount) {
    shader_.use();
    shader_.setMat4("uModel", glm::mat4(1.0f));
    shader_.setMat4("uView", camera.getViewMatrix());
    shader_.setMat4("uProjection", camera.getProjectionMatrix());
    shader_.setVec3("uLightDir", glm::normalize(lightDir));

    shader_.setFloat("uLayerHeight", layerHeight);
    shader_.setFloat("uBaseThickness", baseThickness);
    shader_.setFloat("uTopCoatThickness", 0.1f);
    shader_.setInt("uBaseLayers", 1);
    shader_.setInt("uTopCoatLayers", 1);
    shader_.setInt("uColorCount", colorCount);

    for (int i = 0; i < (int)palette.size() && i < 16; i++) {
        shader_.setVec3("uPalette[" + std::to_string(i) + "]",
                         glm::vec3(palette[i].r, palette[i].g, palette[i].b));
    }

    mesh.drawTriangles();
}

// ============================================================
// NailRenderer
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
    bool ok4 = normalRenderer_.init(shaderDir);
    bool ok5 = printPreviewRenderer_.init(shaderDir);
    return ok1 && ok2 && ok3 && ok4 && ok5;
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
            normalRenderer_.render(mesh, camera);
            break;
        case RenderMode::SlicePreview:
            sliceRenderer_.render(mesh, camera, currentLayer_);
            break;
        case RenderMode::ColorPreview:
            colorRenderer_.render(mesh, camera, palette_);
            break;
        case RenderMode::PrintPreview:
            printPreviewRenderer_.render(mesh, camera, lightDir_,
                                          palette_,
                                          printLayerHeight_,
                                          printBaseThickness_,
                                          printColorCount_);
            break;
        case RenderMode::PatternPreview:
            meshRenderer_.renderWithTexture(mesh, camera, lightDir_,
                                             patternTexture_, pattern_, texXform_);
            break;
    }
}

void NailRenderer::setRenderMode(RenderMode mode) { mode_ = mode; }
void NailRenderer::setPalette(const std::vector<ColorRGBf>& palette) { palette_ = palette; }
void NailRenderer::setCurrentLayer(int layer) { currentLayer_ = layer; }
void NailRenderer::setLightDir(const glm::vec3& dir) { lightDir_ = dir; }

} // namespace NailPrint3D
