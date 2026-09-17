#pragma once
/**
 * @file    deferred_renderer.h
 * @brief   延迟渲染管线
 *
 * 完整管线：
 *  1. Shadow Pass     → 阴影深度图
 *  2. Geometry Pass   → G-Buffer (位置/法线/反照率/材质)
 *  3. SSAO Pass       → 环境光遮蔽
 *  4. Lighting Pass   → PBR 光照 + 阴影
 *  5. Skybox Pass     → 程序化天空
 *  6. Wireframe Pass  → 线框/法线可视化
 */

#include "gl_mesh.h"
#include "mesh_camera.h"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <string>
#include <random>
#include <iostream>

namespace MeshRender {

// ============================================================
// 着色器程序封装
// ============================================================

class ShaderProgram {
public:
    GLuint id = 0;

    ShaderProgram() = default;

    bool loadFromFiles(const std::string& vertPath, const std::string& fragPath) {
        std::string vertSrc = readFile(vertPath);
        std::string fragSrc = readFile(fragPath);
        if (vertSrc.empty() || fragSrc.empty()) return false;
        return compileAndLink(vertSrc.c_str(), fragSrc.c_str());
    }

    bool loadFromFiles(const std::string& vertPath, const std::string& fragPath,
                       const std::string& geomPath) {
        std::string vertSrc = readFile(vertPath);
        std::string fragSrc = readFile(fragPath);
        std::string geomSrc = readFile(geomPath);
        if (vertSrc.empty() || fragSrc.empty()) return false;
        return compileAndLink(vertSrc.c_str(), fragSrc.c_str(),
                              geomSrc.empty() ? nullptr : geomSrc.c_str());
    }

    void use() const { glUseProgram(id); }

    void setMat4(const std::string& name, const glm::mat4& m) const {
        glUniformMatrix4fv(glGetUniformLocation(id, name.c_str()), 1, GL_FALSE, glm::value_ptr(m));
    }
    void setMat3(const std::string& name, const glm::mat3& m) const {
        glUniformMatrix3fv(glGetUniformLocation(id, name.c_str()), 1, GL_FALSE, glm::value_ptr(m));
    }
    void setVec3(const std::string& name, const glm::vec3& v) const {
        glUniform3fv(glGetUniformLocation(id, name.c_str()), 1, glm::value_ptr(v));
    }
    void setVec2(const std::string& name, const glm::vec2& v) const {
        glUniform2fv(glGetUniformLocation(id, name.c_str()), 1, glm::value_ptr(v));
    }
    void setFloat(const std::string& name, float v) const {
        glUniform1f(glGetUniformLocation(id, name.c_str()), v);
    }
    void setInt(const std::string& name, int v) const {
        glUniform1i(glGetUniformLocation(id, name.c_str()), v);
    }
    void setBool(const std::string& name, bool v) const {
        setInt(name, v ? 1 : 0);
    }

    void setVec3Array(const std::string& name, const std::vector<glm::vec3>& arr) const {
        if (arr.empty()) return;
        glUniform3fv(glGetUniformLocation(id, name.c_str()), (GLsizei)arr.size(), glm::value_ptr(arr[0]));
    }
    void setFloatArray(const std::string& name, const std::vector<float>& arr) const {
        glUniform1fv(glGetUniformLocation(id, name.c_str()), (GLsizei)arr.size(), arr.data());
    }
    void setIntArray(const std::string& name, const std::vector<int>& arr) const {
        glUniform1iv(glGetUniformLocation(id, name.c_str()), (GLsizei)arr.size(), arr.data());
    }

private:
    std::string readFile(const std::string& path) {
        FILE* f = nullptr;
        fopen_s(&f, path.c_str(), "rb");
        if (!f) { std::cerr << "无法打开着色器文件: " << path << std::endl; return ""; }
        fseek(f, 0, SEEK_END);
        long len = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::string content(len, '\0');
        fread(&content[0], 1, len, f);
        fclose(f);
        return content;
    }

    bool compileAndLink(const char* vertSrc, const char* fragSrc, const char* geomSrc = nullptr) {
        auto compile = [&](GLenum type, const char* src) -> GLuint {
            GLuint shader = glCreateShader(type);
            glShaderSource(shader, 1, &src, nullptr);
            glCompileShader(shader);
            GLint ok; glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
            if (!ok) {
                char log[1024];
                glGetShaderInfoLog(shader, 1024, nullptr, log);
                std::cerr << "着色器编译错误:\n" << log << std::endl;
                glDeleteShader(shader);
                return 0;
            }
            return shader;
        };

        GLuint vert = compile(GL_VERTEX_SHADER, vertSrc);
        GLuint frag = compile(GL_FRAGMENT_SHADER, fragSrc);
        GLuint geom = geomSrc ? compile(GL_GEOMETRY_SHADER, geomSrc) : 0;
        if (!vert || !frag) return false;

        id = glCreateProgram();
        glAttachShader(id, vert);
        glAttachShader(id, frag);
        if (geom) glAttachShader(id, geom);
        glLinkProgram(id);
        GLint ok; glGetProgramiv(id, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[1024];
            glGetProgramInfoLog(id, 1024, nullptr, log);
            std::cerr << "着色器链接错误:\n" << log << std::endl;
            return false;
        }
        glDeleteShader(vert);
        glDeleteShader(frag);
        if (geom) glDeleteShader(geom);
        return true;
    }
};

// ============================================================
// 光源定义
// ============================================================

enum class LightType { Directional = 0, Point = 1, Spot = 2 };

struct Light {
    LightType type = LightType::Directional;
    glm::vec3 position  = glm::vec3(5, 8, 5);
    glm::vec3 direction = glm::vec3(-0.5f, -0.8f, -0.5f);
    glm::vec3 color     = glm::vec3(1.0f, 0.95f, 0.85f);
    float intensity     = 3.0f;
    float range         = 20.0f;
    float innerCone     = 0.90f;  // cos(25°)
    float outerCone     = 0.80f;  // cos(36°)
};

// ============================================================
// 渲染对象
// ============================================================

struct RenderObject {
    GLMesh mesh;
    glm::mat4 modelMatrix = glm::mat4(1.0f);
    glm::vec3 baseColor   = glm::vec3(0.8f);
    float metallic        = 0.0f;
    float roughness       = 0.5f;
    float ao              = 1.0f;
    glm::vec3 emissive    = glm::vec3(0.0f);
    float emissiveStrength = 0.0f;
    bool visible          = true;
    bool castShadow       = true;
    std::string name;
};

// ============================================================
// G-Buffer 帧缓冲
// ============================================================

struct GBuffer {
    GLuint fbo = 0;
    GLuint textures[4] = {0};  // position, normal, albedoSpec, material
    GLuint depthTexture = 0;
    int width = 0, height = 0;

    void create(int w, int h) {
        width = w; height = h;
        if (fbo) destroy();

        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        // 4 个颜色附件
        GLenum attachments[4] = {
            GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
            GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3
        };

        // position (RGB16F)
        glGenTextures(4, textures);
        for (int i = 0; i < 4; i++) {
            glBindTexture(GL_TEXTURE_2D, textures[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, w, h, 0, GL_RGB, GL_FLOAT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glFramebufferTexture2D(GL_FRAMEBUFFER, attachments[i], GL_TEXTURE_2D, textures[i], 0);
        }

        // 深度
        glGenTextures(1, &depthTexture);
        glBindTexture(GL_TEXTURE_2D, depthTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTexture, 0);

        glDrawBuffers(4, attachments);
        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE)
            std::cerr << "G-Buffer 不完整! 状态: " << status << std::endl;

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void bind() { glBindFramebuffer(GL_FRAMEBUFFER, fbo); }
    void unbind() { glBindFramebuffer(GL_FRAMEBUFFER, 0); }

    void bindTextures() {
        for (int i = 0; i < 4; i++) {
            glActiveTexture(GL_TEXTURE0 + i);
            glBindTexture(GL_TEXTURE_2D, textures[i]);
        }
    }

    void destroy() {
        if (fbo) { glDeleteFramebuffers(1, &fbo); fbo = 0; }
        glDeleteTextures(4, textures);
        glDeleteTextures(1, &depthTexture);
        for (int i = 0; i < 4; i++) textures[i] = 0;
        depthTexture = 0;
    }
};

// ============================================================
// 阴影映射
// ============================================================

struct ShadowMap {
    GLuint fbo = 0;
    GLuint texture = 0;
    int width = 2048, height = 2048;
    glm::mat4 lightSpaceMatrix = glm::mat4(1.0f);

    void create(int w = 2048, int h = 2048) {
        width = w; height = h;
        if (fbo) destroy();

        glGenFramebuffers(1, &fbo);
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, w, h, 0,
                     GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texture, 0);
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void updateLightSpace(const Light& light, const glm::vec3& sceneCenter, float sceneRadius) {
        glm::mat4 lightProjection = glm::ortho(
            -sceneRadius, sceneRadius,
            -sceneRadius, sceneRadius,
            0.1f, sceneRadius * 4.0f);
        glm::vec3 lightPos = sceneCenter - light.direction * sceneRadius * 2.0f;
        glm::mat4 lightView = glm::lookAt(lightPos, sceneCenter, glm::vec3(0, 1, 0));
        lightSpaceMatrix = lightProjection * lightView;
    }

    void bind() { glBindFramebuffer(GL_FRAMEBUFFER, fbo); }
    void unbind() { glBindFramebuffer(GL_FRAMEBUFFER, 0); }
    void destroy() {
        if (fbo) { glDeleteFramebuffers(1, &fbo); fbo = 0; }
        if (texture) { glDeleteTextures(1, &texture); texture = 0; }
    }
};

// ============================================================
// SSAO
// ============================================================

struct SSAOBuffer {
    GLuint fbo = 0, blurFbo = 0;
    GLuint texture = 0, blurTexture = 0;
    GLuint noiseTexture = 0;
    std::vector<glm::vec3> samples;
    int kernelSize = 64;
    float radius = 0.5f;
    float bias = 0.025f;

    void create(int w, int h) {
        if (fbo) destroy();

        // 采样核
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::default_random_engine rng(42);
        samples.clear();
        for (int i = 0; i < kernelSize; i++) {
            glm::vec3 sample(
                dist(rng) * 2.0f - 1.0f,
                dist(rng) * 2.0f - 1.0f,
                dist(rng)
            );
            sample = glm::normalize(sample);
            float scale = (float)i / kernelSize;
            scale = 0.1f + scale * scale * 0.9f;  // 二次分布，靠近相机的采样更多
            sample *= scale;
            samples.push_back(sample);
        }

        // 噪声纹理
        std::vector<glm::vec3> noise(16);
        for (auto& n : noise) {
            n = glm::vec3(dist(rng) * 2.0f - 1.0f, dist(rng) * 2.0f - 1.0f, 0.0f);
        }
        glGenTextures(1, &noiseTexture);
        glBindTexture(GL_TEXTURE_2D, noiseTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 4, 4, 0, GL_RGB, GL_FLOAT, noise.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

        // SSAO FBO
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, w, h, 0, GL_RGB, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // Blur FBO
        glGenFramebuffers(1, &blurFbo);
        glBindFramebuffer(GL_FRAMEBUFFER, blurFbo);
        glGenTextures(1, &blurTexture);
        glBindTexture(GL_TEXTURE_2D, blurTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, w, h, 0, GL_RGB, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, blurTexture, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void destroy() {
        if (fbo) { glDeleteFramebuffers(1, &fbo); fbo = 0; }
        if (blurFbo) { glDeleteFramebuffers(1, &blurFbo); blurFbo = 0; }
        if (texture) { glDeleteTextures(1, &texture); texture = 0; }
        if (blurTexture) { glDeleteTextures(1, &blurTexture); blurTexture = 0; }
        if (noiseTexture) { glDeleteTextures(1, &noiseTexture); noiseTexture = 0; }
    }
};

// ============================================================
// 全屏四边形
// ============================================================

struct FullscreenQuad {
    GLuint vao = 0, vbo = 0;

    void create() {
        float vertices[] = {
            -1, -1, 0, 0,
             1, -1, 1, 0,
             1,  1, 1, 1,
            -1,  1, 0, 1,
        };
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, (void*)8);
        glBindVertexArray(0);
    }

    void draw() {
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
        glBindVertexArray(0);
    }

    void destroy() {
        if (vao) { glDeleteVertexArrays(1, &vao); vao = 0; }
        if (vbo) { glDeleteBuffers(1, &vbo); vbo = 0; }
    }
};

// ============================================================
// 延迟渲染器
// ============================================================

class DeferredRenderer {
public:
    GBuffer gbuffer;
    ShadowMap shadowMap;
    SSAOBuffer ssao;
    FullscreenQuad fullscreenQuad;
    GLMesh skyboxMesh;

    ShaderProgram geometryShader;
    ShaderProgram lightingShader;
    ShaderProgram shadowShader;
    ShaderProgram wireframeShader;
    ShaderProgram ssaoShader;
    ShaderProgram ssaoBlurShader;
    ShaderProgram skyboxShader;

    bool enableShadow = true;
    bool enableSSAO = true;
    bool enableWireframe = false;
    bool enableSkybox = true;
    bool enableFog = false;
    int  wireframeMode = 0;  // 0=线框, 1=法线, 2=深度

    glm::vec3 ambientColor = glm::vec3(0.15f, 0.15f, 0.2f);
    float ambientStrength = 0.3f;
    glm::vec3 fogColor = glm::vec3(0.5f, 0.6f, 0.7f);
    float fogDensity = 0.02f;

    std::string shaderDir;

    bool init(int width, int height, const std::string& shaderDirectory) {
        shaderDir = shaderDirectory;

        // 着色器加载
        bool ok = true;
        ok &= geometryShader.loadFromFiles(shaderDir + "/mesh_geometry_pass.vert",
                                            shaderDir + "/mesh_geometry_pass.frag");
        ok &= lightingShader.loadFromFiles(shaderDir + "/mesh_lighting_pass.vert",
                                            shaderDir + "/mesh_lighting_pass.frag");
        ok &= shadowShader.loadFromFiles(shaderDir + "/mesh_shadow_depth.vert",
                                          shaderDir + "/mesh_shadow_depth.frag");
        ok &= wireframeShader.loadFromFiles(shaderDir + "/mesh_wireframe.vert",
                                             shaderDir + "/mesh_wireframe.frag");
        ok &= ssaoShader.loadFromFiles(shaderDir + "/mesh_ssao.vert",
                                        shaderDir + "/mesh_ssao.frag");
        ok &= ssaoBlurShader.loadFromFiles(shaderDir + "/mesh_ssao.vert",
                                            shaderDir + "/mesh_ssao_blur.frag");
        ok &= skyboxShader.loadFromFiles(shaderDir + "/mesh_skybox.vert",
                                          shaderDir + "/mesh_skybox.frag");
        if (!ok) {
            std::cerr << "着色器加载失败!" << std::endl;
            return false;
        }

        gbuffer.create(width, height);
        shadowMap.create(2048, 2048);
        ssao.create(width, height);
        fullscreenQuad.create();

        // 天空盒立方体
        MeshData skyboxMeshData = createCube(2.0f);
        skyboxMesh.upload(skyboxMeshData);

        return true;
    }

    void resize(int width, int height) {
        gbuffer.create(width, height);
        ssao.create(width, height);
    }

    /** 渲染一帧 */
    void render(const std::vector<RenderObject>& objects,
                const std::vector<Light>& lights,
                const Camera& camera) {

        // 计算场景中心和半径
        glm::vec3 sceneCenter(0);
        float sceneRadius = 5.0f;
        if (!objects.empty()) {
            AABB sceneBox;
            for (const auto& obj : objects) {
                if (!obj.visible) continue;
                // 变换 AABB 的 8 个角点
                const float* mn = obj.mesh.aabb.min;
                const float* mx = obj.mesh.aabb.max;
                glm::vec4 corners[8] = {
                    obj.modelMatrix * glm::vec4(mn[0], mn[1], mn[2], 1.0f),
                    obj.modelMatrix * glm::vec4(mx[0], mn[1], mn[2], 1.0f),
                    obj.modelMatrix * glm::vec4(mn[0], mx[1], mn[2], 1.0f),
                    obj.modelMatrix * glm::vec4(mx[0], mx[1], mn[2], 1.0f),
                    obj.modelMatrix * glm::vec4(mn[0], mn[1], mx[2], 1.0f),
                    obj.modelMatrix * glm::vec4(mx[0], mn[1], mx[2], 1.0f),
                    obj.modelMatrix * glm::vec4(mn[0], mx[1], mx[2], 1.0f),
                    obj.modelMatrix * glm::vec4(mx[0], mx[1], mx[2], 1.0f),
                };
                for (int c = 0; c < 8; c++)
                    sceneBox.expand(glm::value_ptr(corners[c]));
            }
            sceneCenter = glm::vec3(
                (sceneBox.min[0] + sceneBox.max[0]) * 0.5f,
                (sceneBox.min[1] + sceneBox.max[1]) * 0.5f,
                (sceneBox.min[2] + sceneBox.max[2]) * 0.5f
            );
            sceneRadius = sceneBox.diagonal() * 0.5f + 1.0f;
        }

        // 找到主方向光（用于阴影）
        const Light* mainDirLight = nullptr;
        for (const auto& light : lights) {
            if (light.type == LightType::Directional) {
                mainDirLight = &light;
                break;
            }
        }
        if (mainDirLight && enableShadow) {
            shadowMap.updateLightSpace(*mainDirLight, sceneCenter, sceneRadius);
        }

        // ---- Pass 1: Shadow ----
        if (mainDirLight && enableShadow) {
            shadowMap.bind();
            glViewport(0, 0, shadowMap.width, shadowMap.height);
            glClear(GL_DEPTH_BUFFER_BIT);
            glCullFace(GL_FRONT);
            shadowShader.use();
            shadowShader.setMat4("uLightSpaceMatrix", shadowMap.lightSpaceMatrix);
            for (const auto& obj : objects) {
                if (!obj.visible || !obj.castShadow) continue;
                shadowShader.setMat4("uModel", obj.modelMatrix);
                obj.mesh.draw();
            }
            glCullFace(GL_BACK);
            shadowMap.unbind();
        }

        // ---- Pass 2: Geometry ----
        gbuffer.bind();
        glViewport(0, 0, gbuffer.width, gbuffer.height);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        geometryShader.use();

        for (const auto& obj : objects) {
            if (!obj.visible) continue;
            glm::mat4 model = obj.modelMatrix;
            glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(model)));
            geometryShader.setMat4("uModel", model);
            geometryShader.setMat4("uView", camera.getViewMatrix());
            geometryShader.setMat4("uProjection", camera.getProjectionMatrix());
            geometryShader.setMat3("uNormalMatrix", normalMatrix);
            geometryShader.setVec3("uBaseColor", obj.baseColor);
            geometryShader.setFloat("uMetallic", obj.metallic);
            geometryShader.setFloat("uRoughness", obj.roughness);
            geometryShader.setFloat("uAO", obj.ao);
            geometryShader.setVec3("uEmissive", obj.emissive);
            geometryShader.setFloat("uEmissiveStrength", obj.emissiveStrength);
            geometryShader.setInt("uWireframeMode", enableWireframe ? 1 : 0);
            geometryShader.setInt("uHasDiffuseTex", 0);
            geometryShader.setInt("uHasNormalTex", 0);
            geometryShader.setInt("uHasSpecularTex", 0);
            obj.mesh.draw();
        }
        gbuffer.unbind();

        // ---- Pass 3: SSAO ----
        if (enableSSAO) {
            ssaoShader.use();
            glBindFramebuffer(GL_FRAMEBUFFER, ssao.fbo);
            glViewport(0, 0, gbuffer.width, gbuffer.height);
            glClear(GL_COLOR_BUFFER_BIT);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, gbuffer.textures[0]);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, gbuffer.textures[1]);
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, ssao.noiseTexture);
            ssaoShader.setInt("gPosition", 0);
            ssaoShader.setInt("gNormal", 1);
            ssaoShader.setInt("uNoiseTex", 2);
            ssaoShader.setVec2("uScreenSize", glm::vec2(gbuffer.width, gbuffer.height));
            ssaoShader.setMat4("uProjection", camera.getProjectionMatrix());
            ssaoShader.setMat4("uView", camera.getViewMatrix());
            ssaoShader.setInt("uKernelSize", ssao.kernelSize);
            ssaoShader.setFloat("uRadius", ssao.radius);
            ssaoShader.setFloat("uBias", ssao.bias);
            // 采样点
            std::vector<glm::vec3> sampleVec3 = ssao.samples;
            ssaoShader.setVec3Array("uSamples", sampleVec3);
            fullscreenQuad.draw();
            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            // Blur
            ssaoBlurShader.use();
            glBindFramebuffer(GL_FRAMEBUFFER, ssao.blurFbo);
            glViewport(0, 0, gbuffer.width, gbuffer.height);
            glClear(GL_COLOR_BUFFER_BIT);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, ssao.texture);
            ssaoBlurShader.setInt("uSSAOInput", 0);
            fullscreenQuad.draw();
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }

        // ---- Pass 4: Lighting ----
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, gbuffer.width, gbuffer.height);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        lightingShader.use();
        gbuffer.bindTextures();
        lightingShader.setInt("gPosition", 0);
        lightingShader.setInt("gNormal", 1);
        lightingShader.setInt("gAlbedoSpec", 2);
        lightingShader.setInt("gMaterial", 3);

        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, shadowMap.texture);
        lightingShader.setInt("uShadowMap", 4);

        if (enableSSAO) {
            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_2D, ssao.blurTexture);
            lightingShader.setInt("uSSAOTex", 5);
        }

        lightingShader.setVec3("uViewPos", camera.position);
        lightingShader.setInt("uHasShadowMap", (enableShadow && mainDirLight) ? 1 : 0);
        lightingShader.setMat4("uLightSpaceMatrix",
            mainDirLight ? shadowMap.lightSpaceMatrix : glm::mat4(1.0f));
        lightingShader.setInt("uHasSSAO", enableSSAO ? 1 : 0);
        lightingShader.setVec3("uAmbientColor", ambientColor);
        lightingShader.setFloat("uAmbientStrength", ambientStrength);
        lightingShader.setInt("uHasFog", enableFog ? 1 : 0);
        lightingShader.setVec3("uFogColor", fogColor);
        lightingShader.setFloat("uFogDensity", fogDensity);

        // 光源
        int numLights = std::min((int)lights.size(), 16);
        lightingShader.setInt("uNumLights", numLights);
        std::vector<int> types(numLights);
        std::vector<glm::vec3> positions(numLights), directions(numLights), colors(numLights);
        std::vector<float> intensities(numLights), ranges(numLights), innerCone(numLights), outerCone(numLights);
        for (int i = 0; i < numLights; i++) {
            types[i] = (int)lights[i].type;
            positions[i] = lights[i].position;
            directions[i] = lights[i].direction;
            colors[i] = lights[i].color;
            intensities[i] = lights[i].intensity;
            ranges[i] = lights[i].range;
            innerCone[i] = lights[i].innerCone;
            outerCone[i] = lights[i].outerCone;
        }
        lightingShader.setIntArray("uLightTypes", types);
        lightingShader.setVec3Array("uLightPositions", positions);
        lightingShader.setVec3Array("uLightDirections", directions);
        lightingShader.setVec3Array("uLightColors", colors);
        lightingShader.setFloatArray("uLightIntensities", intensities);
        lightingShader.setFloatArray("uLightRanges", ranges);
        lightingShader.setFloatArray("uLightInnerCone", innerCone);
        lightingShader.setFloatArray("uLightOuterCone", outerCone);

        fullscreenQuad.draw();

        // ---- Pass 5: Skybox ----
        if (enableSkybox) {
            glDepthFunc(GL_LEQUAL);
            skyboxShader.use();
            glm::mat4 viewNoTrans = glm::mat4(glm::mat3(camera.getViewMatrix()));
            skyboxShader.setMat4("uView", viewNoTrans);
            skyboxShader.setMat4("uProjection", camera.getProjectionMatrix());
            // 太阳方向 = 主方向光方向
            glm::vec3 sunDir = mainDirLight ? -mainDirLight->direction : glm::vec3(0.5f, 0.4f, 0.3f);
            skyboxShader.setVec3("uSunDir", sunDir);
            skyboxMesh.draw();
            glDepthFunc(GL_LESS);
        }

        // ---- Pass 6: Wireframe overlay ----
        if (enableWireframe) {
            wireframeShader.use();
            wireframeShader.setMat4("uView", camera.getViewMatrix());
            wireframeShader.setMat4("uProjection", camera.getProjectionMatrix());
            wireframeShader.setInt("uMode", wireframeMode);
            wireframeShader.setVec3("uViewPos", camera.position);
            for (const auto& obj : objects) {
                if (!obj.visible) continue;
                wireframeShader.setMat4("uModel", obj.modelMatrix);
                glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(obj.modelMatrix)));
                wireframeShader.setMat3("uNormalMatrix", normalMatrix);
                obj.mesh.drawWireframe();
            }
        }
    }

    void destroy() {
        gbuffer.destroy();
        shadowMap.destroy();
        ssao.destroy();
        fullscreenQuad.destroy();
        skyboxMesh.release();
    }
};

} // namespace MeshRender
