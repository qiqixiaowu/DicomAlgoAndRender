#pragma once
/**
 * @file    shader.h
 * @brief   GLSL 着色器封装
 *
 * 依赖：GLAD, core/color_types.h
 */

#include "core/color_types.h"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>

namespace NailPrint3D {

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

} // namespace NailPrint3D
