/**
 * @file    shader.cpp
 * @brief   GLSL 着色器封装 实现
 */

#include "render/shader.h"
#include <iostream>
#include <fstream>
#include <sstream>

namespace NailPrint3D {

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

} // namespace NailPrint3D
