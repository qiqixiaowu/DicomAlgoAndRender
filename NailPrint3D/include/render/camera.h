#pragma once
/**
 * @file    camera.h
 * @brief   轨道相机（无 GL 依赖，仅使用 GLM 数学库）
 *
 * 依赖：GLM
 */

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>

namespace NailPrint3D {

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

} // namespace NailPrint3D
