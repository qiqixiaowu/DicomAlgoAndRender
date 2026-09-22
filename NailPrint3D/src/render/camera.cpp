/**
 * @file    camera.cpp
 * @brief   轨道相机 实现
 */

#include "render/camera.h"

namespace NailPrint3D {

RenderCamera::RenderCamera()
    : target_(0, 0, 0)
    , distance_(5.0f)
    , angleX_(-45.0f)
    , angleY_(0.0f)
    , fov_(45.0f)
    , aspect_(1.0f)
    , near_(0.1f)
    , far_(100.0f) {
    updatePosition();
}

void RenderCamera::orbit(float dx, float dy) {
    angleY_ += dx * 0.5f;
    angleX_ += dy * 0.5f;
    angleX_ = std::clamp(angleX_, -89.0f, 89.0f);
    updatePosition();
}

void RenderCamera::pan(float dx, float dy) {
    glm::vec3 right = glm::normalize(glm::cross(front_, up_));
    glm::vec3 up = glm::normalize(up_);
    target_ -= right * dx * 0.01f * distance_;
    target_ -= up * dy * 0.01f * distance_;
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
    float radX = glm::radians(angleX_);
    float radY = glm::radians(angleY_);
    position_.x = target_.x + distance_ * std::cos(radX) * std::cos(radY);
    position_.y = target_.y + distance_ * std::cos(radX) * std::sin(radY);
    position_.z = target_.z + distance_ * std::sin(radX);
    front_ = glm::normalize(target_ - position_);
    up_ = glm::vec3(0, 0, 1);
}

} // namespace NailPrint3D
