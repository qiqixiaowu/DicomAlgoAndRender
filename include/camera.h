#pragma once

#include <glm/glm.hpp>
#include <glfw3.h>

class OrbitCamera
{
public:
    OrbitCamera();

    // 设置目标点和距离
    void setTarget(const glm::vec3& target);
    void setDistance(float distance);

    // 处理输入
    void processMouseButton(int button, int action, double x, double y);
    void processMouseMotion(double x, double y);
    void processMouseScroll(double yoffset);

    // 更新相机
    void update(float deltaTime);

    // 获取视图矩阵和位置
    glm::mat4 getViewMatrix() const;
    glm::vec3 getPosition() const;
    glm::vec3 getTarget() const { return target; }

    // 设置限制
    void setPitchLimits(float minPitch, float maxPitch);
    void setDistanceLimits(float minDist, float maxDist);

private:
    void updateCamera();

private:
    glm::vec3 target = glm::vec3(0.0f);
    glm::vec3 position = glm::vec3(0.0f, 0.0f, 5.0f);

    float distance = 5.0f;
    float yaw = 0.0f;
    float pitch = 0.0f;

    float minDistance = 1.0f;
    float maxDistance = 50.0f;
    float minPitch = -89.0f;
    float maxPitch = 89.0f;

    bool rotating = false;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;

    float rotationSpeed = 0.5f;
    float zoomSpeed = 0.5f;
};