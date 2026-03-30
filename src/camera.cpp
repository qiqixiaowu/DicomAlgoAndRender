#include "camera.h"
#include <iostream>
#include<glm/gtc/matrix_transform.hpp>

OrbitCamera::OrbitCamera()
{
    updateCamera();
}

void OrbitCamera::setTarget(const glm::vec3& newTarget)
{
    target = newTarget;
    updateCamera();
}

void OrbitCamera::setDistance(float newDistance)
{
    distance = glm::clamp(newDistance, minDistance, maxDistance);
    updateCamera();
}

void OrbitCamera::processMouseButton(int button, int action, double x, double y)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            rotating = true;
            lastMouseX = x;
            lastMouseY = y;
        }
        else if (action == GLFW_RELEASE)
        {
            rotating = false;
        }
    }
}

void OrbitCamera::processMouseMotion(double x, double y)
{
    if (!rotating) return;

    double dx = x - lastMouseX;
    double dy = y - lastMouseY;

    yaw -= dx * rotationSpeed;
    pitch -= dy * rotationSpeed;

    // 限制俯仰角
    pitch = glm::clamp(pitch, minPitch, maxPitch);

    // 确保偏航角在0-360度范围内
    if (yaw > 360.0f) yaw -= 360.0f;
    if (yaw < 0.0f) yaw += 360.0f;

    lastMouseX = x;
    lastMouseY = y;

    updateCamera();
}

void OrbitCamera::processMouseScroll(double yoffset)
{
    distance -= yoffset * zoomSpeed;
    distance = glm::clamp(distance, minDistance, maxDistance);
    updateCamera();
}

void OrbitCamera::update(float deltaTime)
{
    // 如果需要平滑插值，可以在这里实现
    // 例如：position = glm::mix(position, desiredPosition, deltaTime * smoothSpeed);
}

void OrbitCamera::updateCamera()
{
    // 球坐标转直角坐标
    position.x = target.x + distance * cos(glm::radians(pitch)) * sin(glm::radians(yaw));
    position.y = target.y + distance * sin(glm::radians(pitch));
    position.z = target.z + distance * cos(glm::radians(pitch)) * cos(glm::radians(yaw));
}

glm::mat4 OrbitCamera::getViewMatrix() const
{
    return glm::lookAt(position, target, glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::vec3 OrbitCamera::getPosition() const
{
    return position;
}

void OrbitCamera::setPitchLimits(float min, float max)
{
    minPitch = min;
    maxPitch = max;
    pitch = glm::clamp(pitch, minPitch, maxPitch);
}

void OrbitCamera::setDistanceLimits(float min, float max)
{
    minDistance = min;
    maxDistance = max;
    distance = glm::clamp(distance, minDistance, maxDistance);
}