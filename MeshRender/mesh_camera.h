#pragma once
/**
 * @file    mesh_camera.h
 * @brief   增强版相机（轨道 + FPS 双模式）
 */

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glfw3.h>
#include <cmath>

namespace MeshRender {

enum class CameraMode { Orbit, FPS };

class Camera {
public:
    CameraMode mode = CameraMode::Orbit;

    // 公共参数
    glm::vec3 position = glm::vec3(0.0f, 2.0f, 5.0f);
    glm::vec3 target   = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 up       = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 worldUp  = glm::vec3(0.0f, 1.0f, 0.0f);

    // Orbit 模式参数
    float orbitDistance = 5.0f;
    float yaw   = -90.0f;   // 水平角度
    float pitch = 0.0f;     // 垂直角度

    // FPS 模式参数
    float fpsYaw   = -90.0f;
    float fpsPitch = 0.0f;

    // 投影参数
    float fov       = 45.0f;
    float aspect    = 16.0f / 9.0f;
    float nearPlane = 0.1f;
    float farPlane  = 200.0f;

    // 限制
    float minDistance = 1.0f, maxDistance = 100.0f;
    float minPitch = -89.0f, maxPitch = 89.0f;

    // 交互
    float rotateSpeed = 0.3f;
    float zoomSpeed   = 0.5f;
    float moveSpeed   = 3.0f;

    // FPS 模式移动状态
    bool keys[6] = {false}; // W A S D Shift Space

    // 鼠标状态
    bool mousePressed = false;
    double lastMouseX = 0, lastMouseY = 0;

    void setMode(CameraMode m) {
        mode = m;
        if (m == CameraMode::FPS) {
            fpsYaw = yaw;
            fpsPitch = pitch;
            updateFPSVectors();
        } else {
            updateOrbit();
        }
    }

    void processMouseButton(int button, int action, double x, double y) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            if (action == GLFW_PRESS) {
                mousePressed = true;
                lastMouseX = x;
                lastMouseY = y;
            } else if (action == GLFW_RELEASE) {
                mousePressed = false;
            }
        }
    }

    void processMouseMotion(double x, double y) {
        if (!mousePressed) return;
        double dx = x - lastMouseX;
        double dy = y - lastMouseY;
        lastMouseX = x;
        lastMouseY = y;

        if (mode == CameraMode::Orbit) {
            yaw   -= (float)dx * rotateSpeed;
            pitch += (float)dy * rotateSpeed;
            pitch = glm::clamp(pitch, minPitch, maxPitch);
            if (yaw > 360) yaw -= 360;
            if (yaw < 0)   yaw += 360;
            updateOrbit();
        } else {
            fpsYaw   -= (float)dx * rotateSpeed;
            fpsPitch += (float)dy * rotateSpeed;
            fpsPitch = glm::clamp(fpsPitch, minPitch, maxPitch);
            updateFPSVectors();
        }
    }

    void processMouseScroll(double yoffset) {
        if (mode == CameraMode::Orbit) {
            orbitDistance -= (float)yoffset * zoomSpeed;
            orbitDistance = glm::clamp(orbitDistance, minDistance, maxDistance);
            updateOrbit();
        } else {
            moveSpeed *= (yoffset > 0) ? 1.1f : 0.9f;
            moveSpeed = glm::clamp(moveSpeed, 0.5f, 20.0f);
        }
    }

    void processKey(int key, int action) {
        int idx = -1;
        switch (key) {
            case GLFW_KEY_W: idx = 0; break;
            case GLFW_KEY_A: idx = 1; break;
            case GLFW_KEY_S: idx = 2; break;
            case GLFW_KEY_D: idx = 3; break;
            case GLFW_KEY_LEFT_SHIFT: idx = 4; break;
            case GLFW_KEY_SPACE: idx = 5; break;
        }
        if (idx >= 0) {
            if (action == GLFW_PRESS || action == GLFW_REPEAT)
                keys[idx] = true;
            else if (action == GLFW_RELEASE)
                keys[idx] = false;
        }
    }

    void update(float deltaTime) {
        if (mode != CameraMode::FPS) return;

        float velocity = moveSpeed * deltaTime;
        if (keys[0]) position += front * velocity;       // W
        if (keys[2]) position -= front * velocity;       // S
        if (keys[3]) position += right * velocity;       // D
        if (keys[1]) position -= right * velocity;       // A
        if (keys[5]) position += worldUp * velocity;     // Space
        if (keys[4]) position -= worldUp * velocity;     // Shift
        target = position + front;
    }

    glm::mat4 getViewMatrix() const {
        return glm::lookAt(position, target, up);
    }

    glm::mat4 getProjectionMatrix() const {
        float safeAspect = (aspect > 0.001f) ? aspect : 1.0f;
        return glm::perspective(glm::radians(fov), safeAspect, nearPlane, farPlane);
    }

    glm::vec3 getFront() const { return front; }
    glm::vec3 getRight() const { return right; }

private:
    glm::vec3 front = glm::vec3(0, 0, -1);
    glm::vec3 right = glm::vec3(1, 0, 0);

    void updateOrbit() {
        float rp = glm::radians(pitch);
        float ry = glm::radians(yaw);
        position.x = target.x + orbitDistance * std::cos(rp) * std::cos(ry);
        position.y = target.y + orbitDistance * std::sin(rp);
        position.z = target.z + orbitDistance * std::cos(rp) * std::sin(ry);
        front = glm::normalize(target - position);
        right = glm::normalize(glm::cross(front, worldUp));
        up    = glm::normalize(glm::cross(right, front));
    }

    void updateFPSVectors() {
        float rp = glm::radians(fpsPitch);
        float ry = glm::radians(fpsYaw);
        front.x = std::cos(rp) * std::cos(ry);
        front.y = std::sin(rp);
        front.z = std::cos(rp) * std::sin(ry);
        front = glm::normalize(front);
        right = glm::normalize(glm::cross(front, worldUp));
        up    = glm::normalize(glm::cross(right, front));
        target = position + front;
    }
};

} // namespace MeshRender
