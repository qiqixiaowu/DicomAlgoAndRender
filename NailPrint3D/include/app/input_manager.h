#pragma once
/**
 * @file    input_manager.h
 * @brief   输入管理：GLFW 回调 + 键盘交互
 *
 * 所有 GLFW callback 和键盘处理逻辑集中于此。
 * 通过 Scene 引用操作状态，不使用全局变量。
 */

#include "app/scene.h"

struct GLFWwindow;

namespace NailPrint3D {

/// 注册所有 GLFW 回调
void registerCallbacks(GLFWwindow* window, Scene& scene);

/// 处理键盘输入（在 keyCallback 中调用）
void handleKey(GLFWwindow* window, int key, int action, Scene& scene);

} // namespace NailPrint3D
