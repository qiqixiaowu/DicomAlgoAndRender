#version 430 core

// ============================================================================
// CT 显示 - 顶点着色器
// 全屏四边形, 用于将重建结果渲染到屏幕
// ============================================================================

layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
