#version 430 core

// ================================
// MPR（多平面重建）顶点着色器
// ================================
// 用于渲染2D切面quad，从3D体数据中提取截面

layout (location = 0) in vec2 inPosition;   // 2D顶点坐标 [-1, 1]
layout (location = 1) in vec2 inTexCoord;   // 纹理坐标 [0, 1]

out vec2 vTexCoord;

void main() {
    gl_Position = vec4(inPosition, 0.0, 1.0);
    vTexCoord = inTexCoord;
}
