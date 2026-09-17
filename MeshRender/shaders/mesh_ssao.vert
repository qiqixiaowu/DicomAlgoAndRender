#version 430 core
/**
 * mesh_ssao.vert - SSAO 顶点着色器（全屏四边形）
 */
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inTexcoord;

out vec2 vTexcoord;

void main() {
    vTexcoord = inTexcoord;
    gl_Position = vec4(inPosition, 0.0, 1.0);
}
