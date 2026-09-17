#version 430 core
/**
 * mesh_lighting_pass.vert - 全屏四边形顶点着色器
 */
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inTexcoord;

out vec2 vTexcoord;

void main() {
    vTexcoord = inTexcoord;
    gl_Position = vec4(inPosition, 0.0, 1.0);
}
