#version 430 core
/**
 * mesh_shadow_depth.vert - 阴影映射深度顶点着色器
 */
layout(location = 0) in vec3 inPosition;

uniform mat4 uModel;
uniform mat4 uLightSpaceMatrix;

void main() {
    gl_Position = uLightSpaceMatrix * uModel * vec4(inPosition, 1.0);
}
