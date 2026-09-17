#version 430 core
/**
 * mesh_skybox.vert - 天空盒顶点着色器
 */
layout(location = 0) in vec3 inPosition;

uniform mat4 uProjection;
uniform mat4 uView;

out vec3 vWorldDir;

void main() {
    vWorldDir = inPosition;
    // 去掉平移，让天空盒始终在无穷远
    mat4 viewNoTrans = mat4(mat3(uView));
    vec4 pos = uProjection * viewNoTrans * vec4(inPosition, 1.0);
    gl_Position = pos.xyww;  // z = w → 深度始终为 1（最远）
}
