#version 330 core
/**
 * nail_shadow.vert — 阴影深度 Pass 顶点着色器
 * 只输出裁剪空间坐标，不传递颜色/法线/UV
 */
layout(location = 0) in vec3 aPos;

uniform mat4 uModel;
uniform mat4 uLightSpaceMatrix;  // 投影 * 视图（从光源视角）

void main() {
    gl_Position = uLightSpaceMatrix * uModel * vec4(aPos, 1.0);
}
