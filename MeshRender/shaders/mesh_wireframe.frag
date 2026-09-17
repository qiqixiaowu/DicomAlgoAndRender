#version 430 core
/**
 * mesh_wireframe.frag - 线框/法线可视化片元着色器
 */
in vec3 vWorldPos;
in vec3 vNormal;

uniform int   uMode = 0;        // 0=纯线框, 1=法线可视化, 2=深度可视化
uniform vec3  uLineColor = vec3(0.0, 1.0, 0.4);
uniform float uLineAlpha = 1.0;
uniform vec3  uViewPos;

out vec4 FragColor;

void main() {
    if (uMode == 0) {
        FragColor = vec4(uLineColor, uLineAlpha);
    } else if (uMode == 1) {
        // 法线可视化：RGB = (N+1)/2
        vec3 n = normalize(vNormal);
        FragColor = vec4(n * 0.5 + 0.5, 1.0);
    } else if (uMode == 2) {
        // 深度可视化
        float depth = gl_FragCoord.z;
        FragColor = vec4(vec3(depth), 1.0);
    }
}
