#version 330 core
in vec3 vColor;
in float vZ;

uniform float uCurrentLayer;

out vec4 FragColor;

void main() {
    // 当前层高亮，其他层暗化
    float layerDiff = abs(vZ - uCurrentLayer);
    float highlight = 1.0 / (1.0 + layerDiff * 2.0);
    vec3 color = vColor * highlight;
    FragColor = vec4(color, 0.8);
}
