#version 330 core
in vec3 vNormal;
out vec4 FragColor;

void main() {
    vec3 normal = normalize(vNormal);
    // 法线可视化: 将 [-1,1] 映射到 [0,1]
    vec3 color = normal * 0.5 + 0.5;
    FragColor = vec4(color, 1.0);
}
