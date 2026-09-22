#version 330 core
in vec3 vNormal;
in vec3 vWorldPos;
in vec2 vUV;

out vec4 FragColor;

void main() {
    vec3 normal = normalize(vNormal);

    // 法线可视化: 将 [-1,1] 映射到 [0,1]
    vec3 normalColor = normal * 0.5 + 0.5;

    // 网格线（每0.05单位一条，帮助观察曲面细节）
    vec2 grid = abs(fract(vUV * 20.0) - 0.5);
    float gridLine = smoothstep(0.45, 0.5, max(grid.x, grid.y)) * 0.15;

    // 边缘高亮（法线与视线垂直时加亮）
    vec3 viewDir = normalize(-vWorldPos);
    float edge = 1.0 - abs(dot(normal, viewDir));
    edge = pow(edge, 2.0) * 0.3;

    vec3 color = normalColor;
    color += gridLine;
    color += edge * vec3(1.0, 0.9, 0.8);

    FragColor = vec4(color, 1.0);
}
