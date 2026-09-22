#version 330 core
in vec3 vColor;
in vec3 vNormal;
in vec3 vWorldPos;
in vec2 vUV;

// 调色板（已校准）
uniform vec4 uPalette[16];
uniform int  uPaletteSize;

// 打印配置 — 统一色彩管理
uniform float uLayerHeight;       // 颜色层层高 (mm)
uniform float uBaseThickness;     // 底胶厚度 (mm)
uniform int   uColorCount;        // 颜色数量

out vec4 FragColor;

// ACES 色调映射
vec3 ACESFilm(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// 按层数计算颜色 — 匹配 slicer.cpp assignColors()
vec3 getLayerColor(float z) {
    // 底胶层
    if (z < uBaseThickness) {
        return vec3(0.96, 0.94, 0.92);
    }

    int colorLayer = int((z - uBaseThickness) / max(uLayerHeight, 0.001));
    int colorIndex = (colorLayer / 3) % max(uColorCount, 1);

    if (uColorCount == 0 || uPaletteSize == 0) return vColor;
    if (colorIndex >= uPaletteSize) return vColor;
    return uPalette[colorIndex].rgb;
}

void main() {
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));

    // Half-Lambert 漫反射
    float NdotL = dot(normal, lightDir);
    float halfLambert = NdotL * 0.5 + 0.5;
    float diff = halfLambert * halfLambert;

    // 填充光
    vec3 fillLightDir = normalize(vec3(-lightDir.x, lightDir.y * 0.3, -lightDir.z));
    float fillDiff = max(dot(normal, fillLightDir), 0.0) * 0.3;

    // 高光
    vec3 viewDir = normalize(vec3(0, 0, 5) - vWorldPos);
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), 32.0);
    float specCoat = pow(max(dot(normal, halfDir), 0.0), 128.0);

    // 菲涅尔
    float fresnel = mix(0.04, 1.0, pow(1.0 - max(dot(normal, viewDir), 0.0), 4.0));

    // 环境光
    vec3 ambientColor = mix(
        vec3(0.75, 0.70, 0.68),
        vec3(0.85, 0.88, 0.95),
        normal.z * 0.5 + 0.5
    ) * 0.35;

    // === 按层数获取颜色（统一色彩管理）===
    vec3 paletteColor = getLayerColor(vWorldPos.z);

    // UV 渐变叠加
    vec3 gradTop = mix(paletteColor, vec3(1.0), 0.15 * vUV.y);
    vec3 gradBot = mix(paletteColor, vec3(0.7, 0.6, 0.7), 0.2 * (1.0 - vUV.y));
    vec3 baseColor = mix(gradBot, gradTop, vUV.y);

    vec3 warmLight = vec3(1.0, 0.96, 0.92);
    vec3 coolLight = vec3(0.85, 0.90, 1.0);
    vec3 specColor = vec3(1.0, 0.98, 0.95);

    vec3 diffuse = baseColor * (diff * warmLight + fillDiff * coolLight);
    vec3 ambientPart = baseColor * ambientColor;
    vec3 specular = (spec * 0.4 + specCoat * 0.8) * specColor;
    vec3 rim = fresnel * 0.25 * specColor;

    vec3 color = ambientPart + diffuse + specular + rim;

    // 色调映射 + sRGB（与 nail_mesh.frag 统一）
    color = ACESFilm(color * 1.2);
    color = pow(color, vec3(1.0 / 2.2));

    FragColor = vec4(color, 1.0);
}
