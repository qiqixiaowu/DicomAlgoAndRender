#version 330 core
in vec3 vNormal;
in vec3 vColor;
in vec3 vWorldPos;
in vec2 vUV;

// 打印配置
uniform float uLayerHeight;       // 颜色层层高 (mm)
uniform float uBaseThickness;     // 底胶厚度 (mm)
uniform float uTopCoatThickness;  // 封层厚度 (mm)
uniform int   uBaseLayers;        // 底胶层数
uniform int   uTopCoatLayers;     // 封层层数
uniform int   uColorCount;        // 颜色数量
uniform vec3  uPalette[16];       // 调色板 (已校准)

// 光照
uniform vec3 uLightDir;

out vec4 FragColor;

// ============================================================
// 按层数计算颜色 — 匹配 slicer.cpp assignColors() 逻辑
// ============================================================
vec3 getLayerColor(float z) {
    // 底胶层：使用调色板第一个颜色（或白色底胶）
    if (z < uBaseThickness) {
        return vec3(0.96, 0.94, 0.92); // 底胶：半透明白
    }

    // 封层：透明高光
    float topStart = uBaseThickness + uLayerHeight * 10.0; // 假设最多10层颜色
    if (z > topStart) {
        return vec3(0.98, 0.96, 0.94); // 封层：高光透明
    }

    // 颜色层：每3层换色，匹配 slicer 的 (colorLayer / 3) % colorCount
    int colorLayer = int((z - uBaseThickness) / uLayerHeight);
    int colorIndex = (colorLayer / 3) % max(uColorCount, 1);

    if (uColorCount == 0) return vColor;
    return uPalette[colorIndex];
}

// ============================================================
// 阶梯效应：量化Z到层边界，模拟3D打印的层叠纹理
// ============================================================
float getStaircaseFactor(float z, vec3 normal) {
    // 层边界处的暗线
    float layerBoundary = fract(z / uLayerHeight);
    float stairEdge = smoothstep(0.0, 0.05, layerBoundary) * smoothstep(1.0, 0.95, layerBoundary);

    // 法线水平分量越大（侧面），阶梯越明显
    float sideFactor = 1.0 - abs(normal.z);
    float stairIntensity = (1.0 - stairEdge) * sideFactor * 0.4;

    return stairIntensity;
}

// ============================================================
// ACES 色调映射
// ============================================================
vec3 ACESFilm(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(uLightDir);
    vec3 viewDir = normalize(-vWorldPos);

    // === 按层数获取颜色 ===
    vec3 baseCol = getLayerColor(vWorldPos.z);

    // === 阶梯效应 ===
    float stair = getStaircaseFactor(vWorldPos.z, normal);
    baseCol *= (1.0 - stair);

    // === 光照 ===
    // Half-Lambert 漫反射
    float NdotL = dot(normal, lightDir);
    float halfLambert = NdotL * 0.5 + 0.5;
    float diff = halfLambert * halfLambert;

    // 填充光
    vec3 fillLightDir = normalize(vec3(-lightDir.x, lightDir.y * 0.3, -lightDir.z));
    float fillDiff = max(dot(normal, fillLightDir), 0.0) * 0.3;

    // 高光
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), 32.0);
    float specCoat = pow(max(dot(normal, halfDir), 0.0), 128.0);

    // 菲涅尔
    float NdotV = max(dot(normal, viewDir), 0.0);
    float fresnel = mix(0.04, 1.0, pow(1.0 - NdotV, 4.0));

    // 环境光
    vec3 ambientColor = mix(
        vec3(0.75, 0.70, 0.68),
        vec3(0.85, 0.88, 0.95),
        normal.z * 0.5 + 0.5
    ) * 0.35;

    // SSS
    float sss = pow(1.0 - abs(NdotL), 2.0) * 0.15;

    vec3 warmLight = vec3(1.0, 0.96, 0.92);
    vec3 coolLight = vec3(0.85, 0.90, 1.0);
    vec3 specColor = vec3(1.0, 0.98, 0.95);

    vec3 diffuse = baseCol * (diff * warmLight + fillDiff * coolLight);
    vec3 ambientPart = baseCol * ambientColor;
    vec3 specular = (spec * 0.4 + specCoat * 0.8) * specColor;
    vec3 rim = fresnel * 0.25 * specColor;
    vec3 sssColor = vec3(1.0, 0.75, 0.65) * sss * baseCol;

    vec3 color = ambientPart + diffuse + specular + rim + sssColor;

    // 色调映射 + sRGB
    color = ACESFilm(color * 1.2);
    color = pow(color, vec3(1.0 / 2.2));

    FragColor = vec4(color, 1.0);
}
