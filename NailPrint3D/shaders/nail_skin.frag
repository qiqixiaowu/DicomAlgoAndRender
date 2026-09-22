#version 330 core
/**
 * nail_skin.frag — 皮肤片段着色器
 * 特性：SSS次表面散射 + 透光 + 暖色漫反射 + 油性高光 + PCF阴影
 */

in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUV;
in vec3 vColor;

uniform mat4 uLightSpaceMatrix;
uniform sampler2D uShadowMap;

uniform vec3 uLightDir;
uniform vec3 uViewPos;

layout(location = 0) out vec4 FragColor;

const float PI = 3.14159265;

// ============================================================
// PCF 阴影（3×3 采样）
// ============================================================
float calculateShadow(vec4 lightSpacePos) {
    vec3 projCoords = lightSpacePos.xyz / lightSpacePos.w;
    projCoords = projCoords * 0.5 + 0.5;

    if (projCoords.z > 1.0) return 0.0;  // 超出远平面
    if (projCoords.x < 0.0 || projCoords.x > 1.0) return 0.0;
    if (projCoords.y < 0.0 || projCoords.y > 1.0) return 0.0;

    float currentDepth = projCoords.z;
    float bias = 0.003;
    float shadow = 0.0;

    vec2 texelSize = 1.0 / textureSize(uShadowMap, 0);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float pcfDepth = texture(uShadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += (currentDepth - bias > pcfDepth) ? 1.0 : 0.0;
        }
    }
    shadow /= 9.0;
    return shadow;
}

// ============================================================
// ACES 色调映射 + sRGB
// ============================================================
vec3 ACESFilm(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 toSRGB(vec3 c) {
    return mix(12.92 * c, 1.055 * pow(c, vec3(1.0/2.4)) - 0.055,
               step(0.0031308, c));
}

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightDir);
    vec3 V = normalize(uViewPos - vWorldPos);
    vec3 H = normalize(L + V);

    // ============================================================
    // 1. 阴影
    // ============================================================
    vec4 lightSpacePos = uLightSpaceMatrix * vec4(vWorldPos, 1.0);
    float shadow = calculateShadow(lightSpacePos);
    float shadowFactor = 1.0 - shadow;

    // ============================================================
    // 2. 皮肤基色（带顶点颜色混合）
    // ============================================================
    vec3 skinBase = mix(vec3(0.85, 0.72, 0.62), vColor, 0.5);

    // ============================================================
    // 3. SSS 次表面散射（简化版）
    //    薄肉区域透光 → 暖红色
    // ============================================================
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);

    // 透光：当法线背向光源但视角接近边缘时
    float backLight = max(dot(-N, L), 0.0);
    float fresnel = pow(1.0 - NdotV, 3.0);
    float sss = backLight * fresnel * 0.6;

    // SSS 暖色（血红蛋白吸收 → 偏红橙）
    vec3 sssColor = vec3(1.0, 0.35, 0.2) * sss;

    // 前向散射（软阴影边界）
    float wrapLight = (dot(N, L) + 0.4) / 1.4;
    wrapLight = clamp(wrapLight, 0.0, 1.0);
    vec3 wrapColor = skinBase * vec3(1.0, 0.85, 0.75) * wrapLight;

    // ============================================================
    // 4. 漫反射（Half-Lambert + 暖色偏移）
    // ============================================================
    float halfLambert = NdotL * 0.5 + 0.5;
    vec3 warmTint = vec3(1.0, 0.92, 0.85);  // 血红蛋白吸收蓝绿
    vec3 diffuse = skinBase * halfLambert * warmTint;

    // ============================================================
    // 5. 油性高光（T-zone 光泽）
    // ============================================================
    float specPower = 40.0;
    float specIntensity = 0.3;
    float NdotH = max(dot(N, H), 0.0);
    vec3 specular = vec3(1.0, 0.95, 0.9) * specIntensity * pow(NdotH, specPower);

    // ============================================================
    // 6. 环境光
    // ============================================================
    vec3 ambient = skinBase * vec3(0.25, 0.22, 0.28);

    // ============================================================
    // 7. 合成
    // ============================================================
    vec3 color = ambient
               + (diffuse + wrapColor + specular) * shadowFactor
               + sssColor * shadowFactor;

    // 附加环境遮蔽（简化）
    float ao = mix(0.7, 1.0, NdotV);
    color *= ao;

    // 色调映射 + sRGB
    color = ACESFilm(color);
    color = toSRGB(color);

    FragColor = vec4(color, 1.0);
}
