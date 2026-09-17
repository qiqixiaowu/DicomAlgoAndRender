#version 430 core
/**
 * mesh_lighting_pass.frag - 延迟渲染光照通道片元着色器
 * PBR 光照 + 多光源 + 阴影
 */
out vec4 FragColor;

in vec2 vTexcoord;

// G-Buffer
uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedoSpec;
uniform sampler2D gMaterial;

// 相机
uniform vec3 uViewPos;

// 光源参数
#define MAX_LIGHTS 16
uniform int   uNumLights = 3;
uniform int   uLightTypes[MAX_LIGHTS];     // 0=方向光, 1=点光, 2=聚光
uniform vec3  uLightPositions[MAX_LIGHTS];
uniform vec3  uLightDirections[MAX_LIGHTS];
uniform vec3  uLightColors[MAX_LIGHTS];
uniform float uLightIntensities[MAX_LIGHTS];
uniform float uLightRanges[MAX_LIGHTS];       // 点光/聚光范围
uniform float uLightInnerCone[MAX_LIGHTS];    // 聚光内角 cos
uniform float uLightOuterCone[MAX_LIGHTS];    // 聚光外角 cos

// 环境光
uniform vec3  uAmbientColor = vec3(0.15, 0.15, 0.2);
uniform float uAmbientStrength = 0.3;

// 阴影
uniform int   uHasShadowMap = 0;
uniform sampler2D uShadowMap;
uniform mat4  uLightSpaceMatrix;
uniform float uShadowBias = 0.005;

// SSAO
uniform int   uHasSSAO = 0;
uniform sampler2D uSSAOTex;

// 雾效
uniform int   uHasFog = 0;
uniform vec3  uFogColor = vec3(0.5, 0.6, 0.7);
uniform float uFogDensity = 0.02;

const float PI = 3.14159265359;

// PBR BRDF
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    denom = PI * denom * denom;
    return a2 / max(denom, 0.0001);
}

float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return geometrySchlickGGX(max(dot(N, V), 0.0), roughness)
         * geometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

// 阴影计算
float calculateShadow(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir) {
    if (uHasShadowMap == 0) return 0.0;

    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    if (projCoords.z > 1.0) return 0.0;

    float currentDepth = projCoords.z;
    float bias = max(uShadowBias * (1.0 - dot(normal, lightDir)), uShadowBias * 0.1);

    // PCF 3x3
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

void main() {
    // 读取 G-Buffer
    vec3 fragPos   = texture(gPosition, vTexcoord).rgb;
    vec3 normal    = texture(gNormal, vTexcoord).rgb;
    vec3 albedo    = texture(gAlbedoSpec, vTexcoord).rgb;
    float metallic = texture(gAlbedoSpec, vTexcoord).a;
    float roughness = texture(gMaterial, vTexcoord).g;
    float ao        = texture(gMaterial, vTexcoord).b;

    // 空像素（背景）
    if (length(fragPos) < 0.001) {
        FragColor = vec4(uAmbientColor, 1.0);
        return;
    }

    // SSAO
    if (uHasSSAO == 1) {
        float ssao = texture(uSSAOTex, vTexcoord).r;
        ao *= ssao;
    }

    vec3 N = normalize(normal);
    vec3 V = normalize(uViewPos - fragPos);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    vec3 Lo = vec3(0.0);

    for (int i = 0; i < uNumLights && i < MAX_LIGHTS; i++) {
        vec3 lightPos = uLightPositions[i];
        vec3 lightColor = uLightColors[i] * uLightIntensities[i];
        vec3 L;
        float attenuation = 1.0;

        if (uLightTypes[i] == 0) {
            // 方向光
            L = normalize(-uLightDirections[i]);
        } else {
            // 点光 / 聚光
            vec3 toLight = lightPos - fragPos;
            float distance = length(toLight);
            L = toLight / max(distance, 0.001);
            float range = uLightRanges[i];
            attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
            if (distance > range) attenuation *= max(0.0, 1.0 - (distance / range));

            if (uLightTypes[i] == 2) {
                // 聚光
                float theta = dot(L, normalize(-uLightDirections[i]));
                float epsilon = uLightInnerCone[i] - uLightOuterCone[i];
                float spotIntensity = clamp((theta - uLightOuterCone[i]) / epsilon, 0.0, 1.0);
                attenuation *= spotIntensity;
            }
        }

        vec3 H = normalize(V + L);

        // Cook-Torrance BRDF
        float NDF = distributionGGX(N, H, roughness);
        float G   = geometrySmith(N, V, L, roughness);
        vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3 specular = numerator / denominator;

        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedo / PI + specular) * lightColor * NdotL * attenuation;

        // 阴影（仅方向光）
        if (uLightTypes[i] == 0 && uHasShadowMap == 1) {
            vec4 fragPosLightSpace = uLightSpaceMatrix * vec4(fragPos, 1.0);
            float shadow = calculateShadow(fragPosLightSpace, N, L);
            Lo *= (1.0 - shadow);
        }
    }

    // 环境光
    vec3 ambient = uAmbientColor * uAmbientStrength * albedo * ao;

    vec3 color = ambient + Lo;

    // 雾效
    if (uHasFog == 1) {
        float dist = length(uViewPos - fragPos);
        float fogFactor = 1.0 - exp(-uFogDensity * uFogDensity * dist * dist);
        fogFactor = clamp(fogFactor, 0.0, 1.0);
        color = mix(color, uFogColor, fogFactor);
    }

    // HDR tone mapping
    color = color / (color + 1.0);
    // gamma
    color = pow(color, vec3(1.0 / 2.2));

    FragColor = vec4(color, 1.0);
}
