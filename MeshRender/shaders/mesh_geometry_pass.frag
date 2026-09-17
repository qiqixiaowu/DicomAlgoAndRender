#version 430 core
/**
 * mesh_geometry_pass.frag - 延迟渲染几何通道片元着色器
 * 输出 G-Buffer: 位置、法线、反照率、材质参数
 */
layout(location = 0) out vec4 gPosition;
layout(location = 1) out vec4 gNormal;
layout(location = 2) out vec4 gAlbedoSpec;
layout(location = 3) out vec4 gMaterial;

in VS_OUT {
    vec3 worldPos;
    vec3 normal;
    vec2 texcoord;
    vec3 tangent;
    vec3 bitangent;
} fs_in;

// 材质参数
uniform vec3  uBaseColor     = vec3(0.8, 0.8, 0.8);
uniform float uMetallic      = 0.0;
uniform float uRoughness     = 0.5;
uniform float uAO            = 1.0;
uniform vec3  uEmissive      = vec3(0.0);
uniform float uEmissiveStrength = 0.0;

// 纹理开关
uniform int   uHasDiffuseTex  = 0;
uniform int   uHasNormalTex   = 0;
uniform int   uHasSpecularTex = 0;
uniform sampler2D uDiffuseTex;
uniform sampler2D uNormalTex;
uniform sampler2D uSpecularTex;

// 线框模式
uniform int   uWireframeMode = 0;
uniform vec3  uWireframeColor = vec3(0.0, 1.0, 0.4);

void main() {
    // 位置
    gPosition = vec4(fs_in.worldPos, 1.0);

    // 法线（法线映射）
    vec3 N = normalize(fs_in.normal);
    if (uHasNormalTex == 1) {
        vec3 tangent = normalize(fs_in.tangent);
        vec3 bitangent = normalize(fs_in.bitangent);
        mat3 TBN = mat3(tangent, bitangent, N);
        vec3 sampledNormal = texture(uNormalTex, fs_in.texcoord).rgb * 2.0 - 1.0;
        N = normalize(TBN * sampledNormal);
    }
    gNormal = vec4(N, 1.0);

    // 反照率 + 高光
    vec3 albedo = uBaseColor;
    if (uHasDiffuseTex == 1)
        albedo *= texture(uDiffuseTex, fs_in.texcoord).rgb;

    float specular = uMetallic;
    if (uHasSpecularTex == 1)
        specular *= texture(uSpecularTex, fs_in.texcoord).r;

    gAlbedoSpec = vec4(albedo, specular);

    // 材质参数: metallic, roughness, ao, emissive_strength
    gMaterial = vec4(uMetallic, uRoughness, uAO, uEmissiveStrength);

    // 线框覆盖
    if (uWireframeMode == 1) {
        gAlbedoSpec.rgb = uWireframeColor;
        gMaterial.g = 0.8; // 高粗糙度
    }
}
