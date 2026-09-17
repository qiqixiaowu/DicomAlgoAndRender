#version 430 core
/**
 * mesh_ssao.frag - SSAO 屏幕空间环境光遮蔽
 */
out float outColor;

in vec2 vTexcoord;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D uNoiseTex;

uniform vec2 uScreenSize;
uniform vec3 uSamples[64];
uniform mat4 uProjection;
uniform mat4 uView;
uniform int   uKernelSize = 64;
uniform float uRadius = 0.5;
uniform float uBias = 0.025;

void main() {
    vec3 fragPos   = texture(gPosition, vTexcoord).rgb;
    vec3 normal    = texture(gNormal, vTexcoord).rgb;

    if (length(fragPos) < 0.001) {
        outColor = 1.0;
        return;
    }

    // 视图空间位置
    vec4 viewPos = uView * vec4(fragPos, 1.0);
    fragPos = viewPos.xyz;
    vec4 viewNormal = uView * vec4(normal, 0.0);
    normal = normalize(viewNormal.xyz);

    // 噪声旋转向量
    vec2 noiseScale = uScreenSize / 4.0;
    vec3 randomVec  = texture(uNoiseTex, vTexcoord * noiseScale).xyz;

    vec3 tangent   = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN       = mat3(tangent, bitangent, normal);

    float occlusion = 0.0;
    for (int i = 0; i < uKernelSize; i++) {
        vec3 samplePos = TBN * uSamples[i];
        samplePos = fragPos + samplePos * uRadius;

        vec4 offset = uProjection * vec4(samplePos, 1.0);
        offset.xyz /= offset.w;
        offset.xyz = offset.xyz * 0.5 + 0.5;

        float sampleDepth = (uView * vec4(texture(gPosition, offset.xy).rgb, 1.0)).z;
        occlusion += (sampleDepth >= samplePos.z + uBias ? 1.0 : 0.0);
    }
    occlusion = 1.0 - (occlusion / float(uKernelSize));
    outColor = occlusion;
}
