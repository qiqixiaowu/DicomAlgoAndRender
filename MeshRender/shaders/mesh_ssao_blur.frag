#version 430 core
/**
 * mesh_ssao_blur.frag - SSAO 模糊通道
 */
out float outColor;
in vec2 vTexcoord;

uniform sampler2D uSSAOInput;

void main() {
    vec2 texelSize = 1.0 / vec2(textureSize(uSSAOInput, 0));
    float result = 0.0;
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            result += texture(uSSAOInput, vTexcoord + offset).r;
        }
    }
    outColor = result / 25.0;
}
