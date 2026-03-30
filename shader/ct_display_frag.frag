#version 430 core

// ============================================================================
// CT 正弦图显示 / 重建结果显示 - 片元着色器
// 
// 用于将浮点纹理映射到屏幕显示
// 支持窗宽窗位调整
// ============================================================================

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D displayTex;   // 待显示的纹理
uniform float windowCenter;      // 窗位
uniform float windowWidth;       // 窗宽
uniform int   colorMode;         // 0=灰度, 1=热力图, 2=反色

// 热力图颜色映射 (类似 jet colormap)
vec3 heatmap(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c;
    if (t < 0.25) {
        c = vec3(0.0, 4.0 * t, 1.0);
    } else if (t < 0.5) {
        c = vec3(0.0, 1.0, 1.0 - 4.0 * (t - 0.25));
    } else if (t < 0.75) {
        c = vec3(4.0 * (t - 0.5), 1.0, 0.0);
    } else {
        c = vec3(1.0, 1.0 - 4.0 * (t - 0.75), 0.0);
    }
    return c;
}

void main() {
    float val = texture(displayTex, TexCoord).r;

    // 窗宽窗位映射
    float lower = windowCenter - windowWidth * 0.5;
    float upper = windowCenter + windowWidth * 0.5;
    float mapped = clamp((val - lower) / (upper - lower), 0.0, 1.0);

    vec3 color;
    if (colorMode == 0) {
        color = vec3(mapped);           // 灰度
    } else if (colorMode == 1) {
        color = heatmap(mapped);        // 热力图
    } else {
        color = vec3(1.0 - mapped);     // 反色
    }

    FragColor = vec4(color, 1.0);
}
