#version 330 core
in vec3 vNormal;
in vec3 vColor;
in vec3 vWorldPos;
in vec2 vUV;

uniform vec3 uLightDir;
uniform vec3 uViewPos;
uniform vec4 uBaseColor;

// 纹理相关
uniform sampler2D uTexture;
uniform int   uTextureEnabled;   // 0=无纹理, 1=有纹理
uniform int   uPatternMode;      // 0=Procedural, 1=Photo, 2=Cartoon, 3=FlatColor, 4=Text
uniform float uTexOffsetX;
uniform float uTexOffsetY;
uniform float uTexScale;
uniform float uTexRotation;
uniform float uOpacity;          // 图案不透明度
uniform int   uBlendMode;        // 0=正常, 1=正片叠底, 2=滤色, 3=覆盖

out vec4 FragColor;

// ============================================================
// UV 变换（平移/缩放/旋转）
// ============================================================
vec2 transformUV(vec2 uv) {
    vec2 t = uv - vec2(0.5);
    float c = cos(uTexRotation);
    float s = sin(uTexRotation);
    t = vec2(t.x * c - t.y * s, t.x * s + t.y * c);
    t /= uTexScale;
    t += vec2(0.5);
    t += vec2(uTexOffsetX, uTexOffsetY);
    return t;
}

// ============================================================
// 程序化纹理：在UV空间生成装饰图案
// ============================================================
vec3 proceduralPattern(vec2 uv) {
    // 底色渐变：从根部到指尖
    vec3 baseGrad = mix(
        vec3(0.95, 0.80, 0.85),  // 根部浅粉
        vec3(0.88, 0.65, 0.78),  // 指尖深粉
        uv.y
    );

    // 光泽条纹（模拟指甲自然纹理）
    float stripe = sin(uv.x * 60.0) * 0.5 + 0.5;
    stripe = pow(stripe, 8.0) * 0.08;

    // 月牙（指甲根部半月）
    float lunulaDist = distance(vec2(uv.x * 0.8, uv.y + 0.15), vec2(0.4, 0.0));
    float lunula = 1.0 - smoothstep(0.12, 0.18, lunulaDist);

    // 指尖白色尖端
    float tipDist = uv.y;
    float tip = smoothstep(0.85, 1.0, tipDist) * 0.3;

    vec3 color = baseGrad;
    color += stripe;
    color = mix(color, vec3(0.98, 0.95, 0.90), lunula * 0.4);
    color = mix(color, vec3(0.95, 0.92, 0.95), tip);

    return color;
}

// ============================================================
// 混合模式
// ============================================================
vec3 blendColors(vec3 base, vec3 pattern, int mode, float opacity) {
    vec3 result;
    if (mode == 0) {
        // 正常
        result = pattern;
    } else if (mode == 1) {
        // 正片叠底
        result = base * pattern;
    } else if (mode == 2) {
        // 滤色
        result = 1.0 - (1.0 - base) * (1.0 - pattern);
    } else {
        // 覆盖
        result = mix(1.0 - 2.0 * (1.0 - base) * (1.0 - pattern),
                     2.0 * base * pattern,
                     step(base, vec3(0.5)));
    }
    return mix(base, result, opacity);
}

// ============================================================
// 卡通风格：posterize + 描边
// ============================================================
vec3 cartoonEffect(vec3 color, vec3 normal, vec2 uv) {
    // 色彩量化（posterize）：每通道只保留4个级别
    vec3 quantized = floor(color * 4.0) / 4.0;

    // 边缘检测：用UV空间的导数模拟描边
    vec2 px = vec2(dFdx(uv.x), dFdy(uv.y));
    float edge = length(px) * 50.0;
    edge = smoothstep(0.0, 0.3, edge);

    // 如果接近边缘，画黑线
    if (edge > 0.5) {
        quantized = mix(quantized, vec3(0.1, 0.05, 0.1), 0.6);
    }

    return quantized;
}

// ============================================================
// 主函数
// ============================================================
void main() {
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(uLightDir);
    vec3 viewDir = normalize(uViewPos - vWorldPos);

    // 基础光照计算（所有模式共用）
    float ambient = 0.3;
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), 64.0);
    float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0);

    // === 获取基础颜色 ===
    vec3 baseCol;

    if (uTextureEnabled == 1) {
        // 有纹理：采样纹理
        vec2 texUV = transformUV(vUV);
        vec3 texColor = texture(uTexture, texUV).rgb;

        if (uPatternMode == 1) {
            // === Photo 模式：照片/头像 ===
            // 直接使用纹理颜色，保留完整细节
            baseCol = blendColors(proceduralPattern(vUV), texColor, uBlendMode, uOpacity);

        } else if (uPatternMode == 2) {
            // === Cartoon 模式：卡通风格 ===
            vec3 blended = blendColors(proceduralPattern(vUV), texColor, uBlendMode, uOpacity);
            baseCol = cartoonEffect(blended, normal, vUV);

        } else if (uPatternMode == 3) {
            // === FlatColor 模式：纯色块 ===
            // 量化到更少色阶，去掉渐变
            vec3 flatCol = floor(texColor * 3.0) / 3.0;
            baseCol = blendColors(proceduralPattern(vUV), flatCol, uBlendMode, uOpacity);

        } else if (uPatternMode == 4) {
            // === Text 模式：文字/SDF ===
            // 锐利采样，不做mipmap插值
            vec2 texUV2 = transformUV(vUV);
            vec4 texSample = texture(uTexture, texUV2);
            // 文字部分用纹理颜色，非文字部分用程序化底色
            float textMask = texSample.a;
            vec3 textColor = texSample.rgb;
            baseCol = mix(proceduralPattern(vUV), textColor, textMask * uOpacity);

        } else {
            // 默认：直接使用纹理
            baseCol = blendColors(proceduralPattern(vUV), texColor, uBlendMode, uOpacity);
        }
    } else {
        // 无纹理：程序化纹理 + 顶点颜色
        vec3 procColor = proceduralPattern(vUV);
        baseCol = mix(procColor, vColor, 0.6);
    }

    // 与基础色调混合
    baseCol = mix(baseCol, uBaseColor.rgb, 0.15);

    // === 根据模式调整光照 ===
    vec3 color;

    if (uPatternMode == 3) {
        // FlatColor：仅环境光 + 轻微高光
        color = baseCol * (ambient + 0.2);
        color += spec * 0.3 * vec3(1.0, 0.98, 0.95);

    } else if (uPatternMode == 2) {
        // Cartoon：简化光照（3级量化）
        float lightLevel = ambient + diff * 0.65;
        lightLevel = floor(lightLevel * 3.0) / 3.0;  // 量化光照
        color = baseCol * lightLevel;
        color += spec * 0.4 * vec3(1.0, 0.98, 0.95);

    } else if (uPatternMode == 4) {
        // Text：仅环境光，保持文字清晰
        color = baseCol * (ambient + 0.4);
        color += spec * 0.2 * vec3(1.0, 0.98, 0.95);

    } else {
        // Photo / Procedural：完整 Blinn-Phong
        color = baseCol * (ambient + diff * 0.65);
        color += spec * 0.6 * vec3(1.0, 0.98, 0.95);
        color += fresnel * 0.4 * vec3(1.0, 0.7, 0.8);
    }

    // 轻微伽马校正
    color = pow(color, vec3(0.9));

    FragColor = vec4(color, 1.0);
}
