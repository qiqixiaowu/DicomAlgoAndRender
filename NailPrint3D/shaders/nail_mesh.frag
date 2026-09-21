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
uniform float uReliefHeight;     // 3D浮雕高度（视差映射用）
uniform float uIridescenceIntensity; // 流光溢彩强度 (0=关, 1=最强)
uniform float uTime;             // 动画时间（流光流动）

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
// 流光溢彩（虹彩/珠光效果）
// 原理：根据视角和法线的关系，将波长映射到RGB色相偏移
//       叠加动态流动的彩色光带，模拟珠光甲油
// ============================================================
vec3 iridescence(vec3 normal, vec3 viewDir, vec2 uv, float intensity) {
    // 菲涅尔系数：掠射角时虹彩最强
    float NdotV = max(dot(normal, viewDir), 0.0);
    float fresnel = pow(1.0 - NdotV, 5.0);

    // 薄膜干涉色（基于波长的RGB映射）
    // phase 控制色相偏移，与视角和位置相关
    float phase = (1.0 - NdotV) * 6.0 + uv.y * 3.0 + uTime * 0.5;

    // 用三角函数生成彩虹色环
    vec3 rainbow;
    rainbow.r = 0.5 + 0.5 * sin(phase);
    rainbow.g = 0.5 + 0.5 * sin(phase + 2.094);  // +120度
    rainbow.b = 0.5 + 0.5 * sin(phase + 4.189);  // +240度

    // 流动光带：沿UV方向移动的亮带
    float band = sin(uv.x * 8.0 + uv.y * 4.0 + uTime * 2.0);
    band = pow(abs(band), 3.0) * sign(band);

    // 珠光底色：柔和的粉白渐变
    vec3 pearlBase = mix(
        vec3(0.95, 0.90, 0.93),  // 浅粉白
        vec3(0.88, 0.82, 0.92),  // 淡紫白
        uv.y
    );

    // 混合：珠光底 + 虹彩（受菲涅尔调制）+ 流动光带
    vec3 iriColor = pearlBase;
    iriColor += rainbow * fresnel * 0.6;
    iriColor += rainbow * band * 0.15;
    iriColor += vec3(1.0, 0.95, 0.9) * fresnel * 0.3;  // 高光泛白

    return mix(pearlBase, iriColor, intensity);
}

// ============================================================
// 视差遮蔽映射（Parallax Occlusion Mapping）
// 用纹理亮度模拟表面凹凸，让3D浮雕在像素级也有立体感
// ============================================================
vec2 parallaxMapping(vec2 uv, vec3 viewDir) {
    if (uReliefHeight <= 0.001) return uv;

    // 层数：根据视角动态调整
    float numLayers = 16.0;
    float layerHeight = 1.0 / numLayers;

    // 视角越斜，位移越大
    vec2 P = viewDir.xy * uReliefHeight;
    vec2 deltaUV = P / numLayers;

    // 从最深层开始逐步向上搜索
    float currentLayerHeight = 0.0;
    vec2 currentUV = uv;
    float currentDepth = texture(uTexture, currentUV).a; // 用alpha做深度

    // 如果没有alpha通道，用亮度做深度
    if (currentDepth < 0.01) {
        vec3 c = texture(uTexture, currentUV).rgb;
        currentDepth = dot(c, vec3(0.299, 0.587, 0.114));
    }

    while (currentLayerHeight < currentDepth) {
        currentUV -= deltaUV;
        vec3 c = texture(uTexture, currentUV).rgb;
        currentDepth = texture(uTexture, currentUV).a;
        if (currentDepth < 0.01)
            currentDepth = dot(c, vec3(0.299, 0.587, 0.114));
        currentLayerHeight += layerHeight;
    }

    // 插值平滑
    vec2 prevUV = currentUV + deltaUV;
    float prevDepth = texture(uTexture, prevUV).a;
    if (prevDepth < 0.01) {
        vec3 c = texture(uTexture, prevUV).rgb;
        prevDepth = dot(c, vec3(0.299, 0.587, 0.114));
    }
    float prevLayer = currentLayerHeight - layerHeight;

    float after = currentDepth - currentLayerHeight;
    float before = prevDepth - prevLayer;
    float weight = after / (after - before);

    return mix(currentUV, prevUV, weight);
}

// ============================================================
// 从高度图生成法线扰动（让浮雕边缘有正确的光照变化）
// ============================================================
vec3 getNormalFromHeightMap(vec2 uv, vec3 baseNormal) {
    if (uReliefHeight <= 0.001) return baseNormal;

    float texelSize = 1.0 / 512.0;
    vec2 offset = vec2(texelSize, 0.0);

    // 采样相邻像素的亮度
    float hL = dot(texture(uTexture, uv - offset.xy).rgb, vec3(0.299, 0.587, 0.114));
    float hR = dot(texture(uTexture, uv + offset.xy).rgb, vec3(0.299, 0.587, 0.114));
    float hD = dot(texture(uTexture, uv - offset.yx).rgb, vec3(0.299, 0.587, 0.114));
    float hU = dot(texture(uTexture, uv + offset.yx).rgb, vec3(0.299, 0.587, 0.114));

    // Sobel法线计算
    vec3 normalMap = vec3(hL - hR, hD - hU, 1.0);
    normalMap = normalize(normalMap);

    // 将切线空间法线变换到世界空间（简化版：直接混合）
    float strength = uReliefHeight * 2.0;
    vec3 perturbed = normalize(baseNormal + (normalMap - vec3(0,0,1)) * strength);

    return perturbed;
}

// ============================================================
// 主函数
// ============================================================
void main() {
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(uLightDir);
    vec3 viewDir = normalize(uViewPos - vWorldPos);

    // === 视差映射：如果有纹理且有浮雕高度，偏移UV ===
    vec2 effectiveUV = vUV;
    if (uTextureEnabled == 1 && uReliefHeight > 0.001) {
        vec2 texUV = transformUV(vUV);
        effectiveUV = transformUV(parallaxMapping(texUV, viewDir));
    } else if (uTextureEnabled == 1) {
        effectiveUV = transformUV(vUV);
    }

    // === 法线扰动：从高度图生成凹凸法线 ===
    if (uTextureEnabled == 1 && uReliefHeight > 0.001) {
        normal = getNormalFromHeightMap(effectiveUV, normal);
    }

    // 基础光照计算（所有模式共用）
    float ambient = 0.3;
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), 64.0);
    float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0);

    // === 获取基础颜色 ===
    vec3 baseCol;

    if (uTextureEnabled == 1) {
        // 有纹理：用视差偏移后的UV采样
        vec3 texColor = texture(uTexture, effectiveUV).rgb;

        if (uPatternMode == 1) {
            // === Photo 模式：照片/头像 ===
            // 直接使用纹理颜色，保留完整细节和亮度
            baseCol = texColor;
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
            vec4 texSample = texture(uTexture, effectiveUV);
            // 文字部分用纹理颜色，非文字部分用程序化底色
            float textMask = texSample.a;
            vec3 textColor = texSample.rgb;
            baseCol = mix(proceduralPattern(vUV), textColor, textMask * uOpacity);

        } else if (uPatternMode == 5) {
            // === Iridescent 模式：流光溢彩 ===
            // 虹彩效果与纹理混合
            vec3 iri = iridescence(normal, viewDir, vUV, uIridescenceIntensity);
            if (uTextureEnabled == 1) {
                baseCol = mix(texColor, iri, uIridescenceIntensity);
            } else {
                baseCol = iri;
            }
        } else {
            // 默认：直接使用纹理
            baseCol = blendColors(proceduralPattern(vUV), texColor, uBlendMode, uOpacity);
        }
    } else {
        // 无纹理：顶点颜色为主，程序化纹理为辅
        // 流光溢彩模式：用虹彩替代程序化纹理
        if (uPatternMode == 5) {
            baseCol = iridescence(normal, viewDir, vUV, uIridescenceIntensity);
        } else {
            vec3 procColor = proceduralPattern(vUV);
            baseCol = mix(procColor, vColor, 0.85);
        }
    }

    // 与基础色调混合（仅轻微影响，保留装饰物多色）
    // Photo 模式跳过此步，保留图片原色
    if (uPatternMode != 1) {
        baseCol = mix(baseCol, uBaseColor.rgb, 0.05);
    }

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

    } else if (uPatternMode == 5) {
        // Iridescent：流光溢彩专用光照
        // 增强高光和菲涅尔，让虹彩更耀眼
        color = baseCol * (ambient + diff * 0.4);
        color += spec * 1.2 * vec3(1.0, 0.98, 0.95);
        color += fresnel * 0.8 * baseCol;  // 菲涅尔用自身颜色，增强边缘虹彩

    } else if (uPatternMode == 1) {
        // Photo：照片模式 — 高亮度，保留原图色彩
        // 环境光提到0.8，漫反射满额，让图片明亮清晰
        color = baseCol * (0.8 + diff * 0.3);
        color += spec * 0.3 * vec3(1.0, 0.98, 0.95);
        // 不加菲涅尔边缘色，避免图片偏色

    } else {
        // Photo / Procedural：完整 Blinn-Phong
        color = baseCol * (ambient + diff * 0.65);
        color += spec * 0.6 * vec3(1.0, 0.98, 0.95);
        color += fresnel * 0.4 * vec3(1.0, 0.7, 0.8);
    }

    // 伽马校正：Photo模式用标准2.2→1.0转换避免过亮，其他模式用0.9微调
    if (uPatternMode == 1) {
        color = pow(color, vec3(1.0));  // 线性输出，保留原图亮度
    } else {
        color = pow(color, vec3(0.9));
    }

    FragColor = vec4(color, 1.0);
}
