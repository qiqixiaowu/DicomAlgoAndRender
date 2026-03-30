#version 430 core

// ================================
// 输入和输出
// ================================
in vec4 cameraPos;
in vec4 vertexPos;
in vec3 worldPos;
in vec3 worldNormal;

out vec4 outColor;

// ================================
// Uniform变量
// ================================
// 纹理采样器
uniform sampler3D volume;              // 3D体数据纹理
uniform sampler1D transferFunc;        // 1D传递函数纹理

// 变换矩阵
uniform mat4 projection;
uniform mat4 modelview;

// 体数据参数
uniform vec3 uVolumeSize;              // 体数据尺寸
uniform vec3 uSpacing;                 // 体素间距

// 渲染控制参数
uniform int uCompositeMode;            // 合成模式：0=VolRen, 1=MIP, 2=MinIP, 3=Average
uniform float stepSize;                // 采样步长
uniform float uSampleRate;             // 采样率（全局控制）
uniform float uDensityScale;           // 密度缩放

// 窗宽窗位
uniform float windowLevel;             // 窗位中心
uniform float windowWidth;             // 窗宽

// 光照参数
uniform float brightness;              // 整体亮度
uniform float ambient;                 // 环境光强度
uniform float diffuse;                 // 漫反射强度
uniform float specular;                // 镜面反射强度
uniform float shininess;               // 高光指数
uniform vec3 lightPos;                 // 光源位置
uniform vec3 lightColor;               // 光源颜色

// 高级特性参数
uniform float jitterStrength;          // 抖动强度
uniform float noiseScale;              // 噪声缩放
uniform float time;                    // 时间（用于动画）
uniform vec2 screenSize;               // 屏幕尺寸

// 环境光遮蔽参数
uniform bool enableAO;                 // 启用AO
uniform float aoStrength;              // AO强度
uniform int aoSamples;                 // AO采样数

// 边缘增强参数
uniform bool enableEdgeEnhance;        // 启用边缘增强
uniform float edgeStrength;            // 边缘强度

// 裁剪平面参数
uniform bool enableClipping;           // 启用裁剪
uniform vec4 clipPlane;                // 裁剪平面 (normal.xyz, distance)

// MPR切面叠加参数（在3D视图中显示切面位置）
uniform bool enableMPROverlay;         // 启用MPR切面叠加显示
uniform float mprAxialPos;             // Axial切面位置 [0,1]
uniform float mprSagittalPos;          // Sagittal切面位置 [0,1]
uniform float mprCoronalPos;           // Coronal切面位置 [0,1]
uniform float mprLineWidth;            // MPR线宽（纹理空间）
uniform float mprLineAlpha;            // MPR线透明度

// ================================
// 区域生长分割参数
// ================================
uniform sampler3D segMask;             // 分割掩码纹理（R通道：0=背景，1=分割区域）
uniform bool  enableSegmentation;      // 是否启用分割叠加显示
uniform vec3  segColor;                // 分割区域高亮颜色（默认橙色）
uniform float segOpacity;              // 分割区域叠加不透明度（0~1）
uniform float segBorderWidth;          // 分割边界线宽度（用于边界高亮，0关闭）

// ================================
// 常量定义
// ================================
const float MAX_DISTANCE = 10.0;       // 最大光线行进距离
const int MAX_STEPS = 512;             // 最大采样步数
const float OPACITY_THRESHOLD = 0.98;  // Alpha提前终止阈值
const float GRADIENT_DELTA = 0.005;    // 梯度计算偏移量
const float MIN_DENSITY = 0.01;        // 最小有效密度

// ================================
// 工具函数
// ================================

/**
 * 快速哈希函数（用于伪随机数生成）
 */
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

/**
 * 2D梯度噪声
 */
float gradientNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    float a = hash12(i);
    float b = hash12(i + vec2(1.0, 0.0));
    float c = hash12(i + vec2(0.0, 1.0));
    float d = hash12(i + vec2(1.0, 1.0));
    
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

/**
 * 蓝噪声模拟
 */
float blueNoise(vec2 uv, float timeOffset) {
    float noise1 = gradientNoise(uv * 1.0 + timeOffset);
    float noise2 = gradientNoise(uv * 2.0 + 100.0 + timeOffset);
    float noise3 = gradientNoise(uv * 4.0 + 200.0 + timeOffset);
    
    float noise = noise1 * 0.5 + noise2 * 0.25 + noise3 * 0.125;
    noise += (hash12(uv * 8.0 + 300.0) - 0.5) * 0.0625;
    
    return fract(noise);
}

/**
 * 屏幕空间蓝噪声
 */
float screenSpaceBlueNoise(vec2 pixelCoord) {
    vec2 uv = pixelCoord / screenSize * noiseScale;
    uv += vec2(hash12(floor(pixelCoord * 0.1)), 
               hash12(floor(pixelCoord * 0.1 + 0.5)));
    return blueNoise(uv, 0.0);
}

/**
 * 计算射线与AABB包围盒的交点
 */
bool intersectRayAABB(vec3 rayOrigin, vec3 rayDir, vec3 boxMin, vec3 boxMax,
                     out float tEnter, out float tExit) {
    vec3 invRayDir = 1.0 / (rayDir + 1e-6); // 避免除零
    vec3 t1 = (boxMin - rayOrigin) * invRayDir;
    vec3 t2 = (boxMax - rayOrigin) * invRayDir;
    
    vec3 tMin = min(t1, t2);
    vec3 tMax = max(t1, t2);
    
    float tmin = max(max(tMin.x, tMin.y), tMin.z);
    float tmax = min(min(tMax.x, tMax.y), tMax.z);
    
    if (tmax < max(tmin, 0.0)) return false;
    
    tEnter = max(tmin, 0.0);
    tExit = tmax;
    return true;
}

/**
 * 对3D纹理进行采样
 */
float sampleVolume(vec3 pos) {
    // 边界检查
    if (any(lessThan(pos, vec3(0.0))) || any(greaterThan(pos, vec3(1.0)))) {
        return 0.0;
    }
    return texture(volume, pos).r;
}

/**
 * 应用窗宽窗位
 */
float applyWindow(float density, float level, float width) {
    float windowMin = level - width / 2.0;
    float windowMax = level + width / 2.0;
    
    if (density <= windowMin) return 0.0;
    if (density >= windowMax) return 1.0;
    return (density - windowMin) / width;
}

/**
 * 计算梯度（用于光照和边缘检测）
 */
vec3 computeGradient(vec3 pos) {
    vec3 gradient;
    gradient.x = sampleVolume(pos + vec3(GRADIENT_DELTA, 0.0, 0.0)) 
               - sampleVolume(pos - vec3(GRADIENT_DELTA, 0.0, 0.0));
    gradient.y = sampleVolume(pos + vec3(0.0, GRADIENT_DELTA, 0.0)) 
               - sampleVolume(pos - vec3(0.0, GRADIENT_DELTA, 0.0));
    gradient.z = sampleVolume(pos + vec3(0.0, 0.0, GRADIENT_DELTA)) 
               - sampleVolume(pos - vec3(0.0, 0.0, GRADIENT_DELTA));
    
    return -gradient; // 梯度指向密度减小的方向，取反得到法线
}

/**
 * 改进的环境光遮蔽计算
 */
float computeAmbientOcclusion(vec3 pos, vec3 normal, float stepSize) {
    if (!enableAO) return 1.0;
    
    float occlusion = 0.0;
    int samples = aoSamples;
    float radius = stepSize * 5.0; // AO采样半径
    
    for (int i = 0; i < samples; i++) {
        // 生成半球采样方向
        float theta = float(i) / float(samples) * 6.28318; // 2π
        float phi = acos(1.0 - float(i) / float(samples));
        
        vec3 sampleDir = vec3(
            sin(phi) * cos(theta),
            sin(phi) * sin(theta),
            cos(phi)
        );
        
        // 确保采样方向在法线半球内
        if (dot(sampleDir, normal) < 0.0) {
            sampleDir = -sampleDir;
        }
        
        vec3 samplePos = pos + sampleDir * radius;
        float sampleDensity = sampleVolume(samplePos);
        
        // 累积遮蔽
        occlusion += sampleDensity;
    }
    
    occlusion = occlusion / float(samples);
    occlusion = 1.0 - clamp(occlusion * aoStrength, 0.0, 1.0);
    
    return occlusion;
}

/**
 * Blinn-Phong光照模型
 */
vec3 applyBlinnPhongLighting(vec3 color, vec3 position, vec3 normal, vec3 viewDir, float ao) {
    // 归一化法线和视图方向
    normal = normalize(normal);
    viewDir = normalize(viewDir);
    
    // 光源方向
    vec3 lightDir = normalize(lightPos - position);
    
    // 环境光（受AO影响）
    vec3 ambientColor = ambient * color * ao;
    
    // 漫反射
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuseColor = diff * diffuse * color * lightColor;
    
    // Blinn-Phong高光
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);
    vec3 specularColor = spec * specular * lightColor;
    
    // 距离衰减
    float distance = length(lightPos - position);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
    
    // 边缘光（Rim lighting）
    float rimFactor = 1.0 - max(dot(viewDir, normal), 0.0);
    vec3 rimColor = pow(rimFactor, 3.0) * 0.3 * lightColor;
    
    return ambientColor + (diffuseColor + specularColor + rimColor) * attenuation;
}

/**
 * 裁剪平面测试
 */
bool isClipped(vec3 pos) {
    if (!enableClipping) return false;
    
    // 计算点到平面的距离
    float dist = dot(vec4(pos, 1.0), clipPlane);
    return dist < 0.0;
}

/**
 * 检测采样点是否在MPR切面线上
 * 返回切面颜色（如果在线上），否则返回vec4(0)
 * Axial线 = 红色, Sagittal线 = 蓝色, Coronal线 = 绿色
 */
vec4 getMPROverlayColor(vec3 pos) {
    if (!enableMPROverlay) return vec4(0.0);
    
    float halfWidth = mprLineWidth * 0.5;
    vec4 overlayColor = vec4(0.0);
    
    // Axial切面线（Z轴位置）- 红色
    if (abs(pos.z - mprAxialPos) < halfWidth) {
        overlayColor = vec4(1.0, 0.3, 0.3, mprLineAlpha);
    }
    // Sagittal切面线（X轴位置）- 蓝色
    if (abs(pos.x - mprSagittalPos) < halfWidth) {
        vec4 sagColor = vec4(0.3, 0.3, 1.0, mprLineAlpha);
        // 如果已有颜色则混合（交叉处）
        if (overlayColor.a > 0.0) {
            overlayColor.rgb = mix(overlayColor.rgb, sagColor.rgb, 0.5);
        } else {
            overlayColor = sagColor;
        }
    }
    // Coronal切面线（Y轴位置）- 绿色
    if (abs(pos.y - mprCoronalPos) < halfWidth) {
        vec4 corColor = vec4(0.3, 1.0, 0.3, mprLineAlpha);
        if (overlayColor.a > 0.0) {
            overlayColor.rgb = mix(overlayColor.rgb, corColor.rgb, 0.5);
        } else {
            overlayColor = corColor;
        }
    }
    
    return overlayColor;
}

/**
 * 自适应步长计算（在高密度区域使用更小的步长）
 */
float adaptiveStepSize(float density, float baseStepSize) {
    // 密度越高，步长越小
    float factor = mix(1.5, 0.5, density);
    return baseStepSize * factor;
}

// ================================
// 区域生长分割辅助函数
// ================================

/**
 * 采样分割掩码
 * @return 1.0 表示该位置在分割区域内，0.0 表示背景
 */
float sampleSegMask(vec3 pos) {
    if (!enableSegmentation) return 0.0;
    return texture(segMask, pos).r;
}

/**
 * 检测分割区域边界
 * 通过采样周围6个邻域，若存在内外差异则判定为边界体素
 */
float detectSegBoundary(vec3 pos, float delta) {
    float center = sampleSegMask(pos);
    if (center < 0.5) return 0.0; // 不在分割区域内

    // 采样6个面邻居
    float px = sampleSegMask(pos + vec3( delta, 0.0,   0.0));
    float nx = sampleSegMask(pos + vec3(-delta, 0.0,   0.0));
    float py = sampleSegMask(pos + vec3(0.0,    delta, 0.0));
    float ny = sampleSegMask(pos + vec3(0.0,   -delta, 0.0));
    float pz = sampleSegMask(pos + vec3(0.0,   0.0,    delta));
    float nz = sampleSegMask(pos + vec3(0.0,   0.0,   -delta));

    // 若任意邻居与中心不同（一个在区域内一个在区域外），则为边界
    float minNeighbor = min(min(min(px, nx), min(py, ny)), min(pz, nz));
    return (minNeighbor < 0.5) ? 1.0 : 0.0;
}

/**
 * 计算分割区域的叠加颜色
 * 区域内部：半透明高亮色叠加
 * 区域边界：不透明边界线（更亮）
 */
vec4 getSegmentationOverlay(vec3 pos, float actualStepSize) {
    if (!enableSegmentation) return vec4(0.0);

    float inSeg = sampleSegMask(pos);
    if (inSeg < 0.5) return vec4(0.0);

    // 检查是否在边界
    float isBoundary = 0.0;
    if (segBorderWidth > 0.0) {
        isBoundary = detectSegBoundary(pos, segBorderWidth);
    }

    // 边界：更亮、更不透明
    if (isBoundary > 0.5) {
        return vec4(segColor * 1.5, segOpacity * 1.8);
    }

    // 区域内部：半透明叠加
    return vec4(segColor, segOpacity * 0.6);
}

// ================================
// 渲染模式实现
// ================================

/**
 * 标准体绘制（Volume Rendering）- Alpha合成
 */
vec4 volumeRendering(vec3 rayOrigin, vec3 rayDir, float tEnter, float tExit) {
    vec3 entryPoint = rayOrigin + rayDir * tEnter;
    vec3 exitPoint = rayOrigin + rayDir * tExit;
    float rayLength = distance(entryPoint, exitPoint);
    
    // 计算采样步数
    int numSteps = int(rayLength / (stepSize * uSampleRate)) + 1;
    numSteps = min(numSteps, MAX_STEPS);
    
    float actualStepSize = rayLength / float(numSteps);
    
    vec4 accumulatedColor = vec4(0.0);
    float accumulatedAlpha = 0.0;
    vec3 viewDir = normalize(-rayDir);
    
    // 屏幕空间噪声（用于抖动）
    vec2 pixelCoord = gl_FragCoord.xy;
    float baseNoise = screenSpaceBlueNoise(pixelCoord);
    
    // 光线行进
    for (int i = 0; i < numSteps; i++) {
        if (accumulatedAlpha >= OPACITY_THRESHOLD) break;
        
        // 进度参数
        float t = float(i) / float(numSteps - 1);
        
        // 抖动（减少木纹伪影）
        float noiseValue = screenSpaceBlueNoise(pixelCoord + vec2(float(i) * 0.1));
        float jitter = (noiseValue - 0.5) * jitterStrength * actualStepSize;
        
        // 采样位置
        vec3 samplePos = mix(entryPoint, exitPoint, t) + rayDir * jitter;
        
        // 边界检查
        if (any(lessThan(samplePos, vec3(0.0))) || 
            any(greaterThan(samplePos, vec3(1.0)))) {
            continue;
        }
        
        // 裁剪平面测试
        if (isClipped(samplePos)) continue;
        
        // 采样密度
        float density = sampleVolume(samplePos);
        if (density < MIN_DENSITY) continue;
        
        // 应用窗宽窗位
        float windowedDensity = applyWindow(density, windowLevel, windowWidth);
        if (windowedDensity < MIN_DENSITY) continue;
        
        // 传递函数采样
        vec4 colorSample = texture(transferFunc, windowedDensity);
        
        // 计算梯度法线
        vec3 normal = computeGradient(samplePos);
        float gradientMagnitude = length(normal);
        
        // 只在梯度足够大时应用光照（边界区域）
        if (gradientMagnitude > 0.01) {
            normal = normalize(normal);
            
            // 环境光遮蔽
            float ao = computeAmbientOcclusion(samplePos, normal, actualStepSize);
            
            // Blinn-Phong光照
            vec3 litColor = applyBlinnPhongLighting(
                colorSample.rgb, 
                samplePos, 
                normal, 
                viewDir, 
                ao
            );
            
            colorSample.rgb = litColor;
            
            // 边缘增强
            if (enableEdgeEnhance && gradientMagnitude > 0.1) {
                float edgeFactor = gradientMagnitude * edgeStrength;
                colorSample.rgb += vec3(edgeFactor);
            }
        } else {
            // 内部区域使用简单的环境光
            colorSample.rgb *= ambient;
        }
        
        // 密度缩放
        float alpha = colorSample.a * uDensityScale;
        alpha = clamp(alpha, 0.0, 1.0);
        
        // MPR切面叠加（在采样点上叠加切面线颜色）
        vec4 mprColor = getMPROverlayColor(samplePos);
        if (mprColor.a > 0.0) {
            // 将MPR线颜色与体绘制颜色混合
            colorSample.rgb = mix(colorSample.rgb, mprColor.rgb, mprColor.a * 0.6);
            alpha = max(alpha, mprColor.a * 0.4); // 确保线可见
        }
        
        // 区域生长分割叠加（高亮显示分割区域）
        vec4 segOverlay = getSegmentationOverlay(samplePos, actualStepSize);
        if (segOverlay.a > 0.0) {
            // 将分割高亮颜色与当前体素颜色混合
            colorSample.rgb = mix(colorSample.rgb, segOverlay.rgb, segOverlay.a);
            // 确保分割区域在较低密度时也可见
            alpha = max(alpha, segOverlay.a * 0.5);
        }
        
        // Front-to-back Alpha合成
        vec3 src = colorSample.rgb * alpha;
        accumulatedColor.rgb += (1.0 - accumulatedAlpha) * src;
        accumulatedAlpha += (1.0 - accumulatedAlpha) * alpha;
    }
    
    accumulatedColor.a = accumulatedAlpha;
    return accumulatedColor;
}

/**
 * 最大密度投影（MIP）
 */
vec4 maximumIntensityProjection(vec3 rayOrigin, vec3 rayDir, float tEnter, float tExit) {
    vec3 entryPoint = rayOrigin + rayDir * tEnter;
    vec3 exitPoint = rayOrigin + rayDir * tExit;
    float rayLength = distance(entryPoint, exitPoint);
    
    int numSteps = int(rayLength / (stepSize * uSampleRate)) + 1;
    numSteps = min(numSteps, MAX_STEPS);
    
    float maxIntensity = 0.0;
    vec3 maxPos = entryPoint;
    
    for (int i = 0; i < numSteps; i++) {
        float t = float(i) / float(numSteps - 1);
        vec3 samplePos = mix(entryPoint, exitPoint, t);
        
        if (isClipped(samplePos)) continue;
        
        float density = sampleVolume(samplePos);
        if (density > maxIntensity) {
            maxIntensity = density;
            maxPos = samplePos;
        }
    }
    
    // 应用窗宽窗位
    float windowedValue = applyWindow(maxIntensity, windowLevel, windowWidth);
    
    // 传递函数映射
    vec4 color = texture(transferFunc, windowedValue);
    
    // 可选：为MIP添加简单的深度提示（通过梯度）
    vec3 gradient = computeGradient(maxPos);
    if (length(gradient) > 0.01) {
        vec3 normal = normalize(gradient);
        vec3 viewDir = normalize(-rayDir);
        float NdotV = max(dot(normal, viewDir), 0.0);
        color.rgb *= (0.5 + 0.5 * NdotV); // 简单的深度提示
    }
    
    return color;
}

/**
 * 最小密度投影（MinIP）
 */
vec4 minimumIntensityProjection(vec3 rayOrigin, vec3 rayDir, float tEnter, float tExit) {
    vec3 entryPoint = rayOrigin + rayDir * tEnter;
    vec3 exitPoint = rayOrigin + rayDir * tExit;
    float rayLength = distance(entryPoint, exitPoint);
    
    int numSteps = int(rayLength / (stepSize * uSampleRate)) + 1;
    numSteps = min(numSteps, MAX_STEPS);
    
    float minIntensity = 1.0;
    vec3 minPos = entryPoint;
    
    for (int i = 0; i < numSteps; i++) {
        float t = float(i) / float(numSteps - 1);
        vec3 samplePos = mix(entryPoint, exitPoint, t);
        
        if (isClipped(samplePos)) continue;
        
        float density = sampleVolume(samplePos);
        if (density < minIntensity) {
            minIntensity = density;
            minPos = samplePos;
        }
    }
    
    float windowedValue = applyWindow(minIntensity, windowLevel, windowWidth);
    return texture(transferFunc, windowedValue);
}

/**
 * 平均值投影（Average）
 */
vec4 averageProjection(vec3 rayOrigin, vec3 rayDir, float tEnter, float tExit) {
    vec3 entryPoint = rayOrigin + rayDir * tEnter;
    vec3 exitPoint = rayOrigin + rayDir * tExit;
    float rayLength = distance(entryPoint, exitPoint);
    
    int numSteps = int(rayLength / (stepSize * uSampleRate)) + 1;
    numSteps = min(numSteps, MAX_STEPS);
    
    float sum = 0.0;
    int count = 0;
    
    for (int i = 0; i < numSteps; i++) {
        float t = float(i) / float(numSteps - 1);
        vec3 samplePos = mix(entryPoint, exitPoint, t);
        
        if (isClipped(samplePos)) continue;
        
        float density = sampleVolume(samplePos);
        sum += density;
        count++;
    }
    
    float avgIntensity = count > 0 ? sum / float(count) : 0.0;
    float windowedValue = applyWindow(avgIntensity, windowLevel, windowWidth);
    
    return texture(transferFunc, windowedValue);
}

// ================================
// 主函数
// ================================
void main() {
    const vec3 boxMin = vec3(0.0);
    const vec3 boxMax = vec3(1.0);
    
    // 计算光线
    vec3 rayOrigin = cameraPos.xyz;
    vec3 rayEnd = vertexPos.xyz;
    vec3 rayDir = normalize(rayEnd - rayOrigin);
    
    // 光线与包围盒求交
    float tEnter, tExit;
    if (!intersectRayAABB(rayOrigin, rayDir, boxMin, boxMax, tEnter, tExit)) {
        discard;
    }
    
    tEnter = max(tEnter, 0.0);
    
    // 根据合成模式选择渲染方法
    vec4 color;
    
    if (uCompositeMode == 0) {
        // 标准体绘制
        color = volumeRendering(rayOrigin, rayDir, tEnter, tExit);
    } else if (uCompositeMode == 1) {
        // 最大密度投影
        color = maximumIntensityProjection(rayOrigin, rayDir, tEnter, tExit);
    } else if (uCompositeMode == 2) {
        // 最小密度投影
        color = minimumIntensityProjection(rayOrigin, rayDir, tEnter, tExit);
    } else {
        // 平均值投影
        color = averageProjection(rayOrigin, rayDir, tEnter, tExit);
    }
    
    // 应用亮度
    color.rgb *= brightness;
    
    // Gamma校正
    color.rgb = pow(color.rgb, vec3(1.0 / 2.2));
    
    // 抗锯齿处理
    if (color.a > 0.001 && enableEdgeEnhance) {
        vec2 pixelCoord = gl_FragCoord.xy;
        float edgeNoise = screenSpaceBlueNoise(pixelCoord * 2.0);
        color.rgb += vec3(edgeNoise * 0.02 - 0.01);
    }
    
    // 最终输出检查
    if (color.a < 0.001) {
        discard;
    }
    
    outColor = vec4(color.rgb, color.a);
}
