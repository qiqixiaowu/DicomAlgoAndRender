#version 430 core

uniform sampler3D volume;
uniform sampler1D transferFunc;
uniform float windowLevel = 0.5;
uniform float windowWidth = 1.0;
uniform float stepSize = 0.001;
uniform float brightness = 1.5;
uniform float ambient = 0.3;
uniform float diffuse = 0.7;
uniform float specular = 0.3;
uniform float shininess = 32.0;
uniform vec3 lightPos = vec3(2.0, 2.0, 2.0);
uniform vec3 lightColor = vec3(1.0, 1.0, 0.9);

// 抖动参数
uniform float jitterStrength = 0.5;
uniform float noiseScale = 1.0;
uniform float time = 0.0; // 时间参数用于动画
uniform vec2 screenSize = vec2(1920.0, 1080.0);

in vec4 cameraPos;
in vec4 vertexPos;
in vec3 worldPos;
in vec3 worldNormal;

out vec4 outColor;

// 快速哈希函数（用于伪随机数生成）
float hash12(vec2 p)
{
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

// 改进的哈希函数（更好的分布）
float hash13(vec3 p)
{
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

// 2D 梯度噪声（类似Perlin噪声）
float gradientNoise(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    // 四个角点的随机梯度
    float a = hash12(i);
    float b = hash12(i + vec2(1.0, 0.0));
    float c = hash12(i + vec2(0.0, 1.0));
    float d = hash12(i + vec2(1.0, 1.0));
    
    // 立方体插值（smoothstep的改进版）
    vec2 u = f * f * (3.0 - 2.0 * f);
    
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// 多层分形噪声（用于更好的随机分布）
float fractalNoise(vec2 p, int octaves)
{
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    
    for (int i = 0; i < octaves; i++)
    {
        value += amplitude * gradientNoise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return value;
}

// 蓝噪声模拟（使用泊松圆盘分布的近似）
float blueNoise(vec2 uv, float timeOffset)
{
    // 使用多种噪声层叠加
    float noise1 = gradientNoise(uv * 1.0 + timeOffset);
    float noise2 = gradientNoise(uv * 2.0 + 100.0 + timeOffset);
    float noise3 = gradientNoise(uv * 4.0 + 200.0 + timeOffset);
    
    // 加权组合（低频权重高，高频权重低）
    float noise = noise1 * 0.5 + noise2 * 0.25 + noise3 * 0.125;
    
    // 添加一些高频分量
    noise += (hash12(uv * 8.0 + 300.0) - 0.5) * 0.0625;
    
    // 确保在 [0,1] 范围内
    return fract(noise);
}

// 时间相关的蓝噪声（随时间变化）
float animatedBlueNoise(vec2 uv)
{
    // 基本噪声
    float baseNoise = blueNoise(uv, 0.0);
    
    // 时间相关的扰动
    float timeNoise = sin(time * 0.5) * 0.1;
    
    // 添加次要噪声层（随时间变化）
    float movingNoise = gradientNoise(uv * 0.5 + vec2(time * 0.1, time * 0.2));
    
    // 混合噪声
    float finalNoise = mix(baseNoise, movingNoise, 0.3);
    
    return finalNoise;
}

// 屏幕空间蓝噪声（基于像素坐标）
float screenSpaceBlueNoise(vec2 pixelCoord)
{
    // 归一化像素坐标
    vec2 uv = pixelCoord / screenSize * noiseScale;
    
    // 添加像素级偏移以避免重复模式
    uv += vec2(hash12(floor(pixelCoord * 0.1)), 
               hash12(floor(pixelCoord * 0.1 + 0.5)));
    
    return blueNoise(uv, 0.0);
}

// 计算抖动值（基于像素和步进）
float computeJitter(vec2 pixelCoord, int stepIndex)
{
    // 使用像素坐标和步进索引生成唯一噪声
    vec3 seed = vec3(pixelCoord, float(stepIndex));
    
    // 使用3D哈希函数
    return hash13(seed * 0.1);
}

// 高级抖动函数（结合时间和空间）
float advancedJitter(vec2 pixelCoord, int stepIndex, float temporalOffset)
{
    // 空间分量
    float spatial = screenSpaceBlueNoise(pixelCoord);
    
    // 步进分量
    float stepWise = computeJitter(pixelCoord, stepIndex);
    
    // 时间分量
    float temporal = sin(float(stepIndex) * 0.1 + time + temporalOffset) * 0.5 + 0.5;
    
    // 加权组合
    return spatial * 0.4 + stepWise * 0.4 + temporal * 0.2;
}

bool intersectRayAABB(vec3 rayOrigin, vec3 rayDir, vec3 boxMin, vec3 boxMax,
                     out float tEnter, out float tExit)
{
    vec3 invRayDir = 1.0 / rayDir;
    vec3 t1 = (boxMin - rayOrigin) * invRayDir;
    vec3 t2 = (boxMax - rayOrigin) * invRayDir;
    
    vec3 tMin = min(t1, t2);
    vec3 tMax = max(t1, t2);
    
    float tmin = max(max(tMin.x, tMin.y), tMin.z);
    float tmax = min(min(tMax.x, tMax.y), tMax.z);
    
    if (tmax < max(tmin, 0.0)) return false;
    
    tEnter = tmin;
    tExit = tmax;
    return true;
}

float applyWindow(float density, float level, float width) 
{
    float windowMin = level - width / 2.0;
    float windowMax = level + width / 2.0;

    if (density <= windowMin) return 0.0;
    if (density >= windowMax) return 1.0;
    return (density - windowMin) / width;
}

vec3 computeSimpleGradient(vec3 pos)
{
    const float eps = 0.01;
    vec3 grad;
    
    grad.x = texture(volume, pos + vec3(eps, 0.0, 0.0)).r 
             - texture(volume, pos - vec3(eps, 0.0, 0.0)).r;
    grad.y = texture(volume, pos + vec3(0.0, eps, 0.0)).r 
             - texture(volume, pos - vec3(0.0, eps, 0.0)).r;
    grad.z = texture(volume, pos + vec3(0.0, 0.0, eps)).r 
             - texture(volume, pos - vec3(0.0, 0.0, eps)).r;
    
    return normalize(grad);
}

vec3 applyPhongLighting(vec3 color, vec3 position, vec3 normal, vec3 viewDir)
{
    vec3 lightDir = normalize(lightPos - position);
    
    vec3 ambientColor = ambient * color;
    
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuseColor = diff * diffuse * color * lightColor;
    
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specularColor = spec * specular * lightColor;
    
    float distance = length(lightPos - position);
    float attenuation = 1.0 / (1.0 + 0.1 * distance + 0.01 * distance * distance);
    
    return ambientColor + (diffuseColor + specularColor) * attenuation;
}

void main()
{
    const vec3 boxMin = vec3(0.0);
    const vec3 boxMax = vec3(1.0);
    
    vec3 rayOrigin = cameraPos.xyz;
    vec3 rayEnd = vertexPos.xyz;
    vec3 rayDir = normalize(rayEnd - rayOrigin);
    
    // 计算射线与边界框的交点
    float tEnter, tExit;
    if (!intersectRayAABB(rayOrigin, rayDir, boxMin, boxMax, tEnter, tExit))
    {
        discard;
    }
    
    tEnter = max(tEnter, 0.0);
    vec3 entryPoint = rayOrigin + rayDir * tEnter;
    vec3 exitPoint = rayOrigin + rayDir * tExit;
    
    float rayLength = distance(entryPoint, exitPoint);
    int numSteps = int(rayLength / stepSize) + 1;
    
    vec4 accumulatedColor = vec4(0.0);
    float accumulatedAlpha = 0.0;
    
    // 视图方向
    vec3 viewDir = normalize(-rayDir);
    
    // 像素坐标（用于噪声生成）
    vec2 pixelCoord = gl_FragCoord.xy;
    
    // 主要渲染循环
    for (int i = 0; i < numSteps; ++i)
    {
        if (accumulatedAlpha > 0.99) break;
        
        // 计算抖动（多种方法可选）
        float jitter = 0.0;
        
        // 方法1：简单哈希抖动
        // jitter = computeJitter(pixelCoord, i) * jitterStrength * stepSize;
        
        // 方法2：蓝噪声抖动（推荐）
        float noiseValue = screenSpaceBlueNoise(pixelCoord + vec2(float(i) * 0.1));
        jitter = (noiseValue - 0.5) * jitterStrength * stepSize;
        
        // 方法3：高级抖动（结合时间和空间）
        // jitter = (advancedJitter(pixelCoord, i, 0.0) - 0.5) * jitterStrength * stepSize;
        
        // 计算采样位置（应用抖动）
        float t = float(i) / float(numSteps - 1);
        vec3 basePos = mix(entryPoint, exitPoint, t);
        vec3 samplePos = basePos + rayDir * jitter;
        
        // 边界检查
        if (any(lessThan(samplePos, boxMin - 0.001)) || 
            any(greaterThan(samplePos, boxMax + 0.001)))
        {
            break;
        }
        
        // 密度采样
        float density = texture(volume, samplePos).r;
        float windowedDensity = applyWindow(density, windowLevel, windowWidth);
        
        if (windowedDensity < 0.01) continue;
        
        // 传输函数采样
        vec4 colorSample = texture(transferFunc, windowedDensity);
        
        // 计算法线
        vec3 normal = computeSimpleGradient(samplePos);
        
        // 应用光照
        vec3 litColor = applyPhongLighting(colorSample.rgb, samplePos, normal, viewDir);
        
        // 阴影因子
        vec3 lightDir = normalize(lightPos - samplePos);
        float shadowFactor = max(dot(normal, lightDir), 0.0);
        litColor *= mix(0.5, 1.0, shadowFactor);
        
        // 透明度调整
        float alpha = colorSample.a * windowedDensity * 1.5;
        alpha = clamp(alpha, 0.0, 1.0);
        
        // 前向Alpha混合
        accumulatedColor.rgb += (1.0 - accumulatedAlpha) * alpha * litColor;
        accumulatedAlpha += (1.0 - accumulatedAlpha) * alpha;
    }
    
    // 后处理：抗锯齿边缘增强
    if (accumulatedAlpha > 0.1)
    {
        // 基于噪声的边缘检测（减少摩尔纹）
        float edgeNoise = screenSpaceBlueNoise(pixelCoord * 2.0);
        float edgeEnhance = 1.0 + 0.2 * edgeNoise;
        accumulatedColor.rgb *= edgeEnhance;
    }
    
    // 应用亮度
    accumulatedColor.rgb *= brightness;
    accumulatedColor.a = accumulatedAlpha;
    
    // 色调映射（可选，减少色带）
    accumulatedColor.rgb = pow(accumulatedColor.rgb, vec3(1.0/2.2));
    
    // 最终输出检查
    if (accumulatedAlpha < 0.001)
    {
        discard;
    }
    
    outColor = vec4(accumulatedColor.rgb, accumulatedAlpha);
}