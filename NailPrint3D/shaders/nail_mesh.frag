#version 330 core
in vec3 vNormal;
in vec3 vColor;
in vec3 vWorldPos;
in vec2 vUV;

uniform vec3 uLightDir;
uniform vec3 uViewPos;
uniform vec4 uBaseColor;

out vec4 FragColor;

// 程序化纹理：在UV空间生成装饰图案
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

void main() {
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(uLightDir);

    // 环境光
    float ambient = 0.3;
    // 漫反射
    float diff = max(dot(normal, lightDir), 0.0);
    // 镜面反射 (Blinn-Phong)
    vec3 viewDir = normalize(uViewPos - vWorldPos);
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), 64.0);

    // 菲涅尔效应 (美甲光泽)
    float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0);

    // 程序化纹理 + 顶点颜色混合
    vec3 procColor = proceduralPattern(vUV);
    vec3 baseCol = mix(procColor, vColor, 0.6);
    baseCol = mix(baseCol, uBaseColor.rgb, 0.15);

    // 最终光照
    vec3 color = baseCol * (ambient + diff * 0.65);
    
    // 高光：白色光泽
    color += spec * 0.6 * vec3(1.0, 0.98, 0.95);
    
    // 菲涅尔边缘光：粉红光泽
    color += fresnel * 0.4 * vec3(1.0, 0.7, 0.8);

    // 轻微伽马校正
    color = pow(color, vec3(0.9));

    FragColor = vec4(color, 1.0);
}
