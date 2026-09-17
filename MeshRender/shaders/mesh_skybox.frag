#version 430 core
/**
 * mesh_skybox.frag - 天空盒片元着色器（程序化渐变天空）
 */
in vec3 vWorldDir;
out vec4 FragColor;

uniform vec3 uTopColor    = vec3(0.3, 0.5, 0.8);
uniform vec3 uBottomColor = vec3(0.7, 0.8, 0.95);
uniform vec3 uSunDir      = normalize(vec3(0.5, 0.4, 0.3));
uniform vec3 uSunColor    = vec3(1.0, 0.9, 0.7);

void main() {
    vec3 dir = normalize(vWorldDir);
    float t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 sky = mix(uBottomColor, uTopColor, smoothstep(0.0, 0.6, t));

    // 太阳光晕
    float sunDot = max(dot(dir, uSunDir), 0.0);
    float sunGlow = pow(sunDot, 64.0) * 0.5 + pow(sunDot, 8.0) * 0.2;
    sky += uSunColor * sunGlow;

    FragColor = vec4(sky, 1.0);
}
