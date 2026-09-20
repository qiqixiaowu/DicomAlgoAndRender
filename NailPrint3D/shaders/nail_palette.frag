#version 330 core
in vec3 vColor;
in vec3 vNormal;
in vec3 vWorldPos;
in vec2 vUV;

uniform vec4 uPalette[16];
uniform int uPaletteSize;

out vec4 FragColor;

void main() {
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));

    float diff = max(dot(normal, lightDir), 0.0);
    float ambient = 0.35;

    vec3 viewDir = normalize(vec3(0, 0, 5) - vWorldPos);
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), 48.0);
    float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0);

    // 找到最接近的调色板颜色
    float minDist = 999.0;
    vec3 paletteColor = vColor;
    for (int i = 0; i < uPaletteSize; i++) {
        float d = distance(vColor, uPalette[i].rgb);
        if (d < minDist) {
            minDist = d;
            paletteColor = uPalette[i].rgb;
        }
    }

    // UV 渐变叠加
    vec3 gradTop = mix(paletteColor, vec3(1.0), 0.15 * vUV.y);
    vec3 gradBot = mix(paletteColor, vec3(0.7, 0.6, 0.7), 0.2 * (1.0 - vUV.y));
    vec3 baseColor = mix(gradBot, gradTop, vUV.y);

    vec3 color = baseColor * (ambient + diff * 0.65);
    color += spec * 0.5 * vec3(1.0, 0.98, 0.95);
    color += fresnel * 0.3 * paletteColor;

    FragColor = vec4(color, 1.0);
}
