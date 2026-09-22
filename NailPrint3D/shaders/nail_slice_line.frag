#version 330 core
in vec3 vColor;
in float vZ;

uniform float uCurrentLayer;
uniform float uLayerHeight;       // 颜色层层高
uniform float uBaseThickness;     // 底胶厚度
uniform int   uColorCount;        // 颜色数量
uniform vec3  uPalette[16];       // 调色板 (已校准)

out vec4 FragColor;

// 按层数计算颜色 — 匹配 slicer.cpp assignColors()
vec3 getLayerColor(float z) {
    // 底胶层
    if (z < uBaseThickness) {
        return vec3(0.96, 0.94, 0.92);
    }

    int colorLayer = int((z - uBaseThickness) / max(uLayerHeight, 0.001));
    int colorIndex = (colorLayer / 3) % max(uColorCount, 1);

    if (uColorCount == 0) return vColor;
    return uPalette[colorIndex];
}

void main() {
    // 按层数着色
    vec3 layerColor = getLayerColor(vZ);

    // 当前层高亮，其他层暗化
    float layerDiff = abs(vZ - uCurrentLayer);
    float highlight = 1.0 / (1.0 + layerDiff * 2.0);

    // 当前层用全色，其他层暗化
    vec3 color = layerColor * (0.3 + highlight * 0.7);

    FragColor = vec4(color, 0.4 + highlight * 0.6);
}
