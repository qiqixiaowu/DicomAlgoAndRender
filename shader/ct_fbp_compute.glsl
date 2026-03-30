#version 430 core

// ============================================================================
// CT 滤波反投影 (FBP) - GPU Compute Shader
//
// 工作方式: 每个工作项计算一个输出像素的反投影累加值
// 输入: 滤波后的正弦图纹理
// 输出: 重建图像纹理
// ============================================================================

layout(local_size_x = 16, local_size_y = 16) in;

// 滤波后的正弦图 [numAngles x numDetectors]
layout(binding = 0, r32f) readonly  uniform image2D sinogramTex;

// 输出重建图像 [outputSize x outputSize]
layout(binding = 1, r32f) writeonly uniform image2D reconImage;

// 重建参数
uniform int   numAngles;         // 投影角度数
uniform int   numDetectors;      // 探测器数
uniform int   outputSize;        // 输出图像尺寸
uniform float angleStart;        // 起始角度 (rad)
uniform float angleStep;         // 角度步长 (rad)
uniform float detectorSpacing;   // 探测器间距

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= outputSize || pixel.y >= outputSize) return;

    float halfOutput = float(outputSize) / 2.0;
    float halfDet    = float(numDetectors) / 2.0;

    // 像素物理坐标 (以图像中心为原点)
    float px = float(pixel.x) - halfOutput + 0.5;
    float py = float(pixel.y) - halfOutput + 0.5;

    float sum = 0.0;

    for (int a = 0; a < numAngles; ++a) {
        float theta    = angleStart + float(a) * angleStep;
        float cosTheta = cos(theta);
        float sinTheta = sin(theta);

        // 当前角度下, 该像素对应的探测器位置
        float t = px * cosTheta + py * sinTheta;

        // 转为探测器索引 (浮点)
        float detIdx = t / detectorSpacing + halfDet - 0.5;

        // 线性插值
        int   d0   = int(floor(detIdx));
        int   d1   = d0 + 1;
        float frac = detIdx - float(d0);

        float val = 0.0;
        if (d0 >= 0 && d1 < numDetectors) {
            float v0 = imageLoad(sinogramTex, ivec2(d0, a)).r;
            float v1 = imageLoad(sinogramTex, ivec2(d1, a)).r;
            val = mix(v0, v1, frac);
        } else if (d0 >= 0 && d0 < numDetectors) {
            val = imageLoad(sinogramTex, ivec2(d0, a)).r;
        }

        sum += val * angleStep;
    }

    imageStore(reconImage, pixel, vec4(sum, 0.0, 0.0, 1.0));
}
