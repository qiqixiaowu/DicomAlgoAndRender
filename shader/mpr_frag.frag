#version 430 core

// ================================
// MPR（多平面重建）片段着色器
// ================================
// 从3D体数据纹理中提取正交切面并渲染

in vec2 vTexCoord;
out vec4 outColor;

// 纹理采样器
uniform sampler3D volume;           // 3D体数据纹理
uniform sampler1D transferFunc;     // 1D传递函数纹理

// 切面参数
uniform int uSliceAxis;             // 切面轴向：0=Axial(Z), 1=Sagittal(X), 2=Coronal(Y)
uniform float uSlicePosition;       // 切面位置 [0, 1]

// 窗宽窗位
uniform float windowLevel;
uniform float windowWidth;

// 显示参数
uniform float brightness;           // 亮度
uniform bool useTransferFunc;       // 是否使用传递函数着色（否则灰度显示）

// 十字线参数
uniform bool showCrosshair;         // 是否显示十字线
uniform vec2 crosshairPos;          // 十字线位置 [0, 1]
uniform vec3 crosshairColorH;       // 水平十字线颜色
uniform vec3 crosshairColorV;       // 垂直十字线颜色
uniform float crosshairThickness;   // 十字线粗细（像素）
uniform vec2 viewportSize;          // 当前视口大小（像素）

// 边框参数
uniform bool showBorder;            // 是否显示边框
uniform vec3 borderColor;           // 边框颜色
uniform float borderThickness;      // 边框粗细（像素）

/**
 * 应用窗宽窗位
 */
float applyWindow(float density) {
    float windowMin = windowLevel - windowWidth / 2.0;
    float windowMax = windowLevel + windowWidth / 2.0;
    
    if (density <= windowMin) return 0.0;
    if (density >= windowMax) return 1.0;
    return (density - windowMin) / windowWidth;
}

/**
 * 根据切面轴向构建3D纹理坐标
 * Axial   (Z轴): texCoord = (u, v, slice)
 * Sagittal(X轴): texCoord = (slice, v, u)
 * Coronal (Y轴): texCoord = (u, slice, v)
 */
vec3 buildTexCoord3D(vec2 uv, float slicePos, int axis) {
    if (axis == 0) {
        // Axial - XY平面，沿Z轴切片
        return vec3(uv.x, uv.y, slicePos);
    } else if (axis == 1) {
        // Sagittal - YZ平面，沿X轴切片
        return vec3(slicePos, uv.y, uv.x);
    } else {
        // Coronal - XZ平面，沿Y轴切片
        return vec3(uv.x, slicePos, uv.y);
    }
}

void main() {
    // 构建3D纹理坐标
    vec3 texCoord3D = buildTexCoord3D(vTexCoord, uSlicePosition, uSliceAxis);
    
    // 采样3D体数据
    float density = texture(volume, texCoord3D).r;
    
    // 应用窗宽窗位
    float windowedDensity = applyWindow(density);
    
    // 选择着色方式
    vec4 color;
    if (useTransferFunc) {
        // 使用传递函数着色
        color = texture(transferFunc, windowedDensity);
    } else {
        // 灰度显示
        color = vec4(vec3(windowedDensity), 1.0);
    }
    
    // 应用亮度
    color.rgb *= brightness;
    
    // Gamma校正
    color.rgb = pow(color.rgb, vec3(1.0 / 2.2));
    
    // ---- 绘制十字线 ----
    if (showCrosshair) {
        // 计算像素坐标
        vec2 pixelPos = vTexCoord * viewportSize;
        vec2 crossPixel = crosshairPos * viewportSize;
        float halfThick = crosshairThickness * 0.5;
        
        // 水平线
        if (abs(pixelPos.y - crossPixel.y) < halfThick) {
            color.rgb = mix(color.rgb, crosshairColorH, 0.8);
        }
        // 垂直线
        if (abs(pixelPos.x - crossPixel.x) < halfThick) {
            color.rgb = mix(color.rgb, crosshairColorV, 0.8);
        }
    }
    
    // ---- 绘制边框 ----
    if (showBorder) {
        vec2 pixelPos = vTexCoord * viewportSize;
        float bw = borderThickness;
        
        if (pixelPos.x < bw || pixelPos.x > viewportSize.x - bw ||
            pixelPos.y < bw || pixelPos.y > viewportSize.y - bw) {
            color.rgb = borderColor;
        }
    }
    
    outColor = color;
}
