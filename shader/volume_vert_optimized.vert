#version 430 core

// ================================
// 输入属性
// ================================
layout (location = 0) in vec3 inPosition;

// ================================
// Uniform变量
// ================================
uniform mat4 projection;
uniform mat4 modelview;

// ================================
// 输出到Fragment Shader
// ================================
out vec4 cameraPos;
out vec4 vertexPos;
out vec3 worldPos;
out vec3 worldNormal;

// ================================
// 立方体顶点法线（用于光照提示）
// ================================
const vec3 cubeNormals[8] = vec3[](
    vec3(-0.577, -0.577, -0.577),  // v0: (-1,-1,-1) 归一化
    vec3( 0.577, -0.577, -0.577),  // v1: ( 1,-1,-1) 归一化
    vec3( 0.577,  0.577, -0.577),  // v2: ( 1, 1,-1) 归一化
    vec3(-0.577,  0.577, -0.577),  // v3: (-1, 1,-1) 归一化
    vec3(-0.577, -0.577,  0.577),  // v4: (-1,-1, 1) 归一化
    vec3( 0.577, -0.577,  0.577),  // v5: ( 1,-1, 1) 归一化
    vec3( 0.577,  0.577,  0.577),  // v6: ( 1, 1, 1) 归一化
    vec3(-0.577,  0.577,  0.577)   // v7: (-1, 1, 1) 归一化
);

void main() {
    // 计算裁剪空间位置
    gl_Position = projection * modelview * vec4(inPosition, 1.0);
    
    // 计算相机位置（世界空间）
    // 通过逆变换将相机从视图空间转到世界空间
    cameraPos = inverse(modelview) * vec4(0.0, 0.0, 0.0, 1.0);
    
    // 顶点位置（纹理空间，范围[0,1]）
    vertexPos = vec4(inPosition, 1.0);
    
    // 世界空间位置
    worldPos = inPosition;
    
    // 顶点法线（基于顶点索引）
    int vertexID = gl_VertexID % 8;
    worldNormal = normalize(cubeNormals[vertexID]);
}
