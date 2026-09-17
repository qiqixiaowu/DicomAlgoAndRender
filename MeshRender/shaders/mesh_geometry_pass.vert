#version 430 core
/**
 * mesh_geometry_pass.vert - 延迟渲染几何通道顶点着色器
 */
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexcoord;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBitangent;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform mat3 uNormalMatrix;

out VS_OUT {
    vec3 worldPos;
    vec3 normal;
    vec2 texcoord;
    vec3 tangent;
    vec3 bitangent;
} vs_out;

void main() {
    vec4 worldPos = uModel * vec4(inPosition, 1.0);
    vs_out.worldPos = worldPos.xyz;
    vs_out.normal   = normalize(uNormalMatrix * inNormal);
    vs_out.texcoord = inTexcoord;
    vs_out.tangent  = normalize(uNormalMatrix * inTangent);
    vs_out.bitangent = normalize(uNormalMatrix * inBitangent);

    gl_Position = uProjection * uView * worldPos;
}
