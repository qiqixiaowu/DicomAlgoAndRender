#version 430 core
layout (location = 0) in vec3 inPosition;

uniform mat4 projection;
uniform mat4 modelview;

out vec4 cameraPos;
out vec4 vertexPos;
out vec3 worldPos;
out vec3 worldNormal;

const vec3 cubeNormals[8] = vec3[](
    vec3(-0.577, -0.577, -0.577),  // v0
    vec3(0.577, -0.577, -0.577),   // v1
    vec3(0.577, 0.577, -0.577),    // v2
    vec3(-0.577, 0.577, -0.577),   // v3
    vec3(-0.577, -0.577, 0.577),   // v4
    vec3(0.577, -0.577, 0.577),    // v5
    vec3(0.577, 0.577, 0.577),     // v6
    vec3(-0.577, 0.577, 0.577)     // v7
);

void main() 
{
    gl_Position = projection * modelview * vec4(inPosition, 1.0);
    cameraPos = inverse(modelview) * vec4(0.0, 0.0, 0.0, 1.0);
    vertexPos = vec4(inPosition, 1.0);
    
    worldPos = inPosition;
    
    int vertexID = gl_VertexID % 8;
    worldNormal = normalize(cubeNormals[vertexID]);
}