#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aColor;
layout (location = 3) in vec2 aUV;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

out vec3 vColor;
out float vZ;

void main() {
    vColor = aColor;
    vZ = aPos.z;
    gl_Position = uProjection * uView * uModel * vec4(aPos, 1.0);
}
