//#include "transferFunction.h"
//
//#include <glm/glm.hpp>
//#include <glm/gtc/matrix_transform.hpp>
//#include <glm/gtc/type_ptr.hpp>
//
//#include <windows.h>
//#include <iostream>
//#include <filesystem>
//#include <vector>
//#include <string>
//
//#include "camera.h"
//#include "dicom_utils.hpp"
//#include "volume_build.hpp"
//#include "post_Processor.h"
//
//#pragma region Camera init
//
//OrbitCamera orbitCamera;
//bool mousePressed = false;
//bool mouseBtnPressed = false;
//float lastX = 400, lastY = 300;
//bool firstMouse = true;
//
//#pragma endregion
//
//float deltaTime = 0.0f;
//
//void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
//{
//    if (button == GLFW_MOUSE_BUTTON_LEFT)
//    {
//        if (action == GLFW_PRESS)
//        {
//            double xpos, ypos;
//            glfwGetCursorPos(window, &xpos, &ypos);
//            orbitCamera.processMouseButton(button, action, xpos, ypos);
//            mousePressed = true;
//        }
//        else if (action == GLFW_RELEASE)
//        {
//            orbitCamera.processMouseButton(button, action, 0, 0);
//            mousePressed = false;
//        }
//    }
//}
//
//void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
//{
//    if (mousePressed)
//    {
//        orbitCamera.processMouseMotion(xpos, ypos);
//    }
//}
//
//void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
//{
//    orbitCamera.processMouseScroll(yoffset);
//}
//
//void framebuffer_size_callback(GLFWwindow* window, int width, int height)
//{
//    glViewport(0, 0, width, height);
//}
//
//struct CubeGeometry 
//{
//    GLuint VAO, VBO, EBO;
//    GLsizei indexCount;
//    std::vector<float> normals;
//};
//
//CubeGeometry createCubeGeometry() 
//{
//    float vertices[] = {
//        0.0f, 0.0f, 0.0f,  // v0
//        1.0f, 0.0f, 0.0f,  // v1
//        1.0f, 1.0f, 0.0f,  // v2
//        0.0f, 1.0f, 0.0f,  // v3
//        0.0f, 0.0f, 1.0f,  // v4
//        1.0f, 0.0f, 1.0f,  // v5
//        1.0f, 1.0f, 1.0f,  // v6
//        0.0f, 1.0f, 1.0f   // v7
//    };
//
//    unsigned int indices[] = {
//        // 前
//        0, 1, 2,  0, 2, 3,
//        // 右
//        1, 5, 6,  1, 6, 2,
//        // 后
//        5, 4, 7,  5, 7, 6,
//        // 左
//        4, 0, 3,  4, 3, 7,
//        // 底
//        3, 2, 6,  3, 6, 7,
//        // 顶
//        4, 5, 1,  4, 1, 0
//    };
//
//    float normals[] = {
//        // 前面法线 (0,0,1)
//        0.0f, 0.0f, 1.0f,  0.0f, 0.0f, 1.0f,  0.0f, 0.0f, 1.0f,  0.0f, 0.0f, 1.0f,
//        // 右面法线 (1,0,0)
//        1.0f, 0.0f, 0.0f,  1.0f, 0.0f, 0.0f,  1.0f, 0.0f, 0.0f,  1.0f, 0.0f, 0.0f,
//        // 后面法线 (0,0,-1)
//        0.0f, 0.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, -1.0f,
//        // 左面法线 (-1,0,0)
//        -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f,
//        // 顶面法线 (0,1,0)
//        0.0f, 1.0f, 0.0f,  0.0f, 1.0f, 0.0f,  0.0f, 1.0f, 0.0f,  0.0f, 1.0f, 0.0f,
//        // 底面法线 (0,-1,0)
//        0.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f
//    };
//
//    CubeGeometry cube;
//    glGenVertexArrays(1, &cube.VAO);
//    glGenBuffers(1, &cube.VBO);
//    glGenBuffers(1, &cube.EBO);
//
//    glBindVertexArray(cube.VAO);
//
//    glBindBuffer(GL_ARRAY_BUFFER, cube.VBO);
//    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
//
//    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cube.EBO);
//    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
//
//    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
//    glEnableVertexAttribArray(0);
//
//    glBindVertexArray(0);
//
//    cube.indexCount = sizeof(indices) / sizeof(indices[0]);
//    return cube;
//}
//
//GLuint create3DTextureFromVolume(const VolumeBuildResult& volumeData) {
//    GLuint textureID;
//    glGenTextures(1, &textureID);
//    glBindTexture(GL_TEXTURE_3D, textureID);
//
//    // 设置纹理参数
//    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
//    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
//    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
//    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
//    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//
//    // 将8位数据转换为浮点数组用于上传
//    std::vector<float> floatData(volumeData.buffer.size());
//    for (size_t i = 0; i < volumeData.buffer.size(); ++i) {
//        floatData[i] = static_cast<float>(volumeData.buffer[i]) / 255.0f;
//    }
//
//    // 上传3D纹理数据
//    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F,
//        volumeData.width, volumeData.height, volumeData.depth,
//        0, GL_RED, GL_FLOAT, floatData.data());
//
//    //纹理贴图的时候需要,体渲染
//    glGenerateMipmap(GL_TEXTURE_3D);
//
//    glBindTexture(GL_TEXTURE_3D, 0); // 解绑
//    return textureID;
//}
//
//int main(int argc, char** argv) {
//#ifdef _WIN32
//    HWND hWnd = GetConsoleWindow();
//    ShowWindow(hWnd, SW_HIDE);
//#endif
//
//    if (argc < 2) {
//        return 1;
//    }
//
//    SeriesData series = collectSeries(argv[1]);
//    VolumeBuildResult volumeData = buildVolume_none(series);
//    if (volumeData.buffer.empty()) {
//        std::cerr << "错误：未能加载DICOM体积数据或数据为空。" << std::endl;
//        return 1;
//    }
//
//    if (!glfwInit()) {
//        std::cerr << "无法初始化GLFW" << std::endl;
//        return -1;
//    }
//
//    orbitCamera.setTarget(glm::vec3(0.5f, 0.5f, 0.5f));
//    orbitCamera.setDistance(3.0f);
//    orbitCamera.setPitchLimits(-89.0f, 89.0f);
//    orbitCamera.setDistanceLimits(1.0f, 10.0f);
//
//    glfwWindowHint(GLFW_SAMPLES, 4);  
//    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
//    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
//    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
// 
//    GLFWwindow* window = glfwCreateWindow(800, 600, "DICOM OpenGL", NULL, NULL);
//    if (!window) {
//        std::cerr << "无法创建GLFW窗口" << std::endl;
//        glfwTerminate();
//        return -1;
//    }
//
//    glfwSetWindowPos(window, 0, 150);
//    glfwMakeContextCurrent(window);
//
//    glfwSetMouseButtonCallback(window, mouse_button_callback);
//    glfwSetCursorPosCallback(window, cursor_position_callback);
//    glfwSetScrollCallback(window, scroll_callback);
//    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
//
//    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
//        std::cerr << "无法初始化GLAD" << std::endl;
//        glfwTerminate();
//        return -1;
//    }
//
//    int width, height;
//    glfwGetFramebufferSize(window, &width, &height);
//    //PostProcessor postProcessor(width, height);
//
//    glEnable(GL_DEPTH_TEST);
//    glDepthFunc(GL_LEQUAL);
//
//    glEnable(GL_BLEND);
//    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//
//    // 对于体绘制，通常不剔除或剔除背面
//    //glDisable(GL_CULL_FACE);  // 禁用剔除，or glCullFace(GL_BACK)
//
//    // 启用多采样抗锯齿
//    glEnable(GL_MULTISAMPLE);
//
//    std::unique_ptr<Shader> shaderProgram = std::make_unique<Shader>("shader/volume_vert_glsl.vert", "shader/volume_frag_glsl.frag");
//    if (shaderProgram == 0) 
//    {
//        std::cerr << "无法创建着色器程序" << std::endl;
//        glfwTerminate();
//        return -1;
//    }
//
//    GLuint volumeTexture = create3DTextureFromVolume(volumeData);
//
//    TransferFunction transferFunction;
//    float windowLevel = 0.5f;  // 对应HU=0
//    float windowWidth = 1.0f;  // 全范围
//    windowLevel = 0.35f;  // 对应HU=400
//    windowWidth = 0.1f;   // 对应窗宽=400HU
//    transferFunction.update(windowLevel, windowWidth);
//
//    CubeGeometry cube = createCubeGeometry();
//
//    GLint projLoc = glGetUniformLocation(shaderProgram->ID, "projection");
//    GLint mvLoc = glGetUniformLocation(shaderProgram->ID, "modelview");
//    GLint volumeSamplerLoc = glGetUniformLocation(shaderProgram->ID, "volume");
//    GLint transferFuncLoc = glGetUniformLocation(shaderProgram->ID, "transferFunc");
//    GLint windowLevelLoc = glGetUniformLocation(shaderProgram->ID, "windowLevel");
//    GLint windowWidthLoc = glGetUniformLocation(shaderProgram->ID, "windowWidth");
//    GLint stepSizeLoc = glGetUniformLocation(shaderProgram->ID, "stepSize");
//    GLint densityScaleLoc = glGetUniformLocation(shaderProgram->ID, "densityScale");
//    GLint brightnessLoc = glGetUniformLocation(shaderProgram->ID, "brightness");
//
//    GLint ambientLoc = glGetUniformLocation(shaderProgram->ID, "ambient");
//    GLint diffuseLoc = glGetUniformLocation(shaderProgram->ID, "diffuse");
//    GLint specularLoc = glGetUniformLocation(shaderProgram->ID, "specular");
//    GLint shininessLoc = glGetUniformLocation(shaderProgram->ID, "shininess");
//    GLint lightPosLoc = glGetUniformLocation(shaderProgram->ID, "lightPos");
//    GLint lightColorLoc = glGetUniformLocation(shaderProgram->ID, "lightColor");
//
//    // 抖动相关uniform
//    GLint jitterStrengthLoc = glGetUniformLocation(shaderProgram->ID, "jitterStrength");
//    GLint noiseScaleLoc = glGetUniformLocation(shaderProgram->ID, "noiseScale");
//    GLint timeLoc = glGetUniformLocation(shaderProgram->ID, "time");
//    GLint screenSizeLoc = glGetUniformLocation(shaderProgram->ID, "screenSize");
//
//    shaderProgram->use();
//    glUniform1f(ambientLoc, 0.3f);
//    glUniform1f(diffuseLoc, 0.7f);
//    glUniform1f(specularLoc, 0.3f);
//    glUniform1f(shininessLoc, 32.0f);
//    glUniform3f(lightPosLoc, 5.0f, 5.0f, 5.0f);
//    glUniform3f(lightColorLoc, 1.0f, 1.0f, 0.9f);
//    glUniform1f(stepSizeLoc, 0.0015f);  
//    glUniform1f(densityScaleLoc, 1.2f);  
//    glUniform1f(brightnessLoc, 1.5f);    
//
//
//    // 抖动参数设置
//    glUniform1f(jitterStrengthLoc, 0.5f); // 中等抖动强度
//    glUniform1f(noiseScaleLoc, 2.0f); // 噪声缩放
//    glUniform1f(timeLoc, glfwGetTime()); // 当前时间
//
//    glUniform2f(screenSizeLoc, (float)width, (float)height);
//
//    glm::mat4 viewMat = glm::lookAt(
//        glm::vec3(2.0f, 2.0f, 2.0f),    
//        glm::vec3(0.5f, 0.5f, 0.5f),   
//        glm::vec3(0.0f, 0.0f, 1.0f)     
//    );
//    glm::mat4 projection = glm::perspective(
//        glm::radians(45.0f),
//        800.0f / 600.0f,
//        0.1f,
//        100.0f
//    );
//    glm::mat4 model = glm::mat4(1.0f);
//
//    while (!glfwWindowShouldClose(window)) 
//    {
//        glfwPollEvents();
//
//        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
//            glfwSetWindowShouldClose(window, true);
//        static bool lightChanged = false;
//        static bool windowChanged = false;
//        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
//            windowLevel += 0.003f;
//            windowLevel = glm::clamp(windowLevel, 0.01f, 1.0f);
//            windowChanged = true;
//        }
//        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
//            windowLevel -= 0.003f;
//            windowLevel = glm::clamp(windowLevel, 0.01f, 1.0f);
//            windowChanged = true;
//        }
//        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
//            windowWidth += 0.003f;
//            windowWidth = glm::clamp(windowWidth, 0.001f, 2.0f);
//            windowChanged = true;
//        }   
//        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
//            windowWidth -= 0.003f;
//            windowWidth = glm::clamp(windowWidth, 0.001f, 2.0f);
//            windowChanged = true;
//        }
//        if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
//            windowLevel = 0.26f;  // HU = 40
//            windowWidth = 0.1f;   // 窗宽 = 400HU
//            windowChanged = true;
//        }
//        if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
//            windowLevel = 0.35f;  // HU = 400
//            windowWidth = 0.5f;   // 窗宽 = 2000HU
//            windowChanged = true;
//        }
//        if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) {
//            windowLevel = 0.3f;   // HU = 200
//            windowWidth = 0.08f;  // 窗宽 = 80HU
//            windowChanged = true;
//        }
//
//        if (windowChanged) {
//            transferFunction.update(windowLevel, windowWidth);
//            shaderProgram->use();
//            glUniform1f(windowLevelLoc, windowLevel);
//            glUniform1f(windowWidthLoc, windowWidth);
//
//            windowChanged = false;
//        }
//
//        // 控制光照强度
//        static float ambientStrength = 0.3f;
//        static float diffuseStrength = 0.7f;
//        if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) {
//            ambientStrength += 0.01f;
//            ambientStrength = glm::clamp(ambientStrength, 0.0f, 1.0f);
//            shaderProgram->use();
//            glUniform1f(ambientLoc, ambientStrength);
//            lightChanged = true;
//        }
//        if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) {
//            ambientStrength -= 0.01f;
//            ambientStrength = glm::clamp(ambientStrength, 0.0f, 1.0f);
//            shaderProgram->use();
//            glUniform1f(ambientLoc, ambientStrength);
//            lightChanged = true;
//        }
//        if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) {
//            diffuseStrength += 0.01f;
//            diffuseStrength = glm::clamp(diffuseStrength, 0.0f, 1.0f);
//            shaderProgram->use();
//            glUniform1f(diffuseLoc, diffuseStrength);
//            lightChanged = true;
//        }
//        if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) {
//            diffuseStrength -= 0.01f;
//            diffuseStrength = glm::clamp(diffuseStrength, 0.0f, 1.0f);
//            shaderProgram->use();
//            glUniform1f(diffuseLoc, diffuseStrength);
//            lightChanged = true;
//        }
//
//        // 切换光照模式
//        if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS) {
//            static bool nprMode = false;
//            nprMode = !nprMode;
//            //通过设置不同的uniform来切换着色器中的光照模式
//        }
//
//        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
//        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//        // ====== 第一遍：渲染到后处理FBO ======
//        // postProcessor.beginRender();
//
//        shaderProgram->use();
//
//        glm::mat4 view = orbitCamera.getViewMatrix();
//        glm::mat4 modelview = view * model;
//
//        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
//        glUniformMatrix4fv(mvLoc, 1, GL_FALSE, glm::value_ptr(modelview));
//
//        glActiveTexture(GL_TEXTURE0);
//        glBindTexture(GL_TEXTURE_3D, volumeTexture);
//        glUniform1i(volumeSamplerLoc, 0); 
//
//        glActiveTexture(GL_TEXTURE1);
//        glBindTexture(GL_TEXTURE_1D, transferFunction.getTextureID());
//        glUniform1i(transferFuncLoc, 1);
//
//        glBindVertexArray(cube.VAO);
//        glDrawElements(GL_TRIANGLES, cube.indexCount, GL_UNSIGNED_INT, 0);
//        glBindVertexArray(0);
//
//        /* 帧缓冲
//            postProcessor.endRender();
//            // ====== 第二遍：应用FXAA抗锯齿 ======
//            postProcessor.applyFXAA();
//        */
//        glfwSwapBuffers(window);
//    }
//
//    glDeleteProgram(shaderProgram->ID);
//    glDeleteTextures(1, &volumeTexture);
//    glDeleteVertexArrays(1, &cube.VAO);
//    glDeleteBuffers(1, &cube.VBO);
//    glDeleteBuffers(1, &cube.EBO);
//
//    glfwTerminate();
//    return 0;
//}