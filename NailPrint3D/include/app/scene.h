#pragma once
/**
 * @file    scene.h
 * @brief   场景状态（原 main_nail_print.cpp 中的 30+ 全局变量）
 *
 * 集中管理所有渲染、网格、切片、颜色、装饰物状态。
 */

#include "render/camera.h"
#include "render/opengl/gl_renderer.h"
#include "render/texture.h"
#include "render/mesh_buffer.h"
#include "core/geometry.h"
#include "core/print_types.h"
#include "core/render_types.h"
#include "core/color_types.h"
#include "mesh/mesh_io.h"
#include "mesh/nail_generator.h"
#include "mesh/ornament.h"
#include "slicing/slicer.h"
#include "app/ui_help.h"

#include <glm/glm.hpp>
#include <vector>

namespace NailPrint3D {

/// 渲染模式索引（用于键盘切换）
enum RenderModeIndex {
    RM_Solid = 0,
    RM_Wireframe = 1,
    RM_Slice = 2,
    RM_Color = 3,
    RM_Pattern = 4
};

/// 场景状态
struct Scene {
    // 渲染
    RenderCamera camera;
    NailRenderer renderer;
    GLNailMesh glMesh;
    GLNailMesh glSliceMesh;

    // 网格数据
    Mesh currentMesh;
    Mesh originalMesh;       // 位移映射前
    Mesh nailBaseMesh;       // 纯甲片（无装饰物）

    // 切片
    std::vector<SliceLayer> sliceLayers;
    int currentLayer = 0;
    int renderMode = 0;

    // 颜色
    std::vector<ColorRGBf> palette;

    // 鼠标
    bool mouseLeftDown = false;
    bool mouseRightDown = false;
    double lastMouseX = 0, lastMouseY = 0;

    // 纹理
    GLTexture texCartoon;
    GLTexture texPortrait;
    GLTexture texGeometric;
    GLTexture texText;
    GLTexture texUserPhoto;
    bool userPhotoLoaded = false;
    RenderPattern currentPattern = RenderPattern::Procedural;

    // 3D浮雕
    float reliefHeight = 0.0f;
    bool displacementApplied = false;

    // 3D装饰物
    int currentOrnament = -1;
    float ornamentSize = 15.0f;
    float ornamentU = 0.5f;
    float ornamentV = 0.5f;
    float ornamentRotation = 0.0f;

    // 配色
    int currentColorScheme = 0;

    // 打印配置
    PrintConfig printConfig;

    /// 将 SliceLayer 转换为 LineSegment3D 并上传到 glSliceMesh
    void uploadSlicePaths();
};

} // namespace NailPrint3D
