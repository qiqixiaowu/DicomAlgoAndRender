#pragma once
/**
 * @file    core/render_types.h
 * @brief   核心渲染类型（无项目依赖，无 GL 依赖）
 *
 * 包含：RenderPattern, TextureTransform, RenderMode
 */

namespace NailPrint3D {

// ============================================================
// 渲染图案模式
// ============================================================

/** @brief 渲染图案模式 */
enum class RenderPattern {
    Procedural = 0,   ///< 程序化纹理（shader内生成：渐变/条纹/月牙）
    Photo      = 1,   ///< 照片/头像（纹理采样 + 完整光照）
    Cartoon    = 2,   ///< 卡通风格（posterize + 描边）
    FlatColor  = 3,   ///< 纯色块（无渐变，仅环境光）
    Text       = 4,   ///< 文字/SDF（锐利边缘）
    Iridescent = 5    ///< 流光溢彩（虹彩/珠光效果）
};

/** @brief 纹理变换参数（让用户调整图案位置/大小/旋转） */
struct TextureTransform {
    float offsetX = 0.0f;   ///< UV平移 X
    float offsetY = 0.0f;   ///< UV平移 Y
    float scale   = 1.0f;   ///< UV缩放
    float rotation = 0.0f;  ///< UV旋转（弧度）
    float opacity = 1.0f;   ///< 图案不透明度
    int   blendMode = 0;    ///< 混合模式: 0=正常, 1=正片叠底, 2=滤色, 3=覆盖
    float reliefHeight = 0.0f; ///< 3D浮雕高度（0=平面, >0=凸起）
    float uvAspect = 1.0f;  ///< UV纵横比校正（1=不校正, <1=横向压缩, >1=纵向压缩）
    int   uvCorrectMode = 0; ///< UV校正模式: 0=不校正, 1=纵横比校正, 2=宽边校正(指尖收窄补偿)
};

/** @brief 渲染模式 */
enum class RenderMode {
    Solid,          ///< 实体渲染
    Wireframe,      ///< 线框渲染
    Normal,         ///< 法线渲染
    SlicePreview,   ///< 切片预览
    ColorPreview,   ///< 颜色预览
    PatternPreview, ///< 图案预览
    PrintPreview    ///< 打印预览（按层数着色 + 阶梯效应）
};

} // namespace NailPrint3D
