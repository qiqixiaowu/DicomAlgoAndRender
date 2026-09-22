#pragma once
/**
 * @file    print_pipeline.h
 * @brief   美甲打印完整流程（5步：网格→修复→切片→色彩→G-code）
 *          以及浮雕位移和装饰物放置
 */

#include "app/scene.h"

namespace NailPrint3D {

/// 运行完整美甲打印流程
void runNailPrintPipeline(Scene& scene);

/// 对网格应用浮雕位移映射
void applyDisplacementToMesh(Scene& scene, float height);

/// 在甲片上放置/移除3D装饰物
void applyOrnamentToMesh(Scene& scene);

} // namespace NailPrint3D
