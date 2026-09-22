#pragma once
/**
 * @file    ui_help.h
 * @brief   UI 辅助：配色方案、装饰物名称、帮助文本
 */

#include "core/color_types.h"
#include <string>

namespace NailPrint3D {

/// 装饰物类型名称
extern const char* ornamentNames[];

/// 配色方案
struct ColorScheme {
    const char* name;
    ColorRGBf colors[8];
};

extern ColorScheme colorSchemes[];
extern const int numColorSchemes;

/// 打印操作说明
void printHelpText();

} // namespace NailPrint3D
