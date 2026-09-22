/**
 * @file    ui_help.cpp
 * @brief   UI 辅助 实现
 */

#include "app/ui_help.h"
#include <iostream>

namespace NailPrint3D {

const char* ornamentNames[] = {
    "小蜘蛛", "立体花朵", "五角星", "蝴蝶", "爱心", "蝴蝶结", "小皇冠", "宝石"
};

ColorScheme colorSchemes[] = {
    { "鲜艳",
      { ColorRGBf(0.15f, 0.08f, 0.08f),
        ColorRGBf(0.9f, 0.3f, 0.5f),
        ColorRGBf(1.0f, 0.8f, 0.2f),
        ColorRGBf(0.6f, 0.3f, 0.8f),
        ColorRGBf(0.9f, 0.2f, 0.3f),
        ColorRGBf(0.8f, 0.2f, 0.4f),
        ColorRGBf(1.0f, 0.85f, 0.3f),
        ColorRGBf(0.3f, 0.6f, 0.9f) } },
    { "马卡龙",
      { ColorRGBf(0.3f, 0.3f, 0.35f),
        ColorRGBf(0.95f, 0.6f, 0.7f),
        ColorRGBf(0.9f, 0.75f, 0.3f),
        ColorRGBf(0.6f, 0.8f, 0.9f),
        ColorRGBf(0.9f, 0.5f, 0.6f),
        ColorRGBf(0.8f, 0.6f, 0.9f),
        ColorRGBf(0.95f, 0.8f, 0.4f),
        ColorRGBf(0.5f, 0.8f, 0.7f) } },
    { "宝石",
      { ColorRGBf(0.1f, 0.1f, 0.15f),
        ColorRGBf(0.8f, 0.1f, 0.2f),
        ColorRGBf(0.9f, 0.7f, 0.1f),
        ColorRGBf(0.1f, 0.4f, 0.8f),
        ColorRGBf(0.9f, 0.15f, 0.3f),
        ColorRGBf(0.2f, 0.6f, 0.4f),
        ColorRGBf(0.85f, 0.7f, 0.2f),
        ColorRGBf(0.3f, 0.5f, 0.9f) } },
    { "糖果",
      { ColorRGBf(0.25f, 0.15f, 0.1f),
        ColorRGBf(1.0f, 0.4f, 0.6f),
        ColorRGBf(1.0f, 0.9f, 0.3f),
        ColorRGBf(0.5f, 0.2f, 0.9f),
        ColorRGBf(1.0f, 0.3f, 0.4f),
        ColorRGBf(0.9f, 0.3f, 0.5f),
        ColorRGBf(1.0f, 0.8f, 0.2f),
        ColorRGBf(0.2f, 0.7f, 0.9f) } },
};
const int numColorSchemes = sizeof(colorSchemes) / sizeof(colorSchemes[0]);

void printHelpText() {
    std::cout << "操作说明:" << std::endl;
    std::cout << "  鼠标左键拖动: 旋转视角" << std::endl;
    std::cout << "  鼠标右键拖动: 平移视角" << std::endl;
    std::cout << "  鼠标滚轮: 缩放" << std::endl;
    std::cout << "  键盘 1: 实体渲染" << std::endl;
    std::cout << "  键盘 2: 线框渲染" << std::endl;
    std::cout << "  键盘 3: 切片预览 (↑↓ 切换层)" << std::endl;
    std::cout << "  键盘 4: 颜色预览" << std::endl;
    std::cout << "  键盘 5: 图案预览 — 程序化纹理" << std::endl;
    std::cout << "  键盘 6: 图案预览 — 照片/头像" << std::endl;
    std::cout << "  键盘 7: 图案预览 — 卡通风格" << std::endl;
    std::cout << "  键盘 8: 图案预览 — 纯色块" << std::endl;
    std::cout << "  键盘 9: 图案预览 — 文字图案" << std::endl;
    std::cout << "  键盘 T: 流光溢彩（虹彩/珠光效果）" << std::endl;
    std::cout << "  键盘 P: 导入外部图片（PNG/JPG/BMP）" << std::endl;
    std::cout << "  键盘 U: 切换UV变形校正（不校正/纵横比/宽边校正）" << std::endl;
    std::cout << "  键盘 R: 增加浮雕高度 (+0.2mm)" << std::endl;
    std::cout << "  键盘 F: 减少浮雕高度 (-0.2mm)" << std::endl;
    std::cout << "  键盘 0: 重置浮雕（平面）" << std::endl;
    std::cout << "  --- 3D立体装饰物 ---" << std::endl;
    std::cout << "  F1: 小蜘蛛    F2: 立体花朵  F3: 五角星" << std::endl;
    std::cout << "  F4: 蝴蝶      F5: 爱心      F6: 蝴蝶结" << std::endl;
    std::cout << "  F7: 小皇冠    F8: 宝石      F9: 移除装饰物" << std::endl;
    std::cout << "  F10: 实体渲染模式（查看3D装饰物）" << std::endl;
    std::cout << "  [ ]: 调整装饰物大小  I/J/K/L: 移动位置  O: 旋转" << std::endl;
    std::cout << "  C: 切换配色方案（鲜艳/马卡龙/宝石/糖果）" << std::endl;
    std::cout << "  ESC: 退出" << std::endl;
}

} // namespace NailPrint3D
