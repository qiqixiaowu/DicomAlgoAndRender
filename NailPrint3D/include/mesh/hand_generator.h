#pragma once
/**
 * @file    mesh/hand_generator.h
 * @brief   程序化手部网格生成器
 *
 * 生成简化手部模型（手掌 + 5根手指），用于与美甲联合渲染。
 * 指甲网格可贴合到指定手指上。
 */

#include "core/geometry.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace NailPrint3D {

/// 手指索引
enum FingerIndex {
    FINGER_THUMB = 0,   // 拇指
    FINGER_INDEX = 1,   // 食指
    FINGER_MIDDLE = 2,  // 中指
    FINGER_RING = 3,    // 无名指
    FINGER_PINKY = 4    // 小指
};

/** @brief 手部网格生成器 */
class HandGenerator {
public:
    /**
     * @brief 生成简化手部网格
     * @param scale 手部整体缩放（1.0 = 真实大小，单位 mm）
     * @return 手部 Mesh（含手掌 + 5根手指，已计算法线）
     */
    static Mesh generateHand(float scale = 1.0f);

    /**
     * @brief 获取指定手指的指甲贴合变换矩阵
     * @param finger 手指索引
     * @param scale 手部缩放
     * @return 指甲应放置的变换（位置 + 旋转）
     */
    static glm::mat4 getNailTransform(FingerIndex finger, float scale = 1.0f);

    /**
     * @brief 获取手指指尖世界坐标
     * @param finger 手指索引
     * @param scale 手部缩放
     * @return 指尖位置
     */
    static Vec3 getFingerTip(FingerIndex finger, float scale = 1.0f);
};

} // namespace NailPrint3D
