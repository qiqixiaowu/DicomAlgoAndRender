#pragma once
/**
 * @file    mesh/mesh_io.h
 * @brief   STL文件加载器（ASCII + 二进制）
 *
 * 支持：
 *  - 二进制STL解析（含属性字节颜色）
 *  - ASCII STL解析
 *  - 模型修复：法线一致化、孔洞检测
 *  - 边界框计算
 */

#include <string>

#include "core/geometry.h"   // Mesh

namespace NailPrint3D {

// ============================================================
// STL加载器
// ============================================================

class STLLoader {
public:
    /** @brief 加载STL文件（自动检测ASCII/二进制） */
    static Mesh load(const std::string& filepath);

    /** @brief 加载二进制STL */
    static Mesh loadBinary(const std::string& filepath);

    /** @brief 加载ASCII STL */
    static Mesh loadASCII(const std::string& filepath);

    /** @brief 保存为二进制STL */
    static bool saveBinary(const std::string& filepath, const Mesh& mesh);

    /** @brief 保存为ASCII STL */
    static bool saveASCII(const std::string& filepath, const Mesh& mesh);

    /** @brief 检测文件是否为ASCII格式 */
    static bool isASCII(const std::string& filepath);

private:
    /** @brief 计算法线（如果STL法线为0） */
    static void ensureNormals(Mesh& mesh);

    /** @brief 法线一致化（使相邻三角形法线方向一致） */
    static void unifyNormals(Mesh& mesh);

    /** @brief 检测孔洞（边只被一个三角形引用） */
    static int detectHoles(const Mesh& mesh);
};

} // namespace NailPrint3D
