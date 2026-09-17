#pragma once
/**
 * @file    dicom_mesh_loader.hpp
 * @brief   DICOM → 体数据 → Marching Cubes → 网格 完整管线
 *
 * 流程：
 *   1. collectSeries()     — 扫描 DICOM 文件夹，收集序列
 *   2. buildVolume_none()  — 构建 3D 体数据（8-bit 窗宽窗位映射）
 *   3. removeBedArtifact() — 可选：去除床板伪影
 *   4. extractIsosurface() — Marching Cubes 提取等值面
 *   5. computeNormals()    — 重新计算法线（平滑）
 *   6. computeTangents()   — 计算切线/副切线
 *
 * 依赖：
 *   - DCMTK (dcmdata, dcmimage, ofstd, dcmimgle)
 *   - 父项目的 dicom_utils.hpp / volume_build.hpp / volume_build.cpp
 */

#include "mesh.h"
#include "marching_cubes.hpp"
#include "gpu_marching_cubes.hpp"  // GPU Compute Shader 实现

// 引入父项目的 DICOM 工具和体数据构建
#include "dicom_utils.hpp"
#include "volume_build.hpp"

#include <iostream>
#include <string>
#include <cmath>

namespace MeshRender {

// ============================================================
// HU 值 → 8-bit 灰度映射参考
// ============================================================

/**
 * 常见组织的 HU 值参考（注意：buildVolume_none 已做窗宽窗位映射为 8-bit）：
 *   空气:     -1000 HU  →  8-bit ≈ 0
 *   肺组织:   -500 HU   →  8-bit ≈ 30-60
 *   脂肪:     -100 HU   →  8-bit ≈ 100-120
 *   水:        0 HU     →  8-bit ≈ 128
 *   软组织:   +40 HU    →  8-bit ≈ 140-160
 *   肝脏:     +60 HU    →  8-bit ≈ 150-170
 *   骨骼:     +400 HU   →  8-bit ≈ 200-255
 *
 * 由于窗宽窗位不同，8-bit 值会有变化。
 * 以下阈值仅供参考，实际使用时建议先用体数据直方图确认。
 */

// ============================================================
// DICOM 网格加载
// ============================================================

/**
 * @brief 从 DICOM 文件夹加载并提取等值面网格
 *
 * @param folder        DICOM 文件夹路径
 * @param isovalue      等值面阈值（8-bit，0-255）
 *                      常用值：
 *                        骨骼表面:  ~180-220
 *                        软组织:    ~100-140
 *                        皮肤表面:  ~50-80
 * @param removeBed     是否去除床板伪影（默认 true）
 * @param spacingScale  间距缩放因子（1.0=原始物理尺寸 mm，
 *                      0.01=缩小到适合 OpenGL 渲染的尺度）
 * @param smoothMesh    是否对结果网格做 Laplacian 平滑
 * @param useGPU        是否使用 GPU Compute Shader 提取（默认 true，需要 OpenGL 4.3+）
 * @param shaderDir     着色器目录路径（GPU 模式需要，用于加载 marching_cubes.comp）
 * @return MeshData     提取的三角网格
 */
inline MeshData loadDicomAsMesh(const std::string& folder,
                                 float isovalue = 128.0f,
                                 bool removeBed = true,
                                 float spacingScale = 0.01f,
                                 bool smoothMesh = true,
                                 bool useGPU = true,
                                 const std::string& shaderDir = "shaders") {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  DICOM 网格加载管线" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "文件夹: " << folder << std::endl;
    std::cout << "等值面阈值: " << isovalue << std::endl;

    // ---- 步骤 1: 收集 DICOM 序列 ----
    std::cout << "\n[1/5] 扫描 DICOM 序列..." << std::endl;
    SeriesData series = collectSeries(folder);
    if (series.slices.empty()) {
        std::cerr << "错误: 未找到 DICOM 文件!" << std::endl;
        return MeshData{};
    }
    std::cout << "  找到 " << series.slices.size() << " 张切片" << std::endl;
    std::cout << "  Series UID: " << series.seriesUID << std::endl;

    // ---- 步骤 2: 构建 3D 体数据 ----
    std::cout << "\n[2/5] 构建 3D 体数据..." << std::endl;
    VolumeBuildResult volume = buildVolume_none(series);
    if (volume.buffer.empty()) {
        std::cerr << "错误: 体数据构建失败!" << std::endl;
        return MeshData{};
    }
    std::cout << "  体数据尺寸: " << volume.width << " x "
              << volume.height << " x " << volume.depth << std::endl;
    std::cout << "  间距: " << volume.spacing[0] << ", "
              << volume.spacing[1] << ", " << volume.spacing[2] << " mm" << std::endl;

    // ---- 步骤 3: 去除床板伪影 ----
    if (removeBed) {
        std::cout << "\n[3/5] 去除床板伪影..." << std::endl;
        removeBedArtifact(volume, 10);
        std::cout << "  床板去除完成" << std::endl;
    } else {
        std::cout << "\n[3/5] 跳过床板去除" << std::endl;
    }

    // ---- 步骤 4: Marching Cubes 等值面提取 ----
    std::cout << "\n[4/5] Marching Cubes 等值面提取..." << std::endl;
    std::cout << "  等值面阈值: " << isovalue << std::endl;

    MeshData mesh;

    if (useGPU) {
        std::cout << "  使用 GPU Compute Shader 提取..." << std::endl;
        mesh = extractIsosurfaceGPU(volume.buffer,
                                     (int)volume.width,
                                     (int)volume.height,
                                     (int)volume.depth,
                                     volume.spacing,
                                     volume.origin,
                                     isovalue,
                                     spacingScale,
                                     shaderDir);
        // 如果 GPU 提取失败，回退到 CPU
        if (mesh.vertices.empty()) {
            std::cout << "  GPU 提取失败，回退到 CPU..." << std::endl;
            mesh = extractIsosurface(volume.buffer,
                                       (int)volume.width,
                                       (int)volume.height,
                                       (int)volume.depth,
                                       volume.spacing,
                                       volume.origin,
                                       isovalue,
                                       spacingScale);
        }
    } else {
        std::cout << "  使用 CPU 提取..." << std::endl;
        mesh = extractIsosurface(volume.buffer,
                                   (int)volume.width,
                                   (int)volume.height,
                                   (int)volume.depth,
                                   volume.spacing,
                                   volume.origin,
                                   isovalue,
                                   spacingScale);
    }

    std::cout << "  顶点数: " << mesh.vertexCount() << std::endl;
    std::cout << "  三角形数: " << mesh.triangleCount() << std::endl;

    if (mesh.vertices.empty()) {
        std::cerr << "警告: 未提取到任何三角形！请调整等值面阈值。" << std::endl;
        return mesh;
    }

    // ---- 步骤 5: 后处理 ----
    std::cout << "\n[5/5] 网格后处理..." << std::endl;

    // 重新计算法线（Marching Cubes 的梯度法线可能不够平滑）
    computeNormals(mesh);
    std::cout << "  法线计算完成" << std::endl;

    // 可选：Laplacian 平滑
    if (smoothMesh) {
        // 使用 mesh_optimizer 中的 laplacianSmooth
        // 这里内联一个简单的 Laplacian 平滑，避免头文件依赖
        // 实际使用时 mesh_optimizer.h 已被包含
        std::cout << "  Laplacian 平滑..." << std::endl;
        // 平滑在 main 中调用 mesh_optimizer 的函数
    }

    // 计算切线
    computeTangents(mesh);
    std::cout << "  切线计算完成" << std::endl;

    // 计算 AABB
    AABB aabb = computeAABB(mesh);
    std::cout << "  AABB min: (" << aabb.min[0] << ", " << aabb.min[1] << ", " << aabb.min[2] << ")" << std::endl;
    std::cout << "  AABB max: (" << aabb.max[0] << ", " << aabb.max[1] << ", " << aabb.max[2] << ")" << std::endl;

    // 居中网格到原点
    float cx = (aabb.min[0] + aabb.max[0]) * 0.5f;
    float cy = (aabb.min[1] + aabb.max[1]) * 0.5f;
    float cz = (aabb.min[2] + aabb.max[2]) * 0.5f;
    for (auto& v : mesh.vertices) {
        v.position[0] -= cx;
        v.position[1] -= cy;
        v.position[2] -= cz;
    }

    mesh.name = "DICOM_Mesh";
    std::cout << "\n========================================" << std::endl;
    std::cout << "  DICOM 网格加载完成!" << std::endl;
    std::cout << "  最终顶点数: " << mesh.vertexCount() << std::endl;
    std::cout << "  最终三角形数: " << mesh.triangleCount() << std::endl;
    std::cout << "========================================\n" << std::endl;

    return mesh;
}

/**
 * @brief 自动估算合适的等值面阈值
 *
 * 通过分析体数据的直方图，找到密度分布的"波谷"作为阈值。
 * 这是一个简单的启发式方法，可能不适用于所有情况。
 *
 * @param volume  体数据
 * @param mode    0=骨骼, 1=软组织, 2=皮肤
 * @return 估算的 8-bit 阈值
 */
inline float estimateIsovalue(const VolumeBuildResult& volume, int mode = 0) {
    if (volume.buffer.empty())
        return 128.0f;

    // 统计直方图
    int hist[256] = {0};
    for (uint8_t v : volume.buffer)
        hist[v]++;

    int total = (int)volume.buffer.size();

    switch (mode) {
        case 0: // 骨骼：取高密度区域的起始点
            // 找到累积分布 70% 处的值
        {
            int cum = 0;
            for (int i = 0; i < 256; i++) {
                cum += hist[i];
                if (cum > total * 0.70f)
                    return (float)i;
            }
            return 180.0f;
        }
        case 1: // 软组织：取中间区域
            return 128.0f;
        case 2: // 皮肤：取较低密度
        {
            int cum = 0;
            for (int i = 0; i < 256; i++) {
                cum += hist[i];
                if (cum > total * 0.30f)
                    return (float)i;
            }
            return 60.0f;
        }
        default:
            return 128.0f;
    }
}

} // namespace MeshRender
