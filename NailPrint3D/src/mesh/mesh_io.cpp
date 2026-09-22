/**
 * @file    mesh_io.cpp
 * @brief   STL文件加载器 实现
 */

#include "mesh/mesh_io.h"
#include <cstring>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <map>

namespace NailPrint3D {

// ============================================================
// STLLoader
// ============================================================

bool STLLoader::isASCII(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) return false;
    char header[80] = {};
    file.read(header, 80);
    // ASCII STL 以 "solid" 开头（但二进制也可能以 solid 开头，需进一步检查）
    if (strncmp(header, "solid", 5) != 0) return false;
    // 检查文件大小是否匹配二进制格式
    file.seekg(0, std::ios::end);
    size_t fileSize = (size_t)file.tellg();
    size_t binarySize = 84 + ((fileSize - 84) / 50) * 50;
    return (binarySize != fileSize);
}

Mesh STLLoader::load(const std::string& filepath) {
    if (isASCII(filepath))
        return loadASCII(filepath);
    return loadBinary(filepath);
}

Mesh STLLoader::loadBinary(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    Mesh mesh;
    if (!file) {
        std::cerr << "[STL] 无法打开文件: " << filepath << std::endl;
        return mesh;
    }

    // 跳过80字节头部
    char header[80];
    file.read(header, 80);

    // 读取三角面片数量
    uint32_t count = 0;
    file.read(reinterpret_cast<char*>(&count), 4);
    if (count == 0 || count > 100000000) {
        std::cerr << "[STL] 无效的三角面片数量: " << count << std::endl;
        return mesh;
    }

    mesh.triangles.resize(count);

    // 逐个读取（每个三角形50字节）
    for (uint32_t i = 0; i < count; i++) {
        // 法线
        float n[3];
        file.read(reinterpret_cast<char*>(n), 12);
        mesh.triangles[i].normal = Vec3(n[0], n[1], n[2]);

        // 三个顶点
        for (int j = 0; j < 3; j++) {
            float v[3];
            file.read(reinterpret_cast<char*>(v), 12);
            mesh.triangles[i].v[j] = Vec3(v[0], v[1], v[2]);
        }

        // 属性字节
        uint16_t attr = 0;
        file.read(reinterpret_cast<char*>(&attr), 2);
        mesh.triangles[i].attr = attr;
    }

    ensureNormals(mesh);
    mesh.computeBBox();

    std::cout << "[STL] 加载完成: " << count << " 个三角面片" << std::endl;
    return mesh;
}

Mesh STLLoader::loadASCII(const std::string& filepath) {
    std::ifstream file(filepath);
    Mesh mesh;
    if (!file) {
        std::cerr << "[STL] 无法打开文件: " << filepath << std::endl;
        return mesh;
    }

    std::string token;
    while (file >> token) {
        if (token == "facet") {
            file >> token; // "normal"
            float nx, ny, nz;
            file >> nx >> ny >> nz;

            Triangle tri;
            tri.normal = Vec3(nx, ny, nz);

            file >> token; // "outer"
            file >> token; // "loop"

            for (int j = 0; j < 3; j++) {
                file >> token; // "vertex"
                float x, y, z;
                file >> x >> y >> z;
                tri.v[j] = Vec3(x, y, z);
            }

            file >> token; // "endloop"
            file >> token; // "endfacet"
            mesh.triangles.push_back(tri);
        }
    }

    ensureNormals(mesh);
    mesh.computeBBox();

    std::cout << "[STL] ASCII加载完成: " << mesh.triangles.size() << " 个三角面片" << std::endl;
    return mesh;
}

bool STLLoader::saveBinary(const std::string& filepath, const Mesh& mesh) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) return false;

    char header[80] = {};
    strncpy(header, "NailPrint3D Binary STL", 79);
    file.write(header, 80);

    uint32_t count = (uint32_t)mesh.triangles.size();
    file.write(reinterpret_cast<const char*>(&count), 4);

    for (const auto& tri : mesh.triangles) {
        file.write(reinterpret_cast<const char*>(&tri.normal.x), 12);
        for (int j = 0; j < 3; j++)
            file.write(reinterpret_cast<const char*>(&tri.v[j].x), 12);
        uint16_t attr = tri.attr;
        file.write(reinterpret_cast<const char*>(&attr), 2);
    }

    return true;
}

bool STLLoader::saveASCII(const std::string& filepath, const Mesh& mesh) {
    std::ofstream file(filepath);
    if (!file) return false;

    file << "solid NailPrint3D\n";
    file << std::fixed << std::setprecision(6);

    for (const auto& tri : mesh.triangles) {
        file << "  facet normal " << tri.normal.x << " " << tri.normal.y << " " << tri.normal.z << "\n";
        file << "    outer loop\n";
        for (int j = 0; j < 3; j++)
            file << "      vertex " << tri.v[j].x << " " << tri.v[j].y << " " << tri.v[j].z << "\n";
        file << "    endloop\n";
        file << "  endfacet\n";
    }
    file << "endsolid NailPrint3D\n";
    return true;
}

void STLLoader::ensureNormals(Mesh& mesh) {
    for (auto& tri : mesh.triangles) {
        if (tri.normal.lengthSq() < 1e-10f) {
            Vec3 e1 = tri.v[1] - tri.v[0];
            Vec3 e2 = tri.v[2] - tri.v[0];
            tri.normal = e1.cross(e2).normalized();
        }
    }
}

void STLLoader::unifyNormals(Mesh& mesh) {
    // 简化版：使所有法线朝向 bbox 中心外侧
    Vec3 center = mesh.bbox.center();
    for (auto& tri : mesh.triangles) {
        Vec3 faceCenter = (tri.v[0] + tri.v[1] + tri.v[2]) * (1.0f / 3.0f);
        Vec3 outward = faceCenter - center;
        if (tri.normal.dot(outward) < 0)
            tri.normal = tri.normal * -1.0f;
    }
}

int STLLoader::detectHoles(const Mesh& mesh) {
    // 统计每条边被引用的次数
    std::map<std::pair<uint64_t, uint64_t>, int> edgeCount;
    for (const auto& tri : mesh.triangles) {
        for (int i = 0; i < 3; i++) {
            // 用顶点坐标量化作为键
            auto makeKey = [](const Vec3& v) -> uint64_t {
                return ((uint64_t)(v.x * 10000) & 0x1FFFFF) |
                       (((uint64_t)(v.y * 10000) & 0x1FFFFF) << 21) |
                       (((uint64_t)(v.z * 10000) & 0x1FFFFF) << 42);
            };
            uint64_t k1 = makeKey(tri.v[i]);
            uint64_t k2 = makeKey(tri.v[(i + 1) % 3]);
            auto key = k1 < k2 ? std::make_pair(k1, k2) : std::make_pair(k2, k1);
            edgeCount[key]++;
        }
    }
    int holes = 0;
    for (const auto& [edge, count] : edgeCount)
        if (count == 1) holes++;
    return holes;
}

} // namespace NailPrint3D
