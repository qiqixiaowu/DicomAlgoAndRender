/**
 * @file    stl_loader.cpp
 * @brief   STL文件加载器 + 美甲网格生成 实现
 */

#include "stl_loader.h"
#include <cstring>
#include <iomanip>

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

// ============================================================
// NailMeshGenerator
// ============================================================

void NailMeshGenerator::buildVertexIndex(Mesh& mesh) {
    mesh.vertices.clear();
    mesh.normals.clear();
    mesh.uvs.clear();
    mesh.colors.clear();
    std::map<uint64_t, uint32_t> vertMap;
    auto makeKey = [](const Vec3& v) -> uint64_t {
        return ((uint64_t)(v.x * 10000) & 0x1FFFFF) |
               (((uint64_t)(v.y * 10000) & 0x1FFFFF) << 21) |
               (((uint64_t)(v.z * 10000) & 0x1FFFFF) << 42);
    };
    for (auto& tri : mesh.triangles) {
        for (int i = 0; i < 3; i++) {
            uint64_t key = makeKey(tri.v[i]);
            auto it = vertMap.find(key);
            if (it == vertMap.end()) {
                uint32_t idx = (uint32_t)mesh.vertices.size();
                vertMap[key] = idx;
                mesh.vertices.push_back(tri.v[i]);
                mesh.uvs.push_back(tri.uv[i]);
                mesh.colors.push_back(ColorRGBf{0.9f, 0.75f, 0.8f}); // 默认美甲粉色
                tri.idx[i] = idx;
            } else {
                tri.idx[i] = it->second;
            }
        }
    }
}

void NailMeshGenerator::addQuad(Mesh& mesh, const Vec3& v0, const Vec3& v1,
                                 const Vec3& v2, const Vec3& v3) {
    Triangle t1, t2;
    t1.v[0] = v0; t1.v[1] = v1; t1.v[2] = v2;
    t2.v[0] = v0; t2.v[1] = v2; t2.v[2] = v3;
    Vec3 e1 = v1 - v0, e2 = v2 - v0;
    Vec3 n = e1.cross(e2).normalized();
    t1.normal = n; t2.normal = n;
    mesh.triangles.push_back(t1);
    mesh.triangles.push_back(t2);
}

/** @brief 带UV的四边形拆分 */
static void addQuadUV(Mesh& mesh, const Vec3& v0, const Vec3& v1,
                       const Vec3& v2, const Vec3& v3,
                       const Vec2UV& uv0, const Vec2UV& uv1,
                       const Vec2UV& uv2, const Vec2UV& uv3) {
    Triangle t1, t2;
    t1.v[0] = v0; t1.v[1] = v1; t1.v[2] = v2;
    t2.v[0] = v0; t2.v[1] = v2; t2.v[2] = v3;
    t1.uv[0] = uv0; t1.uv[1] = uv1; t1.uv[2] = uv2;
    t2.uv[0] = uv0; t2.uv[1] = uv2; t2.uv[2] = uv3;
    Vec3 e1 = v1 - v0, e2 = v2 - v0;
    Vec3 n = e1.cross(e2).normalized();
    t1.normal = n; t2.normal = n;
    mesh.triangles.push_back(t1);
    mesh.triangles.push_back(t2);
}

Mesh NailMeshGenerator::generateNailPatch(float width, float length,
                                           float curvature, float thickness,
                                           int segmentsU, int segmentsV) {
    Mesh mesh;
    mesh.triangles.clear();

    float halfW = width * 0.5f;
    float halfL = length * 0.5f;

    // 美甲形状：椭圆形弧面
    // - 后端（v=0）平直（指甲根部）
    // - 前端（v=1）圆弧收窄（指尖自由边）
    // - 横向弧度模拟指甲穹起

    std::vector<std::vector<Vec3>> topVerts(segmentsV + 1, std::vector<Vec3>(segmentsU + 1));
    std::vector<std::vector<Vec3>> botVerts(segmentsV + 1, std::vector<Vec3>(segmentsU + 1));
    std::vector<std::vector<Vec2UV>> uvVerts(segmentsV + 1, std::vector<Vec2UV>(segmentsU + 1));

    for (int j = 0; j <= segmentsV; j++) {
        float v = (float)j / segmentsV;
        // y 从后端(-halfL)到前端(+halfL)
        float y = -halfL + v * length;

        // 前端收窄：宽度随 v 变化（后端全宽，前端收窄到 70%）
        float widthScale = 1.0f - 0.3f * v * v;
        float curHalfW = halfW * widthScale;

        for (int i = 0; i <= segmentsU; i++) {
            float u = (float)i / segmentsU;
            float x = -curHalfW + u * (2.0f * curHalfW);

            // 横向弧度：抛物线穹起（中间高，两侧低）
            float lateralArc = curvature * (1.0f - (2.0f * u - 1.0f) * (2.0f * u - 1.0f));

            // 纵向弧度：从前到后微微拱起
            float longArc = curvature * 0.4f * (1.0f - (2.0f * v - 1.0f) * (2.0f * v - 1.0f));

            float z = lateralArc + longArc;

            topVerts[j][i] = Vec3(x, y, z);
            botVerts[j][i] = Vec3(x, y, z - thickness);

            // UV 坐标：直接使用 u,v 参数
            uvVerts[j][i] = Vec2UV(u, v);
        }
    }

    // 顶面三角形（带UV）
    for (int j = 0; j < segmentsV; j++) {
        for (int i = 0; i < segmentsU; i++) {
            addQuadUV(mesh, topVerts[j][i], topVerts[j][i + 1],
                      topVerts[j + 1][i + 1], topVerts[j + 1][i],
                      uvVerts[j][i], uvVerts[j][i + 1],
                      uvVerts[j + 1][i + 1], uvVerts[j + 1][i]);
        }
    }

    // 底面三角形（翻转法线，UV相同）
    for (int j = 0; j < segmentsV; j++) {
        for (int i = 0; i < segmentsU; i++) {
            addQuadUV(mesh, botVerts[j + 1][i], botVerts[j + 1][i + 1],
                      botVerts[j][i + 1], botVerts[j][i],
                      uvVerts[j + 1][i], uvVerts[j + 1][i + 1],
                      uvVerts[j][i + 1], uvVerts[j][i]);
        }
    }

    // 四条侧边（带UV）
    for (int j = 0; j < segmentsV; j++) {
        // 左边
        addQuadUV(mesh, botVerts[j][0], topVerts[j][0],
                  topVerts[j + 1][0], botVerts[j + 1][0],
                  Vec2UV(0, (float)j / segmentsV), Vec2UV(0, (float)j / segmentsV),
                  Vec2UV(0, (float)(j + 1) / segmentsV), Vec2UV(0, (float)(j + 1) / segmentsV));
        // 右边
        addQuadUV(mesh, botVerts[j + 1][segmentsU], topVerts[j + 1][segmentsU],
                  topVerts[j][segmentsU], botVerts[j][segmentsU],
                  Vec2UV(1, (float)(j + 1) / segmentsV), Vec2UV(1, (float)(j + 1) / segmentsV),
                  Vec2UV(1, (float)j / segmentsV), Vec2UV(1, (float)j / segmentsV));
    }
    for (int i = 0; i < segmentsU; i++) {
        // 前边（v=0，指甲根部）
        addQuadUV(mesh, botVerts[0][i], topVerts[0][i],
                  topVerts[0][i + 1], botVerts[0][i + 1],
                  Vec2UV((float)i / segmentsU, 0), Vec2UV((float)i / segmentsU, 0),
                  Vec2UV((float)(i + 1) / segmentsU, 0), Vec2UV((float)(i + 1) / segmentsU, 0));
        // 后边（v=1，指尖）
        addQuadUV(mesh, botVerts[segmentsV][i + 1], topVerts[segmentsV][i + 1],
                  topVerts[segmentsV][i], botVerts[segmentsV][i],
                  Vec2UV((float)(i + 1) / segmentsU, 1), Vec2UV((float)(i + 1) / segmentsU, 1),
                  Vec2UV((float)i / segmentsU, 1), Vec2UV((float)i / segmentsU, 1));
    }

    mesh.computeBBox();
    mesh.triangles[0].attr = 0;

    // 构建去重顶点列表 + 索引
    buildVertexIndex(mesh);

    mesh.computeNormals();
    return mesh;
}

Mesh NailMeshGenerator::addReliefPattern(Mesh& base, int patternType, float patternHeight) {
    // 使用 UV 坐标做图案映射，修改顶点 Z 偏移和颜色
    const float PI = 3.14159265358979f;

    for (size_t vi = 0; vi < base.vertices.size(); vi++) {
        float u = base.uvs[vi].u;
        float v = base.uvs[vi].v;
        float zOffset = 0.0f;
        ColorRGBf color{0.9f, 0.75f, 0.8f}; // 默认美甲底色

        if (patternType == 0) {
            // === 花朵图案 ===
            // 中心花朵：5瓣花
            float cx = 0.5f, cy = 0.4f;
            float dx = u - cx, dy = v - cy;
            float dist = std::sqrt(dx * dx + dy * dy);
            float angle = std::atan2(dy, dx);
            // 5瓣花：r = 0.15 * |cos(2.5 * theta)|
            float petalR = 0.12f * std::abs(std::cos(2.5f * angle));
            if (dist < petalR + 0.03f) {
                float t = dist / (petalR + 0.03f);
                zOffset = patternHeight * (1.0f - t) * 0.8f;
                // 花瓣颜色：粉红渐变
                color = ColorRGBf{1.0f - t * 0.3f, 0.3f + t * 0.3f, 0.5f + t * 0.2f};
            }
            // 叶片装饰
            for (int k = 0; k < 3; k++) {
                float la = k * 2.0f * PI / 3.0f + PI / 6.0f;
                float lx = cx + 0.25f * std::cos(la);
                float ly = cy + 0.25f * std::sin(la);
                float ld = std::sqrt((u - lx) * (u - lx) + (v - ly) * (v - ly));
                if (ld < 0.06f) {
                    zOffset += patternHeight * 0.3f * (1.0f - ld / 0.06f);
                    color = ColorRGBf{0.3f, 0.6f, 0.4f};
                }
            }
            // 边框装饰
            float edgeDist = std::min(u, std::min(1.0f - u, std::min(v, 1.0f - v)));
            if (edgeDist < 0.05f) {
                zOffset += patternHeight * 0.2f;
                color = ColorRGBf{0.85f, 0.65f, 0.75f};
            }
        } else if (patternType == 1) {
            // === 几何菱格图案 ===
            float scale = 8.0f;
            float gu = u * scale;
            float gv = v * scale;
            // 菱格：|frac(gu) - 0.5| + |frac(gv) - 0.5| < 0.3
            float fu = gu - std::floor(gu);
            float fv = gv - std::floor(gv);
            float diamond = std::abs(fu - 0.5f) + std::abs(fv - 0.5f);
            if (diamond < 0.25f) {
                zOffset = patternHeight * 0.6f;
                // 交替颜色
                int checker = ((int)std::floor(gu) + (int)std::floor(gv)) % 2;
                color = checker ? ColorRGBf{0.9f, 0.5f, 0.6f} : ColorRGBf{0.5f, 0.7f, 0.9f};
            }
            // 中心徽章
            float cdist = std::sqrt((u - 0.5f) * (u - 0.5f) + (v - 0.4f) * (v - 0.4f));
            if (cdist < 0.12f) {
                zOffset = patternHeight * (1.0f - cdist / 0.12f);
                color = ColorRGBf{1.0f, 0.85f, 0.3f};
            }
        } else if (patternType == 2) {
            // === 星空图案 ===
            // 星星点阵
            for (int s = 0; s < 12; s++) {
                float seed = (float)s * 7.13f;
                float sx = 0.1f + 0.8f * std::fmod(seed * 1.7f, 1.0f);
                float sy = 0.1f + 0.8f * std::fmod(seed * 2.3f, 1.0f);
                float sd = std::sqrt((u - sx) * (u - sx) + (v - sy) * (v - sy));
                if (sd < 0.04f) {
                    float t = sd / 0.04f;
                    zOffset = std::max(zOffset, patternHeight * (1.0f - t));
                    color = ColorRGBf{1.0f, 0.95f, 0.6f};
                }
            }
            // 渐变底色
            color.r = 0.2f + u * 0.3f;
            color.g = 0.15f + v * 0.25f;
            color.b = 0.4f + (1.0f - u) * 0.3f;
        }

        base.vertices[vi].z += zOffset;
        base.colors[vi] = color;
    }

    base.computeNormals();
    base.computeBBox();
    return base;
}

Mesh NailMeshGenerator::generateCylinder(float radius, float height, int segments) {
    Mesh mesh;
    std::vector<Vec3> topRing(segments), botRing(segments);
    for (int i = 0; i < segments; i++) {
        float angle = (float)i / segments * 2.0f * 3.14159265f;
        float x = radius * std::cos(angle);
        float y = radius * std::sin(angle);
        topRing[i] = Vec3(x, y, height);
        botRing[i] = Vec3(x, y, 0);
    }
    // 侧面
    for (int i = 0; i < segments; i++) {
        int next = (i + 1) % segments;
        addQuad(mesh, botRing[i], botRing[next], topRing[next], topRing[i]);
    }
    // 底面（扇形三角化）
    Vec3 centerBot(0, 0, 0), centerTop(0, 0, height);
    for (int i = 0; i < segments; i++) {
        int next = (i + 1) % segments;
        Triangle t1;
        t1.v[0] = centerBot; t1.v[1] = botRing[i]; t1.v[2] = botRing[next];
        mesh.triangles.push_back(t1);
        Triangle t2;
        t2.v[0] = centerTop; t2.v[1] = topRing[next]; t2.v[2] = topRing[i];
        mesh.triangles.push_back(t2);
    }
    mesh.computeBBox();
    buildVertexIndex(mesh);
    mesh.computeNormals();
    return mesh;
}

Mesh NailMeshGenerator::generateBox(float w, float h, float d) {
    Mesh mesh;
    float hw = w * 0.5f, hh = h * 0.5f, hd = d * 0.5f;
    Vec3 v000(-hw, -hh, -hd), v100(hw, -hh, -hd), v110(hw, hh, -hd), v010(-hw, hh, -hd);
    Vec3 v001(-hw, -hh, hd), v101(hw, -hh, hd), v111(hw, hh, hd), v011(-hw, hh, hd);

    addQuad(mesh, v000, v100, v110, v010); // 底
    addQuad(mesh, v001, v011, v111, v101); // 顶
    addQuad(mesh, v000, v010, v011, v001); // 左
    addQuad(mesh, v100, v101, v111, v110); // 右
    addQuad(mesh, v000, v001, v101, v100); // 前
    addQuad(mesh, v010, v110, v111, v011); // 后

    mesh.computeBBox();
    buildVertexIndex(mesh);
    mesh.computeNormals();
    return mesh;
}

} // namespace NailPrint3D
