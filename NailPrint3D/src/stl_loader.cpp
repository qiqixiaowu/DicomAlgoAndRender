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
                // 保留三角形顶点颜色，无则默认美甲粉色
                ColorRGBf c = tri.vcolor[i];
                if (c.r == 0 && c.g == 0 && c.b == 0)
                    c = ColorRGBf{0.9f, 0.75f, 0.8f};
                mesh.colors.push_back(c);
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

// ============================================================
// 位移映射：用图片亮度作为高度图，修改网格顶点Z坐标
// 实现"任何图片→3D浮雕"的核心功能
// ============================================================
void NailMeshGenerator::applyDisplacementMap(Mesh& base,
                                             const unsigned char* imageData,
                                             int imgW, int imgH,
                                             float displacementHeight,
                                             float uvScaleU, float uvScaleV) {
    if (!imageData || imgW <= 0 || imgH <= 0 || base.vertices.empty()) {
        std::cerr << "[Displacement] 无效参数" << std::endl;
        return;
    }

    std::cout << "[Displacement] 开始位移映射: " << base.vertices.size()
              << " 顶点, 图片 " << imgW << "x" << imgH
              << ", 高度=" << displacementHeight << "mm" << std::endl;

    int displacedCount = 0;

    for (size_t vi = 0; vi < base.vertices.size(); vi++) {
        // 获取顶点UV坐标
        if (vi >= base.uvs.size()) continue;

        float u = base.uvs[vi].u * uvScaleU;
        float v = base.uvs[vi].v * uvScaleV;

        // Wrap模式：取小数部分
        u = u - std::floor(u);
        v = v - std::floor(v);

        // 翻转V（图片原点在左上，UV原点在左下）
        v = 1.0f - v;

        // 采样图片像素
        int px = std::min((int)(u * imgW), imgW - 1);
        int py = std::min((int)(v * imgH), imgH - 1);
        int pixelIdx = (py * imgW + px) * 4;  // RGBA

        // 计算亮度: 0.299R + 0.587G + 0.114B
        float r = imageData[pixelIdx] / 255.0f;
        float g = imageData[pixelIdx + 1] / 255.0f;
        float b = imageData[pixelIdx + 2] / 255.0f;
        float luminance = 0.299f * r + 0.587f * g + 0.114f * b;

        // 亮度→高度位移（亮的地方凸起，暗的地方不变）
        float zOffset = luminance * displacementHeight;

        // 只对顶面顶点做位移（Z > 底面）
        // 底面顶点的Z通常 < 0（thickness以下），顶面Z > 0
        if (base.vertices[vi].z > -0.01f) {
            base.vertices[vi].z += zOffset;
            displacedCount++;

            // 同时更新顶点颜色：用图片颜色混合
            base.colors[vi] = ColorRGBf{r, g, b};
        }
    }

    // 重新计算法线（位移后表面法线会改变，这对光照至关重要）
    base.computeNormals();
    base.computeBBox();

    std::cout << "[Displacement] 完成: " << displacedCount << " 顶点已位移"
              << ", Z范围: [" << base.bbox.min.z << ", " << base.bbox.max.z << "]"
              << std::endl;
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

// ============================================================
// 3D立体装饰物生成器实现
// ============================================================

Mesh OrnamentGenerator::generateSphere(float radius, int segments, int rings) {
    Mesh mesh;
    for (int r = 0; r <= rings; r++) {
        float phi = (float)r / rings * 3.14159265f;
        for (int s = 0; s <= segments; s++) {
            float theta = (float)s / segments * 2.0f * 3.14159265f;
            float x = radius * std::sin(phi) * std::cos(theta);
            float y = radius * std::sin(phi) * std::sin(theta);
            float z = radius * std::cos(phi);
            Triangle t;
            // 用顶点直接构建三角形
            (void)x; (void)y; (void)z;
        }
    }
    // 正式构建球面三角形
    mesh.triangles.clear();
    std::vector<std::vector<Vec3>> verts(rings + 1, std::vector<Vec3>(segments + 1));
    for (int r = 0; r <= rings; r++) {
        float phi = (float)r / rings * 3.14159265f;
        for (int s = 0; s <= segments; s++) {
            float theta = (float)s / segments * 2.0f * 3.14159265f;
            verts[r][s] = Vec3(
                radius * std::sin(phi) * std::cos(theta),
                radius * std::sin(phi) * std::sin(theta),
                radius * std::cos(phi));
        }
    }
    for (int r = 0; r < rings; r++) {
        for (int s = 0; s < segments; s++) {
            NailMeshGenerator::addQuad(mesh, verts[r][s], verts[r][s + 1],
                    verts[r + 1][s + 1], verts[r + 1][s]);
        }
    }
    mesh.computeBBox();
    NailMeshGenerator::buildVertexIndex(mesh);
    mesh.computeNormals();
    return mesh;
}

Mesh OrnamentGenerator::generateCone(float radius, float height, int segments) {
    Mesh mesh;
    Vec3 apex(0, 0, height);
    std::vector<Vec3> base(segments);
    for (int i = 0; i < segments; i++) {
        float a = (float)i / segments * 2.0f * 3.14159265f;
        base[i] = Vec3(radius * std::cos(a), radius * std::sin(a), 0);
    }
    // 侧面
    for (int i = 0; i < segments; i++) {
        int next = (i + 1) % segments;
        Triangle t;
        t.v[0] = base[i]; t.v[1] = base[next]; t.v[2] = apex;
        mesh.triangles.push_back(t);
    }
    // 底面
    Vec3 center(0, 0, 0);
    for (int i = 0; i < segments; i++) {
        int next = (i + 1) % segments;
        Triangle t;
        t.v[0] = center; t.v[1] = base[next]; t.v[2] = base[i];
        mesh.triangles.push_back(t);
    }
    mesh.computeBBox();
    NailMeshGenerator::buildVertexIndex(mesh);
    mesh.computeNormals();
    return mesh;
}

Mesh OrnamentGenerator::generateTorus(float majorR, float minorR, int majorSeg, int minorSeg) {
    Mesh mesh;
    std::vector<std::vector<Vec3>> verts(majorSeg, std::vector<Vec3>(minorSeg));
    for (int i = 0; i < majorSeg; i++) {
        float u = (float)i / majorSeg * 2.0f * 3.14159265f;
        for (int j = 0; j < minorSeg; j++) {
            float v = (float)j / minorSeg * 2.0f * 3.14159265f;
            verts[i][j] = Vec3(
                (majorR + minorR * std::cos(v)) * std::cos(u),
                (majorR + minorR * std::cos(v)) * std::sin(u),
                minorR * std::sin(v));
        }
    }
    for (int i = 0; i < majorSeg; i++) {
        int ni = (i + 1) % majorSeg;
        for (int j = 0; j < minorSeg; j++) {
            int nj = (j + 1) % minorSeg;
            NailMeshGenerator::addQuad(mesh, verts[i][j], verts[ni][j],
                    verts[ni][nj], verts[i][nj]);
        }
    }
    mesh.computeBBox();
    NailMeshGenerator::buildVertexIndex(mesh);
    mesh.computeNormals();
    return mesh;
}

Mesh OrnamentGenerator::mergeMeshes(const Mesh& a, const Mesh& b) {
    Mesh result = a;
    for (const auto& tri : b.triangles)
        result.triangles.push_back(tri);
    result.computeBBox();
    NailMeshGenerator::buildVertexIndex(result);
    result.computeNormals();
    return result;
}

void OrnamentGenerator::translateMesh(Mesh& m, float x, float y, float z) {
    for (auto& tri : m.triangles)
        for (int i = 0; i < 3; i++) {
            tri.v[i].x += x; tri.v[i].y += y; tri.v[i].z += z;
        }
    for (auto& v : m.vertices) { v.x += x; v.y += y; v.z += z; }
    m.computeBBox();
}

void OrnamentGenerator::rotateMeshZ(Mesh& m, float angle) {
    float c = std::cos(angle), s = std::sin(angle);
    for (auto& tri : m.triangles)
        for (int i = 0; i < 3; i++) {
            float x = tri.v[i].x, y = tri.v[i].y;
            tri.v[i].x = x * c - y * s;
            tri.v[i].y = x * s + y * c;
        }
    for (auto& v : m.vertices) {
        float x = v.x, y = v.y;
        v.x = x * c - y * s;
        v.y = x * s + y * c;
    }
    m.computeBBox();
}

void OrnamentGenerator::scaleMesh(Mesh& m, float s) {
    for (auto& tri : m.triangles)
        for (int i = 0; i < 3; i++)
            tri.v[i] = tri.v[i] * s;
    for (auto& v : m.vertices) v = v * s;
    m.computeBBox();
}

void OrnamentGenerator::colorMesh(Mesh& m, const ColorRGBf& c) {
    // 在每个三角形的顶点上设置颜色
    for (auto& tri : m.triangles)
        for (int i = 0; i < 3; i++)
            tri.vcolor[i] = c;
    m.colors.assign(m.vertices.size(), c);
}

// ============================================================
// 小蜘蛛
// ============================================================
Mesh OrnamentGenerator::generateSpider(float size, ColorRGBf color) {
    Mesh spider;
    float s = size * 0.1f;  // 缩放因子
    ColorRGBf bodyColor(0.15f, 0.08f, 0.08f);   // 深黑色身体
    ColorRGBf legColor(0.2f, 0.1f, 0.1f);       // 深灰色腿
    ColorRGBf eyeColor(1.0f, 0.9f, 0.2f);       // 金黄色眼睛
    ColorRGBf eyeBackColor(0.8f, 0.1f, 0.1f);   // 红色眼背

    // 身体：两个球体（头胸部 + 腹部）
    Mesh head = generateSphere(s * 0.6f, 12, 8);
    translateMesh(head, 0, s * 0.5f, s * 0.3f);
    colorMesh(head, bodyColor);
    Mesh abdomen = generateSphere(s * 1.0f, 12, 8);
    translateMesh(abdomen, 0, -s * 0.6f, s * 0.4f);
    colorMesh(abdomen, bodyColor);
    spider = mergeMeshes(head, abdomen);

    // 8条腿（每侧4条，用细长圆锥）
    for (int side = 0; side < 2; side++) {
        float dir = (side == 0) ? 1.0f : -1.0f;
        for (int i = 0; i < 4; i++) {
            Mesh leg = generateCone(s * 0.08f, s * 1.5f, 6);
            float angle = (float)(i - 1.5) * 0.4f;
            rotateMeshZ(leg, angle);
            translateMesh(leg, dir * s * 0.5f, s * 0.3f - i * s * 0.3f, s * 0.2f);
            colorMesh(leg, legColor);
            spider = mergeMeshes(spider, leg);
        }
    }

    // 两个眼睛（小球）— 黄色眼球
    Mesh eyeL = generateSphere(s * 0.12f, 8, 6);
    translateMesh(eyeL, -s * 0.2f, s * 0.7f, s * 0.7f);
    colorMesh(eyeL, eyeColor);
    Mesh eyeR = generateSphere(s * 0.12f, 8, 6);
    translateMesh(eyeR, s * 0.2f, s * 0.7f, s * 0.7f);
    colorMesh(eyeR, eyeColor);
    spider = mergeMeshes(spider, mergeMeshes(eyeL, eyeR));

    return spider;
}

// ============================================================
// 立体花朵
// ============================================================
Mesh OrnamentGenerator::generateFlower(float size, ColorRGBf color) {
    Mesh flower;
    float s = size * 0.1f;
    ColorRGBf centerColor(1.0f, 0.75f, 0.1f);   // 橙黄色花心
    ColorRGBf petalColor = color;                // 用户指定花瓣色
    ColorRGBf leafColor(0.2f, 0.6f, 0.2f);      // 绿色叶子

    // 花心：球体
    Mesh center = generateSphere(s * 0.4f, 12, 8);
    colorMesh(center, centerColor);
    flower = center;

    // 6片花瓣：每片是扁球体，颜色渐变
    for (int i = 0; i < 6; i++) {
        float angle = (float)i / 6.0f * 2.0f * 3.14159265f;
        Mesh petal = generateSphere(s * 0.5f, 10, 6);
        for (auto& tri : petal.triangles)
            for (int j = 0; j < 3; j++) tri.v[j].z *= 0.3f;
        for (auto& v : petal.vertices) v.z *= 0.3f;
        translateMesh(petal, s * 0.7f * std::cos(angle), s * 0.7f * std::sin(angle), s * 0.1f);
        // 花瓣颜色微调，每片略有差异
        ColorRGBf pc(petalColor.r * (0.85f + 0.15f * i / 6.0f),
                     petalColor.g * (0.85f + 0.15f * i / 6.0f),
                     petalColor.b);
        colorMesh(petal, pc);
        flower = mergeMeshes(flower, petal);
    }

    // 两片绿色叶子
    for (int side = 0; side < 2; side++) {
        float dir = (side == 0) ? -1.0f : 1.0f;
        Mesh leaf = generateSphere(s * 0.35f, 8, 5);
        for (auto& tri : leaf.triangles)
            for (int j = 0; j < 3; j++) { tri.v[j].z *= 0.2f; tri.v[j].y *= 1.8f; }
        for (auto& v : leaf.vertices) { v.z *= 0.2f; v.y *= 1.8f; }
        translateMesh(leaf, dir * s * 0.9f, -s * 0.3f, s * 0.05f);
        rotateMeshZ(leaf, dir * 0.5f);
        colorMesh(leaf, leafColor);
        flower = mergeMeshes(flower, leaf);
    }

    return flower;
}

// ============================================================
// 五角星
// ============================================================
Mesh OrnamentGenerator::generateStar(float size, ColorRGBf color) {
    Mesh star;
    float s = size * 0.1f;
    float thickness = s * 0.3f;
    ColorRGBf starColor = color;
    ColorRGBf edgeColor(1.0f, 0.95f, 0.3f);  // 亮金色边缘

    // 5个尖角 + 5个凹点
    std::vector<Vec3> outerPts, innerPts;
    for (int i = 0; i < 5; i++) {
        float outerAngle = (float)i / 5.0f * 2.0f * 3.14159265f - 1.5708f;
        float innerAngle = outerAngle + 3.14159265f / 5.0f;
        outerPts.push_back(Vec3(s * std::cos(outerAngle), s * std::sin(outerAngle), 0));
        innerPts.push_back(Vec3(s * 0.4f * std::cos(innerAngle), s * 0.4f * std::sin(innerAngle), 0));
    }

    Vec3 centerTop(0, 0, thickness * 0.5f);
    Vec3 centerBot(0, 0, -thickness * 0.5f);
    for (int i = 0; i < 5; i++) {
        int ni = (i + 1) % 5;
        // 顶面三角形：outer[i] -> inner[i] -> outer[ni]
        Triangle top1, top2;
        top1.v[0] = Vec3(outerPts[i].x, outerPts[i].y, thickness * 0.5f);
        top1.v[1] = Vec3(innerPts[i].x, innerPts[i].y, thickness * 0.5f);
        top1.v[2] = Vec3(outerPts[ni].x, outerPts[ni].y, thickness * 0.5f);
        for (int j = 0; j < 3; j++) top1.vcolor[j] = starColor;
        star.triangles.push_back(top1);
        // 底面（反向）
        top2.v[0] = Vec3(outerPts[i].x, outerPts[i].y, -thickness * 0.5f);
        top2.v[1] = Vec3(outerPts[ni].x, outerPts[ni].y, -thickness * 0.5f);
        top2.v[2] = Vec3(innerPts[i].x, innerPts[i].y, -thickness * 0.5f);
        for (int j = 0; j < 3; j++) top2.vcolor[j] = starColor;
        star.triangles.push_back(top2);
        // 侧面：outer[i]到outer[ni]的厚度 — 用亮金色
        Triangle side1, side2;
        side1.v[0] = Vec3(outerPts[i].x, outerPts[i].y, -thickness * 0.5f);
        side1.v[1] = Vec3(outerPts[i].x, outerPts[i].y, thickness * 0.5f);
        side1.v[2] = Vec3(outerPts[ni].x, outerPts[ni].y, thickness * 0.5f);
        for (int j = 0; j < 3; j++) side1.vcolor[j] = edgeColor;
        star.triangles.push_back(side1);
        side2.v[0] = Vec3(outerPts[i].x, outerPts[i].y, -thickness * 0.5f);
        side2.v[1] = Vec3(outerPts[ni].x, outerPts[ni].y, thickness * 0.5f);
        side2.v[2] = Vec3(outerPts[ni].x, outerPts[ni].y, -thickness * 0.5f);
        for (int j = 0; j < 3; j++) side2.vcolor[j] = edgeColor;
        star.triangles.push_back(side2);
        // 侧面：inner[i]到outer[ni]
        Triangle side3, side4;
        side3.v[0] = Vec3(innerPts[i].x, innerPts[i].y, -thickness * 0.5f);
        side3.v[1] = Vec3(innerPts[i].x, innerPts[i].y, thickness * 0.5f);
        side3.v[2] = Vec3(outerPts[ni].x, outerPts[ni].y, thickness * 0.5f);
        for (int j = 0; j < 3; j++) side3.vcolor[j] = edgeColor;
        star.triangles.push_back(side3);
        side4.v[0] = Vec3(innerPts[i].x, innerPts[i].y, -thickness * 0.5f);
        side4.v[1] = Vec3(outerPts[ni].x, outerPts[ni].y, thickness * 0.5f);
        side4.v[2] = Vec3(outerPts[ni].x, outerPts[ni].y, -thickness * 0.5f);
        for (int j = 0; j < 3; j++) side4.vcolor[j] = edgeColor;
        star.triangles.push_back(side4);
    }

    star.computeBBox();
    NailMeshGenerator::buildVertexIndex(star);
    star.computeNormals();
    return star;
}

// ============================================================
// 蝴蝶
// ============================================================
Mesh OrnamentGenerator::generateButterfly(float size, ColorRGBf color) {
    Mesh butterfly;
    float s = size * 0.1f;
    ColorRGBf bodyColor(0.15f, 0.08f, 0.05f);   // 深棕色身体
    ColorRGBf wingUpperColor = color;             // 上翅颜色
    ColorRGBf wingLowerColor(color.r * 0.7f, color.g * 0.7f, color.b * 1.1f); // 下翅偏深
    ColorRGBf spotColor(1.0f, 0.85f, 0.2f);      // 翅膀斑点金色
    ColorRGBf antennaColor(0.2f, 0.1f, 0.05f);   // 触角色

    // 身体：细长椭球
    Mesh body = generateSphere(s * 0.15f, 10, 6);
    for (auto& tri : body.triangles)
        for (int i = 0; i < 3; i++) tri.v[i].y *= 2.5f;
    for (auto& v : body.vertices) v.y *= 2.5f;
    colorMesh(body, bodyColor);
    butterfly = body;

    // 4个翅膀：上翅大、下翅小，用压扁的球体
    float wingZ = s * 0.15f;
    // 左上翅
    Mesh wingLU = generateSphere(s * 0.8f, 12, 8);
    for (auto& tri : wingLU.triangles) for (int i = 0; i < 3; i++) { tri.v[i].z *= 0.15f; tri.v[i].x *= 0.7f; }
    for (auto& v : wingLU.vertices) { v.z *= 0.15f; v.x *= 0.7f; }
    translateMesh(wingLU, -s * 0.6f, s * 0.3f, wingZ);
    colorMesh(wingLU, wingUpperColor);
    butterfly = mergeMeshes(butterfly, wingLU);

    // 右上翅
    Mesh wingRU = generateSphere(s * 0.8f, 12, 8);
    for (auto& tri : wingRU.triangles) for (int i = 0; i < 3; i++) { tri.v[i].z *= 0.15f; tri.v[i].x *= 0.7f; }
    for (auto& v : wingRU.vertices) { v.z *= 0.15f; v.x *= 0.7f; }
    translateMesh(wingRU, s * 0.6f, s * 0.3f, wingZ);
    colorMesh(wingRU, wingUpperColor);
    butterfly = mergeMeshes(butterfly, wingRU);

    // 左下翅
    Mesh wingLD = generateSphere(s * 0.5f, 10, 6);
    for (auto& tri : wingLD.triangles) for (int i = 0; i < 3; i++) { tri.v[i].z *= 0.15f; tri.v[i].x *= 0.6f; }
    for (auto& v : wingLD.vertices) { v.z *= 0.15f; v.x *= 0.6f; }
    translateMesh(wingLD, -s * 0.4f, -s * 0.4f, wingZ);
    colorMesh(wingLD, wingLowerColor);
    butterfly = mergeMeshes(butterfly, wingLD);

    // 右下翅
    Mesh wingRD = generateSphere(s * 0.5f, 10, 6);
    for (auto& tri : wingRD.triangles) for (int i = 0; i < 3; i++) { tri.v[i].z *= 0.15f; tri.v[i].x *= 0.6f; }
    for (auto& v : wingRD.vertices) { v.z *= 0.15f; v.x *= 0.6f; }
    translateMesh(wingRD, s * 0.4f, -s * 0.4f, wingZ);
    colorMesh(wingRD, wingLowerColor);
    butterfly = mergeMeshes(butterfly, wingRD);

    // 翅膀斑点（4个小金球）
    Mesh spotLU = generateSphere(s * 0.12f, 8, 6);
    translateMesh(spotLU, -s * 0.7f, s * 0.4f, wingZ + s * 0.02f);
    colorMesh(spotLU, spotColor);
    butterfly = mergeMeshes(butterfly, spotLU);
    Mesh spotRU = generateSphere(s * 0.12f, 8, 6);
    translateMesh(spotRU, s * 0.7f, s * 0.4f, wingZ + s * 0.02f);
    colorMesh(spotRU, spotColor);
    butterfly = mergeMeshes(butterfly, spotRU);

    // 两根触角（细圆锥）
    Mesh antL = generateCone(s * 0.03f, s * 0.5f, 4);
    translateMesh(antL, -s * 0.08f, s * 1.2f, s * 0.1f);
    rotateMeshZ(antL, -0.3f);
    colorMesh(antL, antennaColor);
    butterfly = mergeMeshes(butterfly, antL);
    Mesh antR = generateCone(s * 0.03f, s * 0.5f, 4);
    translateMesh(antR, s * 0.08f, s * 1.2f, s * 0.1f);
    rotateMeshZ(antR, 0.3f);
    colorMesh(antR, antennaColor);
    butterfly = mergeMeshes(butterfly, antR);

    return butterfly;
}

// ============================================================
// 爱心
// ============================================================
Mesh OrnamentGenerator::generateHeart(float size, ColorRGBf color) {
    Mesh heart;
    float s = size * 0.1f;
    ColorRGBf heartColor = color;
    ColorRGBf highlightColor(1.0f, 0.9f, 0.9f);  // 高光白色偏粉

    int segments = 32;
    float thickness = s * 0.4f;
    std::vector<Vec3> topPts(segments), botPts(segments);

    for (int i = 0; i < segments; i++) {
        float t = (float)i / segments * 2.0f * 3.14159265f;
        float x = 16.0f * std::pow(std::sin(t), 3);
        float y = 13.0f * std::cos(t) - 5.0f * std::cos(2 * t) - 2.0f * std::cos(3 * t) - std::cos(4 * t);
        x *= s * 0.06f;
        y *= s * 0.06f;
        topPts[i] = Vec3(x, y, thickness * 0.5f);
        botPts[i] = Vec3(x, y, -thickness * 0.5f);
    }

    Vec3 centerTop(0, 0, thickness * 0.5f);
    Vec3 centerBot(0, 0, -thickness * 0.5f);
    for (int i = 0; i < segments; i++) {
        int ni = (i + 1) % segments;
        // 顶面 — 用高光色（顶部更亮）
        Triangle t1;
        t1.v[0] = centerTop; t1.v[1] = topPts[i]; t1.v[2] = topPts[ni];
        for (int j = 0; j < 3; j++) t1.vcolor[j] = highlightColor;
        heart.triangles.push_back(t1);
        // 底面（反向）— 用主色
        Triangle t2;
        t2.v[0] = centerBot; t2.v[1] = botPts[ni]; t2.v[2] = botPts[i];
        for (int j = 0; j < 3; j++) t2.vcolor[j] = heartColor;
        heart.triangles.push_back(t2);
        // 侧面 — 用主色
        Triangle s1, s2;
        s1.v[0] = botPts[i]; s1.v[1] = topPts[i]; s1.v[2] = topPts[ni];
        for (int j = 0; j < 3; j++) s1.vcolor[j] = heartColor;
        heart.triangles.push_back(s1);
        s2.v[0] = botPts[i]; s2.v[1] = topPts[ni]; s2.v[2] = botPts[ni];
        for (int j = 0; j < 3; j++) s2.vcolor[j] = heartColor;
        heart.triangles.push_back(s2);
    }

    heart.computeBBox();
    NailMeshGenerator::buildVertexIndex(heart);
    heart.computeNormals();
    return heart;
}

// ============================================================
// 蝴蝶结
// ============================================================
Mesh OrnamentGenerator::generateBow(float size, ColorRGBf color) {
    Mesh bow;
    float s = size * 0.1f;
    ColorRGBf centerColor(0.4f, 0.05f, 0.15f);  // 深红色中心结
    ColorRGBf wingColor = color;                 // 翅膀主色
    ColorRGBf ribbonColor(color.r * 0.8f, color.g * 0.6f, color.b * 0.8f); // 飘带偏深
    ColorRGBf highlightColor(1.0f, 0.95f, 0.9f); // 高光

    // 中心结：小球
    Mesh center = generateSphere(s * 0.25f, 10, 8);
    colorMesh(center, centerColor);
    bow = center;

    // 左右两个翅膀：压扁的球体
    for (int side = 0; side < 2; side++) {
        float dir = (side == 0) ? -1.0f : 1.0f;
        Mesh wing = generateSphere(s * 0.6f, 12, 8);
        for (auto& tri : wing.triangles)
            for (int i = 0; i < 3; i++) {
                tri.v[i].z *= 0.2f;
                tri.v[i].x *= 1.2f;
            }
        for (auto& v : wing.vertices) { v.z *= 0.2f; v.x *= 1.2f; }
        translateMesh(wing, dir * s * 0.6f, 0, s * 0.1f);
        colorMesh(wing, wingColor);
        bow = mergeMeshes(bow, wing);

        // 翅膀上的高光小球
        Mesh hl = generateSphere(s * 0.1f, 8, 6);
        translateMesh(hl, dir * s * 0.5f, s * 0.15f, s * 0.12f);
        colorMesh(hl, highlightColor);
        bow = mergeMeshes(bow, hl);
    }

    // 两条飘带（细长扁盒）
    for (int side = 0; side < 2; side++) {
        float dir = (side == 0) ? -1.0f : 1.0f;
        Mesh ribbon = generateSphere(s * 0.15f, 8, 6);
        for (auto& tri : ribbon.triangles)
            for (int i = 0; i < 3; i++) {
                tri.v[i].z *= 0.15f;
                tri.v[i].y *= 3.0f;
            }
        for (auto& v : ribbon.vertices) { v.z *= 0.15f; v.y *= 3.0f; }
        translateMesh(ribbon, dir * s * 0.3f, -s * 0.8f, s * 0.05f);
        rotateMeshZ(ribbon, dir * 0.3f);
        colorMesh(ribbon, ribbonColor);
        bow = mergeMeshes(bow, ribbon);
    }

    return bow;
}

// ============================================================
// 小皇冠
// ============================================================
Mesh OrnamentGenerator::generateCrown(float size, ColorRGBf color) {
    Mesh crown;
    float s = size * 0.1f;
    ColorRGBf baseColor = color;                          // 皇冠主体金色
    ColorRGBf spikeColor(0.9f, 0.75f, 0.1f);             // 尖角深金色
    ColorRGBf tipColor(0.9f, 0.2f, 0.3f);                // 尖顶宝石红色
    ColorRGBf gemColor(0.2f, 0.5f, 0.9f);                // 底环宝石蓝色

    // 底环：圆环
    Mesh baseRing = generateTorus(s * 0.6f, s * 0.12f, 20, 8);
    colorMesh(baseRing, baseColor);
    crown = baseRing;

    // 底环上镶嵌4颗蓝色宝石
    for (int i = 0; i < 4; i++) {
        float a = (float)i / 4.0f * 2.0f * 3.14159265f + 0.4f;
        Mesh gem = generateSphere(s * 0.08f, 8, 6);
        translateMesh(gem, s * 0.6f * std::cos(a), s * 0.6f * std::sin(a), s * 0.12f);
        colorMesh(gem, gemColor);
        crown = mergeMeshes(crown, gem);
    }

    // 5个尖角（圆锥）
    for (int i = 0; i < 5; i++) {
        float angle = (float)i / 5.0f * 2.0f * 3.14159265f;
        Mesh spike = generateCone(s * 0.15f, s * 0.6f, 8);
        translateMesh(spike, s * 0.6f * std::cos(angle), s * 0.6f * std::sin(angle), s * 0.3f);
        colorMesh(spike, spikeColor);
        crown = mergeMeshes(crown, spike);

        // 每个尖角顶部加一颗红色宝石球
        Mesh tip = generateSphere(s * 0.1f, 8, 6);
        translateMesh(tip, s * 0.6f * std::cos(angle), s * 0.6f * std::sin(angle), s * 0.9f);
        colorMesh(tip, tipColor);
        crown = mergeMeshes(crown, tip);
    }

    return crown;
}

// ============================================================
// 宝石（多面切割）
// ============================================================
Mesh OrnamentGenerator::generateGem(float size, ColorRGBf color) {
    Mesh gem;
    float s = size * 0.1f;
    ColorRGBf tableColor = color;                         // 台面主色
    ColorRGBf crownColor(color.r * 1.1f, color.g * 1.05f, color.b * 1.1f); // 冠部偏亮
    ColorRGBf pavilionColor(color.r * 0.6f, color.g * 0.55f, color.b * 0.7f); // 亭部偏深

    int n = 6;
    float topR = s * 0.7f;
    float midR = s * 0.9f;
    float bottomR = s * 0.3f;
    float topZ = s * 0.4f;
    float midZ = 0;
    float bottomZ = -s * 0.5f;

    std::vector<Vec3> topRing(n), midRing(n), botTip(1);

    for (int i = 0; i < n; i++) {
        float a = (float)i / n * 2.0f * 3.14159265f;
        topRing[i] = Vec3(topR * std::cos(a), topR * std::sin(a), topZ);
        midRing[i] = Vec3(midR * std::cos(a), midR * std::sin(a), midZ);
    }
    Vec3 bottom(0, 0, bottomZ);
    Vec3 topCenter(0, 0, topZ);

    // 顶面（台面）— 最亮
    for (int i = 0; i < n; i++) {
        int ni = (i + 1) % n;
        Triangle t;
        t.v[0] = topCenter; t.v[1] = topRing[i]; t.v[2] = topRing[ni];
        for (int j = 0; j < 3; j++) t.vcolor[j] = tableColor;
        gem.triangles.push_back(t);
    }
    // 冠部斜面（topRing -> midRing）— 中等亮度
    for (int i = 0; i < n; i++) {
        int ni = (i + 1) % n;
        // 用 addQuad 但手动设置颜色
        Vec3 v0 = topRing[i], v1 = topRing[ni], v2 = midRing[ni], v3 = midRing[i];
        Triangle t1, t2;
        t1.v[0] = v0; t1.v[1] = v1; t1.v[2] = v2;
        t2.v[0] = v0; t2.v[1] = v2; t2.v[2] = v3;
        for (int j = 0; j < 3; j++) { t1.vcolor[j] = crownColor; t2.vcolor[j] = crownColor; }
        gem.triangles.push_back(t1);
        gem.triangles.push_back(t2);
    }
    // 亭部（midRing -> bottom 尖点）— 最暗
    for (int i = 0; i < n; i++) {
        int ni = (i + 1) % n;
        Triangle t1;
        t1.v[0] = midRing[i]; t1.v[1] = midRing[ni]; t1.v[2] = bottom;
        for (int j = 0; j < 3; j++) t1.vcolor[j] = pavilionColor;
        gem.triangles.push_back(t1);
    }

    gem.computeBBox();
    NailMeshGenerator::buildVertexIndex(gem);
    gem.computeNormals();
    return gem;
}

// ============================================================
// 统一生成入口
// ============================================================
Mesh OrnamentGenerator::generate(OrnamentType type, float size, ColorRGBf color) {
    switch (type) {
        case OrnamentType::Spider:    return generateSpider(size, color);
        case OrnamentType::Flower:    return generateFlower(size, color);
        case OrnamentType::Star:      return generateStar(size, color);
        case OrnamentType::Butterfly: return generateButterfly(size, color);
        case OrnamentType::Heart:     return generateHeart(size, color);
        case OrnamentType::Bow:       return generateBow(size, color);
        case OrnamentType::Crown:     return generateCrown(size, color);
        case OrnamentType::Gem:       return generateGem(size, color);
        default: return generateStar(size, color);
    }
}

// ============================================================
// 将装饰物放置到甲片上
// ============================================================
Mesh OrnamentGenerator::placeOnNail(const Mesh& nailMesh, const Mesh& ornament,
                                     float u, float v, float scale, float rotation) {
    Mesh result = nailMesh;
    Mesh orn = ornament;

    // 缩放
    scaleMesh(orn, scale);
    // 旋转
    rotateMeshZ(orn, rotation);

    // 在甲片网格中查找最接近 (u,v) 的顶点，获取真实表面Z值
    // 甲面是弧形曲面，不同位置的Z值不同，必须贴合实际表面
    float bestDist = 1e30f;
    float surfaceX = 0, surfaceY = 0, surfaceZ = 0;

    if (!nailMesh.uvs.empty()) {
        // 有UV数据：直接用UV匹配
        for (size_t i = 0; i < nailMesh.uvs.size(); i++) {
            float du = nailMesh.uvs[i].u - u;
            float dv = nailMesh.uvs[i].v - v;
            float dist = du * du + dv * dv;
            if (dist < bestDist) {
                bestDist = dist;
                surfaceX = nailMesh.vertices[i].x;
                surfaceY = nailMesh.vertices[i].y;
                surfaceZ = nailMesh.vertices[i].z;
            }
        }
    } else {
        // 无UV数据：用包围盒插值估算
        float nailW = nailMesh.bbox.max.x - nailMesh.bbox.min.x;
        float nailL = nailMesh.bbox.max.y - nailMesh.bbox.min.y;
        surfaceX = nailMesh.bbox.min.x + u * nailW;
        surfaceY = nailMesh.bbox.min.y + v * nailL;
        // 在顶点中找最接近 (surfaceX, surfaceY) 的点
        for (size_t i = 0; i < nailMesh.vertices.size(); i++) {
            float dx = nailMesh.vertices[i].x - surfaceX;
            float dy = nailMesh.vertices[i].y - surfaceY;
            float dist = dx * dx + dy * dy;
            if (dist < bestDist) {
                bestDist = dist;
                surfaceZ = nailMesh.vertices[i].z;
            }
        }
    }

    // 装饰物底部Z值
    float ornBotZ = orn.bbox.min.z;
    // 装饰物中心XY
    float ornCenterX = orn.bbox.center().x;
    float ornCenterY = orn.bbox.center().y;

    // 将装饰物贴合到甲面表面：底部对齐表面Z
    translateMesh(orn, surfaceX - ornCenterX,
                       surfaceY - ornCenterY,
                       surfaceZ - ornBotZ);

    // 合并到结果
    result = mergeMeshes(result, orn);

    std::cout << "[装饰物] 已贴合甲面 (" << u << ", " << v << ")"
              << ", 表面坐标=(" << surfaceX << "," << surfaceY << "," << surfaceZ << ")"
              << ", 缩放=" << scale
              << ", 装饰物三角形数=" << orn.triangles.size()
              << std::endl;

    return result;
}

} // namespace NailPrint3D
