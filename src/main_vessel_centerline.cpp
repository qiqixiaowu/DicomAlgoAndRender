/**
 * @file main_vessel_centerline.cpp
 * @brief Demo: 3D Vessel Centerline Extraction
 *
 * This demo shows how to use the VascularCenterline library to extract
 * a centerline tree from a 3D vessel mask volume.
 *
 * Build: g++ -std=c++17 -O2 -fopenmp main_vessel_centerline.cpp -o vessel_centerline
 *        (CUDA build: nvcc -std=c++17 -O2 main_vessel_centerline.cpp -o vessel_centerline_gpu)
 *
 * Usage: vessel_centerline <mask_file> <seed_x> <seed_y> <seed_z>
 */

#include "vessel_centerline.hpp"
#include <fstream>
#include <sstream>

using namespace vessel;

// ============================================================================
//  Simple MHD reader for mask volumes (simplified)
// ============================================================================

struct VolumeData {
    std::vector<unsigned char> mask;
    std::vector<short> image;
    int size[3];
    double spacing[3];
};

bool ReadMHD(const std::string& filename, VolumeData& vol) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open: " << filename << std::endl;
        return false;
    }

    std::string line;
    std::string rawFile;
    vol.spacing[0] = vol.spacing[1] = vol.spacing[2] = 1.0;
    vol.size[0] = vol.size[1] = vol.size[2] = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key;
        if (std::getline(iss, key, '=')) {
            // Trim whitespace
            key.erase(key.find_last_not_of(" \t") + 1);
            std::string value;
            std::getline(iss, value);
            value.erase(0, value.find_first_not_of(" \t"));

            if (key == "DimSize") {
                std::istringstream vs(value);
                vs >> vol.size[0] >> vol.size[1] >> vol.size[2];
            } else if (key == "ElementSpacing") {
                std::istringstream vs(value);
                vs >> vol.spacing[0] >> vol.spacing[1] >> vol.spacing[2];
            } else if (key == "ElementDataFile") {
                rawFile = value;
            }
        }
    }
    file.close();

    if (rawFile.empty() || vol.size[0] == 0) return false;

    // Read raw data
    std::string rawPath = filename.substr(0, filename.find_last_of("/\\") + 1) + rawFile;
    std::ifstream raw(rawPath, std::ios::binary);
    if (!raw.is_open()) {
        std::cerr << "Cannot open raw: " << rawPath << std::endl;
        return false;
    }

    int total = vol.size[0] * vol.size[1] * vol.size[2];
    vol.mask.resize(total);
    raw.read(reinterpret_cast<char*>(vol.mask.data()), total);
    raw.close();

    // Create a synthetic image (gradient) for medialness
    vol.image.resize(total);
    for (int k = 0; k < vol.size[2]; k++) {
        for (int j = 0; j < vol.size[1]; j++) {
            for (int i = 0; i < vol.size[0]; i++) {
                vol.image[k * vol.size[0] * vol.size[1] + j * vol.size[0] + i] =
                    static_cast<short>(100 + 50 * std::sin(i * 0.1) * std::cos(j * 0.1));
            }
        }
    }

    return true;
}

// ============================================================================
//  Generate synthetic vessel for testing
// ============================================================================

void GenerateSyntheticVessel(VolumeData& vol) {
    // Create a simple tubular vessel along a curved path
    vol.size[0] = 128;
    vol.size[1] = 128;
    vol.size[2] = 128;
    vol.spacing[0] = vol.spacing[1] = vol.spacing[2] = 0.5;

    int total = vol.size[0] * vol.size[1] * vol.size[2];
    vol.mask.resize(total, 0);
    vol.image.resize(total, 0);

    // Vessel center path: a curve from (64, 64, 0) to (64, 64, 127)
    // with some lateral displacement
    auto vesselCenter = [](int z) -> std::pair<double, double> {
        double t = static_cast<double>(z) / 127.0;
        double x = 64 + 20 * std::sin(t * PI * 2);
        double y = 64 + 15 * std::cos(t * PI * 3);
        return {x, y};
    };

    double vesselRadius = 4.0;  // mm

    for (int k = 0; k < vol.size[2]; k++) {
        auto [cx, cy] = vesselCenter(k);
        for (int j = 0; j < vol.size[1]; j++) {
            for (int i = 0; i < vol.size[0]; i++) {
                int idx = k * vol.size[0] * vol.size[1] + j * vol.size[0] + i;
                double dx = i - cx;
                double dy = j - cy;
                double dist = std::sqrt(dx * dx + dy * dy);

                if (dist <= vesselRadius) {
                    vol.mask[idx] = 7;  // vessel label
                    // Bright vessel, dark background (for medialness)
                    vol.image[idx] = static_cast<short>(200 - dist * 30);
                } else {
                    vol.image[idx] = 50;
                }
            }
        }
    }
}

// ============================================================================
//  Print centerline tree
// ============================================================================

void PrintTree(const std::shared_ptr<CenterlineNode>& node, int depth = 0) {
    std::string indent(depth * 2, ' ');
    std::cout << indent << "Branch #" << node->id
              << " (" << node->points.size() << " points)";
    if (!node->points.empty()) {
        auto& p0 = node->points[0];
        auto& pN = node->points.back();
        std::cout << " [(" << p0.x << "," << p0.y << "," << p0.z << ") -> ("
                  << pN.x << "," << pN.y << "," << pN.z << ")]";
    }
    std::cout << "\n";

    for (auto& child : node->children) {
        PrintTree(child, depth + 1);
    }
}

// ============================================================================
//  Export centerline to JSON for visualization
// ============================================================================

void ExportJSON(const std::shared_ptr<CenterlineNode>& root,
                const std::string& filename)
{
    std::ofstream out(filename);
    out << "{\n  \"centerlines\": [\n";

    bool first = true;
    auto nodes = root->PreOrder();
    for (auto* node : nodes) {
        if (node->points.empty()) continue;
        if (!first) out << ",\n";
        first = false;
        out << "    {\n";
        out << "      \"id\": " << node->id << ",\n";
        out << "      \"points\": [";
        for (size_t i = 0; i < node->points.size(); i++) {
            if (i > 0) out << ", ";
            out << "[" << node->points[i].x << ", "
                << node->points[i].y << ", "
                << node->points[i].z << "]";
        }
        out << "]\n    }";
    }

    out << "\n  ]\n}\n";
    out.close();
    std::cout << "Centerline exported to: " << filename << std::endl;
}

// ============================================================================
//  Main
// ============================================================================

int main(int argc, char* argv[])
{
    std::cout << "========================================\n";
    std::cout << "  3D Vessel Centerline Extraction Demo\n";
    std::cout << "  Profile Medialness + Dijkstra + GPU\n";
    std::cout << "========================================\n\n";

    VolumeData vol;

    if (argc >= 2) {
        // Read from MHD file
        std::cout << "Reading volume from: " << argv[1] << "\n";
        if (!ReadMHD(argv[1], vol)) {
            std::cerr << "Failed to read volume!\n";
            return 1;
        }
    } else {
        // Generate synthetic vessel
        std::cout << "Generating synthetic vessel volume (128x128x128)...\n";
        GenerateSyntheticVessel(vol);
    }

    std::cout << "Volume size: " << vol.size[0] << "x" << vol.size[1]
              << "x" << vol.size[2] << "\n";
    std::cout << "Spacing: " << vol.spacing[0] << "x" << vol.spacing[1]
              << "x" << vol.spacing[2] << " mm\n";

    // Count vessel voxels
    int vesselCount = 0;
    for (auto& v : vol.mask) if (v == 7) vesselCount++;
    std::cout << "Vessel voxels: " << vesselCount << "\n\n";

    // Create centerline extractor
    VascularCenterline extractor;
    unsigned int size[3] = {(unsigned int)vol.size[0],
                            (unsigned int)vol.size[1],
                            (unsigned int)vol.size[2]};
    extractor.SetVolume(vol.image.data(), vol.mask.data(), size, vol.spacing);
    extractor.SetVesselLabel(7);

    // Set seed point (start of vessel)
    Point3Dd seed(64 * vol.spacing[0], 64 * vol.spacing[1], 0);
    if (argc >= 5) {
        seed.x = std::stod(argv[2]);
        seed.y = std::stod(argv[3]);
        seed.z = std::stod(argv[4]);
    }
    std::cout << "Seed point: (" << seed.x << ", " << seed.y << ", " << seed.z << ")\n\n";

    // Extract centerline (CPU)
    std::cout << "--- CPU Extraction ---\n";
    auto root = std::make_shared<CenterlineNode>();
    auto t1 = std::chrono::steady_clock::now();
    extractor.ExtractAllCenterlines({seed}, root);
    auto t2 = std::chrono::steady_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(t2 - t1).count();

    std::cout << "\nCPU Total time: " << cpuTime << " ms\n\n";

    // Print result
    std::cout << "--- Centerline Tree ---\n";
    PrintTree(root);

    // Export to JSON
    ExportJSON(root, "centerline_result.json");

    // Summary statistics
    auto allNodes = root->PreOrder();
    int totalPoints = 0;
    int branchCount = 0;
    for (auto* n : allNodes) {
        totalPoints += static_cast<int>(n->points.size());
        if (!n->points.empty()) branchCount++;
    }
    std::cout << "\n--- Summary ---\n";
    std::cout << "Branches: " << branchCount << "\n";
    std::cout << "Total points: " << totalPoints << "\n";
    std::cout << "CPU time: " << cpuTime << " ms\n";

    return 0;
}
