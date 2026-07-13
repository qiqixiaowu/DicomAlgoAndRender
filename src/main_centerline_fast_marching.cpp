/**
 * main_centerline_fast_marching.cpp
 *
 * Demo program for 3D centerline extraction via Fast Marching Method.
 *
 * Generates a synthetic Y-shaped vessel (two branches merging into a trunk),
 * extracts the centerline using the Fast Marching + Upwind Eikonal solver,
 * and exports results as JSON for visualization.
 *
 * Build:  g++ -std=c++17 -O2 -o centerline_fm main_centerline_fast_marching.cpp
 *         (add -fopenmp for parallel build)
 */

#include "centerline_fast_marching.hpp"
#include <fstream>
#include <iomanip>
#include <chrono>

using namespace cfm;

// ============================================================================
//  Synthetic vessel generator: Y-shaped tube
// ============================================================================
void GenerateYShapedVessel(Volume3D<unsigned char>& mask, int label = 1) {
    int W = mask.dim[0], H = mask.dim[1], D = mask.dim[2];
    float sx = mask.spacing[0], sy = mask.spacing[1], sz = mask.spacing[2];

    // Trunk: vertical tube from z=D*0.2 to z=D*0.55, centered at (W/2, H/2)
    // Branch A: from z=D*0.55 to z=D*0.9, offset to the left
    // Branch B: from z=D*0.55 to z=D*0.9, offset to the right
    float trunkRadius = 5.0f;   // in voxels
    float branchRadius = 3.5f;

    for (int z = 0; z < D; ++z) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float dx = x - W / 2.0f;
                float dy = y - H / 2.0f;
                float r2 = dx * dx + dy * dy;

                if (z < D * 0.55) {
                    // Trunk
                    if (r2 <= trunkRadius * trunkRadius && z >= D * 0.2)
                        mask[mask.Get1DIndex(x, y, z)] = label;
                } else {
                    // Branch A (left) and B (right)
                    float zT = (z - D * 0.55f) / (D * 0.35f);  // 0→1
                    float offsetA = -zT * W * 0.15f;
                    float offsetB =  zT * W * 0.15f;

                    float dxA = x - (W / 2.0f + offsetA);
                    float dyA = y - H / 2.0f;
                    if (dxA * dxA + dyA * dyA <= branchRadius * branchRadius)
                        mask[mask.Get1DIndex(x, y, z)] = label;

                    float dxB = x - (W / 2.0f + offsetB);
                    float dyB = y - H / 2.0f;
                    if (dxB * dxB + dyB * dyB <= branchRadius * branchRadius)
                        mask[mask.Get1DIndex(x, y, z)] = label;
                }
            }
        }
    }
}

// ============================================================================
//  Export centerline to JSON
// ============================================================================
void ExportJSON(const CenterlineExtractor& extractor, const std::string& filename) {
    std::ofstream ofs(filename);
    ofs << std::fixed << std::setprecision(4);
    ofs << "{\n  \"centerlines\": [\n";

    for (size_t i = 0; i < extractor.centerlines.size(); ++i) {
        const auto& line = extractor.centerlines[i];
        ofs << "    [";
        for (size_t j = 0; j < line.size(); ++j) {
            int x, y, z;
            extractor.mask.Get3DIndex(line[j], x, y, z);
            ofs << "[" << x * extractor.mask.spacing[0] << ","
                << y * extractor.mask.spacing[1] << ","
                << z * extractor.mask.spacing[2] << "]";
            if (j + 1 < line.size()) ofs << ", ";
        }
        ofs << "]";
        if (i + 1 < extractor.centerlines.size()) ofs << ",";
        ofs << "\n";
    }

    ofs << "  ]\n}\n";
    ofs.close();
    std::cout << "[CFM] Centerline exported to " << filename << std::endl;
}

// ============================================================================
//  Export mask to simple MHD format for ITK/Slicer visualization
// ============================================================================
void ExportMHD(const Volume3D<unsigned char>& mask, const std::string& baseName) {
    std::string rawFile = baseName + ".raw";
    std::string mhdFile = baseName + ".mhd";

    // Write raw
    std::ofstream raw(rawFile, std::ios::binary);
    raw.write(reinterpret_cast<const char*>(mask.data.data()),
              mask.totalSize * sizeof(unsigned char));
    raw.close();

    // Write mhd
    std::ofstream mhd(mhdFile);
    mhd << "ObjectType = Image\n"
        << "NDims = 3\n"
        << "BinaryData = True\n"
        << "BinaryDataByteOrderMSB = False\n"
        << "CompressedData = False\n"
        << "TransformMatrix = 1 0 0 0 1 0 0 0 1\n"
        << "Offset = 0 0 0\n"
        << "CenterOfRotation = 0 0 0\n"
        << "ElementSpacing = "
        << mask.spacing[0] << " " << mask.spacing[1] << " " << mask.spacing[2] << "\n"
        << "DimSize = "
        << mask.dim[0] << " " << mask.dim[1] << " " << mask.dim[2] << "\n"
        << "AnatomicalOrientation = RAS\n"
        << "ElementType = MET_UCHAR\n"
        << "ElementDataFile = " << rawFile << "\n";
    mhd.close();

    std::cout << "[CFM] Mask exported to " << mhdFile << std::endl;
}

// ============================================================================
//  Print centerline tree
// ============================================================================
void PrintCenterlines(const CenterlineExtractor& extractor) {
    std::cout << "\n=== Centerline Extraction Results ===\n";
    std::cout << "Total centerlines: " << extractor.centerlines.size() << "\n\n";

    for (size_t i = 0; i < extractor.centerlines.size(); ++i) {
        const auto& line = extractor.centerlines[i];
        std::cout << "Centerline #" << i << " (length=" << line.size() << " voxels):\n";

        // Print first 5 and last 5 points
        int n = (int)line.size();
        int show = std::min(n, 5);
        for (int j = 0; j < show; ++j) {
            int x, y, z;
            extractor.mask.Get3DIndex(line[j], x, y, z);
            std::cout << "  [" << j << "] voxel(" << x << "," << y << "," << z << ")"
                      << " = phys(" << x * extractor.mask.spacing[0] << ","
                      << y * extractor.mask.spacing[1] << ","
                      << z * extractor.mask.spacing[2] << ") mm\n";
        }
        if (n > 10) std::cout << "  ... (" << n - 10 << " more points)\n";
        for (int j = std::max(show, n - 5); j < n; ++j) {
            int x, y, z;
            extractor.mask.Get3DIndex(line[j], x, y, z);
            std::cout << "  [" << j << "] voxel(" << x << "," << y << "," << z << ")"
                      << " = phys(" << x * extractor.mask.spacing[0] << ","
                      << y * extractor.mask.spacing[1] << ","
                      << z * extractor.mask.spacing[2] << ") mm\n";
        }
        std::cout << "\n";
    }

    // Print source point
    if (extractor.sourcePoint >= 0) {
        int x, y, z;
        extractor.mask.Get3DIndex(extractor.sourcePoint, x, y, z);
        std::cout << "Source point (center): voxel(" << x << "," << y << "," << z << ")\n";
    }
}

// ============================================================================
//  Main
// ============================================================================
int main() {
    std::cout << "=== 3D Centerline Extraction via Fast Marching Method ===\n";
    std::cout << "Algorithm: Upwind Eikonal solver + Fast Marching + Steepest Descent\n\n";

    // Create synthetic Y-shaped vessel
    const int W = 64, H = 64, D = 80;
    const float sx = 0.5f, sy = 0.5f, sz = 0.5f;  // 0.5mm isotropic

    Volume3D<unsigned char> mask;
    mask.SetSize(W, H, D, sx, sy, sz);
    GenerateYShapedVessel(mask, 1);

    // Count vessel voxels
    int vesselCount = 0;
    for (int i = 0; i < mask.Length(); ++i)
        if (mask[i] == 1) ++vesselCount;
    std::cout << "Vessel mask: " << W << "x" << H << "x" << D
              << " (" << vesselCount << " vessel voxels)\n\n";

    // Export mask for visualization
    ExportMHD(mask, "vessel_mask_fm");

    // Extract centerline
    auto t0 = std::chrono::high_resolution_clock::now();

    CenterlineExtractor extractor;
    extractor.SetMask(mask.data.data(), W, H, D, sx, sy, sz);
    extractor.minCenterlineLength = 30;
    extractor.smallBranchRatio = 0.05;
    extractor.nearCenterRatio = 0.10;

    bool ok = extractor.Extract(1);

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    if (!ok) {
        std::cerr << "Centerline extraction failed!\n";
        return 1;
    }

    std::cout << "Extraction time: " << elapsed << " seconds\n";

    // Print results
    PrintCenterlines(extractor);

    // Export to JSON
    ExportJSON(extractor, "centerline_fm_result.json");

    // Export centerline as labeled mask
    Volume3D<unsigned char> clMask;
    clMask.SetSize(W, H, D, sx, sy, sz);
    for (int idx : extractor.centerlineSet) {
        clMask[idx] = 5;  // centerline label
    }
    ExportMHD(clMask, "centerline_fm_mask");

    std::cout << "\nDone. Use 3D Slicer or ITK-SNAP to view the .mhd files.\n";
    return 0;
}
