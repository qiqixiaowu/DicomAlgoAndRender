/**
 * centerline_fast_marching.hpp
 *
 * 3D Centerline Extraction via Fast Marching Method + Upwind Eikonal Solver
 *
 * Algorithm pipeline:
 *   1. Unit-speed wavefront (F=1) from boundary → distance field D(x), max = source point
 *   2. Moderate-speed wavefront (F=1/√D) from source → branch endpoints
 *   3. Fast-speed wavefront (F=1/exp(D)) from source → steepest descent backtrace = centerline
 *
 * The Eikonal equation |∇T|·F = 1 is discretized with a first-order upwind scheme
 * and solved via Fast Marching Method (min-heap driven causal ordering).
 *
 * Zero third-party dependencies. C++17. OpenMP optional.
 *
 * Original reference: McsfAlgoView3DCenterLine (United-Imaging, 2013)
 */

#ifndef CENTERLINE_FAST_MARCHING_HPP
#define CENTERLINE_FAST_MARCHING_HPP

#include <vector>
#include <set>
#include <map>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <iostream>

namespace cfm {

// ============================================================================
//  Volume3D  —  lightweight 3D image wrapper (1D-indexed for cache efficiency)
// ============================================================================
template <typename T>
class Volume3D {
public:
    std::vector<T> data;
    int dim[3];          // [x, y, z]
    float spacing[3];    // physical spacing
    long sliceSize;      // dim[0] * dim[1]
    long totalSize;      // dim[0] * dim[1] * dim[2]

    Volume3D() : sliceSize(0), totalSize(0) {
        dim[0] = dim[1] = dim[2] = 0;
        spacing[0] = spacing[1] = spacing[2] = 1.0f;
    }

    void SetSize(int x, int y, int z, float sx = 1.f, float sy = 1.f, float sz = 1.f) {
        dim[0] = x; dim[1] = y; dim[2] = z;
        spacing[0] = sx; spacing[1] = sy; spacing[2] = sz;
        sliceSize = (long)x * y;
        totalSize = sliceSize * z;
        data.assign(totalSize, T{});
    }

    int Length() const { return static_cast<int>(totalSize); }

    T& operator[](int idx) { return data[idx]; }
    const T& operator[](int idx) const { return data[idx]; }

    int Get1DIndex(int x, int y, int z) const {
        return static_cast<int>(z * sliceSize + (long)y * dim[0] + x);
    }

    bool Get1DIndex(int x, int y, int z, int& idx) const {
        if (x < 0 || y < 0 || z < 0 || x >= dim[0] || y >= dim[1] || z >= dim[2])
            return false;
        idx = static_cast<int>(z * sliceSize + (long)y * dim[0] + x);
        return true;
    }

    bool Get3DIndex(int idx, int& x, int& y, int& z) const {
        if (idx < 0 || idx >= totalSize) return false;
        z = static_cast<int>(idx / sliceSize);
        int rem = static_cast<int>(idx - z * sliceSize);
        y = rem / dim[0];
        x = rem % dim[0];
        return true;
    }

    /// 6-neighbors: [-x, +x, -y, +y, -z, +z], -1 if out of bounds
    bool GetNeighbor6(int idx, int nb[6]) const {
        int x, y, z;
        if (!Get3DIndex(idx, x, y, z)) return false;
        nb[0] = (x > 0)          ? idx - 1           : -1;
        nb[1] = (x < dim[0] - 1) ? idx + 1           : -1;
        nb[2] = (y > 0)          ? idx - dim[0]      : -1;
        nb[3] = (y < dim[1] - 1) ? idx + dim[0]      : -1;
        nb[4] = (z > 0)          ? static_cast<int>(idx - sliceSize) : -1;
        nb[5] = (z < dim[2] - 1) ? static_cast<int>(idx + sliceSize) : -1;
        return true;
    }

    /// 26-neighbors
    bool GetNeighbor26(int idx, int nb[26]) const {
        int x, y, z;
        if (!Get3DIndex(idx, x, y, z)) return false;
        int n = 0;
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    int ni;
                    if (Get1DIndex(x + dx, y + dy, z + dz, ni))
                        nb[n] = ni;
                    else
                        nb[n] = -1;
                    ++n;
                }
        return true;
    }
};

// ============================================================================
//  SparseHashMap  —  open-addressing hash map for sparse voxel storage
//  Key = voxel 1D index, Value = float/double distance value
//  Used by Fast Marching to store only "alive" / "trial" voxels (sparse)
// ============================================================================
class SparseHashMap {
public:
    std::unordered_map<int, double> m_map;

    void Clear() { m_map.clear(); }
    bool Empty() const { return m_map.empty(); }
    int  Size() const { return static_cast<int>(m_map.size()); }

    bool Find(int idx) const { return m_map.find(idx) != m_map.end(); }

    bool Find(int idx, double& val) const {
        auto it = m_map.find(idx);
        if (it == m_map.end()) return false;
        val = it->second;
        return true;
    }

    void Insert(int idx, double val) { m_map[idx] = val; }
    void Erase(int idx) { m_map.erase(idx); }
    double Get(int idx) const {
        auto it = m_map.find(idx);
        return (it != m_map.end()) ? it->second : 1e8;
    }
};

// ============================================================================
//  MinHeap  —  binary min-heap with index tracking for Fast Marching Method
//  Stores (voxelIndex, arrivalTime) pairs, supports DecreaseKey via hash map
// ============================================================================
class MinHeap {
public:
    struct Entry { int index; double value; };
    std::vector<Entry> heap;
    std::unordered_map<int, int> posMap;  // voxelIndex → position in heap

    bool IsEmpty() const { return heap.empty(); }
    int  Size() const { return static_cast<int>(heap.size()); }

    void Clear() { heap.clear(); posMap.clear(); }

    void Insert(int idx, double val) {
        auto it = posMap.find(idx);
        if (it != posMap.end()) {
            // DecreaseKey: update if new value is smaller
            if (val < heap[it->second].value) {
                heap[it->second].value = val;
                SiftUp(it->second);
            }
        } else {
            int pos = static_cast<int>(heap.size());
            heap.push_back({idx, val});
            posMap[idx] = pos;
            SiftUp(pos);
        }
    }

    bool Extract(int& idx, double& val) {
        if (heap.empty()) return false;
        idx = heap[0].index;
        val = heap[0].value;
        posMap.erase(idx);

        int last = static_cast<int>(heap.size()) - 1;
        if (last > 0) {
            heap[0] = heap[last];
            posMap[heap[0].index] = 0;
        }
        heap.pop_back();
        if (!heap.empty()) SiftDown(0);
        return true;
    }

private:
    void SiftUp(int pos) {
        while (pos > 0) {
            int parent = (pos - 1) / 2;
            if (heap[pos].value < heap[parent].value) {
                Swap(pos, parent);
                pos = parent;
            } else break;
        }
    }

    void SiftDown(int pos) {
        int n = static_cast<int>(heap.size());
        while (true) {
            int left = 2 * pos + 1, right = 2 * pos + 2;
            int smallest = pos;
            if (left < n && heap[left].value < heap[smallest].value) smallest = left;
            if (right < n && heap[right].value < heap[smallest].value) smallest = right;
            if (smallest == pos) break;
            Swap(pos, smallest);
            pos = smallest;
        }
    }

    void Swap(int a, int b) {
        std::swap(heap[a], heap[b]);
        posMap[heap[a].index] = a;
        posMap[heap[b].index] = b;
    }
};

// ============================================================================
//  UpwindSolver  —  first-order upwind discretization of the Eikonal equation
//
//  Solves |∇T| = 1/F  (i.e. F|∇T| = 1) at a voxel using its 6 face-neighbors.
//
//  For each axis d ∈ {x, y, z}, take the upwind neighbor (smaller T value):
//      t_d = min(T(neighbor-), T(neighbor+))
//
//  Sort t_x ≤ t_y ≤ t_z, then try:
//    (a) 3D solution: solve  (T-t_x)² + (T-t_y)² + (T-t_z)² = (1/F)²
//    (b) 2D solution: solve  (T-t_x)² + (T-t_y)² = (1/F)²
//    (c) 1D solution: T = t_x + 1/F
// ============================================================================
class UpwindSolver {
public:
    /// Unit-speed upwind (F = 1, solves |∇T| = 1 → distance field)
    static double SolveUnit(const Volume3D<unsigned char>& vol, int idx,
                            const SparseHashMap& alive) {
        return Solve(vol, idx, alive, 1.0);
    }

    /// Moderate-speed upwind: F = 1/sqrt(D(x)), solves |∇T| = sqrt(D(x))
    static double SolveModerate(const Volume3D<unsigned char>& vol, int idx,
                                const SparseHashMap& alive,
                                const SparseHashMap& distField) {
        double D = distField.Get(idx);
        if (D < 1e-6) D = 1e-6;
        double speed = 1.0 / std::sqrt(D);   // F = 1/√D
        return Solve(vol, idx, alive, speed);
    }

    /// Fast-speed upwind: F = 1/exp(D(x)), solves |∇T| = exp(D(x))
    static double SolveFast(const Volume3D<unsigned char>& vol, int idx,
                            const SparseHashMap& alive,
                            const SparseHashMap& distField) {
        double D = distField.Get(idx);
        double speed = 1.0 / std::exp(D);    // F = 1/exp(D)
        return Solve(vol, idx, alive, speed);
    }

private:
    /// Core upwind solver: solves F|∇T| = 1 at voxel idx
    /// speed = F (wave propagation speed at this voxel)
    static double Solve(const Volume3D<unsigned char>& vol, int idx,
                        const SparseHashMap& alive, double speed) {
        int nb[6];
        vol.GetNeighbor6(idx, nb);

        // Collect upwind values for each axis
        double tAxis[3];
        for (int axis = 0; axis < 3; ++axis) {
            double tNeg = 1e8, tPos = 1e8;
            int iNeg = nb[2 * axis];
            int iPos = nb[2 * axis + 1];

            if (iNeg != -1) {
                double v;
                if (alive.Find(iNeg, v)) tNeg = v;
            }
            if (iPos != -1) {
                double v;
                if (alive.Find(iPos, v)) tPos = v;
            }

            // If one side is out of bounds, mirror the other
            if (iNeg == -1 && iPos != -1) tNeg = tPos;
            if (iPos == -1 && iNeg != -1) tPos = tNeg;

            tAxis[axis] = std::min(tNeg, tPos);
        }

        // Sort ascending: t0 ≤ t1 ≤ t2
        std::sort(tAxis, tAxis + 3);

        double invF = speed;  // 1/F = travel time per unit distance
        double T;

        // --- Try 3D solution ---
        // (T-t0)² + (T-t1)² + (T-t2)² = invF²
        // 3T² - 2(t0+t1+t2)T + (t0²+t1²+t2² - invF²) = 0
        // discriminant form (see Sethian 1996):
        double sum = tAxis[0] + tAxis[1] + tAxis[2];
        double a = tAxis[0] * (tAxis[1] - tAxis[0]) +
                   tAxis[1] * (tAxis[2] - tAxis[1]) +
                   tAxis[2] * (tAxis[0] - tAxis[2]) +
                   1.5 * invF * invF;
        if (a >= 0) {
            T = sum / 3.0 + std::sqrt(a * 2.0 / 9.0);
            if (T > tAxis[2]) return T;
        }

        // --- Try 2D solution ---
        // (T-t0)² + (T-t1)² = invF²
        double b = -(tAxis[0] - tAxis[1]) * (tAxis[0] - tAxis[1]) + 2.0 * invF * invF;
        if (b >= 0) {
            T = (tAxis[0] + tAxis[1] + std::sqrt(b)) / 2.0;
            if (T > tAxis[1]) return T;
        }

        // --- 1D solution ---
        return tAxis[0] + invF;
    }
};

// ============================================================================
//  CenterlineExtractor  —  main algorithm class
//
//  Pipeline:
//    1. Extract vessel boundary (6-neighbor boundary detection)
//    2. Extract vessel inside (region growing from seed, excluding surface)
//    3. PointSourceExtract: Fast Marching with F=1 from boundary → D(x)
//       Source point = argmax D(x) (farthest from boundary = center)
//    4. BoundaryPointExtract: Fast Marching with F=1/√D from source → endpoints
//    5. FastLevelSet: Fast Marching with F=1/exp(D) from source → T(x)
//    6. SteepestDescent: from each endpoint, follow -∇T back to source = centerline
//    7. Post-processing: cut small branches, remove duplicates
// ============================================================================
class CenterlineExtractor {
public:
    Volume3D<unsigned char> mask;       // binary vessel mask
    int sourcePoint;                     // computed source (center) point
    SparseHashMap distField;             // D(x): unit-speed distance from boundary
    SparseHashMap fastLevelSet;          // T(x): fast-speed arrival time from source
    std::vector<std::vector<int>> centerlines;  // extracted centerlines (voxel indices)
    std::set<int> centerlineSet;         // all voxels on any centerline

    // Parameters
    int hashRatio = 2000;
    int minCenterlineLength = 50;
    double smallBranchRatio = 0.05;
    double nearCenterRatio = 0.10;

    CenterlineExtractor() : sourcePoint(-1) {}

    void SetMask(const unsigned char* maskData, int x, int y, int z,
                 float sx, float sy, float sz) {
        mask.SetSize(x, y, z, sx, sy, sz);
        std::copy(maskData, maskData + mask.totalSize, mask.data.begin());
    }

    /// Main entry: extract centerline tree from binary mask
    bool Extract(unsigned char label = 1) {
        // Step 1: collect vessel voxels
        SparseHashMap aliveVessel;
        for (int i = 0; i < mask.Length(); ++i) {
            if (mask[i] == label)
                aliveVessel.Insert(i, 0.0);
        }
        if (aliveVessel.Empty()) return false;

        // Step 2: compute distance field D(x) via unit-speed Fast Marching
        if (!PointSourceExtract(aliveVessel)) return false;
        std::cout << "[CFM] Source point: " << sourcePoint << std::endl;

        // Step 3: find branch endpoints via moderate-speed Fast Marching
        std::vector<int> boundaryPoints;
        if (!BoundaryPointExtract(boundaryPoints)) return false;
        std::cout << "[CFM] Found " << boundaryPoints.size() << " boundary points" << std::endl;

        // Step 4: compute fast arrival time T(x) via fast-speed Fast Marching
        if (!ComputeFastLevelSet()) return false;

        // Step 5: steepest descent from each endpoint to source
        centerlines.resize(boundaryPoints.size());
        for (size_t i = 0; i < boundaryPoints.size(); ++i) {
            SteepestDescent(boundaryPoints[i], sourcePoint, fastLevelSet,
                            centerlines[i]);
        }

        // Step 6: post-processing
        PostProcess();

        return true;
    }

    /// Extract centerline between two specific points
    bool ExtractBetween(int ptA, int ptB) {
        if (fastLevelSet.Empty()) return false;

        // Snap points to fastLevelSet if needed
        ptA = SnapToField(ptA);
        ptB = SnapToField(ptB);
        if (ptA < 0 || ptB < 0) return false;

        std::vector<int> lineA, lineB;
        if (!SteepestDescent(ptA, sourcePoint, fastLevelSet, lineA)) return false;
        if (!SteepestDescent(ptB, sourcePoint, fastLevelSet, lineB)) return false;

        // Merge two paths: find common prefix, combine non-common parts
        centerlines.clear();
        centerlines.push_back(MergePaths(lineA, lineB));
        return true;
    }

    /// Get centerline as 3D points (physical coordinates)
    void GetCenterlinePoints(int lineIdx, std::vector<float>& pts) const {
        if (lineIdx < 0 || lineIdx >= (int)centerlines.size()) return;
        pts.clear();
        for (int idx : centerlines[lineIdx]) {
            int x, y, z;
            mask.Get3DIndex(idx, x, y, z);
            pts.push_back(x * mask.spacing[0]);
            pts.push_back(y * mask.spacing[1]);
            pts.push_back(z * mask.spacing[2]);
        }
    }

private:
    // ------------------------------------------------------------------------
    //  Step 2: PointSourceExtract
    //  Fast Marching with unit speed (F=1) from vessel boundary inward.
    //  Produces D(x) = Euclidean distance from boundary (approximate).
    //  Source point = argmax D(x) = point farthest from boundary = vessel center.
    // ------------------------------------------------------------------------
    bool PointSourceExtract(const SparseHashMap& aliveVessel) {
        // 2a: Extract vessel boundary (voxels with at least one non-vessel neighbor)
        std::vector<int> surface;
        for (const auto& kv : aliveVessel.m_map) {
            int idx = kv.first;
            int nb[6];
            mask.GetNeighbor6(idx, nb);
            bool isBoundary = false;
            for (int j = 0; j < 6; ++j) {
                if (nb[j] == -1) { isBoundary = true; break; }
                if (!aliveVessel.Find(nb[j])) { isBoundary = true; break; }
            }
            if (isBoundary) surface.push_back(idx);
        }
        if (surface.empty()) return false;

        // 2b: Extract vessel inside (region growing from seed, excluding surface)
        SparseHashMap inside;
        // Find a seed point: a vessel voxel whose 6-neighbors are all vessel
        int seed = -1;
        for (const auto& kv : aliveVessel.m_map) {
            int idx = kv.first;
            int nb[6];
            mask.GetNeighbor6(idx, nb);
            bool allInside = true;
            for (int j = 0; j < 6; ++j) {
                if (nb[j] == -1 || !aliveVessel.Find(nb[j])) {
                    allInside = false; break;
                }
            }
            if (allInside) { seed = idx; break; }
        }
        if (seed == -1) seed = surface[0];  // fallback

        // Region growing to mark "inside" voxels (not on surface)
        std::vector<bool> isSurface(mask.Length(), false);
        for (int idx : surface) isSurface[idx] = true;

        std::queue<int> q;
        q.push(seed);
        inside.Insert(seed, 0);
        while (!q.empty()) {
            int cur = q.front(); q.pop();
            int nb[6];
            mask.GetNeighbor6(cur, nb);
            for (int j = 0; j < 6; ++j) {
                if (nb[j] == -1) continue;
                if (isSurface[nb[j]]) continue;
                if (inside.Find(nb[j])) continue;
                if (!aliveVessel.Find(nb[j])) continue;
                inside.Insert(nb[j], 0);
                q.push(nb[j]);
            }
        }

        // 2c: Fast Marching with F=1 from surface into inside
        SparseHashMap trial;   // trial set (in heap)
        MinHeap heap;

        // Initialize: surface voxels have T=0, their inside neighbors are trial
        for (int idx : surface) {
            distField.Insert(idx, 0.0);
        }
        for (int idx : surface) {
            int nb[6];
            mask.GetNeighbor6(idx, nb);
            for (int j = 0; j < 6; ++j) {
                if (nb[j] == -1) continue;
                if (!inside.Find(nb[j])) continue;
                if (distField.Find(nb[j])) continue;
                double val = UpwindSolver::SolveUnit(mask, nb[j], distField);
                trial.Insert(nb[j], val);
                heap.Insert(nb[j], val);
            }
        }

        // March
        int maxIdx = -1;
        double maxVal = 0;
        while (!heap.IsEmpty()) {
            int idx;
            double val;
            if (!heap.Extract(idx, val)) break;

            distField.Insert(idx, val);
            if (val > maxVal) { maxVal = val; maxIdx = idx; }

            int nb[6];
            mask.GetNeighbor6(idx, nb);
            for (int j = 0; j < 6; ++j) {
                if (nb[j] == -1) continue;
                if (!inside.Find(nb[j])) continue;
                if (distField.Find(nb[j])) continue;
                double newVal = UpwindSolver::SolveUnit(mask, nb[j], distField);
                trial.Insert(nb[j], newVal);
                heap.Insert(nb[j], newVal);
            }
        }

        sourcePoint = maxIdx;
        return (sourcePoint >= 0);
    }

    // ------------------------------------------------------------------------
    //  Step 3: BoundaryPointExtract
    //  Fast Marching with F=1/√D(x) from source point outward.
    //  Finds "branch endpoints" = local maxima of the arrival time field.
    // ------------------------------------------------------------------------
    bool BoundaryPointExtract(std::vector<int>& boundaryPoints) {
        SparseHashMap aliveHash;
        SparseHashMap trial;
        MinHeap heap;

        // Initialize from source point
        aliveHash.Insert(sourcePoint, 0.0);
        int nb[6];
        mask.GetNeighbor6(sourcePoint, nb);
        for (int j = 0; j < 6; ++j) {
            if (nb[j] == -1) continue;
            if (!distField.Find(nb[j])) continue;
            if (aliveHash.Find(nb[j])) continue;
            double D = distField.Get(nb[j]);
            double speed = 1.0 / std::sqrt(std::max(D, 1e-6));
            double val = UpwindSolver::Solve(mask, nb[j], aliveHash, speed);
            trial.Insert(nb[j], val);
            heap.Insert(nb[j], val);
        }

        // March with moderate speed
        while (!heap.IsEmpty()) {
            int idx;
            double val;
            if (!heap.Extract(idx, val)) break;
            aliveHash.Insert(idx, val);

            int nb2[6];
            mask.GetNeighbor6(idx, nb2);
            for (int j = 0; j < 6; ++j) {
                if (nb2[j] == -1) continue;
                if (!distField.Find(nb2[j])) continue;
                if (aliveHash.Find(nb2[j])) continue;
                double D = distField.Get(nb2[j]);
                double speed = 1.0 / std::sqrt(std::max(D, 1e-6));
                double newVal = UpwindSolver::Solve(mask, nb2[j], aliveHash, speed);
                trial.Insert(nb2[j], newVal);
                heap.Insert(nb2[j], newVal);
            }
        }

        // Find local maxima: cluster voxels by floor(val/15), find boundary of each cluster
        // Simplified: find voxels whose all neighbors have ≤ value (local maxima)
        std::map<int, std::vector<int>> clusters;
        for (const auto& kv : aliveHash.m_map) {
            int clusterId = static_cast<int>(kv.second / 15.0);
            clusters[clusterId].push_back(kv.first);
        }

        for (auto& [clusterId, voxels] : clusters) {
            // Find the voxel with maximum arrival time in this cluster
            // that has no neighbor with higher clusterId (i.e., it's a "top edge")
            int maxIdx = -1;
            double maxVal = -1;
            for (int idx : voxels) {
                double val = aliveHash.Get(idx);
                // Check if any neighbor has higher clusterId
                bool isTopEdge = true;
                int nb3[6];
                mask.GetNeighbor6(idx, nb3);
                for (int j = 0; j < 6; ++j) {
                    if (nb3[j] == -1) continue;
                    double nVal = aliveHash.Get(nb3[j]);
                    if (nVal > val + 1e-6) {
                        isTopEdge = false;
                        break;
                    }
                }
                if (isTopEdge && val > maxVal) {
                    maxVal = val;
                    maxIdx = idx;
                }
            }
            if (maxIdx >= 0) {
                boundaryPoints.push_back(maxIdx);
            }
        }

        return !boundaryPoints.empty();
    }

    // ------------------------------------------------------------------------
    //  Step 4: ComputeFastLevelSet
    //  Fast Marching with F=1/exp(D(x)) from source point.
    //  Produces T(x) = arrival time used for steepest descent.
    // ------------------------------------------------------------------------
    bool ComputeFastLevelSet() {
        SparseHashMap trial;
        MinHeap heap;

        // Initialize from source
        fastLevelSet.Insert(sourcePoint, 0.0);
        int nb[6];
        mask.GetNeighbor6(sourcePoint, nb);
        for (int j = 0; j < 6; ++j) {
            if (nb[j] == -1) continue;
            if (!distField.Find(nb[j])) continue;
            if (fastLevelSet.Find(nb[j])) continue;
            double val = UpwindSolver::SolveFast(mask, nb[j], fastLevelSet, distField);
            trial.Insert(nb[j], val);
            heap.Insert(nb[j], val);
        }

        // March
        while (!heap.IsEmpty()) {
            int idx;
            double val;
            if (!heap.Extract(idx, val)) break;
            fastLevelSet.Insert(idx, val);

            int nb2[6];
            mask.GetNeighbor6(idx, nb2);
            for (int j = 0; j < 6; ++j) {
                if (nb2[j] == -1) continue;
                if (!distField.Find(nb2[j])) continue;
                if (fastLevelSet.Find(nb2[j])) continue;
                double newVal = UpwindSolver::SolveFast(mask, nb2[j], fastLevelSet, distField);
                trial.Insert(nb2[j], newVal);
                heap.Insert(nb2[j], newVal);
            }
        }

        return !fastLevelSet.Empty();
    }

    // ------------------------------------------------------------------------
    //  Step 5: SteepestDescent
    //  From startPoint, follow -∇T (steepest descent of arrival time) to endPoint.
    //  This traces the centerline path.
    // ------------------------------------------------------------------------
    bool SteepestDescent(int startPoint, int endPoint,
                         const SparseHashMap& field,
                         std::vector<int>& line) {
        // Ensure start has higher T than end (descend from high to low)
        double tStart = field.Get(startPoint);
        double tEnd = field.Get(endPoint);
        if (tStart < tEnd) std::swap(startPoint, endPoint);

        line.clear();
        int cur = startPoint;
        line.push_back(cur);
        centerlineSet.insert(cur);

        int maxIter = mask.Length();  // safety limit
        while (maxIter-- > 0) {
            if (cur == endPoint) {
                std::reverse(line.begin(), line.end());
                return true;
            }

            int nb[6];
            mask.GetNeighbor6(cur, nb);

            // Find neighbor with maximum T difference (steepest descent)
            int nextIdx = -1;
            double maxDrop = 0;
            double curVal = field.Get(cur);
            for (int j = 0; j < 6; ++j) {
                if (nb[j] == -1) continue;
                if (!field.Find(nb[j])) continue;
                double nVal = field.Get(nb[j]);
                double drop = curVal - nVal;
                if (drop > maxDrop) {
                    maxDrop = drop;
                    nextIdx = nb[j];
                }
            }

            if (nextIdx < 0 || maxDrop < 0) return false;

            cur = nextIdx;
            line.push_back(cur);
            centerlineSet.insert(cur);
        }

        return false;
    }

    // ------------------------------------------------------------------------
    //  Step 6: Post-processing
    //  Remove short branches and duplicates.
    // ------------------------------------------------------------------------
    void PostProcess() {
        if (centerlines.empty()) return;

        // Find max line length
        int maxLen = 0;
        for (auto& line : centerlines)
            maxLen = std::max(maxLen, (int)line.size());

        if (maxLen < minCenterlineLength) {
            centerlines.clear();
            return;
        }

        int smallBranch = std::max((int)(maxLen * smallBranchRatio), 20);
        int nearCenter = std::max((int)(maxLen * nearCenterRatio), 30);

        // Cut small branches near center
        CutSmallBranches(nearCenter);

        // Cut sub-branches (run twice)
        SubBranchCutDown(smallBranch);
        SubBranchCutDown(smallBranch);

        // Sort by length (descending)
        std::sort(centerlines.begin(), centerlines.end(),
                  [](const std::vector<int>& a, const std::vector<int>& b) {
                      return a.size() > b.size();
                  });

        // Rebuild centerlineSet
        centerlineSet.clear();
        for (auto& line : centerlines)
            for (int idx : line)
                centerlineSet.insert(idx);
    }

    void CutSmallBranches(int minLen) {
        std::vector<std::vector<int>> filtered;
        for (auto& line : centerlines) {
            if ((int)line.size() >= minLen)
                filtered.push_back(line);
        }
        centerlines = filtered;
    }

    /// Remove branches that are sub-paths of longer branches
    void SubBranchCutDown(int minBranchLen) {
        if (centerlines.empty()) return;

        // Find branch node for each line (first point that differs from all others)
        std::vector<int> branchNode(centerlines.size(), 0);
        for (size_t i = 0; i < centerlines.size(); ++i) {
            for (int k = 0; k < (int)centerlines[i].size(); ++k) {
                bool isNode = true;
                for (size_t j = 0; j < centerlines.size(); ++j) {
                    if (j == i) continue;
                    if (k < (int)centerlines[j].size() &&
                        centerlines[i][k] == centerlines[j][k]) {
                        isNode = false;
                        break;
                    }
                }
                if (isNode) {
                    branchNode[i] = k;
                    break;
                }
            }
        }

        // Cut branches shorter than minBranchLen after the branch node
        std::vector<std::vector<int>> filtered;
        for (size_t i = 0; i < centerlines.size(); ++i) {
            int afterNode = (int)centerlines[i].size() - branchNode[i];
            if (afterNode >= minBranchLen) {
                filtered.push_back(centerlines[i]);
            }
        }
        centerlines = filtered;
    }

    /// Merge two steepest-descent paths into a single centerline
    std::vector<int> MergePaths(const std::vector<int>& lineA,
                                const std::vector<int>& lineB) {
        // Both paths go from endpoint → source (reversed after SteepestDescent)
        // Find common suffix (shared path to source)
        int minLen = std::min(lineA.size(), lineB.size());
        int split = 0;
        for (int i = 0; i < minLen; ++i) {
            if (lineA[lineA.size()-1-i] != lineB[lineB.size()-1-i]) {
                split = i;
                break;
            }
            split = i + 1;
        }

        std::vector<int> merged;
        // A (reversed, from endpoint to split point)
        for (int i = (int)lineA.size() - 1; i >= (int)lineA.size() - split; --i)
            merged.push_back(lineA[i]);
        // A (from split to endpoint, but the part before split)
        for (int i = (int)lineA.size() - split - 1; i >= 0; --i)
            merged.push_back(lineA[i]);
        // B (from split point to its endpoint)
        for (int i = (int)lineB.size() - split - 1; i >= 0; --i)
            merged.push_back(lineB[i]);

        return merged;
    }

    /// Snap a point to the nearest voxel in the fastLevelSet field
    int SnapToField(int pt) {
        if (fastLevelSet.Find(pt)) return pt;
        int nb[26];
        mask.GetNeighbor26(pt, nb);
        for (int j = 0; j < 26; ++j) {
            if (nb[j] == -1) continue;
            if (fastLevelSet.Find(nb[j])) return nb[j];
        }
        return -1;
    }
};

} // namespace cfm

#endif // CENTERLINE_FAST_MARCHING_HPP
