#pragma once
/**
 * ============================================================
 *  Graph Cut 图割分割
 * ============================================================
 *
 * 【算法原理】
 *   将图像分割建模为能量最小化问题，通过图论中的最小割/最大流定理求解。
 *
 *   能量函数：
 *     E(L) = Σ_p D_p(L_p) + λ·Σ_{p,q∈N} V_{pq}(L_p, L_q)
 *
 *   其中：
 *     D_p(L_p): 数据项（unary / T-link），表示像素 p 被标记为 L_p 的代价
 *     V_{pq}:   平滑项（pairwise / N-link），惩罚相邻像素标签不一致
 *     λ:        平滑系数
 *     N:        相邻像素对集合
 *
 * 【图的构建】
 *   - 源点 S (source) = 前景标签
 *   - 汇点 T (sink)   = 背景标签
 *   - T-link: 每个像素到 S/T 的边，权重 = D_p(fg) / D_p(bg)
 *   - N-link: 相邻像素之间的边，权重 = V_{pq}
 *
 *   Unary 代价（交互式）：
 *     对前景/背景种子构建灰度直方图 P_fg(I), P_bg(I)
 *     D_p(fg) = -log(P_fg(I_p) + ε)
 *     D_p(bg) = -log(P_bg(I_p) + ε)
 *
 *   Pairwise 代价：
 *     V_{pq} = exp(-(I_p - I_q)² / (2·σ²))
 *     灰度越接近的像素，切割代价越高（倾向同类）
 *
 * 【最大流算法】
 *   这里实现了 Boykov-Kolmogorov 风格的 BFS 增广路径最大流算法。
 *   原始 BK 算法使用搜索树加速，这里用简化版 BFS 增广：
 *     1. BFS 从源到汇找增广路径
 *     2. 沿路径减去瓶颈流量
 *     3. 重复直到无增广路径
 *     4. 源侧可达节点 = 前景，其余 = 背景
 *
 * 【参考文献】
 *   Boykov Y, Kolmogorov V. "An experimental comparison of min-cut/max-flow
 *   algorithms for energy minimization in vision." IEEE TPAMI, 2004.
 *   Boykov Y, Jolly M-P. "Interactive graph cuts for optimal boundary &
 *   region segmentation." ICCV, 2001.
 * ============================================================
 */

#include "image2d.hpp"

namespace medseg {

// ============================================================
// 参数
// ============================================================
struct GraphCutParams {
    float lambda     = 10.0f;   // 平滑项权重
    float sigma      = 30.0f;   // 邻域权重中的 σ（灰度差异控制）
    int   histBins   = 256;     // 直方图 bin 数
    int   connectivity = 4;     // 邻域连通性 (4 或 8)
};

// ============================================================
// 图的边结构
// ============================================================
struct GraphEdge {
    int to;
    float capacity;
    float flow;
    int rev; // 反向边索引
};

// ============================================================
// 最大流求解器（BFS 增广路径）
// ============================================================
class MaxFlowSolver {
public:
    MaxFlowSolver() = default;
    explicit MaxFlowSolver(int numNodes) : adj(numNodes) {}

    void init(int numNodes) {
        adj.assign(numNodes, {});
    }

    /** 添加有向边（自动添加反向边） */
    void addEdge(int from, int to, float cap) {
        int fromIdx = static_cast<int>(adj[from].size());
        int toIdx   = static_cast<int>(adj[to].size());
        adj[from].push_back({to, cap, 0.0f, toIdx});
        adj[to].push_back({from, 0.0f, 0.0f, fromIdx}); // 反向边容量为0
    }

    /** 添加双向边（两个方向都有容量） */
    void addBidirectionalEdge(int from, int to, float cap) {
        int fromIdx = static_cast<int>(adj[from].size());
        int toIdx   = static_cast<int>(adj[to].size());
        adj[from].push_back({to, cap, 0.0f, toIdx});
        adj[to].push_back({from, cap, 0.0f, fromIdx});
    }

    /** BFS 增广路径最大流 (Edmonds-Karp) */
    float maxflow(int source, int sink) {
        int n = static_cast<int>(adj.size());
        float totalFlow = 0.0f;

        while (true) {
            // BFS 寻找增广路径
            std::vector<int> parent(n, -1);
            std::vector<int> parentEdge(n, -1);
            parent[source] = source;

            std::queue<int> q;
            q.push(source);

            while (!q.empty() && parent[sink] == -1) {
                int u = q.front(); q.pop();
                for (int i = 0; i < static_cast<int>(adj[u].size()); ++i) {
                    auto& e = adj[u][i];
                    if (parent[e.to] == -1 && e.capacity - e.flow > 1e-8f) {
                        parent[e.to] = u;
                        parentEdge[e.to] = i;
                        q.push(e.to);
                    }
                }
            }

            if (parent[sink] == -1) break; // 无增广路径

            // 找瓶颈
            float bottleneck = 1e30f;
            for (int v = sink; v != source; v = parent[v]) {
                auto& e = adj[parent[v]][parentEdge[v]];
                bottleneck = std::min(bottleneck, e.capacity - e.flow);
            }

            // 更新流量
            for (int v = sink; v != source; v = parent[v]) {
                auto& e = adj[parent[v]][parentEdge[v]];
                e.flow += bottleneck;
                adj[e.to][e.rev].flow -= bottleneck;
            }

            totalFlow += bottleneck;
        }

        return totalFlow;
    }

    /** 最大流后获取最小割（源侧可达节点） */
    std::vector<bool> minCut(int source) {
        int n = static_cast<int>(adj.size());
        std::vector<bool> visited(n, false);
        std::queue<int> q;
        q.push(source);
        visited[source] = true;

        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (auto& e : adj[u]) {
                if (!visited[e.to] && e.capacity - e.flow > 1e-8f) {
                    visited[e.to] = true;
                    q.push(e.to);
                }
            }
        }
        return visited; // true = 前景（源侧）
    }

private:
    std::vector<std::vector<GraphEdge>> adj;
};

// ============================================================
// Graph Cut 分割器
// ============================================================
class GraphCutSegmentor {
public:
    /**
     * 交互式图割分割
     * @param image      灰度图像 (float)
     * @param fgSeeds    前景种子掩码 (>0 为前景种子)
     * @param bgSeeds    背景种子掩码 (>0 为背景种子)
     * @param params     算法参数
     * @return           分割掩码 (255=前景, 0=背景)
     */
    Image2D<uint8_t> segment(
        const Image2D<float>& image,
        const Image2D<uint8_t>& fgSeeds,
        const Image2D<uint8_t>& bgSeeds,
        const GraphCutParams& params = {});

private:
    // 构建前景/背景灰度直方图
    void buildHistograms(
        const Image2D<float>& image,
        const Image2D<uint8_t>& fgSeeds,
        const Image2D<uint8_t>& bgSeeds,
        int bins, float minVal, float maxVal,
        std::vector<float>& fgHist,
        std::vector<float>& bgHist);

    // 计算 Unary 代价：-log(P(I) + ε)
    float unaryCost(float intensity, const std::vector<float>& hist,
                    float minVal, float maxVal, int bins);

    // 计算 Pairwise 代价：exp(-(I_p - I_q)² / (2σ²))
    float pairwiseCost(float Ip, float Iq, float sigma);
};

// ============================================================
//                      实现
// ============================================================

inline void GraphCutSegmentor::buildHistograms(
    const Image2D<float>& image,
    const Image2D<uint8_t>& fgSeeds,
    const Image2D<uint8_t>& bgSeeds,
    int bins, float minVal, float maxVal,
    std::vector<float>& fgHist,
    std::vector<float>& bgHist)
{
    fgHist.assign(bins, 0.0f);
    bgHist.assign(bins, 0.0f);
    float range = maxVal - minVal;
    if (range < 1e-8f) range = 1.0f;

    int fgCount = 0, bgCount = 0;

    for (int i = 0; i < image.size(); ++i) {
        float val = std::max(minVal, std::min(maxVal, image[i]));
        int bin = static_cast<int>((val - minVal) / range * (bins - 1));
        bin = std::max(0, std::min(bins - 1, bin));

        if (fgSeeds[i] > 0) { fgHist[bin] += 1.0f; fgCount++; }
        if (bgSeeds[i] > 0) { bgHist[bin] += 1.0f; bgCount++; }
    }

    // 归一化为概率分布
    if (fgCount > 0) for (auto& v : fgHist) v /= fgCount;
    if (bgCount > 0) for (auto& v : bgHist) v /= bgCount;
}

inline float GraphCutSegmentor::unaryCost(
    float intensity, const std::vector<float>& hist,
    float minVal, float maxVal, int bins)
{
    float range = maxVal - minVal;
    if (range < 1e-8f) range = 1.0f;

    int bin = static_cast<int>((std::max(minVal, std::min(maxVal, intensity)) - minVal) / range * (bins - 1));
    bin = std::max(0, std::min(bins - 1, bin));

    // -log(P(I) + ε)，ε=0.01 防止 log(0)
    return -std::log(hist[bin] + 0.01f);
}

inline float GraphCutSegmentor::pairwiseCost(float Ip, float Iq, float sigma) {
    float diff = Ip - Iq;
    return std::exp(-(diff * diff) / (2.0f * sigma * sigma));
}

inline Image2D<uint8_t> GraphCutSegmentor::segment(
    const Image2D<float>& image,
    const Image2D<uint8_t>& fgSeeds,
    const Image2D<uint8_t>& bgSeeds,
    const GraphCutParams& params)
{
    int W = image.width, H = image.height;
    int N = W * H;

    // 求图像值域
    float minVal = *std::min_element(image.data.begin(), image.data.end());
    float maxVal = *std::max_element(image.data.begin(), image.data.end());

    // 1. 构建前景/背景直方图
    std::vector<float> fgHist, bgHist;
    buildHistograms(image, fgSeeds, bgSeeds, params.histBins, minVal, maxVal, fgHist, bgHist);

    // 2. 构建图
    // 节点 0..N-1 = 像素, N = source, N+1 = sink
    int source = N, sink = N + 1;
    MaxFlowSolver solver(N + 2);

    const float INFINITY_CAP = 1e8f;

    // 2a. T-link（数据项）
    for (int i = 0; i < N; ++i) {
        float I = image[i];

        if (fgSeeds[i] > 0) {
            // 硬约束：前景种子，source→pixel 无穷大，pixel→sink 为0
            solver.addEdge(source, i, INFINITY_CAP);
            solver.addEdge(i, sink, 0.0f);
        } else if (bgSeeds[i] > 0) {
            // 硬约束：背景种子
            solver.addEdge(source, i, 0.0f);
            solver.addEdge(i, sink, INFINITY_CAP);
        } else {
            // 非种子：基于直方图的 unary 代价
            float costFG = unaryCost(I, bgHist, minVal, maxVal, params.histBins);
            float costBG = unaryCost(I, fgHist, minVal, maxVal, params.histBins);
            solver.addEdge(source, i, costFG);
            solver.addEdge(i, sink, costBG);
        }
    }

    // 2b. N-link（平滑项）
    // 4-邻域偏移
    int dx4[] = {1, 0};
    int dy4[] = {0, 1};
    // 8-邻域额外偏移
    int dx8[] = {1, 0, 1, -1};
    int dy8[] = {0, 1, 1,  1};

    int numDirs = (params.connectivity == 8) ? 4 : 2;
    int* dxArr = (params.connectivity == 8) ? dx8 : dx4;
    int* dyArr = (params.connectivity == 8) ? dy8 : dy4;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int idx = y * W + x;
            float Ip = image[idx];

            for (int d = 0; d < numDirs; ++d) {
                int nx = x + dxArr[d], ny = y + dyArr[d];
                if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                int nidx = ny * W + nx;
                float Iq = image[nidx];

                float weight = params.lambda * pairwiseCost(Ip, Iq, params.sigma);
                solver.addBidirectionalEdge(idx, nidx, weight);
            }
        }
    }

    // 3. 求解最大流
    solver.maxflow(source, sink);

    // 4. 最小割得到分割
    auto inSource = solver.minCut(source);

    Image2D<uint8_t> mask(W, H, 0);
    for (int i = 0; i < N; ++i) {
        mask[i] = inSource[i] ? 255 : 0;
    }
    return mask;
}

} // namespace medseg
