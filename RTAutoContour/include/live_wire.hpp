#pragma once
/**
 * ============================================================
 *  Live-Wire 智能剪刀 (Intelligent Scissors / Smart Contour)
 * ============================================================
 *
 * 提炼自 McsfAlgoAutoContour 工程中的 IntellSci / SmartContour。
 *
 * 【算法原理】
 *   Live-Wire 是一种经典的交互式图像分割工具：
 *   1. 用户在图像上点击若干"锚点"(anchor points)
 *   2. 算法自动找到相邻锚点之间的"最优"边界路径
 *   3. 最优路径 = 梯度代价图上的最短路径（Dijkstra）
 *
 *   代价图通过图像梯度构造：
 *     - 梯度越大（边缘越强）的方向代价越低
 *     - 路径倾向于沿着图像边缘走
 *
 * 【工作流程】
 *   ```
 *   computeCostMap(image)     → 计算梯度代价图
 *   setAnchor(x, y)          → 用户点击锚点
 *   computePaths(anchor)     → Dijkstra 传播（从锚点出发）
 *   getLiveWire(target)      → 回溯得到锚点到鼠标的最优路径
 *   commitSegment()          → 确认当前路径段
 *   closeContour()           → 闭合轮廓并填充内部
 *   ```
 *
 * 【核心算法：Dijkstra 传播】
 *   ```
 *   for each pixel p in PriorityQueue:
 *       for each 8-neighbor q of p:
 *           cost = totalCost[p] + edgeCost(p→q)
 *           if cost < totalCost[q]:
 *               totalCost[q] = cost
 *               parent[q] = p
 *               PQ.push(q, cost)
 *   ```
 *
 * 【边缘代价函数】
 *   edgeCost(p→q) = 1.0 / (1.0 + |gradient(q)|)
 *   梯度越大 → 代价越小 → 路径倾向于沿边缘走
 *
 * 【参考】
 *   原工程: McsfAlgoRTSmartContour.cpp — IntellSci, computePaths, liveWire
 *   论文: Mortensen & Barrett, "Intelligent Scissors for Image Composition",
 *         SIGGRAPH 1995.
 * ============================================================
 */

#include "rt_types.hpp"

namespace rtac {

// ============================================================
//  梯度计算
// ============================================================

/**
 * 计算图像的梯度幅值（Sobel 3x3）
 *
 * Gx = [[-1,0,1],[-2,0,2],[-1,0,1]] * I
 * Gy = [[-1,-2,-1],[0,0,0],[1,2,1]] * I
 * |G| = sqrt(Gx² + Gy²)
 */
inline Image2D<float> ComputeGradientMagnitude(const Image2D<float>& image) {
    int W = image.width, H = image.height;
    Image2D<float> grad(W, H, 0.f);

    for (int y = 1; y < H - 1; ++y)
    for (int x = 1; x < W - 1; ++x) {
        float gx = -image.atClamped(x-1, y-1) - 2*image.atClamped(x-1, y) - image.atClamped(x-1, y+1)
                   +image.atClamped(x+1, y-1) + 2*image.atClamped(x+1, y) + image.atClamped(x+1, y+1);
        float gy = -image.atClamped(x-1, y-1) - 2*image.atClamped(x, y-1) - image.atClamped(x+1, y-1)
                   +image.atClamped(x-1, y+1) + 2*image.atClamped(x, y+1) + image.atClamped(x+1, y+1);
        grad.at(x, y) = std::sqrt(gx * gx + gy * gy);
    }
    return grad;
}

// ============================================================
//  Live-Wire 引擎
// ============================================================
class LiveWire {
public:
    /// 初始化代价图
    void initialize(const Image2D<float>& image) {
        width_  = image.width;
        height_ = image.height;
        size_t N = static_cast<size_t>(width_) * height_;

        // 计算梯度
        gradient_ = ComputeGradientMagnitude(image);

        // 最大梯度（用于归一化）
        maxGrad_ = *std::max_element(gradient_.data.begin(), gradient_.data.end());
        if (maxGrad_ < 1e-6f) maxGrad_ = 1.0f;

        // 初始化状态
        totalCost_.assign(N, std::numeric_limits<float>::max());
        parent_.assign(N, -1);
        visited_.assign(N, false);

        // 清空已确认路径
        confirmedPath_.clear();
        anchors_.clear();
    }

    /// 设置锚点并从该点做 Dijkstra 传播
    void setAnchor(int x, int y) {
        if (x < 0 || x >= width_ || y < 0 || y >= height_) return;

        anchors_.push_back({x, y});
        computePaths(x, y);
    }

    /**
     * 获取 Live-Wire 路径：从最近锚点到目标点的最短路径
     *
     * 通过回溯 parent 链实现
     */
    std::vector<Point2i> getLiveWire(int targetX, int targetY) const {
        std::vector<Point2i> path;
        if (targetX < 0 || targetX >= width_ || targetY < 0 || targetY >= height_)
            return path;

        int idx = targetY * width_ + targetX;
        if (parent_[idx] < 0 && (targetX != anchors_.back().x || targetY != anchors_.back().y))
            return path;

        // 回溯
        while (idx >= 0) {
            int x = idx % width_, y = idx / width_;
            path.push_back({x, y});
            if (parent_[idx] == idx) break; // 到达锚点
            idx = parent_[idx];
        }

        std::reverse(path.begin(), path.end());
        return path;
    }

    /// 确认当前 live-wire 段（将路径加入已确认列表）
    void commitSegment(int targetX, int targetY) {
        auto segment = getLiveWire(targetX, targetY);
        for (const auto& p : segment) {
            confirmedPath_.push_back(p);
        }
        // 以目标点为新锚点
        setAnchor(targetX, targetY);
    }

    /// 获取已确认的完整路径
    const std::vector<Point2i>& getConfirmedPath() const { return confirmedPath_; }

    /// 获取锚点列表
    const std::vector<Point2i>& getAnchors() const { return anchors_; }

    /**
     * 闭合轮廓：连接最后一点与第一个锚点，生成填充掩膜
     *
     * 使用 floodfill 填充闭合轮廓内部
     */
    Image2D<uint8_t> closeAndFill(uint8_t maskID = 255) {
        if (confirmedPath_.empty() || anchors_.size() < 2) {
            return Image2D<uint8_t>(width_, height_, 0);
        }

        // 获取最后一段到第一个锚点的路径
        auto closing = getLiveWire(anchors_.front().x, anchors_.front().y);
        std::vector<Point2i> fullPath = confirmedPath_;
        for (const auto& p : closing) {
            fullPath.push_back(p);
        }

        // 光栅化路径为边缘
        Image2D<uint8_t> edgeMask(width_, height_, 0);
        for (const auto& p : fullPath) {
            if (p.x >= 0 && p.x < width_ && p.y >= 0 && p.y < height_)
                edgeMask.at(p.x, p.y) = 1;
        }

        // Floodfill 外部
        std::vector<bool> ext(edgeMask.size(), false);
        std::queue<int> queue;

        auto tryPush = [&](int x, int y) {
            if (x < 0 || x >= width_ || y < 0 || y >= height_) return;
            int idx = y * width_ + x;
            if (!ext[idx] && edgeMask[idx] == 0) {
                ext[idx] = true;
                queue.push(idx);
            }
        };

        for (int x = 0; x < width_; ++x) { tryPush(x, 0); tryPush(x, height_ - 1); }
        for (int y = 0; y < height_; ++y) { tryPush(0, y); tryPush(width_ - 1, y); }

        while (!queue.empty()) {
            int idx = queue.front(); queue.pop();
            int x = idx % width_, y = idx / width_;
            tryPush(x - 1, y); tryPush(x + 1, y);
            tryPush(x, y - 1); tryPush(x, y + 1);
        }

        // 非外部 = 内部
        Image2D<uint8_t> result(width_, height_, 0);
        for (size_t i = 0; i < result.size(); ++i) {
            if (!ext[i]) result[i] = maskID;
        }
        return result;
    }

    /// 重置
    void reset() {
        confirmedPath_.clear();
        anchors_.clear();
        totalCost_.assign(totalCost_.size(), std::numeric_limits<float>::max());
        parent_.assign(parent_.size(), -1);
        visited_.assign(visited_.size(), false);
    }

private:
    /**
     * Dijkstra 传播：从 (sx, sy) 出发，计算到所有像素的最短路径
     *
     * 代价函数: edgeCost(p→q) = 1.0 / (1.0 + gradient(q)/maxGrad)
     * 即梯度越大的位置代价越小，路径倾向于沿边缘走
     */
    void computePaths(int sx, int sy) {
        size_t N = static_cast<size_t>(width_) * height_;
        totalCost_.assign(N, std::numeric_limits<float>::max());
        parent_.assign(N, -1);
        visited_.assign(N, false);

        // 8邻域偏移
        static const int dx8[] = { 1, 1, 0,-1,-1,-1, 0, 1};
        static const int dy8[] = { 0, 1, 1, 1, 0,-1,-1,-1};
        // 对角线代价乘以 sqrt(2)
        static const float dist8[] = {1.f, 1.414f, 1.f, 1.414f, 1.f, 1.414f, 1.f, 1.414f};

        // 优先队列 (cost, index)
        using PQItem = std::pair<float, int>;
        std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> pq;

        int startIdx = sy * width_ + sx;
        totalCost_[startIdx] = 0.f;
        parent_[startIdx] = startIdx;
        pq.push({0.f, startIdx});

        while (!pq.empty()) {
            auto [cost, idx] = pq.top(); pq.pop();

            if (visited_[idx]) continue;
            visited_[idx] = true;

            int cx = idx % width_, cy = idx / width_;

            for (int d = 0; d < 8; ++d) {
                int nx = cx + dx8[d], ny = cy + dy8[d];
                if (nx < 0 || nx >= width_ || ny < 0 || ny >= height_) continue;

                int nIdx = ny * width_ + nx;
                if (visited_[nIdx]) continue;

                // 代价 = 1 / (1 + 归一化梯度)，梯度大则代价小
                float gradNorm = gradient_[nIdx] / maxGrad_;
                float edgeCost = dist8[d] / (1.0f + gradNorm * 10.0f);

                float newCost = cost + edgeCost;
                if (newCost < totalCost_[nIdx]) {
                    totalCost_[nIdx] = newCost;
                    parent_[nIdx] = idx;
                    pq.push({newCost, nIdx});
                }
            }
        }
    }

    int width_ = 0, height_ = 0;
    float maxGrad_ = 1.0f;
    Image2D<float> gradient_;
    std::vector<float> totalCost_;
    std::vector<int> parent_;
    std::vector<bool> visited_;
    std::vector<Point2i> confirmedPath_;
    std::vector<Point2i> anchors_;
};

} // namespace rtac
