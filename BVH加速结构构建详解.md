# BVH（层次包围盒）加速结构构建详解

> 本文档系统讲解 BVH（Bounding Volume Hierarchy）树的构建原理、策略与实现，
> 涵盖从基础概念到 SAH（表面积启发式）高级构建算法，并给出完整 C++ 实现代码。
> 适用于光线追踪、切片加速、碰撞检测等场景。

---

## 目录

- [1. BVH 基本概念](#1-bvh-基本概念)
- [2. AABB 包围盒](#2-aabb-包围盒)
- [3. BVH 构建策略总览](#3-bvh-构建策略总览)
- [4. 中位数分割构建法](#4-中位数分割构建法)
- [5. SAH（表面积启发式）构建法](#5-sah表面积启发式构建法)
- [6. SBVH（空间分割BVH）](#6-sbvh空间分割bvh)
- [7. BVH 遍历与查询](#7-bvh-遍历与查询)
- [8. GPU 上的 BVH](#8-gpu-上的-bvh)
- [9. 与其他加速结构的对比](#9-与其他加速结构的对比)
- [10. 完整实现代码](#10-完整实现代码)
- [11. 性能分析与调优](#11-性能分析与调优)

---

## 1. BVH 基本概念

### 1.1 什么是 BVH

BVH（Bounding Volume Hierarchy）是一种**树形空间加速结构**。核心思想：

1. 为每个图元（三角形/体素）计算一个**轴对齐包围盒（AABB）**
2. 将相邻图元的包围盒**合并**成更大的包围盒，形成层次结构
3. 查询时，先测试外层包围盒，**快速剔除**不可能相交的大区域

```
原始三角形分布:                    BVH 树结构:

   T1  T2        T3 T4              Root (AABB: 全部)
    \  /          |  /               ├── Left (AABB: T1,T2,T3,T4)
     \/           | /                │   ├── Left (AABB: T1,T2)
     合并区域     |/                 │   │   ├── Leaf: T1
                  /                  │   │   └── Leaf: T2
                 /                   │   └── Right (AABB: T3,T4)
               合并                      │   ├── Leaf: T3
                                         │   └── Leaf: T4
                                         └── (空)
```

### 1.2 BVH vs 其他加速结构

| 特性 | BVH | KD-Tree | 八叉树 | 均匀网格 |
|------|-----|---------|--------|----------|
| **空间划分方式** | 对象划分 | 空间划分 | 空间划分 | 空间划分 |
| **图元重复存储** | 无（每个图元只在一个叶子中） | 有（图元可能跨多个区域） | 有 | 有 |
| **构建速度** | 快~中 | 慢（SAH） | 中 | 快 |
| **查询速度** | 快 | 最快 | 中 | 场景均匀时快 |
| **动态更新** | 支持增量更新 | 需重建 | 需重建 | 需重建 |
| **内存占用** | 低 | 中 | 中 | 低 |
| **适用场景** | 通用、动态场景 | 静态高质量渲染 | 体素、空间查询 | 均匀分布数据 |

> **关键区别：** BVH 是**对象划分（Object Partitioning）**——把图元分组，不切分空间；
> KD-Tree 是**空间划分（Space Partitioning）**——切分空间，图元可能出现在多个叶子中。

### 1.3 BVH 的关键属性

```
属性1: 每个图元恰好出现在一个叶子节点中（无重复）
属性2: 内部节点的 AABB 包含其所有子节点的 AABB
属性3: 兄弟节点的 AABB 可以重叠（不像 KD-Tree 那样严格分割空间）
属性4: 叶子节点包含少量图元（通常 1~16 个）
```

---

## 2. AABB 包围盒

### 2.1 AABB 定义

AABB（Axis-Aligned Bounding Box）是沿坐标轴对齐的最小长方体：

```cpp
struct AABB {
    glm::vec3 min;  // 最小角
    glm::vec3 max;  // 最大角

    // 合并两个 AABB
    AABB merge(const AABB& other) const {
        return {
            glm::min(min, other.min),
            glm::max(max, other.max)
        };
    }

    // 表面积（SAH 用）
    float surfaceArea() const {
        glm::vec3 d = max - min;
        return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
    }

    // 体积
    float volume() const {
        glm::vec3 d = max - min;
        return d.x * d.y * d.z;
    }

    // 射线相交测试（Slab 方法）
    bool intersectRay(const glm::vec3& origin, const glm::vec3& dir,
                      float& tEnter, float& tExit) const {
        glm::vec3 invDir = 1.0f / dir;
        glm::vec3 t1 = (min - origin) * invDir;
        glm::vec3 t2 = (max - origin) * invDir;
        glm::vec3 tMin = glm::min(t1, t2);
        glm::vec3 tMax = glm::max(t1, t2);
        tEnter = glm::max(glm::max(tMin.x, tMin.y), tMin.z);
        tExit  = glm::min(glm::min(tMax.x, tMax.y), tMax.z);
        return tExit >= tEnter && tExit >= 0.0f;
    }

    // Z 平面相交测试（切片用）
    bool intersectsZ(float z0) const {
        return z0 >= min.z && z0 <= max.z;
    }
};
```

### 2.2 为三角形计算 AABB

```cpp
AABB computeTriangleAABB(const Triangle& tri) {
    return {
        glm::min(glm::min(tri.v[0], tri.v[1]), tri.v[2]),
        glm::max(glm::max(tri.v[0], tri.v[1]), tri.v[2])
    };
}
```

### 2.3 射线-AABB 相交的数学原理（Slab 方法）

对每个轴，AABB 形成一个"平板"（slab）：

$$
\text{对 X 轴: } t_{1x} = \frac{x_{\min} - o_x}{d_x}, \quad t_{2x} = \frac{x_{\max} - o_x}{d_x}
$$

射线进入所有三个平板的**交集**时才与 AABB 相交：

$$
t_{\text{enter}} = \max(t_{\min,x},\ t_{\min,y},\ t_{\min,z})
$$
$$
t_{\text{exit}} = \min(t_{\max,x},\ t_{\max,y},\ t_{\max,z})
$$

相交条件：$t_{\text{exit}} \geq t_{\text{enter}}$ 且 $t_{\text{exit}} \geq 0$。

```
射线方向 →
         t1x        t2x
          |          |
  ────────┤██████████├────────── X 轴平板
          │          │
     t_enter    t_exit
          |          |
          ▼          ▼
  ═════════════════════════════ 射线
```

---

## 3. BVH 构建策略总览

### 3.1 构建的核心问题

BVH 构建本质上是一个**递归二分问题**：

```
输入: 一组图元（三角形）及其 AABB
输出: BVH 树

递归过程:
  1. 计算所有图元的总 AABB → 当前节点
  2. 选择一个分割轴（X/Y/Z）和分割位置
  3. 将图元分为左右两组
  4. 对左右两组递归构建子树
  5. 当图元数量 ≤ 阈值时，创建叶子节点
```

### 3.2 构建策略对比

| 策略 | 分割轴选择 | 分割位置 | 构建复杂度 | 查询质量 | 适用场景 |
|------|-----------|---------|-----------|---------|---------|
| **中位数分割** | 最大跨度轴 | 质心中位数 | O(n log n) | 中等 | 快速构建、简单场景 |
| **等量分割** | 轮换轴 | 中间索引 | O(n log n) | 中等 | 简单实现 |
| **SAH（表面积启发式）** | SAH 最优 | SAH 最优 | O(n log²n) | **最优** | 高质量渲染 |
| **Morton 排序（LBVH）** | Morton 码隐式 | Morton 码中点 | O(n log n) | 良好 | GPU 并行构建 |
| **HLBVH + SAH** | Morton 码 + SAH | 混合 | O(n log n) | 优秀 | GPU 构建首选 |

### 3.3 构建质量指标

BVH 的"质量"主要通过**查询时遍历的节点数**来衡量。一个好的 BVH 应该：

1. **最小化叶子 AABB 的表面积**（减少射线命中概率）
2. **最小化兄弟节点的重叠**（减少同时遍历两个分支的概率）
3. **保持树的平衡性**（避免退化成链表）

---

## 4. 中位数分割构建法

### 4.1 算法思路

最简单有效的 BVH 构建方法：

1. 选择**包围盒跨度最大的轴**作为分割轴
2. 沿该轴对图元**按质心排序**
3. 取**中位数**作为分割点，保证左右子树平衡

```
分割前（沿 X 轴）:
  T1  T2  T3  T4  T5  T6  T7  T8
  ●   ●   ●   ●   ●   ●   ●   ●
  ───────────┼───────────
       左半        右半
  (T1~T4)      (T5~T8)

选择最大跨度轴的原因:
  - 沿最长维度分割，能最有效地减小子节点 AABB 体积
  - 避免在"薄"的维度上分割导致子节点高度重叠
```

### 4.2 伪代码

```
function buildBVH_median(triangles, indices, start, end):
    node = new BVHNode()
    node.bbox = computeBounds(triangles, indices, start, end)

    count = end - start
    if count <= LEAF_SIZE:
        node.leaf = true
        node.indices = indices[start..end]
        return node

    // 选择最大跨度轴
    centroidBBox = computeCentroidBounds(triangles, indices, start, end)
    axis = longestAxis(centroidBBox)

    // 沿该轴排序
    sort(indices[start..end], by: triangles[i].centroid[axis])

    // 中位数分割
    mid = start + count / 2

    node.left  = buildBVH_median(triangles, indices, start, mid)
    node.right = buildBVH_median(triangles, indices, mid, end)
    return node
```

### 4.3 为什么选择最大跨度轴

```
情况A: 沿最长轴分割（正确）          情况B: 沿最短轴分割（错误）

  Y                                    Y
  ↑     ┌────────┐                    ↑  ┌──┬──┐
  │     │   左   │                    │  │左│右│  ← 高度重叠！
  │     ├────────┤                    │  └──┴──┘
  │     │   右   │                    │
  └──────────────→ X                  └──────────────→ X

  子节点 AABB 几乎不重叠              子节点 AABB 大面积重叠
  查询效率高                          查询时可能同时进入两个分支
```

---

## 5. SAH（表面积启发式）构建法

### 5.1 SAH 核心思想

SAH（Surface Area Heuristic）基于**概率论**估计查询代价，选择使期望代价最小的分割方案。

**核心假设：** 射线在空间中均匀分布且方向随机。一个子节点的 AABB 被射线命中的概率与其**表面积**成正比。

### 5.2 SAH 代价公式

对于一次分割，期望查询代价为：

$$
C_{\text{split}} = C_{\text{traversal}} + \frac{S_L}{S_N} \cdot N_L \cdot C_{\text{intersect}} + \frac{S_R}{S_N} \cdot N_R \cdot C_{\text{intersect}}
$$

其中：
- $C_{\text{traversal}}$ — 遍历内部节点的代价（通常设为 1）
- $C_{\text{intersect}}$ — 射线-三角形相交测试的代价（通常设为 1.2~2）
- $S_N$ — 当前节点 AABB 的表面积
- $S_L, S_R$ — 左右子节点 AABB 的表面积
- $N_L, N_R$ — 左右子节点的图元数量
- $\frac{S_L}{S_N}$ — 射线命中左子节点的概率

**目标：** 找到使 $C_{\text{split}}$ 最小的分割方案。

### 5.3 不分割的代价（创建叶子节点）

$$
C_{\text{leaf}} = N \cdot C_{\text{intersect}}
$$

当 $C_{\text{leaf}} < C_{\text{split}}$ 时，创建叶子节点更优。

### 5.4 SAH 构建流程

```
function buildBVH_SAH(triangles, indices, start, end):
    node = new BVHNode()
    node.bbox = computeBounds(triangles, indices, start, end)

    count = end - start

    // 1. 计算不分割（叶子）的代价
    costLeaf = count * C_INTERSECT

    // 2. 对每个轴尝试分割，找最优
    bestCost = INFINITY
    bestAxis = -1
    bestSplit = -1

    for axis in {X, Y, Z}:
        // 沿轴排序
        sort(indices[start..end], by: centroid[axis])

        // 从左到右扫描，计算前缀 AABB
        prefixBBox[0] = triangles[indices[start]].bbox
        for i in 1..count-1:
            prefixBBox[i] = merge(prefixBBox[i-1], triangles[indices[start+i]].bbox)

        // 从右到左扫描，计算后缀 AABB
        suffixBBox[count-1] = triangles[indices[start+count-1]].bbox
        for i in count-2..0:
            suffixBBox[i] = merge(suffixBBox[i+1], triangles[indices[start+i]].bbox)

        // 尝试每个分割位置
        for split in 1..count-1:
            leftBBox  = prefixBBox[split-1]
            rightBBox = suffixBBox[split]
            leftCount = split
            rightCount = count - split

            cost = C_TRAVERSAL
                 + (leftBBox.surfaceArea()  / node.bbox.surfaceArea()) * leftCount  * C_INTERSECT
                 + (rightBBox.surfaceArea() / node.bbox.surfaceArea()) * rightCount * C_INTERSECT

            if cost < bestCost:
                bestCost = cost
                bestAxis = axis
                bestSplit = split

    // 3. 比较分割 vs 叶子
    if bestCost > costLeaf or count <= 1:
        node.leaf = true
        node.indices = indices[start..end]
        return node

    // 4. 执行最优分割
    sort(indices[start..end], by: centroid[bestAxis])
    mid = start + bestSplit
    node.left  = buildBVH_SAH(triangles, indices, start, mid)
    node.right = buildBVH_SAH(triangles, indices, mid, end)
    return node
```

### 5.5 SAH 分桶优化（Binning）

对大量图元，尝试每个分割位置代价太高（O(n²)）。使用**分桶（Binning）**将搜索降到 O(n)：

```
将图元沿轴分成 B 个桶（通常 B=12 或 16）

  桶:  |  B1  |  B2  |  B3  |  B4  |  B5  |  B6  |
       |------|------|------|------|------|------|
  图元: ●●●   ●●    ●●●●  ●     ●●●   ●●

  只在桶边界处尝试分割（B-1 个候选位置）
  每个桶维护: 图元数 + 合并 AABB

  复杂度: O(n) 扫描 + O(B) 尝试分割 = O(n)
```

```cpp
// SAH 分桶实现
struct Bin {
    AABB bbox;
    int count = 0;
};

BVHSplit findBestSplitSAH(const std::vector<Triangle>& tris,
                          const std::vector<int>& indices,
                          int start, int end,
                          const AABB& nodeBBox)
{
    constexpr int NUM_BINS = 12;
    constexpr float C_TRAVERSAL = 1.0f;
    constexpr float C_INTERSECT = 1.2f;

    float bestCost = std::numeric_limits<float>::max();
    int bestAxis = -1, bestBin = 0;

    for (int axis = 0; axis < 3; axis++) {
        // 计算质心范围
        float centroidMin = FLT_MAX, centroidMax = -FLT_MAX;
        for (int i = start; i < end; i++) {
            float c = tris[indices[i]].centroid[axis];
            centroidMin = std::min(centroidMin, c);
            centroidMax = std::max(centroidMax, c);
        }
        if (centroidMax <= centroidMin) continue;  // 该轴无跨度

        // 初始化桶
        Bin bins[NUM_BINS];
        float scale = NUM_BINS / (centroidMax - centroidMin);

        for (int i = start; i < end; i++) {
            float c = tris[indices[i]].centroid[axis];
            int b = std::min(NUM_BINS - 1, (int)((c - centroidMin) * scale));
            bins[b].bbox = bins[b].bbox.merge(tris[indices[i]].bbox);
            bins[b].count++;
        }

        // 前缀/后缀扫描
        AABB prefixBBox[NUM_BINS];
        int prefixCount[NUM_BINS];
        prefixBBox[0] = bins[0].bbox;
        prefixCount[0] = bins[0].count;
        for (int i = 1; i < NUM_BINS; i++) {
            prefixBBox[i] = prefixBBox[i-1].merge(bins[i].bbox);
            prefixCount[i] = prefixCount[i-1] + bins[i].count;
        }

        AABB suffixBBox[NUM_BINS];
        int suffixCount[NUM_BINS];
        suffixBBox[NUM_BINS-1] = bins[NUM_BINS-1].bbox;
        suffixCount[NUM_BINS-1] = bins[NUM_BINS-1].count;
        for (int i = NUM_BINS-2; i >= 0; i--) {
            suffixBBox[i] = suffixBBox[i+1].merge(bins[i].bbox);
            suffixCount[i] = suffixCount[i+1] + bins[i].count;
        }

        // 尝试每个桶边界分割
        float invNodeSA = 1.0f / nodeBBox.surfaceArea();
        for (int b = 1; b < NUM_BINS; b++) {
            int leftCount  = prefixCount[b-1];
            int rightCount = suffixCount[b];
            if (leftCount == 0 || rightCount == 0) continue;

            float cost = C_TRAVERSAL
                + prefixBBox[b-1].surfaceArea() * invNodeSA * leftCount  * C_INTERSECT
                + suffixBBox[b].surfaceArea()   * invNodeSA * rightCount * C_INTERSECT;

            if (cost < bestCost) {
                bestCost = cost;
                bestAxis = axis;
                bestBin = b;
            }
        }
    }

    return { bestAxis, bestBin, bestCost };
}
```

### 5.6 SAH 的直觉理解

```
场景: 一群密集的小三角形 + 一个孤立的大三角形

中位数分割:                        SAH 分割:
┌─────────────────────┐           ┌──────────────┐┌──┐
│ ●●●●●●●●●●●●●  ●   │           │ ●●●●●●●●●●●●●││ ●│
│ (密集区)      (大)  │           │  (密集区)    ││大│
└─────────────────────┘           └──────────────┘└──┘
     左7个  右7个                      左13个      右1个

问题: 大三角形的 AABB 很大，               SAH 正确地将大三角形
导致右子树 AABB 覆盖大片区域，             单独分到右子树，
射线频繁误命中右子树                      减少了不必要的遍历
```

---

## 6. SBVH（空间分割BVH）

### 6.1 动机

标准 BVH 是对象划分，当图元的 AABB 跨越大范围时，子节点 AABB 会严重重叠：

```
问题场景: 一个大三角形横跨整个场景

  ┌──────────────────────────────┐
  │         大三角形 AABB          │
  │  ┌──┐  ┌──┐  ┌──┐  ┌──┐    │
  │  │T1│  │T2│  │T3│  │T4│    │
  │  └──┘  └──┘  └──┘  └──┘    │
  └──────────────────────────────┘

  对象划分: 大三角形只能分到左或右，导致另一侧 AABB 仍然包含它
  → 左右子树 AABB 高度重叠
```

### 6.2 SBVH 方案

SBVH 结合了对象划分和空间划分：

1. 先用 SAH 计算对象划分的代价
2. 再用 SAH 计算空间划分（沿分割面切割图元 AABB）的代价
3. 选择代价更小的方案

```
空间划分: 沿分割面切割大三角形

  分割面
    │
  ┌─┼──────────┐
  │ │大三角形左半│  → 左子树
  └─┼──────────┘
    │
  ┌─┼──────────┐
  │ │大三角形右半│  → 右子树
  └─┼──────────┘

  代价: 图元被引用两次（内存增加）
  收益: 子节点 AABB 不重叠（查询更快）
```

> SBVH 在实际光线追踪中比纯 SAH-BVH 快约 20%~40%，但构建更复杂、内存更大。

---

## 7. BVH 遍历与查询

### 7.1 射线查询（最近交点）

```cpp
// 返回最近交点的三角形索引和参数 t
bool rayIntersectBVH(const BVHNode* root,
                     const glm::vec3& origin,
                     const glm::vec3& dir,
                     float& tHit, int& triIndex)
{
    tHit = std::numeric_limits<float>::max();
    bool found = false;

    // 使用栈进行迭代遍历（避免递归开销）
    std::stack<const BVHNode*> stack;
    stack.push(root);

    while (!stack.empty()) {
        const BVHNode* node = stack.top();
        stack.pop();

        // AABB 相交测试
        float tEnter, tExit;
        if (!node->bbox.intersectRay(origin, dir, tEnter, tExit))
            continue;

        // 早终止：如果 AABB 的最近交点已经比当前最近交点远
        if (tEnter > tHit)
            continue;

        if (node->isLeaf) {
            // 叶子节点：测试每个三角形
            for (int idx : node->indices) {
                float t;
                if (rayTriangleIntersect(origin, dir, triangles[idx], t) && t < tHit) {
                    tHit = t;
                    triIndex = idx;
                    found = true;
                }
            }
        } else {
            // 内部节点：压入子节点
            // 优化：按射线方向决定压入顺序（先近后远）
            if (dir[node->splitAxis] < 0) {
                stack.push(node->left);
                stack.push(node->right);  // right 在栈顶，先弹出
            } else {
                stack.push(node->right);
                stack.push(node->left);   // left 在栈顶，先弹出
            }
        }
    }
    return found;
}
```

### 7.2 遍历顺序优化

```
射线方向 →
         Root
        /    \
     Near    Far
     / \     / \
    ...

先访问 Near 子树（射线先到达的 AABB）
如果 Near 中找到交点且 t < Far 的 tEnter，可以跳过 Far
→ 这就是"早终止"优化的原理
```

### 7.3 Z 平面查询（切片专用）

```cpp
// 切片场景：查询与 Z=z0 平面相交的所有三角形
std::vector<int> queryZPlane(const BVHNode* root, float z0)
{
    std::vector<int> result;
    std::stack<const BVHNode*> stack;
    stack.push(root);

    while (!stack.empty()) {
        const BVHNode* node = stack.top();
        stack.pop();

        // Z 范围快速剔除
        if (!node->bbox.intersectsZ(z0))
            continue;

        if (node->isLeaf) {
            for (int idx : node->indices) {
                // 精确检查三角形是否与 Z=z0 相交
                const auto& tri = triangles[idx];
                float zmin = std::min({tri.v[0].z, tri.v[1].z, tri.v[2].z});
                float zmax = std::max({tri.v[0].z, tri.v[1].z, tri.v[2].z});
                if (z0 >= zmin && z0 <= zmax) {
                    result.push_back(idx);
                }
            }
        } else {
            stack.push(node->left);
            stack.push(node->right);
        }
    }
    return result;
}
```

### 7.4 遍历复杂度分析

| 场景 | 暴力遍历 | BVH 遍历 |
|------|---------|---------|
| N 个三角形 | O(N) | O(log N) ~ O(log²N) |
| 100 万三角形 | ~100 万次测试 | ~20~40 次测试 |
| 查询 1000 层切片 | 10 亿次测试 | ~2~4 万次测试 |

---

## 8. GPU 上的 BVH

### 8.1 线性化 BVH（扁平化存储）

GPU 不适合指针遍历，需要将 BVH 扁平化为数组：

```cpp
// 线性化 BVH 节点（GPU 友好）
struct alignas(32) LinearBVHNode {
    glm::vec4 bboxMin;      // xyz=min, w=0
    glm::vec4 bboxMax;      // xyz=max, w=0
    int leftChild;          // 左子节点索引（-1 表示无）
    int rightChild;         // 右子节点索引（-1 表示无）
    int firstIndex;         // 叶子: 第一个图元索引
    int indexCount;         // 叶子: 图元数量（0=内部节点）
    int splitAxis;          // 分割轴（0=X, 1=Y, 2=Z）
    int padding[2];         // 对齐填充
};

// 深度优先扁平化
int flattenBVH(const BVHNode* node, std::vector<LinearBVHNode>& linear,
               std::vector<int>& indices, int& currentIndex)
{
    int myIndex = currentIndex++;
    linear[myIndex].bboxMin = glm::vec4(node->bbox.min, 0.0f);
    linear[myIndex].bboxMax = glm::vec4(node->bbox.max, 0.0f);

    if (node->isLeaf) {
        linear[myIndex].firstIndex = (int)indices.size();
        linear[myIndex].indexCount = (int)node->indices.size();
        for (int idx : node->indices)
            indices.push_back(idx);
        linear[myIndex].leftChild = -1;
        linear[myIndex].rightChild = -1;
    } else {
        // 先递归左子树（深度优先保证内存局部性）
        linear[myIndex].leftChild = flattenBVH(node->left, linear, indices, currentIndex);
        linear[myIndex].rightChild = flattenBVH(node->right, linear, indices, currentIndex);
        linear[myIndex].indexCount = 0;
        linear[myIndex].splitAxis = node->splitAxis;
    }
    return myIndex;
}
```

### 8.2 GPU 遍历（GLSL）

```glsl
// BVH 纹理: 两个 SSBO
// nodesBuffer: LinearBVHNode[]
// indicesBuffer: int[] (三角形索引)

layout(std430, binding = 0) readonly buffer BVHNodes {
    vec4 nodes[];  // 每个节点用 4 个 vec4 表示
};

layout(std430, binding = 1) readonly buffer BVHIndices {
    int triIndices[];
};

bool intersectRayAABB(vec3 origin, vec3 invDir, vec3 bmin, vec3 bmax,
                      out float tEnter, out float tExit)
{
    vec3 t1 = (bmin - origin) * invDir;
    vec3 t2 = (bmax - origin) * invDir;
    vec3 tmin = min(t1, t2);
    vec3 tmax = max(t1, t2);
    tEnter = max(max(tmin.x, tmin.y), tmin.z);
    tExit  = min(min(tmax.x, tmax.y), tmax.z);
    return tExit >= tEnter && tExit >= 0.0;
}

// GPU 上的 BVH 遍历（使用栈模拟递归）
bool rayIntersectBVH_GPU(vec3 origin, vec3 dir, out float tHit)
{
    tHit = 1e30;
    bool found = false;
    vec3 invDir = 1.0 / dir;

    // 小型栈（GPU 上栈深度有限）
    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0;  // root index

    while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];

        // 读取节点（每个节点 4 个 vec4 = 64 字节）
        int base = nodeIdx * 4;
        vec3 bmin = nodes[base].xyz;
        vec3 bmax = nodes[base + 1].xyz;
        int leftChild   = floatBitsToInt(nodes[base + 2].x);
        int rightChild  = floatBitsToInt(nodes[base + 2].y);
        int firstIndex  = floatBitsToInt(nodes[base + 2].z);
        int indexCount  = floatBitsToInt(nodes[base + 2].w);
        int splitAxis   = floatBitsToInt(nodes[base + 3].x);

        float tEnter, tExit;
        if (!intersectRayAABB(origin, invDir, bmin, bmax, tEnter, tExit))
            continue;
        if (tEnter > tHit)
            continue;

        if (indexCount > 0) {
            // 叶子节点
            for (int i = 0; i < indexCount; i++) {
                int triIdx = triIndices[firstIndex + i];
                float t;
                if (rayTriangleIntersect(origin, dir, triIdx, t) && t < tHit) {
                    tHit = t;
                    found = true;
                }
            }
        } else {
            // 内部节点：按方向决定压入顺序
            if (dir[splitAxis] >= 0.0) {
                if (rightChild >= 0) stack[stackPtr++] = rightChild;
                if (leftChild  >= 0) stack[stackPtr++] = leftChild;
            } else {
                if (leftChild  >= 0) stack[stackPtr++] = leftChild;
                if (rightChild >= 0) stack[stackPtr++] = rightChild;
            }
        }
    }
    return found;
}
```

### 8.3 LBVH（基于 Morton 码的并行构建）

LBVH（Linear BVH）利用 **Morton 码（Z-order）** 实现 GPU 并行构建：

```
步骤1: 计算每个图元质心的 Morton 码
步骤2: 按 Morton 码排序（GPU 并行基数排序）
步骤3: 相邻 Morton 码有公共前缀的图元归为一组
步骤4: 自底向上并行构建树

Morton 码 (Z-order curve):
  将 3D 坐标交错编码为 1D 值
  空间上相邻的点 → Morton 码相近

  Y
  3──15  7──19
  │      │
  2──14  6──18
  │      │
  1──13  5──17
  │      │
  0──12  4──16  ──→ X

  Morton 码保持了空间局部性
```

```cpp
// 计算 30 位 Morton 码
uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z)
{
    x = expandBits(x);
    y = expandBits(y << 1);
    z = expandBits(z << 2);
    return x | y | z;
}

uint32_t expandBits(uint32_t v)
{
    v = (v | v << 16) & 0x030000FF;
    v = (v | v << 8)  & 0x0300F00F;
    v = (v | v << 4)  & 0x030C30C3;
    v = (v | v << 2)  & 0x09249249;
    return v;
}
```

---

## 9. 与其他加速结构的对比

### 9.1 BVH vs KD-Tree

```
BVH (对象划分):                    KD-Tree (空间划分):

  ┌────────┬────────┐              ┌────────┬────────┐
  │  T1,T2 │  T3,T4 │              │  区域A  │  区域B  │
  │  ●●    │    ●●  │              │  ●●    │    ●●  │
  │        │        │              │        │        │
  └────────┴────────┘              └────────┴────────┘
  左子树 AABB 包含 T1,T2           左区域 = 空间左半
  右子树 AABB 包含 T3,T4           右区域 = 空间右半
  AABB 可以重叠                     空间严格不重叠
  T1 只在左叶子中                   T1 可能跨左右两个区域
                                    → T1 被两个叶子引用
```

| 指标 | BVH | KD-Tree |
|------|-----|---------|
| 构建后内存 | 每个图元存一次 | 图元可能存多次 |
| 射线查询 | 快 | 略快（空间不重叠） |
| 动态更新 | 容易（Refit） | 困难（需重建） |
| GPU 友好度 | 高 | 中 |

### 9.2 BVH vs 八叉树

```
八叉树: 递归将空间分成 8 个子立方体

         ┌───┬───┐
         │ 1 │ 2 │
    ┌───┬───┼───┼───┐
    │ 0 │ 1 │ 2 │ 3 │
    ├───┼───┼───┼───┤
    │ 4 │ 5 │ 6 │ 7 │
    └───┴───┴───┴───┘

  每层固定 8 叉，空间均匀分割
  适合: 体素数据、均匀分布
  BVH 是二叉，自适应分割
  适合: 不均匀分布的三角形网格
```

---

## 10. 完整实现代码

### 10.1 BVH 类定义

```cpp
#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <stack>
#include <algorithm>
#include <limits>
#include <cstdint>

// ─── 前向声明 ───
struct Triangle;

// ─── AABB ───
struct AABB {
    glm::vec3 min{std::numeric_limits<float>::max()};
    glm::vec3 max{std::numeric_limits<float>::lowest()};

    AABB() = default;
    AABB(const glm::vec3& mn, const glm::vec3& mx) : min(mn), max(mx) {}

    void expand(const glm::vec3& p) {
        min = glm::min(min, p);
        max = glm::max(max, p);
    }

    void expand(const AABB& other) {
        min = glm::min(min, other.min);
        max = glm::max(max, other.max);
    }

    glm::vec3 centroid() const { return (min + max) * 0.5f; }

    float surfaceArea() const {
        glm::vec3 d = max - min;
        return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
    }

    int longestAxis() const {
        glm::vec3 d = max - min;
        if (d.x >= d.y && d.x >= d.z) return 0;
        if (d.y >= d.z) return 1;
        return 2;
    }

    bool intersectsZ(float z0) const {
        return z0 >= min.z && z0 <= max.z;
    }

    bool intersectRay(const glm::vec3& origin, const glm::vec3& dir,
                      float& tEnter, float& tExit) const {
        glm::vec3 invDir = 1.0f / dir;
        glm::vec3 t1 = (min - origin) * invDir;
        glm::vec3 t2 = (max - origin) * invDir;
        glm::vec3 tMin = glm::min(t1, t2);
        glm::vec3 tMax = glm::max(t1, t2);
        tEnter = glm::max(glm::max(tMin.x, tMin.y), tMin.z);
        tExit  = glm::min(glm::min(tMax.x, tMax.y), tMax.z);
        return tExit >= tEnter && tExit >= 0.0f;
    }
};

// ─── BVH 节点 ───
struct BVHNode {
    AABB bbox;
    int leftChild  = -1;  // 子节点索引（在节点数组中）
    int rightChild = -1;
    int firstIndex = -1;  // 叶子: 第一个图元索引
    int indexCount = 0;   // 叶子: 图元数量；内部节点: 0
    int splitAxis  = 0;   // 内部节点的分割轴

    bool isLeaf() const { return indexCount > 0; }
};

// ─── 构建信息 ───
struct TriangleInfo {
    int index;          // 原始三角形索引
    AABB bbox;          // 三角形 AABB
    glm::vec3 centroid; // 三角形质心
};

// ─── BVH 构建器 ───
class BVHBuilder {
public:
    enum class BuildStrategy { MEDIAN, SAH_BINNED };

    std::vector<BVHNode> nodes;
    std::vector<int> orderedIndices;  // 重排后的三角形索引

    // 构建入口
    void build(const std::vector<Triangle>& triangles,
               BuildStrategy strategy = BuildStrategy::SAH_BINNED,
               int maxLeafSize = 4)
    {
        // 1. 准备构建信息
        std::vector<TriangleInfo> infos(triangles.size());
        for (size_t i = 0; i < triangles.size(); i++) {
            const auto& tri = triangles[i];
            infos[i].index = (int)i;
            infos[i].bbox.min = glm::min(glm::min(tri.v[0], tri.v[1]), tri.v[2]);
            infos[i].bbox.max = glm::max(glm::max(tri.v[0], tri.v[1]), tri.v[2]);
            infos[i].centroid = infos[i].bbox.centroid();
        }

        // 2. 预分配节点空间
        nodes.reserve(triangles.size() * 2);
        orderedIndices.reserve(triangles.size());

        // 3. 递归构建
        if (strategy == BuildStrategy::MEDIAN)
            buildMedian(infos, 0, (int)infos.size(), maxLeafSize);
        else
            buildSAH(infos, 0, (int)infos.size(), maxLeafSize);
    }

private:
    // ─── 中位数分割构建 ───
    int buildMedian(std::vector<TriangleInfo>& infos,
                    int start, int end, int maxLeafSize)
    {
        int nodeIdx = (int)nodes.size();
        nodes.push_back(BVHNode{});

        // 计算总 AABB
        AABB bbox;
        AABB centroidBBox;
        for (int i = start; i < end; i++) {
            bbox.expand(infos[i].bbox);
            centroidBBox.expand(infos[i].centroid);
        }

        int count = end - start;
        if (count <= maxLeafSize) {
            // 创建叶子
            int firstIdx = (int)orderedIndices.size();
            for (int i = start; i < end; i++)
                orderedIndices.push_back(infos[i].index);
            nodes[nodeIdx].bbox = bbox;
            nodes[nodeIdx].firstIndex = firstIdx;
            nodes[nodeIdx].indexCount = count;
            return nodeIdx;
        }

        // 选择最大跨度轴
        int axis = centroidBBox.longestAxis();

        // 沿轴排序
        std::sort(infos.begin() + start, infos.begin() + end,
            [axis](const TriangleInfo& a, const TriangleInfo& b) {
                return a.centroid[axis] < b.centroid[axis];
            });

        int mid = start + count / 2;

        // 递归构建子树
        int leftIdx  = buildMedian(infos, start, mid, maxLeafSize);
        int rightIdx = buildMedian(infos, mid, end, maxLeafSize);

        nodes[nodeIdx].bbox = bbox;
        nodes[nodeIdx].leftChild = leftIdx;
        nodes[nodeIdx].rightChild = rightIdx;
        nodes[nodeIdx].splitAxis = axis;
        nodes[nodeIdx].indexCount = 0;
        return nodeIdx;
    }

    // ─── SAH 分桶构建 ───
    int buildSAH(std::vector<TriangleInfo>& infos,
                 int start, int end, int maxLeafSize)
    {
        constexpr int NUM_BINS = 12;
        constexpr float C_TRAVERSAL = 1.0f;
        constexpr float C_INTERSECT = 1.2f;

        int nodeIdx = (int)nodes.size();
        nodes.push_back(BVHNode{});

        // 计算总 AABB 和质心 AABB
        AABB bbox, centroidBBox;
        for (int i = start; i < end; i++) {
            bbox.expand(infos[i].bbox);
            centroidBBox.expand(infos[i].centroid);
        }

        int count = end - start;
        float leafCost = count * C_INTERSECT;

        if (count <= maxLeafSize) {
            int firstIdx = (int)orderedIndices.size();
            for (int i = start; i < end; i++)
                orderedIndices.push_back(infos[i].index);
            nodes[nodeIdx].bbox = bbox;
            nodes[nodeIdx].firstIndex = firstIdx;
            nodes[nodeIdx].indexCount = count;
            return nodeIdx;
        }

        // ── SAH 搜索最优分割 ──
        float bestCost = std::numeric_limits<float>::max();
        int bestAxis = -1, bestBin = 0;

        struct Bin {
            AABB bbox;
            int count = 0;
        };

        for (int axis = 0; axis < 3; axis++) {
            float cMin = centroidBBox.min[axis];
            float cMax = centroidBBox.max[axis];
            if (cMax <= cMin) continue;

            float scale = NUM_BINS / (cMax - cMin);
            Bin bins[NUM_BINS];

            for (int i = start; i < end; i++) {
                int b = std::min(NUM_BINS - 1,
                    (int)((infos[i].centroid[axis] - cMin) * scale));
                bins[b].bbox.expand(infos[i].bbox);
                bins[b].count++;
            }

            // 前缀扫描
            AABB prefixBBox[NUM_BINS];
            int prefixCount[NUM_BINS];
            prefixBBox[0] = bins[0].bbox;
            prefixCount[0] = bins[0].count;
            for (int i = 1; i < NUM_BINS; i++) {
                prefixBBox[i] = prefixBBox[i-1];
                prefixBBox[i].expand(bins[i].bbox);
                prefixCount[i] = prefixCount[i-1] + bins[i].count;
            }

            // 后缀扫描
            AABB suffixBBox[NUM_BINS];
            int suffixCount[NUM_BINS];
            suffixBBox[NUM_BINS-1] = bins[NUM_BINS-1].bbox;
            suffixCount[NUM_BINS-1] = bins[NUM_BINS-1].count;
            for (int i = NUM_BINS-2; i >= 0; i--) {
                suffixBBox[i] = suffixBBox[i+1];
                suffixBBox[i].expand(bins[i].bbox);
                suffixCount[i] = suffixCount[i+1] + bins[i].count;
            }

            // 尝试分割
            float invSA = 1.0f / bbox.surfaceArea();
            for (int b = 1; b < NUM_BINS; b++) {
                int lc = prefixCount[b-1];
                int rc = suffixCount[b];
                if (lc == 0 || rc == 0) continue;

                float cost = C_TRAVERSAL
                    + prefixBBox[b-1].surfaceArea() * invSA * lc * C_INTERSECT
                    + suffixBBox[b].surfaceArea()   * invSA * rc * C_INTERSECT;

                if (cost < bestCost) {
                    bestCost = cost;
                    bestAxis = axis;
                    bestBin = b;
                }
            }
        }

        // 比较分割 vs 叶子
        if (bestAxis == -1 || bestCost > leafCost) {
            int firstIdx = (int)orderedIndices.size();
            for (int i = start; i < end; i++)
                orderedIndices.push_back(infos[i].index);
            nodes[nodeIdx].bbox = bbox;
            nodes[nodeIdx].firstIndex = firstIdx;
            nodes[nodeIdx].indexCount = count;
            return nodeIdx;
        }

        // 执行分割
        float cMin = centroidBBox.min[bestAxis];
        float cMax = centroidBBox.max[bestAxis];
        float scale = NUM_BINS / (cMax - cMin);

        auto getBin = [&](const TriangleInfo& info) {
            return std::min(NUM_BINS - 1,
                (int)((info.centroid[bestAxis] - cMin) * scale));
        };

        int mid = start;
        for (int i = start; i < end; i++) {
            if (getBin(infos[i]) < bestBin) {
                std::swap(infos[i], infos[mid]);
                mid++;
            }
        }

        // 安全检查
        if (mid == start || mid == end) {
            mid = start + count / 2;
        }

        int leftIdx  = buildSAH(infos, start, mid, maxLeafSize);
        int rightIdx = buildSAH(infos, mid, end, maxLeafSize);

        nodes[nodeIdx].bbox = bbox;
        nodes[nodeIdx].leftChild = leftIdx;
        nodes[nodeIdx].rightChild = rightIdx;
        nodes[nodeIdx].splitAxis = bestAxis;
        nodes[nodeIdx].indexCount = 0;
        return nodeIdx;
    }
};
```

### 10.2 查询接口

```cpp
// ─── BVH 查询器 ───
class BVHQuery {
public:
    const std::vector<BVHNode>& nodes;
    const std::vector<int>& indices;
    const std::vector<Triangle>& triangles;

    BVHQuery(const BVHBuilder& bvh, const std::vector<Triangle>& tris)
        : nodes(bvh.nodes), indices(bvh.orderedIndices), triangles(tris) {}

    // 射线查询（最近交点）
    bool rayIntersect(const glm::vec3& origin, const glm::vec3& dir,
                      float& tHit, int& triIdx) const
    {
        tHit = std::numeric_limits<float>::max();
        bool found = false;
        glm::vec3 invDir = 1.0f / dir;

        int stack[64];
        int stackPtr = 0;
        stack[stackPtr++] = 0;  // root

        while (stackPtr > 0) {
            int nodeIdx = stack[--stackPtr];
            const BVHNode& node = nodes[nodeIdx];

            float tEnter, tExit;
            if (!node.bbox.intersectRay(origin, dir, tEnter, tExit))
                continue;
            if (tEnter > tHit)
                continue;

            if (node.isLeaf()) {
                for (int i = 0; i < node.indexCount; i++) {
                    int idx = indices[node.firstIndex + i];
                    float t;
                    if (rayTriangleIntersect(origin, dir, triangles[idx], t) && t < tHit) {
                        tHit = t;
                        triIdx = idx;
                        found = true;
                    }
                }
            } else {
                // 按方向决定遍历顺序
                if (dir[node.splitAxis] >= 0.0f) {
                    if (node.rightChild >= 0) stack[stackPtr++] = node.rightChild;
                    if (node.leftChild  >= 0) stack[stackPtr++] = node.leftChild;
                } else {
                    if (node.leftChild  >= 0) stack[stackPtr++] = node.leftChild;
                    if (node.rightChild >= 0) stack[stackPtr++] = node.rightChild;
                }
            }
        }
        return found;
    }

    // Z 平面查询（切片加速）
    std::vector<int> queryZPlane(float z0) const
    {
        std::vector<int> result;

        int stack[64];
        int stackPtr = 0;
        stack[stackPtr++] = 0;

        while (stackPtr > 0) {
            int nodeIdx = stack[--stackPtr];
            const BVHNode& node = nodes[nodeIdx];

            if (!node.bbox.intersectsZ(z0))
                continue;

            if (node.isLeaf()) {
                for (int i = 0; i < node.indexCount; i++) {
                    int idx = indices[node.firstIndex + i];
                    const auto& tri = triangles[idx];
                    float zmin = std::min({tri.v[0].z, tri.v[1].z, tri.v[2].z});
                    float zmax = std::max({tri.v[0].z, tri.v[1].z, tri.v[2].z});
                    if (z0 >= zmin && z0 <= zmax)
                        result.push_back(idx);
                }
            } else {
                if (node.leftChild  >= 0) stack[stackPtr++] = node.leftChild;
                if (node.rightChild >= 0) stack[stackPtr++] = node.rightChild;
            }
        }
        return result;
    }

private:
    // Möller-Trumbore 射线-三角形相交
    static bool rayTriangleIntersect(const glm::vec3& origin,
                                      const glm::vec3& dir,
                                      const Triangle& tri,
                                      float& t)
    {
        constexpr float EPS = 1e-8f;
        glm::vec3 e1 = tri.v[1] - tri.v[0];
        glm::vec3 e2 = tri.v[2] - tri.v[0];
        glm::vec3 p = glm::cross(dir, e2);
        float det = glm::dot(e1, p);
        if (std::abs(det) < EPS) return false;
        float invDet = 1.0f / det;
        glm::vec3 tv = origin - tri.v[0];
        float u = glm::dot(tv, p) * invDet;
        if (u < 0.0f || u > 1.0f) return false;
        glm::vec3 q = glm::cross(tv, e1);
        float v = glm::dot(dir, q) * invDet;
        if (v < 0.0f || u + v > 1.0f) return false;
        t = glm::dot(e2, q) * invDet;
        return t > 0.0f;
    }
};
```

---

## 11. 性能分析与调优

### 11.1 构建性能

| 方法 | 10万三角形 | 100万三角形 | 构建质量 |
|------|-----------|------------|---------|
| 暴力遍历 | — | — | — |
| 中位数分割 | ~50ms | ~600ms | 中等 |
| SAH 分桶 | ~80ms | ~900ms | 优秀 |
| SAH 精确 | ~500ms | ~6s | 最优 |
| LBVH (GPU) | ~5ms | ~30ms | 良好 |

### 11.2 查询性能

| 场景 | 无加速 | 中位数 BVH | SAH BVH |
|------|--------|-----------|---------|
| 射线查询 (100万三角形) | ~2ms | ~0.05ms | ~0.02ms |
| 切片查询 (1000层) | ~2s | ~50ms | ~20ms |
| 加速比 | 1× | 40× | 100× |

### 11.3 调优建议

```
1. 叶子大小 (maxLeafSize):
   - 光线追踪: 2~4（减少叶子内线性扫描）
   - 切片加速: 8~16（三角形-Z平面测试很快，可以多放些）
   - 碰撞检测: 4~8

2. 分桶数量 (NUM_BINS):
   - 12 是经典选择（pbrt 默认值）
   - 更多桶 → 更精确但构建更慢
   - 8~16 是合理范围

3. 代价常数:
   - C_TRAVERSAL = 1.0 (基准)
   - C_INTERSECT = 1.2 (三角形相交比 AABB 稍贵)
   - 可根据实际测量调整

4. 内存布局:
   - 深度优先扁平化 → 缓存友好
   - 节点大小对齐到 32/64 字节
   - 图元索引连续存储

5. 动态场景:
   - 使用 BVH Refit（不重建，只更新 AABB）
   - 当质量下降到阈值时再重建
   - 或使用 BVH 重建 + Refit 混合策略
```

### 11.4 BVH Refit（动态更新）

当物体移动但拓扑不变时，可以自底向上更新 AABB 而不重新分割：

```cpp
void refitBVH(std::vector<BVHNode>& nodes, int nodeIdx,
              const std::vector<Triangle>& triangles,
              const std::vector<int>& indices)
{
    BVHNode& node = nodes[nodeIdx];
    if (node.isLeaf()) {
        AABB bbox;
        for (int i = 0; i < node.indexCount; i++) {
            int idx = indices[node.firstIndex + i];
            bbox.expand(triangles[idx].v[0]);
            bbox.expand(triangles[idx].v[1]);
            bbox.expand(triangles[idx].v[2]);
        }
        node.bbox = bbox;
    } else {
        refitBVH(nodes, node.leftChild, triangles, indices);
        refitBVH(nodes, node.rightChild, triangles, indices);
        node.bbox = nodes[node.leftChild].bbox;
        node.bbox.expand(nodes[node.rightChild].bbox);
    }
}
```

```
Refit 的局限:
  - 不改变树的拓扑结构
  - 物体大幅移动后 AABB 重叠加剧 → 查询变慢
  - 需要定期重建（通常每 N 帧或质量下降到阈值时）
```

---

## 附录：与本项目切片算法的集成

将 BVH 集成到现有切片流程中：

```cpp
// 切片流水线集成 BVH
std::vector<std::vector<LineSegment2D>> sliceWithBVH(
    const Mesh& mesh,
    const std::vector<float>& layerHeights)
{
    // 1. 构建 BVH（一次性）
    BVHBuilder builder;
    builder.build(mesh.triangles, BVHBuilder::BuildStrategy::SAH_BINNED, 8);
    BVHQuery query(builder, mesh.triangles);

    // 2. 每层切片
    std::vector<std::vector<LineSegment2D>> layers(layerHeights.size());

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)layerHeights.size(); i++) {
        // BVH 加速查询相关三角形
        auto candidateTris = query.queryZPlane(layerHeights[i]);

        // 精确求交
        for (int triIdx : candidateTris) {
            LineSegment2D seg;
            if (sliceTriangle(mesh.triangles[triIdx], layerHeights[i], seg)) {
                layers[i].push_back(seg);
            }
        }
    }

    return layers;
}
```

```
性能对比（100万三角形，1000层切片）:

无加速:     ~2.0s    (每层遍历全部 100 万三角形)
Z 排序:     ~0.3s    (二分查找 Z 范围)
BVH 中位数: ~0.05s   (O(log n) 查询)
BVH SAH:    ~0.02s   (更优的树结构)
```

---

**相关文档：**
- [3D打印切片算法与颜色管理_完整学习手册.md](3D打印切片算法与颜色管理_完整学习手册.md) — 切片算法总览
- [改进建议与优化方案.md](改进建议与优化方案.md) — 八叉树加速结构
- [光线投射渲染流程详解.md](光线投射渲染流程详解.md) — 光线投射中的 AABB 相交

**最后更新：** 2026年9月16日
