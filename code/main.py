
# class Solution:
#     def twoSum(self, nums: List[int], target: int) -> List[int]:
#         dic_value = {}
#         for i in range(len(nums)):
#             need_value = target - nums[i]
#             if need_value in dic_value:
#                 return [dic_value[need_value], i]
#             dic_value[nums[i]] = i
#         return []

import SimpleITK as sitk
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

# 1. 生成合成血管mask（一个弯曲的圆柱体）
def create_tube_mask(shape=(128,128,128), radius=5):
    mask = np.zeros(shape, dtype=np.uint8)
    # 生成曲线参数
    t = np.linspace(0, 1, 50)
    x = 64 + 30*np.sin(np.pi*t)
    y = 64 + 30*np.cos(2*np.pi*t)
    z = 10 + 100*t
    # 离散化
    pts = np.round(np.stack([x,y,z], axis=1)).astype(int)
    # 在mask中绘制线段并膨胀成圆柱（简单方法：每个点画球）
    from scipy.spatial import KDTree
    coords = np.array(np.where(mask==0)).T
    tree = KDTree(pts)
    dist, _ = tree.query(coords)
    mask_flat = (dist <= radius).reshape(shape).astype(np.uint8)
    return mask_flat

mask = create_tube_mask()
# 2. 距离场
distance = distance_transform_edt(mask)
# 3. 手动选择起点终点（取两端点）
coords = np.argwhere(mask > 0)
z_min = coords[np.argmin(coords[:,2])]  # Z最小端
z_max = coords[np.argmax(coords[:,2])]  # Z最大端
start = tuple(z_min)
end = tuple(z_max)

# 4. Dijkstra提取路径（简化版，只使用邻域）
def dijkstra_path(distance, start, end):
    shape = distance.shape
    # 邻居6
    neighbors = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    import heapq
    start_idx = start[0]*shape[1]*shape[2] + start[1]*shape[2] + start[2]
    end_idx = end[0]*shape[1]*shape[2] + end[1]*shape[2] + end[2]
    dist = {start_idx: 0}
    prev = {}
    heap = [(0, start_idx)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == end_idx: break
        z = u // (shape[1]*shape[2])
        y = (u % (shape[1]*shape[2])) // shape[2]
        x = u % shape[2]
        for dz,dy,dx in neighbors:
            nz,ny,nx = z+dz, y+dy, x+dx
            if 0<=nz<shape[0] and 0<=ny<shape[1] and 0<=nx<shape[2]:
                v = nz*shape[1]*shape[2] + ny*shape[2] + nx
                cost = 1.0 / (distance[nz,ny,nx] + 0.1)
                nd = d + cost
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
    # 回溯
    path = []
    cur = end_idx
    while cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.append(start_idx)
    path.reverse()
    pts = []
    for p in path:
        z = p // (shape[1]*shape[2])
        y = (p % (shape[1]*shape[2])) // shape[2]
        x = p % shape[2]
        pts.append([z,y,x])
    return np.array(pts)

centerline = dijkstra_path(distance, start, end)

# 5. 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(centerline[:,2], centerline[:,1], centerline[:,0], c='r', s=10)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
plt.title('Extracted Centerline')
plt.show()