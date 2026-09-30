# Open3D / PCL / CGAL 快速学习手册

> 三大 3D 几何处理库的实战指南：点云 → 网格 → 计算几何

---

## 目录

- [1. 三库总览与选型](#1-三库总览与选型)
- [2. 环境搭建](#2-环境搭建)
- [3. Open3D 快速入门](#3-open3d-快速入门)
- [4. PCL 快速入门](#4-pcl-快速入门)
- [5. CGAL 快速入门](#5-cgal-快速入门)
- [6. 三库对比与协作](#6-三库对比与协作)
- [7. 实战项目路线](#7-实战项目路线)
- [8. 常见问题速查](#8-常见问题速查)

---

## 1. 三库总览与选型

### 一句话定位

| 库 | 语言 | 定位 | 类比 |
|---|---|---|---|
| **Open3D** | Python / C++ | 现代3D数据处理，易用优先 | 3D界的 NumPy |
| **PCL** | C++ (有Python绑定) | 工业级点云处理，功能最全 | 3D界的 OpenCV |
| **CGAL** | C++ (头为主) | 计算几何算法库，精度优先 | 几何界的 STL |

### 选型决策树

```
你的需求是什么？
├─ 快速原型 / 可视化 / 深度学习预处理 → Open3D
├─ 工业点云（滤波/配准/分割/表面重建） → PCL
├─ 网格布尔运算 / Voronoi / Delaunay / 精确几何 → CGAL
└─ 全流程（点云→网格→几何分析）→ Open3D + CGAL 组合
```

### 核心能力矩阵

| 能力 | Open3D | PCL | CGAL |
|---|---|---|---|
| 点云IO | ✅ 简单 | ✅ 全面 | ❌ 不专注 |
| 点云滤波 | ✅ 基础 | ✅ 最全 | ❌ |
| 点云配准(ICP) | ✅ 易用 | ✅ 全面 | ✅ 精确 |
| 表面重建 | ✅ Poisson/Delaunay | ✅ Greedy/MLS | ✅ 最精确 |
| 网格处理 | ✅ 基础 | ❌ 弱 | ✅ 最强 |
| 布尔运算 | ❌ | ❌ | ✅ 核心 |
| Voronoi/Delaunay | ✅ 基础 | ❌ | ✅ 最强 |
| 可视化 | ✅ 最方便 | ⚠️ 需VTK | ⚠️ 需配置 |
| Python支持 | ✅ 原生 | ⚠️ 绑定 | ⚠️ 绑定 |
| 学习曲线 | 低 | 中高 | 高 |

---

## 2. 环境搭建

### 2.1 Open3D（Python，最快上手）

```bash
pip install open3d
```

验证：
```python
import open3d as o3d
print(o3d.__version__)
```

### 2.2 PCL（C++，推荐 vcpkg）

```bash
# 方式1: vcpkg（推荐Windows）
vcpkg install pcl

# 方式2: Ubuntu
sudo apt install libpcl-dev

# 方式3: Python绑定
pip install python-pcl  # 功能有限，推荐用C++
```

CMake 配置：
```cmake
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
target_link_libraries(your_app ${PCL_LIBRARIES})
```

### 2.3 CGAL（C++，头文件库为主）

```bash
# vcpkg
vcpkg install cgal

# Ubuntu
sudo apt install libcgal-dev
```

CMake 配置：
```cmake
find_package(CGAL REQUIRED)
target_link_libraries(your_app CGAL::CGAL)
# 需要布尔运算时额外加
find_package(CGAL COMPONENTS Qt5)
```

---

## 3. Open3D 快速入门

### 3.1 核心数据结构

```python
import open3d as o3d
import numpy as np

# === 点云 ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
pcd.colors = o3d.utility.Vector3dVector(np.random.rand(100, 3))  # RGB 0~1
pcd.normals = o3d.utility.Vector3dVector(np.random.rand(100, 3))

# IO
o3d.io.write_point_cloud("cloud.ply", pcd)
pcd = o3d.io.read_point_cloud("cloud.ply")

# === 三角网格 ===
mesh = o3d.geometry.TriangleMesh()
# 从点云创建（需要进一步重建）
# 或直接读取
mesh = o3d.io.read_triangle_mesh("model.ply")
mesh.compute_vertex_normals()  # 计算法线

# === 体素网格 ===
voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
```

### 3.2 可视化（最常用）

```python
# 基本可视化
o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")

# 网格可视化
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

# 多对象 + 自定义窗口
vis = o3d.visualization.Visualizer()
vis.create_window(width=1280, height=720)
vis.add_geometry(pcd)
vis.add_geometry(mesh)
# 改变渲染风格
opt = vis.get_render_option()
opt.point_size = 3
opt.background_color = np.array([0.1, 0.1, 0.1])
vis.run()
vis.destroy_window()
```

### 3.3 点云处理核心操作

```python
# === 降采样 ===
pcd_down = pcd.voxel_down_sample(voxel_size=0.05)
pcd_down = pcd.uniform_down_sample(every_k_points=3)

# === 滤波 ===
pcd_filtered = pcd_down.remove_statistical_outlier(
    nb_neighbors=20, std_ratio=2.0
)[0]  # 统计滤波去噪
pcd_filtered = pcd_down.remove_radius_outlier(
    nb_points=16, radius=0.05
)[0]  # 半径滤波

# === 法线估计 ===
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)
pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))

# === KNN / 半径搜索 ===
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
[k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[0], 50)      # K近邻
[k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[0], 0.1)  # 半径搜索
```

### 3.4 点云配准（ICP）

```python
source = o3d.io.read_point_cloud("source.ply")
target = o3d.io.read_point_cloud("target.ply")
source_down = source.voxel_down_sample(0.05)
target_down = target.voxel_down_sample(0.05)
source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# === 全局配准（RANSAC + FPFH特征） ===
def preprocess_fpfh(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    return pcd_down, fpfh

voxel_size = 0.05
s_down, s_fpfh = preprocess_fpfh(source, voxel_size)
t_down, t_fpfh = preprocess_fpfh(target, voxel_size)

result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    s_down, t_down, s_fpfh, t_fpfh, True,
    voxel_size * 1.5,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    4,
    [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
    ],
    o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))

# === 精配准（Point-to-Plane ICP） ===
result_icp = o3d.pipelines.registration.registration_icp(
    source_down, target_down, 0.02,
    result_ransac.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane())

source.transform(result_icp.transformation)
```

### 3.5 表面重建

```python
# === Poisson 重建（最常用） ===
pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9)
# 去掉低密度区域（修剪）
vertices_to_remove = densities < np.quantile(densities, 0.05)
mesh.remove_vertices_by_mask(vertices_to_remove)

# === Alpha Shape 重建 ===
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.03)

# === Ball Pivoting 重建 ===
pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
radii = [0.005, 0.01, 0.02]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))
```

### 3.6 网格处理

```python
mesh = o3d.io.read_triangle_mesh("model.ply")
mesh.compute_vertex_normals()

# 简化（二次误差度量）
mesh_simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=5000)

# 细分
mesh_subdivided = mesh.subdivide_loop(number_of_iterations=1)

# 裁剪
bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1,-1,-1), max_bound=(1,1,1))
mesh_cropped = mesh.crop(bbox)

# 凸包
hull, _ = mesh.compute_convex_hull()

# 网格分割（连通分量）
mesh.cluster_connected_triangles()  # 返回标签和数量
```

---

## 4. PCL 快速入门

### 4.1 核心数据结构

```cpp
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

// 基本点类型
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
// 带颜色
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
// 带法线
pcl::PointCloud<pcl::PointNormal>::Ptr cloud_n(new pcl::PointCloud<pcl::PointNormal>);

// 填充点
pcl::PointXYZ pt;
pt.x = 1.0; pt.y = 2.0; pt.z = 3.0;
cloud->push_back(pt);

// 或从数组转换
cloud->width = 100;
cloud->height = 1;
cloud->points.resize(100);
for (auto& p : cloud->points) {
    p.x = ((rand() % 100) / 100.0);
    p.y = ((rand() % 100) / 100.0);
    p.z = ((rand() % 100) / 100.0);
}
```

### 4.2 IO 操作

```cpp
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

// 读取
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
if (pcl::io::loadPCDFile<pcl::PointXYZ>("cloud.pcd", *cloud) == -1) {
    PCL_ERROR("Couldn't read file\n");
    return -1;
}
// PLY格式
pcl::io::loadPLYFile<pcl::PointXYZ>("cloud.ply", *cloud);

// 保存
pcl::io::savePCDFileBinary("output.pcd", *cloud);
pcl::io::savePLYFileBinary("output.ply", *cloud);
```

### 4.3 滤波（PCL最强项）

```cpp
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>

// === 体素降采样 ===
pcl::VoxelGrid<pcl::PointXYZ> voxel;
voxel.setInputCloud(cloud);
voxel.setLeafSize(0.05f, 0.05f, 0.05f);
pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
voxel.filter(*filtered);

// === 统计滤波去噪 ===
pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
sor.setInputCloud(cloud);
sor.setMeanK(50);
sor.setStddevMulThresh(1.0);
sor.filter(*filtered);

// === 直通滤波（按轴裁剪） ===
pcl::PassThrough<pcl::PointXYZ> pass;
pass.setInputCloud(cloud);
pass.setFilterFieldName("z");
pass.setFilterLimits(0.0, 1.0);
pass.filter(*filtered);

// === 半径滤波 ===
pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
ror.setInputCloud(cloud);
ror.setRadiusSearch(0.1);
ror.setMinNeighborsInRadius(10);
ror.filter(*filtered);
```

### 4.4 法线估计

```cpp
#include <pcl/features/normal_3d.h>

pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
ne.setInputCloud(cloud);
pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
ne.setSearchMethod(tree);
ne.setRadiusSearch(0.03);

pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
ne.compute(*normals);

// 合并点+法线
pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
```

### 4.5 关键点与特征描述子

```cpp
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/fpfh.h>

// === FPFH 特征（配准用） ===
pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
fpfh.setInputCloud(cloud);
fpfh.setInputNormals(normals);
fpfh.setSearchMethod(tree);
fpfh.setRadiusSearch(0.05);

pcl::PointCloud<pcl::FPFHSignature33>::Ptr features(new pcl::PointCloud<pcl::FPFHSignature33>);
fpfh.compute(*features);
```

### 4.6 点云配准

```cpp
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>

// === ICP 精配准 ===
pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
icp.setInputSource(source_cloud);
icp.setInputTarget(target_cloud);
icp.setMaxCorrespondenceDistance(0.05);
icp.setMaximumIterations(50);
icp.setTransformationEpsilon(1e-8);
icp.setEuclideanFitnessEpsilon(1e-5);

pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
icp.align(*aligned);

if (icp.hasConverged()) {
    std::cout << "Score: " << icp.getFitnessScore() << std::endl;
    Eigen::Matrix4f transform = icp.getFinalTransformation();
}

// === Point-to-Plane ICP（更精确） ===
pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> icp_plane;
// 用法类似，输入需要带法线

// === NDT 配准（大范围初始对齐） ===
pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
ndt.setInputSource(source_cloud);
ndt.setInputTarget(target_cloud);
ndt.setResolution(1.0);
ndt.setStepSize(0.1);
ndt.setMaximumIterations(50);
ndt.align(*aligned);
```

### 4.7 表面重建

```cpp
#include <pcl/surface/mls.h>        // Moving Least Squares 平滑
#include <pcl/surface/gp3.h>        // Greedy Projection Triangulation
#include <pcl/surface/poisson.h>    // Poisson 重建

// === MLS 平滑（上采样+法线） ===
pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
mls.setInputCloud(cloud);
mls.setComputeNormals(true);
mls.setPolynomialOrder(2);
mls.setSearchRadius(0.03);
pcl::PointCloud<pcl::PointNormal>::Ptr mls_points(new pcl::PointCloud<pcl::PointNormal>);
mls.process(*mls_points);

// === Greedy 三角化 ===
pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
gp3.setInputCloud(mls_points);
gp3.setSearchMethod(tree);
gp3.setMaximumNearestNeighbors(20);
gp3.setMaximumSurfaceAngle(M_PI / 4);
gp3.setMaximumAngle(2 * M_PI / 3);
gp3.setNormalConsistency(true);
pcl::PolygonMesh mesh;
gp3.reconstruct(mesh);

// === Poisson 重建 ===
pcl::Poisson<pcl::PointNormal> poisson;
poisson.setInputCloud(mls_points);
poisson.setDepth(9);
poisson.reconstruct(mesh);

// 保存网格
pcl::io::saveOBJFile("mesh.obj", mesh);
```

### 4.8 分割

```cpp
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

// === RANSAC 平面分割 ===
pcl::SACSegmentation<pcl::PointXYZ> seg;
seg.setOptimizeCoefficients(true);
seg.setModelType(pcl::SACMODEL_PLANE);
seg.setMethodType(pcl::SAC_RANSAC);
seg.setDistanceThreshold(0.01);

pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
seg.setInputCloud(cloud);
seg.segment(*inliers, *coefficients);

// 提取平面上的点
pcl::ExtractIndices<pcl::PointXYZ> extract;
extract.setInputCloud(cloud);
extract.setIndices(inliers);
extract.setNegative(false);  // true = 提取非平面点
extract.filter(*filtered);

// === 欧式聚类分割 ===
pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
ec.setClusterTolerance(0.02);
ec.setMinClusterSize(100);
ec.setMaxClusterSize(25000);
ec.setSearchMethod(tree);
ec.setInputCloud(cloud);
std::vector<pcl::PointIndices> cluster_indices;
ec.extract(cluster_indices);
```

### 4.9 可视化

```cpp
#include <pcl/visualization/pcl_visualizer.h>

pcl::visualization::PCLVisualizer::Ptr viewer(
    new pcl::visualization::PCLVisualizer("3D Viewer"));
viewer->setBackgroundColor(0.05, 0.05, 0.05);
viewer->addPointCloud<pcl::PointXYZ>(cloud, "cloud");
viewer->setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
viewer->addCoordinateSystem(1.0);
viewer->initCameraParameters();

// 添加法线
viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(
    cloud, normals, 10, 0.05, "normals");

// 添加网格
viewer->addPolygonMesh(mesh, "mesh");

while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
}
```

---

## 5. CGAL 快速入门

### 5.1 核心概念

CGAL 的核心设计理念是**模板 + 概念(Concept) + 特性(Traits)**：

```
Kernel（核）—— 决定精度
├─ Exact_predicates_inexact_constructions_kernel (EPICK)  ← 最常用，快速
├─ Exact_predicates_exact_constructions_kernel (EPECK)    ← 需要精确构造时
└─ Cartesian<double> / Cartesian<CGAL::Exact_nt>
```

```cpp
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2    Point_2;
typedef K::Point_3    Point_3;
typedef K::Vector_3   Vector_3;
typedef K::Segment_3  Segment_3;
typedef K::Triangle_3 Triangle_3;
typedef K::Plane_3    Plane_3;
```

### 5.2 基本几何操作

```cpp
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/algorithm.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3 Point_3;
typedef K::Vector_3 Vector_3;
typedef K::Segment_3 Segment_3;
typedef K::Triangle_3 Triangle_3;

// 点和向量
Point_3 p(1.0, 2.0, 3.0);
Point_3 q(4.0, 5.0, 6.0);
Vector_3 v = q - p;  // 向量
double dist = CGAL::squared_distance(p, q);  // 距离平方

// 线段相交
Segment_3 s1(p, q);
Segment_3 s2(Point_3(0,0,0), Point_3(5,5,5));
if (CGAL::do_intersect(s1, s2)) {
    auto result = CGAL::intersection(s1, s2);
    if (const Point_3* ip = boost::get<Point_3>(&*result)) {
        std::cout << "交点: " << *ip << std::endl;
    }
}

// 三角形面积
Triangle_3 tri(p, q, Point_3(0,0,0));
double area = tri.area();
```

### 5.3 2D Delaunay 三角剖分

```cpp
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/point_generators_2.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_2<K> Delaunay;
typedef K::Point_2 Point_2;

// 生成随机点
std::vector<Point_2> points;
CGAL::Random_points_in_square_2<Point_2> gen(1.0);
std::copy_n(gen, 100, std::back_inserter(points));

// 构建 Delaunay 三角剖分
Delaunay dt;
dt.insert(points.begin(), points.end());

// 遍历三角形
for (Delaunay::Finite_faces_iterator fit = dt.finite_faces_begin();
     fit != dt.finite_faces_end(); ++fit) {
    Point_2 p0 = fit->vertex(0)->point();
    Point_2 p1 = fit->vertex(1)->point();
    Point_2 p2 = fit->vertex(2)->point();
    // 处理三角形 (p0, p1, p2)
}

// 最近邻查询
Point_2 query(0.5, 0.5);
Delaunay::Vertex_handle nearest = dt.nearest_vertex(query);
```

### 5.4 3D Delaunay 与 Alpha Shape

```cpp
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Alpha_shape_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Alpha_shape_vertex_base_3<K> Vb;
typedef CGAL::Alpha_shape_cell_base_3<K> Cb;
typedef CGAL::Triangulation_data_structure_3<Vb, Cb> Tds;
typedef CGAL::Alpha_shape_3<Tds> Alpha_shape_3;
typedef K::Point_3 Point_3;

// 读取点云
std::vector<Point_3> points = { /* ... */ };

// Alpha Shape 重建表面
Alpha_shape_3 as(points.begin(), points.end());
as.set_alpha(0.5);  // alpha 参数控制细节

// 提取表面三角形
std::vector<Alpha_shape_3::Facet> facets;
for (auto it = as.alpha_shape_facets_begin(); it != as.alpha_shape_facets_end(); ++it) {
    facets.push_back(*it);
}
```

### 5.5 网格布尔运算（CGAL杀手锏）

```cpp
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/IO/OBJ.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Surface_mesh<K::Point_3> Mesh;

Mesh mesh1, mesh2;
CGAL::IO::read_OBJ("mesh1.obj", mesh1);
CGAL::IO::read_OBJ("mesh2.obj", mesh2);

Mesh result_union, result_inter, result_diff;

// 布尔运算
namespace PMP = CGAL::Polygon_mesh_processing;

// 并集
PMP::corefine_and_compute_union(mesh1, mesh2, result_union);
// 交集
PMP::corefine_and_compute_intersection(mesh1, mesh2, result_inter);
// 差集 (mesh1 - mesh2)
PMP::corefine_and_compute_difference(mesh1, mesh2, result_diff);

CGAL::IO::write_OBJ("union.obj", result_union);
```

### 5.6 Poisson 表面重建

```cpp
#include <CGAL/poisson_surface_reconstruction_3.h>
#include <CGAL/IO/read_points.h>
#include <CGAL/Surface_mesh.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3 Point_3;
typedef K::Vector_3 Vector_3;
typedef std::pair<Point_3, Vector_3> PointVectorPair;
typedef CGAL::Surface_mesh<Point_3> Mesh;

// 读取带法线的点云
std::vector<PointVectorPair> points;
CGAL::IO::read_points("cloud.ply", std::back_inserter(points),
    CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>())
                    .normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>()));

// Poisson 重建
Mesh mesh;
double average_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>(
    points, 6, CGAL::parameters::point_map(
        CGAL::First_of_pair_property_map<PointVectorPair>()));

CGAL::poisson_surface_reconstruction_delaunay(
    points.begin(), points.end(),
    CGAL::First_of_pair_property_map<PointVectorPair>(),
    CGAL::Second_of_pair_property_map<PointVectorPair>(),
    mesh, average_spacing);

CGAL::IO::write_OBJ("reconstructed.obj", mesh);
```

### 5.7 网格简化与优化

```cpp
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/simplify.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Surface_mesh<K::Point_3> Mesh;
namespace PMP = CGAL::Polygon_mesh_processing;

Mesh mesh;
CGAL::IO::read_OBJ("model.obj", mesh);

// === 边折叠简化 ===
SMS::Edge_length_cost<Mesh> cost;
SMS::Edge_length_stop_predicate<double> stop(0.1);  // 目标边长
SMS::Bounded_normal_placement_filter<Mesh> filter;
SMS::edge_collapse(mesh, stop,
    CGAL::parameters::get_cost(cost)
                    .get_placement(SMS::Midpoint_placement<Mesh>())
                    .filter(filter));

// === 各向同性重网格化 ===
PMP::isotropic_remeshing(mesh.faces(), 0.05, mesh,
    PMP::parameters::number_of_iterations(3));

// === 网格平滑 ===
PMP::smooth_mesh(mesh, PMP::parameters::number_of_iterations(10));

// === 自交检测 ===
bool self_intersect = PMP::does_self_intersect(mesh);
```

### 5.8 AABB Tree（空间查询）

```cpp
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>

typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_3 Point;
typedef K::Triangle_3 Triangle;
typedef std::vector<Triangle>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_traits;
typedef CGAL::AABB_tree<AABB_traits> AABB_tree;

// 构建三角形列表
std::vector<Triangle> triangles;
triangles.push_back(Triangle(Point(0,0,0), Point(1,0,0), Point(0,1,0)));
triangles.push_back(Triangle(Point(1,0,0), Point(1,1,0), Point(0,1,0)));

// 构建 AABB Tree
AABB_tree tree(triangles.begin(), triangles.end());

// 最近点查询
Point query(0.5, 0.5, 1.0);
Point closest = tree.closest_point(query);

// 射线相交
K::Ray_3 ray(Point(0.5, 0.5, 5.0), Vector(0, 0, -1));
boost::optional<AABB_tree::Intersection_and_primitive_id<K::Ray_3>::Type>
    hit = tree.first_intersection(ray);
if (hit) {
    Point intersection = boost::get<Point>(&hit->first);
}
```

---

## 6. 三库对比与协作

### 6.1 同一任务的三库实现对比

#### 点云读取 + 降采样 + 法线

| 步骤 | Open3D (Python) | PCL (C++) | CGAL (C++) |
|---|---|---|---|
| 读取 | `o3d.io.read_point_cloud()` | `pcl::io::loadPCDFile()` | `CGAL::IO::read_points()` |
| 降采样 | `pcd.voxel_down_sample(0.05)` | `pcl::VoxelGrid` filter | 手动或用网格 |
| 法线 | `pcd.estimate_normals()` | `pcl::NormalEstimation` | `CGAL::pca_estimate_normals()` |
| 代码量 | ~5行 | ~20行 | ~15行 |

### 6.2 推荐组合方案

```
方案A: 快速原型
  Open3D (Python) → 全流程搞定
  适合: 研究、可视化、深度学习预处理

方案B: 工业级点云
  PCL (C++) → 滤波/配准/分割/重建
  适合: 机器人、自动驾驶、工业检测

方案C: 精确几何
  CGAL (C++) → 布尔运算/网格处理/计算几何
  适合: CAD、医学建模、3D打印

方案D: 全栈组合（推荐）
  Open3D (Python) → 预处理 + 可视化
  CGAL (C++) → 精确网格操作
  通过 pybind11 / 文件交换连接
```

### 6.3 文件格式互操作

```
点云: PLY / PCD / XYZ / LAS
  Open3D ←→ PCL: 用 PLY 格式
  Open3D ←→ CGAL: 用 PLY / OFF 格式

网格: OBJ / PLY / OFF / STL
  CGAL ←→ Open3D: 用 OBJ 或 PLY
  CGAL ←→ PCL: 用 PLY (PolygonMesh)
```

---

## 7. 实战项目路线

### Level 1: 入门（1-2周）

```
□ Open3D: 读取点云 → 降采样 → 可视化
□ Open3D: 法线估计 → Poisson重建 → 网格可视化
□ PCL: PCD文件读写 → 体素降采样 → 统计滤波
□ CGAL: 2D Delaunay三角剖分 → 可视化
```

### Level 2: 进阶（2-4周）

```
□ Open3D: 两片点云ICP配准 → 结果可视化
□ PCL: RANSAC平面分割 → 欧式聚类 → 多目标提取
□ PCL: MLS平滑 → Greedy三角化 → 网格导出
□ CGAL: 两个网格布尔运算（并/交/差）
□ CGAL: Poisson重建 → 网格简化
```

### Level 3: 实战（1-2月）

```
□ 多视角点云拼接: PCL配准 → CGAL重建 → Open3D可视化
□ 3D扫描后处理: Open3D去噪 → CGAL布尔裁剪 → 网格优化
□ 医学影像表面提取: Marching Cubes → CGAL平滑 → 布尔切割
□ 点云分割+重建流水线: PCL分割 → 各物体重建 → Open3D展示
```

---

## 8. 常见问题速查

### Q: Open3D 和 PCL 选哪个？

| 场景 | 推荐 |
|---|---|
| Python 快速原型 | Open3D |
| 需要最全的滤波算法 | PCL |
| 需要NDT配准 | PCL |
| 需要好的可视化 | Open3D |
| 深度学习预处理 | Open3D |
| 工业部署(C++) | PCL |

### Q: CGAL 的 Kernel 怎么选？

```cpp
// 90% 的情况用这个：快速，谓词精确，构造不精确但够用
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

// 布尔运算、需要精确几何构造时用这个
typedef CGAL::Exact_predicates_exact_constructions_kernel K;
```

### Q: PCL 编译太慢怎么办？

```cmake
# 只链接需要的模块
find_package(PCL REQUIRED COMPONENTS common io filters features kdtree registration segmentation surface visualization)

# 用预编译头
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")  # MSVC 多核编译
```

### Q: Open3D C++ 怎么用？

```cmake
# vcpkg 安装
vcpkg install open3d

# CMake
find_package(Open3D REQUIRED)
target_link_libraries(your_app Open3D::Open3D)
```

```cpp
#include <open3d/Open3D.h>
auto pcd = open3d::io::CreatePointCloudFromFile("cloud.ply");
auto mesh = open3d::geometry::TriangleMesh::CreateFromPointCloudPoisson(*pcd);
```

### Q: 三库能混用吗？

可以！通过文件格式中转：
```
PCL处理 → 保存PLY → CGAL读取 → 布尔运算 → 保存OBJ → Open3D可视化
```

或用 Python 统一调度：
```python
import open3d as o3d
import subprocess

# Open3D 预处理
pcd = o3d.io.read_point_cloud("input.ply")
o3d.io.write_point_cloud("cleaned.ply", pcd)

# 调用 CGAL C++ 程序做布尔运算
subprocess.run(["./cgal_boolean", "cleaned.ply", "cut.ply", "result.obj"])

# Open3D 可视化结果
mesh = o3d.io.read_triangle_mesh("result.obj")
o3d.visualization.draw_geometries([mesh])
```

---

## 速查卡

```
┌─────────────────────────────────────────────────────────────┐
│                    快速决策                                  │
├──────────────┬──────────────────────────────────────────────┤
│ 读取点云     │ Open3D: read_point_cloud()                   │
│              │ PCL: loadPCDFile() / loadPLYFile()           │
│              │ CGAL: IO::read_points()                      │
├──────────────┼──────────────────────────────────────────────┤
│ 降采样       │ Open3D: voxel_down_sample()                  │
│              │ PCL: VoxelGrid filter                        │
├──────────────┼──────────────────────────────────────────────┤
│ 去噪         │ Open3D: remove_statistical_outlier()         │
│              │ PCL: StatisticalOutlierRemoval               │
├──────────────┼──────────────────────────────────────────────┤
│ 法线         │ Open3D: estimate_normals()                   │
│              │ PCL: NormalEstimation                        │
│              │ CGAL: pca_estimate_normals()                 │
├──────────────┼──────────────────────────────────────────────┤
│ ICP配准      │ Open3D: registration_icp()                   │
│              │ PCL: IterativeClosestPoint                   │
├──────────────┼──────────────────────────────────────────────┤
│ 表面重建     │ Open3D: create_from_point_cloud_poisson()    │
│              │ PCL: Poisson / GreedyProjectionTriangulation │
│              │ CGAL: poisson_surface_reconstruction_3()     │
├──────────────┼──────────────────────────────────────────────┤
│ 布尔运算     │ CGAL: corefine_and_compute_union()           │
│              │   (仅CGAL支持)                               │
├──────────────┼──────────────────────────────────────────────┤
│ 网格简化     │ Open3D: simplify_quadric_decimation()        │
│              │ CGAL: edge_collapse()                        │
├──────────────┼──────────────────────────────────────────────┤
│ 可视化       │ Open3D: draw_geometries()  ← 最方便          │
│              │ PCL: PCLVisualizer                           │
└──────────────┴──────────────────────────────────────────────┘
```

> **学习建议**: 先用 Open3D(Python) 快速跑通全流程建立直觉，再根据项目需要深入 PCL 或 CGAL 的 C++ 细节。
