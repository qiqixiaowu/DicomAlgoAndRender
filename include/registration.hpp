#pragma once
/**
 * @file    registration.hpp
 * @brief   3D 医疗影像配准算法库（Header-Only，无外部依赖）
 *
 * 数据：3D 体积（short/float HU 值，W×H×D）
 * 算法：
 *  1. Rigid3D      — 6-DOF 刚性配准（tx,ty,tz + rx,ry,rz），梯度下降 + 多分辨率金字塔
 *  2. Demons3D     — 3D Demons 形变配准（Thirion 1998 光流力）
 *  3. MG-Demons3D  — 多格网多分辨率 3D Demons
 *  4. LCC-Demons3D — 局部互相关 Demons（多模态鲁棒）
 *  5. FFD3D        — 三次 B 样条自由形变（解析梯度，高效）
 *
 * 参考：
 *  algo-registration/src/McsfAlgoRegistration/    (刚性)
 *  algo-registration/src/McsfAlgoDemons/          (Demons)
 *  algo-registrationgpu/src/McsfAlgoSynRegG/      (SyN/LCC)
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace MedReg {

// ============================================================
// §1  核心数据结构
// ============================================================

/**
 * @brief 3D 医学图像体积（float，归一化或 HU 值）
 *
 * 存储顺序：Z-slice major（DICOM 顺序）
 *   index = z * W * H + y * W + x
 */
struct Image3D {
    std::vector<float> data;
    int width  = 0;  ///< Columns（X 方向）
    int height = 0;  ///< Rows（Y 方向）
    int depth  = 0;  ///< Slices（Z 方向）
    float spacingX = 1.0f;  ///< 物理间距 mm
    float spacingY = 1.0f;
    float spacingZ = 1.0f;

    int  size() const { return width * height * depth; }
    bool empty() const { return data.empty(); }

    int idx(int x, int y, int z) const {
        return z * width * height + y * width + x;
    }
    float  at(int x, int y, int z) const { return data[idx(x,y,z)]; }
    float& at(int x, int y, int z)       { return data[idx(x,y,z)]; }

    void resize(int w, int h, int d) {
        width = w; height = h; depth = d;
        data.assign(w * h * d, 0.0f);
    }

    void normalize() {
        if (data.empty()) return;
        float mn = *std::min_element(data.begin(), data.end());
        float mx = *std::max_element(data.begin(), data.end());
        float rng = mx - mn + 1e-8f;
        for (auto& v : data) v = (v - mn) / rng;
    }

    /** 获取第 z 层轴位切片（返回 W×H 的 float 数组） */
    std::vector<float> getAxialSlice(int z) const {
        z = std::clamp(z, 0, depth - 1);
        std::vector<float> s(width * height);
        std::copy(data.data() + z * width * height,
                  data.data() + (z+1) * width * height, s.begin());
        return s;
    }
};

/**
 * @brief 3D 密集形变场（体素单位位移）
 *
 * dx/dy/dz 各自独立存储，与源码三通道 pDis[] 对应
 */
struct DeformField3D {
    std::vector<float> dx, dy, dz;
    int width = 0, height = 0, depth = 0;

    int  size() const { return width * height * depth; }
    bool empty() const { return dx.empty(); }

    void resize(int w, int h, int d) {
        width = w; height = h; depth = d;
        int N = w * h * d;
        dx.assign(N, 0.0f); dy.assign(N, 0.0f); dz.assign(N, 0.0f);
    }
    int idx(int x, int y, int z) const {
        return z * width * height + y * width + x;
    }
};

/**
 * @brief 6-DOF 3D 刚性变换参数
 * 对应源码 PARAMETER_t 中的旋转+平移参数
 */
struct RigidParam3D {
    float tx = 0.0f, ty = 0.0f, tz = 0.0f;  ///< 平移（体素单位）
    float rx = 0.0f, ry = 0.0f, rz = 0.0f;  ///< 旋转（弧度，ZYX Euler）
};

// ============================================================
// §2  图像处理工具
// ============================================================

/**
 * @brief 三线性插值（越界返回 0）
 *
 * 对应源码：McsfAlgoRegistration.cpp → ReSampleLinear3D
 */
inline float trilinear(const Image3D& v, float x, float y, float z) {
    if (x < 0 || x > v.width  - 1.0001f ||
        y < 0 || y > v.height - 1.0001f ||
        z < 0 || z > v.depth  - 1.0001f) return 0.0f;

    int xi = std::min((int)x, v.width  - 2);
    int yi = std::min((int)y, v.height - 2);
    int zi = std::min((int)z, v.depth  - 2);
    float fx = x - xi, fy = y - yi, fz = z - zi;

    float v000 = v.at(xi,   yi,   zi  ), v100 = v.at(xi+1, yi,   zi  );
    float v010 = v.at(xi,   yi+1, zi  ), v110 = v.at(xi+1, yi+1, zi  );
    float v001 = v.at(xi,   yi,   zi+1), v101 = v.at(xi+1, yi,   zi+1);
    float v011 = v.at(xi,   yi+1, zi+1), v111 = v.at(xi+1, yi+1, zi+1);

    return ((v000*(1-fx)+v100*fx)*(1-fy) + (v010*(1-fx)+v110*fx)*fy) * (1-fz)
         + ((v001*(1-fx)+v101*fx)*(1-fy) + (v011*(1-fx)+v111*fx)*fy) *    fz;
}

/**
 * @brief 3D 可分离高斯平滑（原地，X→Y→Z 三次 1D 卷积）
 *
 * 对应源码：McsfAlgoAMLFilter.cpp → GaussianFilter3D
 */
inline void gaussianSmooth3D(std::vector<float>& data,
                               int w, int h, int d, float sigma) {
    if (sigma < 0.01f) return;
    int r = static_cast<int>(std::ceil(2.5f * sigma));
    if (r < 1) r = 1;
    int ksz = 2*r + 1;
    std::vector<float> k(ksz);
    float sum = 0.0f;
    for (int i = -r; i <= r; i++) { k[i+r] = std::exp(-0.5f*i*i/(sigma*sigma)); sum += k[i+r]; }
    for (auto& v : k) v /= sum;

    std::vector<float> tmp(w*h*d);

    // X
    for (int z=0;z<d;z++) for (int y=0;y<h;y++) for (int x=0;x<w;x++) {
        float s=0; for (int dd=-r;dd<=r;dd++) s += k[dd+r]*data[z*w*h+y*w+std::clamp(x+dd,0,w-1)];
        tmp[z*w*h+y*w+x] = s;
    }
    data = tmp;
    // Y
    std::fill(tmp.begin(),tmp.end(),0.f);
    for (int z=0;z<d;z++) for (int y=0;y<h;y++) for (int x=0;x<w;x++) {
        float s=0; for (int dd=-r;dd<=r;dd++) s += k[dd+r]*data[z*w*h+std::clamp(y+dd,0,h-1)*w+x];
        tmp[z*w*h+y*w+x] = s;
    }
    data = tmp;
    // Z
    std::fill(tmp.begin(),tmp.end(),0.f);
    for (int z=0;z<d;z++) for (int y=0;y<h;y++) for (int x=0;x<w;x++) {
        float s=0; for (int dd=-r;dd<=r;dd++) s += k[dd+r]*data[std::clamp(z+dd,0,d-1)*w*h+y*w+x];
        tmp[z*w*h+y*w+x] = s;
    }
    data = tmp;
}

/**
 * @brief 3D 图像梯度（中心差分）
 *
 * 对应源码：McsfAlgoRegistration.cpp → ComputeGradient3D
 */
inline void computeGradient3D(const Image3D& vol,
                               std::vector<float>& gx,
                               std::vector<float>& gy,
                               std::vector<float>& gz) {
    int w = vol.width, h = vol.height, d = vol.depth;
    int N = w*h*d;
    gx.assign(N,0.f); gy.assign(N,0.f); gz.assign(N,0.f);
    for (int z=1;z<d-1;z++) for (int y=1;y<h-1;y++) for (int x=1;x<w-1;x++) {
        int i = vol.idx(x,y,z);
        gx[i] = 0.5f*(vol.at(x+1,y,  z  ) - vol.at(x-1,y,  z  ));
        gy[i] = 0.5f*(vol.at(x,  y+1,z  ) - vol.at(x,  y-1,z  ));
        gz[i] = 0.5f*(vol.at(x,  y,  z+1) - vol.at(x,  y,  z-1));
    }
}

/** 3D 体积缩放（三线性插值） */
inline Image3D resizeVolume(const Image3D& src, int dW, int dH, int dD) {
    Image3D dst; dst.width=dW; dst.height=dH; dst.depth=dD;
    dst.spacingX=src.spacingX*src.width/dW;
    dst.spacingY=src.spacingY*src.height/dH;
    dst.spacingZ=src.spacingZ*src.depth/dD;
    dst.data.resize(dW*dH*dD);
    float sx=(float)src.width/dW, sy=(float)src.height/dH, sz=(float)src.depth/dD;
    for (int z=0;z<dD;z++) for (int y=0;y<dH;y++) for (int x=0;x<dW;x++)
        dst.data[z*dW*dH+y*dW+x] = trilinear(src, x*sx, y*sy, z*sz);
    return dst;
}

// ============================================================
// §3  变换与重采样
// ============================================================

/**
 * @brief 从 ZYX Euler 角构造 3×3 旋转矩阵  R = Rz × Ry × Rx
 */
inline void buildRotMat3D(float rx, float ry, float rz, float R[3][3]) {
    float cx=std::cos(rx), sx=std::sin(rx);
    float cy=std::cos(ry), sy=std::sin(ry);
    float cz=std::cos(rz), sz=std::sin(rz);
    R[0][0]=cy*cz;           R[0][1]=cz*sx*sy-cx*sz; R[0][2]=cx*cz*sy+sx*sz;
    R[1][0]=cy*sz;           R[1][1]=cx*cz+sx*sy*sz; R[1][2]=cx*sy*sz-cz*sx;
    R[2][0]=-sy;             R[2][1]=cy*sx;           R[2][2]=cx*cy;
}

/**
 * @brief 3D 刚性变换重采样
 *
 * 逆映射（ref→mov）：m = R^T × (p - center) - t + center
 * 对应源码：McsfAlgoRegistration.cpp → ReSampleRigid3D
 */
inline Image3D applyRigid3D(const Image3D& mov, const Image3D& ref,
                              const RigidParam3D& p) {
    float R[3][3]; buildRotMat3D(p.rx, p.ry, p.rz, R);
    float cx=ref.width*0.5f, cy=ref.height*0.5f, cz=ref.depth*0.5f;

    Image3D out; out.resize(ref.width, ref.height, ref.depth);
    out.spacingX=ref.spacingX; out.spacingY=ref.spacingY; out.spacingZ=ref.spacingZ;

    for (int z=0;z<ref.depth;z++) for (int y=0;y<ref.height;y++) for (int x=0;x<ref.width;x++) {
        float dx=(x-cx)-p.tx, dy=(y-cy)-p.ty, dz=(z-cz)-p.tz;
        float mx = cx + R[0][0]*dx + R[1][0]*dy + R[2][0]*dz;
        float my = cy + R[0][1]*dx + R[1][1]*dy + R[2][1]*dz;
        float mz = cz + R[0][2]*dx + R[1][2]*dy + R[2][2]*dz;
        out.data[out.idx(x,y,z)] = trilinear(mov, mx, my, mz);
    }
    return out;
}

/**
 * @brief 应用 3D 密集形变场
 * 对应源码：McsfAlgoDemonsResampling.cpp → ApplyDeformField3D
 */
inline Image3D applyDeform3D(const Image3D& mov, const DeformField3D& d) {
    Image3D out; out.resize(d.width, d.height, d.depth);
    out.spacingX=mov.spacingX; out.spacingY=mov.spacingY; out.spacingZ=mov.spacingZ;
    for (int z=0;z<d.depth;z++) for (int y=0;y<d.height;y++) for (int x=0;x<d.width;x++) {
        int i=d.idx(x,y,z);
        out.data[i] = trilinear(mov, x+d.dx[i], y+d.dy[i], z+d.dz[i]);
    }
    return out;
}

// ============================================================
// §4  相似度量
// ============================================================

/** 3D MSE（对应 MeanSquareErrorMetric.cpp） */
inline float computeMSE3D(const Image3D& ref, const Image3D& mov) {
    double mse=0.0; int N=ref.size();
    for (int i=0;i<N;i++) { double d=ref.data[i]-mov.data[i]; mse+=d*d; }
    return (float)(mse/N);
}

/** 3D NMI（对应 MutualInformationMetric.cpp） */
inline float computeNMI3D(const Image3D& ref, const Image3D& mov, int bins=64) {
    int N=ref.size();
    float rMin=*std::min_element(ref.data.begin(),ref.data.end());
    float rMax=*std::max_element(ref.data.begin(),ref.data.end());
    float mMin=*std::min_element(mov.data.begin(),mov.data.end());
    float mMax=*std::max_element(mov.data.begin(),mov.data.end());
    float rR=rMax-rMin+1e-8f, mR=mMax-mMin+1e-8f;
    std::vector<double> hR(bins,0),hM(bins,0),hJ(bins*bins,0);
    for (int i=0;i<N;i++) {
        int r=std::clamp((int)((ref.data[i]-rMin)/rR*bins),0,bins-1);
        int m=std::clamp((int)((mov.data[i]-mMin)/mR*bins),0,bins-1);
        hR[r]++; hM[m]++; hJ[r*bins+m]++;
    }
    double Hr=0,Hm=0,Hj=0;
    for (int i=0;i<bins;i++) {
        if (hR[i]>0) Hr-=(hR[i]/N)*std::log(hR[i]/N);
        if (hM[i]>0) Hm-=(hM[i]/N)*std::log(hM[i]/N);
    }
    for (int i=0;i<bins*bins;i++) if (hJ[i]>0) Hj-=(hJ[i]/N)*std::log(hJ[i]/N);
    return (float)((Hr+Hm)/(Hj+1e-10));
}

// ============================================================
// §5  3D 刚性配准（6-DOF 梯度下降 + 多分辨率金字塔）
// ============================================================

struct RigidRegConfig3D {
    int   maxIterations = 150;    ///< 每层最大迭代
    float maxStep       = 2.0f;   ///< 初始步长（体素）
    float minStep       = 0.005f; ///< 收敛阈值
    float relaxFactor   = 0.95f;  ///< 步长松弛系数
    bool  useNMI        = false;  ///< false=MSE，true=NMI
    int   numLevels     = 2;      ///< 多分辨率层数
    float smoothSigma   = 1.0f;   ///< 各层降采样前平滑 σ
};

static float rigidMetric3D(const Image3D& ref, const Image3D& mov,
                             const RigidParam3D& p, bool useNMI) {
    Image3D w = applyRigid3D(mov, ref, p);
    return useNMI ? -computeNMI3D(ref, w) : computeMSE3D(ref, w);
}

/**
 * @brief 单层 6-DOF 3D 刚性配准（有限差分梯度下降）
 * 对应源码：Optimization.cpp → coptimization::Optimize
 */
static RigidParam3D rigidRegSingleLevel3D(const Image3D& ref, const Image3D& mov,
                                           const RigidRegConfig3D& cfg,
                                           RigidParam3D init) {
    RigidParam3D cur = init;
    float step = cfg.maxStep;
    // 旋转梯度缩放（与平移量纲统一）
    float diagLen = std::sqrt((float)(ref.width*ref.width + ref.height*ref.height + ref.depth*ref.depth));
    const float EPS_T = 0.5f, EPS_R = 0.003f;

    for (int it=0; it<cfg.maxIterations && step>cfg.minStep; it++) {
        float base = rigidMetric3D(ref, mov, cur, cfg.useNMI);
        RigidParam3D p = cur;

        // 有限差分（前向差分，每参数 1 次额外求值）
        auto fd = [&](float& param, float eps) {
            param += eps; float r = (rigidMetric3D(ref,mov,p,cfg.useNMI)-base)/eps;
            param = (&param==&p.tx?cur.tx:(&param==&p.ty?cur.ty:
                    (&param==&p.tz?cur.tz:(&param==&p.rx?cur.rx:
                    (&param==&p.ry?cur.ry:cur.rz))))); return r;
        };

        float dTx=0,dTy=0,dTz=0,dRx=0,dRy=0,dRz=0;
        p=cur; p.tx+=EPS_T; dTx=(rigidMetric3D(ref,mov,p,cfg.useNMI)-base)/EPS_T;
        p=cur; p.ty+=EPS_T; dTy=(rigidMetric3D(ref,mov,p,cfg.useNMI)-base)/EPS_T;
        p=cur; p.tz+=EPS_T; dTz=(rigidMetric3D(ref,mov,p,cfg.useNMI)-base)/EPS_T;
        p=cur; p.rx+=EPS_R; dRx=(rigidMetric3D(ref,mov,p,cfg.useNMI)-base)/EPS_R*diagLen;
        p=cur; p.ry+=EPS_R; dRy=(rigidMetric3D(ref,mov,p,cfg.useNMI)-base)/EPS_R*diagLen;
        p=cur; p.rz+=EPS_R; dRz=(rigidMetric3D(ref,mov,p,cfg.useNMI)-base)/EPS_R*diagLen;

        float gn = std::sqrt(dTx*dTx+dTy*dTy+dTz*dTz+dRx*dRx+dRy*dRy+dRz*dRz);
        if (gn < 1e-12f) break;

        RigidParam3D next = cur;
        next.tx -= step*dTx/gn; next.ty -= step*dTy/gn; next.tz -= step*dTz/gn;
        next.rx -= step*dRx/gn/diagLen; next.ry -= step*dRy/gn/diagLen; next.rz -= step*dRz/gn/diagLen;

        if (rigidMetric3D(ref,mov,next,cfg.useNMI) < base) cur = next;
        else step *= cfg.relaxFactor;
    }
    return cur;
}

/**
 * @brief 多分辨率 3D 刚性配准（粗→细金字塔）
 * 对应源码：McsfAlgoRegistration.cpp → ImageRegistrationRigidPyramid3D
 */
inline RigidParam3D rigidRegistration3D(const Image3D& ref, const Image3D& mov,
                                         const RigidRegConfig3D& cfg = RigidRegConfig3D()) {
    int L = std::max(1, cfg.numLevels);
    std::vector<Image3D> rP(L), mP(L);
    rP[0]=ref; mP[0]=mov;
    for (int i=1;i<L;i++) {
        int nW=std::max(8,rP[i-1].width/2), nH=std::max(8,rP[i-1].height/2), nD=std::max(4,rP[i-1].depth/2);
        Image3D rS=rP[i-1], mS=mP[i-1];
        gaussianSmooth3D(rS.data,rS.width,rS.height,rS.depth,cfg.smoothSigma);
        gaussianSmooth3D(mS.data,mS.width,mS.height,mS.depth,cfg.smoothSigma);
        rP[i]=resizeVolume(rS,nW,nH,nD); mP[i]=resizeVolume(mS,nW,nH,nD);
    }
    RigidParam3D param;
    for (int lvl=L-1;lvl>=0;lvl--) {
        float sX=(float)rP[lvl].width/ref.width;
        float sY=(float)rP[lvl].height/ref.height;
        float sZ=(float)rP[lvl].depth/ref.depth;
        RigidParam3D init = param;
        init.tx*=sX; init.ty*=sY; init.tz*=sZ;
        RigidParam3D res = rigidRegSingleLevel3D(rP[lvl],mP[lvl],cfg,init);
        param = res; param.tx/=sX; param.ty/=sY; param.tz/=sZ;
    }
    return param;
}

// ============================================================
// §6  3D Demons 形变配准
// ============================================================

struct DemonsConfig3D {
    float alpha        = 1.0f;
    float sigma        = 2.0f;
    float step         = 0.4f;
    float stopCriteria = 0.001f;
    int   maxIterations = 120;
};

static void computeDemonsForce3D(const Image3D& ref, const Image3D& warped,
                                  const std::vector<float>& gx,
                                  const std::vector<float>& gy,
                                  const std::vector<float>& gz,
                                  std::vector<float>& fx,
                                  std::vector<float>& fy,
                                  std::vector<float>& fz,
                                  float alpha) {
    int N = ref.size();
    fx.resize(N); fy.resize(N); fz.resize(N);
    for (int i=0;i<N;i++) {
        float diff = ref.data[i] - warped.data[i];
        float gn2  = gx[i]*gx[i] + gy[i]*gy[i] + gz[i]*gz[i];
        float den  = gn2 + alpha*alpha*diff*diff;
        if (std::abs(den)<1e-12f) { fx[i]=fy[i]=fz[i]=0.f; continue; }
        fx[i]=diff*gx[i]/den; fy[i]=diff*gy[i]/den; fz[i]=diff*gz[i]/den;
    }
}

/**
 * @brief 3D Demons 配准
 * 对应源码：BaseDemons::DemonsRegistration（BaseDemons.cpp）
 */
inline DeformField3D demonsRegistration3D(const Image3D& ref, const Image3D& mov,
                                           const DemonsConfig3D& cfg = DemonsConfig3D()) {
    int w=ref.width, h=ref.height, d=ref.depth, N=w*h*d;
    DeformField3D disp; disp.resize(w,h,d);

    std::vector<float> gx, gy, gz;
    computeGradient3D(ref, gx, gy, gz);  // 参考图梯度固定

    std::vector<float> fx(N), fy(N), fz(N);
    float prevErr = 1e30f;

    for (int it=0;it<cfg.maxIterations;it++) {
        Image3D warped = applyDeform3D(mov, disp);
        computeDemonsForce3D(ref, warped, gx, gy, gz, fx, fy, fz, cfg.alpha);

        gaussianSmooth3D(fx, w,h,d, cfg.sigma);
        gaussianSmooth3D(fy, w,h,d, cfg.sigma);
        gaussianSmooth3D(fz, w,h,d, cfg.sigma);

        for (int i=0;i<N;i++) {
            disp.dx[i]+=cfg.step*fx[i];
            disp.dy[i]+=cfg.step*fy[i];
            disp.dz[i]+=cfg.step*fz[i];
        }
        gaussianSmooth3D(disp.dx, w,h,d, cfg.sigma);
        gaussianSmooth3D(disp.dy, w,h,d, cfg.sigma);
        gaussianSmooth3D(disp.dz, w,h,d, cfg.sigma);

        float err = computeMSE3D(ref, warped);
        if (it>5 && std::abs(prevErr-err)<cfg.stopCriteria) break;
        prevErr = err;
    }
    return disp;
}

// ============================================================
// §7  3D Multi-Grid Demons
// ============================================================

/**
 * @brief 多格网 3D Demons
 * 对应源码：MultiGridDemons::DoRegistration（MultiGridDemons.cpp）
 */
inline DeformField3D multiGridDemons3D(const Image3D& ref, const Image3D& mov,
                                        int maxDepth=3,
                                        const DemonsConfig3D& cfg=DemonsConfig3D()) {
    if (maxDepth<=1 || ref.width<=12 || ref.height<=12 || ref.depth<=6)
        return demonsRegistration3D(ref, mov, cfg);

    int nW=ref.width/2, nH=ref.height/2, nD=std::max(4,ref.depth/2);
    Image3D rS=resizeVolume(ref,nW,nH,nD), mS=resizeVolume(mov,nW,nH,nD);

    DemonsConfig3D cc=cfg; cc.maxIterations=std::max(20,cfg.maxIterations/2);
    DeformField3D coarse = multiGridDemons3D(rS, mS, maxDepth-1, cc);

    float sX=(float)ref.width/nW, sY=(float)ref.height/nH, sZ=(float)ref.depth/nD;
    DeformField3D up; up.resize(ref.width, ref.height, ref.depth);
    for (int z=0;z<ref.depth;z++) for (int y=0;y<ref.height;y++) for (int x=0;x<ref.width;x++) {
        int xi=std::clamp((int)(x/sX),0,nW-1);
        int yi=std::clamp((int)(y/sY),0,nH-1);
        int zi=std::clamp((int)(z/sZ),0,nD-1);
        int dst=up.idx(x,y,z), src=zi*nW*nH+yi*nW+xi;
        up.dx[dst]=coarse.dx[src]*sX; up.dy[dst]=coarse.dy[src]*sY; up.dz[dst]=coarse.dz[src]*sZ;
    }

    Image3D preWarped = applyDeform3D(mov, up);
    DeformField3D fine = demonsRegistration3D(ref, preWarped, cfg);

    DeformField3D out; out.resize(ref.width,ref.height,ref.depth);
    int N=out.size();
    for (int i=0;i<N;i++) {
        out.dx[i]=up.dx[i]+fine.dx[i];
        out.dy[i]=up.dy[i]+fine.dy[i];
        out.dz[i]=up.dz[i]+fine.dz[i];
    }
    return out;
}

// ============================================================
// §8  3D LCC Demons（局部互相关）
// ============================================================

struct LCCConfig3D {
    float sigma        = 3.0f;
    float smoothSigma  = 2.0f;
    float step         = 0.06f;
    float stopCriteria = 1e-5f;
    int   maxIterations = 60;
    bool  relax         = true;
};

/**
 * @brief 3D LCC 驱动力
 * 对应源码：LCC_Force 类（McsfAlgoLCCDemons.cpp）
 */
static void computeLCCForce3D(const Image3D& ref, const Image3D& warped,
                                std::vector<float>& fx,
                                std::vector<float>& fy,
                                std::vector<float>& fz,
                                float sigma) {
    int w=ref.width, h=ref.height, d=ref.depth, N=w*h*d;
    std::vector<float> muR=ref.data, muM=warped.data, cross(N), ref2(N);
    for (int i=0;i<N;i++) { cross[i]=ref.data[i]*warped.data[i]; ref2[i]=ref.data[i]*ref.data[i]; }
    gaussianSmooth3D(muR,  w,h,d,sigma); gaussianSmooth3D(muM,  w,h,d,sigma);
    gaussianSmooth3D(cross,w,h,d,sigma); gaussianSmooth3D(ref2, w,h,d,sigma);

    std::vector<float> gx, gy, gz;
    computeGradient3D(ref, gx, gy, gz);

    fx.assign(N,0.f); fy.assign(N,0.f); fz.assign(N,0.f);
    for (int i=0;i<N;i++) {
        float Erm = cross[i] - muR[i]*muM[i];
        float Er2 = ref2[i]  - muR[i]*muR[i];
        if (Er2 < 1e-8f) continue;
        float coeff = 2.f*Erm/(Er2+1e-8f);
        float dLCC  = coeff*(warped.data[i]-muM[i]);
        fx[i]=-dLCC*gx[i]; fy[i]=-dLCC*gy[i]; fz[i]=-dLCC*gz[i];
    }
}

/**
 * @brief 3D LCC Demons 配准
 * 对应源码：LCCDemons::DemonsRegistration（McsfAlgoLCCDemons.cpp）
 */
inline DeformField3D lccDemonsRegistration3D(const Image3D& ref, const Image3D& mov,
                                              const LCCConfig3D& cfg=LCCConfig3D()) {
    int w=ref.width, h=ref.height, d=ref.depth, N=w*h*d;
    DeformField3D disp; disp.resize(w,h,d);
    std::vector<float> fx, fy, fz;
    float prevErr=1e30f, curStep=cfg.step; int divCnt=0;

    for (int it=0;it<cfg.maxIterations;it++) {
        Image3D warped = applyDeform3D(mov, disp);
        computeLCCForce3D(ref, warped, fx, fy, fz, cfg.sigma);
        gaussianSmooth3D(fx,w,h,d,cfg.smoothSigma);
        gaussianSmooth3D(fy,w,h,d,cfg.smoothSigma);
        gaussianSmooth3D(fz,w,h,d,cfg.smoothSigma);
        for (int i=0;i<N;i++) { disp.dx[i]+=curStep*fx[i]; disp.dy[i]+=curStep*fy[i]; disp.dz[i]+=curStep*fz[i]; }
        gaussianSmooth3D(disp.dx,w,h,d,cfg.smoothSigma);
        gaussianSmooth3D(disp.dy,w,h,d,cfg.smoothSigma);
        gaussianSmooth3D(disp.dz,w,h,d,cfg.smoothSigma);
        float err = computeMSE3D(ref, warped);
        if (it>0) {
            if (cfg.relax && err>prevErr) { curStep*=0.5f; if (++divCnt>5) break; }
            else divCnt=0;
            if (std::abs(prevErr-err)<cfg.stopCriteria) break;
        }
        prevErr=err;
    }
    return disp;
}

// ============================================================
// §9  3D B-Spline FFD（解析梯度）
// ============================================================

static inline float bsplineBasis(float t, int i) {
    switch (i) {
        case -1: { float s=1.f-t; return s*s*s/6.f; }
        case  0: return (3.f*t*t*t - 6.f*t*t + 4.f)/6.f;
        case  1: return (-3.f*t*t*t + 3.f*t*t + 3.f*t + 1.f)/6.f;
        case  2: return t*t*t/6.f;
        default: return 0.f;
    }
}

struct BSplineGrid3D {
    std::vector<float> cx, cy, cz;
    int gridW=0, gridH=0, gridD=0;
    int imgW=0, imgH=0, imgD=0;

    float cellW() const { return (float)imgW/std::max(1,gridW-3); }
    float cellH() const { return (float)imgH/std::max(1,gridH-3); }
    float cellD() const { return (float)imgD/std::max(1,gridD-3); }
    int gIdx(int i,int j,int k) const { return k*gridW*gridH+j*gridW+i; }
    int gN() const { return gridW*gridH*gridD; }

    void getDisplacement(float px, float py, float pz, float& dx, float& dy, float& dz) const {
        float u=px/cellW(), v=py/cellH(), w=pz/cellD();
        int iu=(int)u+1, iv=(int)v+1, iw=(int)w+1;
        float fu=u-std::floor(u), fv=v-std::floor(v), fw=w-std::floor(w);
        dx=dy=dz=0.f;
        for (int dk=-1;dk<=2;dk++) {
            float bk=bsplineBasis(fw,dk);
            for (int dj=-1;dj<=2;dj++) {
                float bjk=bsplineBasis(fv,dj)*bk;
                for (int di=-1;di<=2;di++) {
                    float wt=bsplineBasis(fu,di)*bjk;
                    int gi=std::clamp(iu+di,0,gridW-1);
                    int gj=std::clamp(iv+dj,0,gridH-1);
                    int gk=std::clamp(iw+dk,0,gridD-1);
                    int idx=gIdx(gi,gj,gk);
                    dx+=wt*cx[idx]; dy+=wt*cy[idx]; dz+=wt*cz[idx];
                }
            }
        }
    }

    DeformField3D toDeformField3D() const {
        DeformField3D d; d.resize(imgW,imgH,imgD);
        for (int z=0;z<imgD;z++) for (int y=0;y<imgH;y++) for (int x=0;x<imgW;x++) {
            float dx,dy,dz; getDisplacement((float)x,(float)y,(float)z,dx,dy,dz);
            int i=d.idx(x,y,z); d.dx[i]=dx; d.dy[i]=dy; d.dz[i]=dz;
        }
        return d;
    }
};

struct FFDConfig3D {
    int   gridSpacing  = 8;
    float stepSize     = 0.5f;
    float stopCriteria = 1e-5f;
    int   maxIterations = 40;
};

/**
 * @brief 3D B-Spline FFD（解析梯度）
 *
 * dMSE/dc_{k,dim} = (2/N) × Σ_x residual(x) × ∇warped_dim(x) × W_k(x)
 * 对应源码：McsfAlgoMultiGridFFD.cpp 梯度计算部分
 */
inline DeformField3D bsplineFFD3D(const Image3D& ref, const Image3D& mov,
                                   const FFDConfig3D& cfg=FFDConfig3D()) {
    int W=ref.width, H=ref.height, D=ref.depth, N=W*H*D;
    BSplineGrid3D grid;
    grid.imgW=W; grid.imgH=H; grid.imgD=D;
    grid.gridW=W/cfg.gridSpacing+4;
    grid.gridH=H/cfg.gridSpacing+4;
    grid.gridD=D/cfg.gridSpacing+4;
    int gN=grid.gN();
    grid.cx.assign(gN,0.f); grid.cy.assign(gN,0.f); grid.cz.assign(gN,0.f);

    float prevMSE=1e30f;
    float cW=grid.cellW(), cH=grid.cellH(), cD=grid.cellD();

    for (int it=0;it<cfg.maxIterations;it++) {
        DeformField3D disp = grid.toDeformField3D();
        Image3D warped = applyDeform3D(mov, disp);
        float mse = computeMSE3D(ref, warped);
        if (it>0 && std::abs(prevMSE-mse)<cfg.stopCriteria) break;
        prevMSE=mse;

        std::vector<float> gx, gy, gz;
        computeGradient3D(warped, gx, gy, gz);

        std::vector<float> gX(gN,0.f), gY(gN,0.f), gZ(gN,0.f);
        for (int z=0;z<D;z++) for (int y=0;y<H;y++) for (int x=0;x<W;x++) {
            int vi=disp.idx(x,y,z);
            float res=warped.data[vi]-ref.data[vi];
            float u=x/cW, v=y/cH, w=z/cD;
            int iu=(int)u+1, iv=(int)v+1, iw=(int)w+1;
            float fu=u-std::floor(u), fv=v-std::floor(v), fw=w-std::floor(w);
            for (int dk=-1;dk<=2;dk++) { float bk=bsplineBasis(fw,dk);
            for (int dj=-1;dj<=2;dj++) { float bjk=bsplineBasis(fv,dj)*bk;
            for (int di=-1;di<=2;di++) {
                float wt=bsplineBasis(fu,di)*bjk;
                int gi=std::clamp(iu+di,0,grid.gridW-1);
                int gj=std::clamp(iv+dj,0,grid.gridH-1);
                int gk=std::clamp(iw+dk,0,grid.gridD-1);
                int ci=grid.gIdx(gi,gj,gk);
                gX[ci]+=res*gx[vi]*wt; gY[ci]+=res*gy[vi]*wt; gZ[ci]+=res*gz[vi]*wt;
            }}}
        }
        float sc=2.f/N;
        for (int i=0;i<gN;i++) { grid.cx[i]-=cfg.stepSize*sc*gX[i]; grid.cy[i]-=cfg.stepSize*sc*gY[i]; grid.cz[i]-=cfg.stepSize*sc*gZ[i]; }
    }
    return grid.toDeformField3D();
}

// ============================================================
// §10  3D Phantom 测试数据生成
// ============================================================

/**
 * @brief 3D Shepp-Logan 风格 CT Phantom（多个椭球叠加）
 */
inline Image3D createPhantom3D(int W, int H, int D, float noiseSigma=0.015f) {
    Image3D img; img.resize(W,H,D);
    img.spacingX=img.spacingY=1.f; img.spacingZ=1.5f;
    float cx=W*0.5f, cy=H*0.5f, cz=D*0.5f;

    auto drawEllipsoid = [&](float ex,float ey,float ez,
                              float rx,float ry,float rz, float val) {
        for (int z=0;z<D;z++) for (int y=0;y<H;y++) for (int x=0;x<W;x++) {
            float dx=(x-ex)/rx, dy=(y-ey)/ry, dz=(z-ez)/rz;
            if (dx*dx+dy*dy+dz*dz<1.f) img.at(x,y,z)=val;
        }
    };

    // 从外到内叠加（模拟 Shepp-Logan 椭球体）
    drawEllipsoid(cx,  cy,        cz,        W*.45f,H*.36f,D*.45f, 0.20f); // 外轮廓
    drawEllipsoid(cx,  cy,        cz,        W*.38f,H*.30f,D*.38f, 0.55f); // 软组织
    drawEllipsoid(cx,  cy-.08f*H, cz,        W*.25f,H*.22f,D*.25f, 0.35f); // 脑室
    drawEllipsoid(cx-W*.16f,cy-.10f*H,cz,   W*.08f,H*.12f,D*.08f, 0.85f); // 左结构
    drawEllipsoid(cx+W*.16f,cy-.10f*H,cz,   W*.08f,H*.12f,D*.08f, 0.85f); // 右结构
    drawEllipsoid(cx, cy+H*.15f,  cz,        W*.06f,H*.06f,D*.06f, 1.00f); // 高亮灶
    drawEllipsoid(cx+W*.06f,cy+H*.10f,cz,   W*.04f,H*.04f,D*.04f, 0.10f); // 暗区（CSF）
    drawEllipsoid(cx, cy-H*.25f,  cz+D*.10f, W*.03f,H*.04f,D*.03f, 0.90f);
    drawEllipsoid(cx+W*.08f,cy-.05f*H,cz-.10f*D,W*.025f,H*.03f,D*.025f,0.70f);

    if (noiseSigma > 0.f) {
        unsigned seed=12345u;
        for (auto& v : img.data) {
            seed=seed*1664525u+1013904223u;
            float u1=((seed>>8)&0xFFFF)/65536.f+1e-6f;
            seed=seed*1664525u+1013904223u;
            float u2=((seed>>8)&0xFFFF)/65536.f;
            v=std::clamp(v+noiseSigma*std::sqrt(-2.f*std::log(u1))*std::cos(6.2832f*u2),0.f,1.f);
        }
    }
    return img;
}

/**
 * @brief 生成 3D 配准测试运动图像（施加已知 6-DOF 刚性 + 局部形变）
 */
inline Image3D createMovingImage3D(const Image3D& ref,
                                    float tx=5.f, float ty=4.f, float tz=2.f,
                                    float rx=.04f,float ry=.03f,float rz=.06f,
                                    float deformScale=2.5f) {
    RigidParam3D p; p.tx=tx;p.ty=ty;p.tz=tz;p.rx=rx;p.ry=ry;p.rz=rz;
    Image3D moved = applyRigid3D(ref, ref, p);

    if (deformScale>0.f) {
        int W=moved.width,H=moved.height,D=moved.depth;
        DeformField3D def; def.resize(W,H,D);
        const float pi=3.14159265f;
        for (int z=0;z<D;z++) for (int y=0;y<H;y++) for (int x=0;x<W;x++) {
            float nx=pi*x/W, ny=pi*y/H, nz=pi*z/D;
            int i=def.idx(x,y,z);
            def.dx[i]=deformScale*std::sin(2*ny)*std::cos(nx+nz);
            def.dy[i]=deformScale*std::sin(nx)*std::cos(2*ny);
            def.dz[i]=deformScale*std::cos(nx+ny)*std::sin(2*nz);
        }
        Image3D tmp=moved; moved=applyDeform3D(tmp,def);
    }
    return moved;
}

} // namespace MedReg
