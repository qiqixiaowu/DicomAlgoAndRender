#pragma once
/**
 * @file pet_suv_unity_filter.h
 * @brief PET SUV 跨设备归一化滤波（EARL Unity Filter）
 *
 * 背景
 * ────
 * 不同厂商 PET 设备、重建算法、迭代次数差异，导致同一病灶在不同设备上
 * 测得的 SUVmax 可相差 20%~40%。为实现多中心临床试验、纵向随访等
 * 跨设备可比性，EARL（European ALARA Research Ltd）提出：
 *
 *   将所有 PET 图像统一平滑至相同的有效分辨率（目标 FWHM），
 *   从而消除设备/重建差异，使 SUV 测量结果可比。
 *
 * 原理
 * ────
 * 设备输出图像的等效点扩散函数（PSF）已经有某个固有分辨率 FWHM_scanner，
 * 目标分辨率为 FWHM_target（通常 EARL 标准为 8mm 或 6mm）。
 * 需要额外施加一个高斯滤波，其 FWHM 满足：
 *
 *   FWHM_additional = sqrt(FWHM_target² - FWHM_scanner²)
 *
 * 本接口传入的 FWHM 即为调用方计算好的 FWHM_additional（或直接指定
 * 目标分辨率，由调用方自行决定语义），对图像施加各向同性 3D 高斯卷积。
 *
 * 实现
 * ────
 * 3D 可分离高斯滤波（Separable Gaussian）：
 *   1. 将 FWHM (mm) 转换为各轴体素空间 σ：
 *        σ_mm   = FWHM / (2√(2ln2))  ≈  FWHM / 2.3548
 *        σ_vox[i] = σ_mm / spacing[i]
 *   2. 对 X、Y、Z 三轴依次做 1D 高斯卷积（利用可分离性，O(N·r) 而非 O(N·r³)）
 *   3. 边界处理：镜像延拓（Mirror Padding），避免边缘信号衰减
 *
 * 时间复杂度：O(3 × N × kernel_radius)，N 为总体素数
 * 典型参数：FWHM=8mm, spacing=4mm → σ_vox≈0.85, radius=3，极快
 */

// ── DLL 导出宏 ────────────────────────────────────────────────────────────────
#ifndef _MCSF_ALGO_MIONC_SEGMENTATION_API_
#  if defined(_WIN32) || defined(_WIN64)
#    ifdef MCSF_ALGO_MIONC_SEGMENTATION_EXPORTS
#      define _MCSF_ALGO_MIONC_SEGMENTATION_API_ __declspec(dllexport)
#    else
#      define _MCSF_ALGO_MIONC_SEGMENTATION_API_ __declspec(dllimport)
#    endif
#  else
#    define _MCSF_ALGO_MIONC_SEGMENTATION_API_
#  endif
#endif

/**
 * @brief 进度/取消接口
 *
 * 调用方可继承此接口实现进度条或取消请求。
 * 若不需要进度反馈，传 nullptr 即可。
 */
struct IProgress {
    /**
     * @brief 报告进度
     * @param progress 当前进度，范围 [dProgressStart, dProgressEnd]
     */
    virtual void SetProgress(double progress) = 0;

    /**
     * @brief 查询是否已请求取消
     * @return true 表示调用方希望中止，算法应尽快返回 false
     */
    virtual bool IsCancelled() const { return false; }

    virtual ~IProgress() = default;
};

/**
 * @brief PET SUV 跨设备归一化高斯滤波（EARL Unity Filter）
 *
 * @param pImage         [in]  原始 PET 图像，unsigned short 线性排列（z 轴慢轴）
 *                              像素值为原始灰度（未转换 SUV），
 *                              线性索引 = z * iSize[1] * iSize[0] + y * iSize[0] + x
 * @param iSize          [in]  图像尺寸 [X, Y, Z]，体素数（均须 > 0）
 * @param dSpacing       [in]  体素间距 [X, Y, Z]，单位 mm（均须 > 0）
 * @param FWHM           [in]  目标高斯滤波 FWHM，单位 mm
 *                              FWHM=0 时跳过滤波直接输出（merely copy）
 *                              FWHM<0 时返回 false
 * @param pOutImage      [out] 滤波后图像，double 类型，调用方负责分配
 *                              大小须 >= iSize[0] * iSize[1] * iSize[2]
 * @param pProgress      [in]  进度/取消接口，可为 nullptr
 * @param dProgressStart [in]  进度起始值（默认 0.0）
 * @param dProgressEnd   [in]  进度终止值（默认 1.0）
 * @return true  成功
 * @return false 参数非法（空指针/尺寸<=0/间距<=0/FWHM<0）或被取消
 *
 * @note 输出值类型为 double，以保存滤波后的亚像素精度，
 *       调用方可按需转换回 SUV 或重新映射到 unsigned short。
 * @note 线程安全：函数内部无全局状态，多线程并发调用不同图像是安全的。
 */
_MCSF_ALGO_MIONC_SEGMENTATION_API_ bool McsfAlgoMMOncUnityFilter(
    const unsigned short* pImage,
    const int             iSize[3],
    const double          dSpacing[3],
    const double&         FWHM,
    double*               pOutImage,
    IProgress*            pProgress      = nullptr,
    double                dProgressStart = 0.0,
    double                dProgressEnd   = 1.0);
