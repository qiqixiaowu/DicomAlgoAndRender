#!/usr/bin/env python3
"""
PET 原始数据标准差 —— 两种计算方式对比验证

方式A（错误）：只用原始 y 的 sum / square_sum，乘斜率，除以 n
方式B（正确）：用斜率+截距算每个点拟合值 → 残差 → 残差标准差（除以 n-2）

用法：
    python pet_std_compare.py
"""

import numpy as np


# ============================================================
# 1. 模拟一组 PET 数据（Patlak / Logan 风格）
# ============================================================
np.random.seed(50)

n = 12  # 时间点个数

# x：Patlak 图的横轴（∫Cp/Cp 之类），单调递增
x = np.linspace(0.5, 6.0, n)

# 真实模型：y = intercept + slope * x + 噪声
true_intercept = 0.35
true_slope = 0.18
noise_std = 0.04

noise = np.random.normal(0, noise_std, size=n)
y = true_intercept + true_slope * x + noise

# ============================================================
# 2. 用最小二乘拟合得到斜率和截距（模拟你已经有的参数）
# ============================================================
slope, intercept = np.polyfit(x, y, 1)  # 1 次多项式
print("=" * 60)
print(f"拟合结果：斜率 = {slope:.6f}，截距 = {intercept:.6f}")
print("=" * 60)

# ============================================================
# 3. 方式A（❌ 错误方式，对应用户的 C++ 代码逻辑）
# ============================================================
sum_y = np.sum(y)
sum_sq_y = np.sum(y * y)

# 用户 C++ 代码的等价 Python 写法：
# sqrt(sum_y² * n - sum_y^2) * slope / n
# 注意：这里用 np.sqrt 里套的是方差的分子部分
var_raw_times_n = sum_sq_y * n - sum_y * sum_y
std_raw = np.sqrt(var_raw_times_n) * slope / n

# 也可以写成更直观的形式：原始 y 的 SD × 斜率
std_y = np.std(y, ddof=0)          # 总体 SD（除以 n）
std_y_ddof1 = np.std(y, ddof=1)    # 样本 SD（除以 n-1）
std_A = std_y * slope               # 等价于用户那行代码

print("\n【方式A ❌ 错误方式】")
print(f"  ∑y        = {sum_y:.6f}")
print(f"  ∑y²       = {sum_sq_y:.6f}")
print(f"  std(y)    = {std_y:.6f}  （原始 y 的总体 SD）")
print(f"  std(y)*slope = {std_A:.6f}")
print(f"  原 C++ 等价  = {std_raw:.6f}")

# ============================================================
# 4. 方式B（✅ 正确方式：拟合值 → 残差 → 残差标准差）
# ============================================================
y_hat = intercept + slope * x          # 每个点的拟合值
residuals = y - y_hat                   # 残差
ss_res = np.sum(residuals ** 2)         # 残差平方和
dof = n - 2                             # 自由度

residual_std = np.sqrt(ss_res / dof)    # 残差标准差
residual_std_n = np.sqrt(ss_res / n)    # 如果坚持用 n 而不是 n-2

# 用 numpy 自带回归也能交叉验证
from numpy.polynomial import polynomial as P
fit = np.polyfit(x, y, 1, full=True)
residuals_np = fit[1][0] if len(fit) > 1 else None

print("\n【方式B ✅ 正确方式】")
print(f"  拟合值 y_hat = {y_hat}")
print(f"  残差        = {residuals}")
print(f"  残差平方和  = {ss_res:.8f}")
print(f"  自由度      = n - 2 = {dof}")
print(f"  残差 SD (÷n-2) = {residual_std:.6f}")
print(f"  残差 SD (÷n)   = {residual_std_n:.6f}")

if residuals_np is not None:
    print(f"  numpy full fit 残差验证 = {residuals_np:.8f}")

# ============================================================
# 5. 对比总结
# ============================================================
print("\n" + "=" * 60)
print("对比总结")
print("=" * 60)
print(f"  方式A（原始SD × 斜率） : {std_A:.6f}")
print(f"  方式B（残差SD ÷ n-2）  : {residual_std:.6f}")
print(f"  两者比值              : {std_A / residual_std:.4f}")
print()
print("结论：")
print("  方式A 没有用到截距，也没有减拟合值，只是对原始 y")
print("  做总体方差再乘斜率 —— 在 PET 线性分析中无意义。")
print("  方式B 才是正确的残差标准差。")
print("=" * 60)

# ============================================================
# 6. 额外验证：如果截距 = 0 会怎样？
# ============================================================
print("\n【补充验证：强制过原点（截距=0）】")
slope_force = np.sum(x * y) / np.sum(x * x)
y_hat_force = slope_force * x
res_force = y - y_hat_force
std_force = np.sqrt(np.sum(res_force ** 2) / (n - 1))  # 此时只估 1 个参数

print(f"  强制过原点斜率 = {slope_force:.6f}")
print(f"  残差 SD (÷n-1) = {std_force:.6f}")
print("  （此情况下方式A 和方式B 也不会相等，因为方式A 仍")
print("   没有做正确的残差计算）")
