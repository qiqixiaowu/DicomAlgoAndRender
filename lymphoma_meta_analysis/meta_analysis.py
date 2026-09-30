#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Meta 分析统计模块
=================
对清洗后的淋巴瘤用药数据进行 meta 分析，包括：

1. 描述性统计
   - 文献特征汇总表
   - 亚组分布

2. 疗效合并分析
   - ORR (总缓解率) 合并 — 比例 meta 分析 (Freeman-Tukey 双反正弦变换)
   - CR (完全缓解率) 合并
   - 中位 PFS / OS 汇总描述

3. 亚组分析
   - 按淋巴瘤亚型
   - 按药物/疗法
   - 按研究类型 (RCT vs 非RCT)
   - 按发表年份

4. 异质性检验
   - Cochran's Q 检验
   - I² 统计量

5. 发表偏倚检验
   - 漏斗图
   - Egger's test (回归法)

6. 敏感性分析
   - 逐一剔除法 (leave-one-out)
"""

import os
import sys
import json
import math
import warnings
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _get_output_dir() -> str:
    """获取输出目录 — PyInstaller 打包后使用 exe 所在目录"""
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "results")


# ============================================================
# 比例 Meta 分析 (Freeman-Tukey 双反正弦变换)
# ============================================================

def ft_transform(p: float, n: int) -> float:
    """
    Freeman-Tukey 双反正弦变换
    将比例 p 转换为近似正态分布的变量

    t = arcsin(sqrt(x/(n+1))) + arcsin(sqrt((x+1)/(n+1)))
    其中 x = p * n
    """
    x = p * n
    # 确保参数在 asin 的定义域 [0, 1] 内
    a = min(max(x / (n + 1), 0.0), 1.0)
    b = min(max((x + 1) / (n + 1), 0.0), 1.0)
    t = math.asin(math.sqrt(a)) + math.asin(math.sqrt(b))
    return t


def ft_back_transform(t: float) -> float:
    """Freeman-Tukey 逆变换，将变换值转回比例"""
    p = (math.sin(t / 2)) ** 2
    return p


def proportion_meta_analysis(
    proportions: List[float],
    sample_sizes: List[int],
    study_names: List[str] = None,
    random_effects: bool = True,
) -> Dict:
    """
    比例 meta 分析（固定/随机效应模型）

    Parameters
    ----------
    proportions : list[float]
        各研究的比例（如 ORR 0.85 表示 85%）
    sample_sizes : list[int]
        各研究样本量
    study_names : list[str]
        研究名称
    random_effects : bool
        是否使用随机效应模型 (DerSimonian-Laird)

    Returns
    -------
    dict : 包含合并比例、CI、异质性等
    """
    k = len(proportions)
    if k == 0:
        return {"error": "无数据"}

    if study_names is None:
        study_names = [f"Study {i+1}" for i in range(k)]

    # Freeman-Tukey 变换
    t_values = []
    variances = []
    weights_fixed = []

    for p, n in zip(proportions, sample_sizes):
        p = min(max(p, 0.0), 1.0)  # 裁剪到 [0, 1]
        n = max(n, 1)
        t = ft_transform(p, n)
        var = 1.0 / (n + 1)  # 变换后方差
        t_values.append(t)
        variances.append(var)
        weights_fixed.append(1.0 / var)

    t_values = np.array(t_values)
    variances = np.array(variances)
    weights_fixed = np.array(weights_fixed)

    # === 固定效应合并 ===
    w_sum = weights_fixed.sum()
    t_pooled_fixed = (weights_fixed * t_values).sum() / w_sum
    se_fixed = math.sqrt(1.0 / w_sum)
    p_pooled_fixed = ft_back_transform(t_pooled_fixed)
    ci_lower_fixed = ft_back_transform(t_pooled_fixed - 1.96 * se_fixed)
    ci_upper_fixed = ft_back_transform(t_pooled_fixed + 1.96 * se_fixed)

    # === 异质性检验 ===
    Q = (weights_fixed * (t_values - t_pooled_fixed) ** 2).sum()
    df_Q = k - 1
    p_heterogeneity = 1.0 - stats.chi2.cdf(Q, df_Q) if df_Q > 0 else 1.0

    # I² 统计量
    if Q > df_Q and df_Q > 0:
        I2 = max(0.0, (Q - df_Q) / Q * 100)
    else:
        I2 = 0.0

    # === 随机效应模型 (DerSimonian-Laird) ===
    if random_effects and k > 1:
        tau2 = max(0.0, (Q - df_Q) / (w_sum - (weights_fixed ** 2).sum() / w_sum))
        weights_random = 1.0 / (variances + tau2)
        w_sum_r = weights_random.sum()
        t_pooled_random = (weights_random * t_values).sum() / w_sum_r
        se_random = math.sqrt(1.0 / w_sum_r)
        p_pooled_random = ft_back_transform(t_pooled_random)
        ci_lower_random = ft_back_transform(t_pooled_random - 1.96 * se_random)
        ci_upper_random = ft_back_transform(t_pooled_random + 1.96 * se_random)
    else:
        tau2 = 0.0
        p_pooled_random = p_pooled_fixed
        ci_lower_random = ci_lower_fixed
        ci_upper_random = ci_upper_fixed
        se_random = se_fixed

    # === 各研究详情 ===
    studies = []
    for i in range(k):
        se_i = math.sqrt(variances[i])
        studies.append({
            "study": study_names[i],
            "proportion": proportions[i],
            "sample_size": sample_sizes[i],
            "weight_fixed": weights_fixed[i] / w_sum * 100,
            "weight_random": (1.0 / (variances[i] + tau2)) / (sum(1.0 / (v + tau2) for v in variances)) * 100 if random_effects else weights_fixed[i] / w_sum * 100,
            "ci_lower": ft_back_transform(t_values[i] - 1.96 * se_i),
            "ci_upper": ft_back_transform(t_values[i] + 1.96 * se_i),
        })

    return {
        "k": k,
        "pooled_proportion_fixed": p_pooled_fixed,
        "ci_fixed": (ci_lower_fixed, ci_upper_fixed),
        "pooled_proportion_random": p_pooled_random,
        "ci_random": (ci_lower_random, ci_upper_random),
        "Q": Q,
        "df_Q": df_Q,
        "p_heterogeneity": p_heterogeneity,
        "I2": I2,
        "tau2": tau2,
        "studies": studies,
        "model": "random" if random_effects else "fixed",
    }


# ============================================================
# HR Meta 分析 (通用效应量合并)
# ============================================================

def hr_meta_analysis(
    hr_values: List[float],
    hr_ci_lower: List[float],
    hr_ci_upper: List[float],
    study_names: List[str] = None,
    random_effects: bool = True,
) -> Dict:
    """
    风险比 (HR) meta 分析 — 对数变换后合并

    Parameters
    ----------
    hr_values : list[float]
        HR 点估计值
    hr_ci_lower, hr_ci_upper : list[float]
        95% CI 上下界
    study_names : list[str]
        研究名称

    Returns
    -------
    dict : 合并 HR、CI、异质性等
    """
    k = len(hr_values)
    if k == 0:
        return {"error": "无数据"}

    if study_names is None:
        study_names = [f"Study {i+1}" for i in range(k)]

    # 对数变换
    log_hr = np.log(np.array(hr_values))
    se = (np.log(np.array(hr_ci_upper)) - np.log(np.array(hr_ci_lower))) / (2 * 1.96)
    se = np.maximum(se, 1e-6)  # 避免除零

    weights_fixed = 1.0 / se ** 2

    # 固定效应
    w_sum = weights_fixed.sum()
    log_hr_pooled_fixed = (weights_fixed * log_hr).sum() / w_sum
    se_fixed = math.sqrt(1.0 / w_sum)

    # 异质性
    Q = (weights_fixed * (log_hr - log_hr_pooled_fixed) ** 2).sum()
    df_Q = k - 1
    p_heterogeneity = 1.0 - stats.chi2.cdf(Q, df_Q) if df_Q > 0 else 1.0
    I2 = max(0.0, (Q - df_Q) / Q * 100) if Q > df_Q and df_Q > 0 else 0.0

    # 随机效应
    if random_effects and k > 1:
        tau2 = max(0.0, (Q - df_Q) / (w_sum - (weights_fixed ** 2).sum() / w_sum))
        weights_random = 1.0 / (se ** 2 + tau2)
        w_sum_r = weights_random.sum()
        log_hr_pooled_random = (weights_random * log_hr).sum() / w_sum_r
        se_random = math.sqrt(1.0 / w_sum_r)
    else:
        tau2 = 0.0
        log_hr_pooled_random = log_hr_pooled_fixed
        se_random = se_fixed

    hr_pooled = math.exp(log_hr_pooled_random)
    hr_ci_lo = math.exp(log_hr_pooled_random - 1.96 * se_random)
    hr_ci_hi = math.exp(log_hr_pooled_random + 1.96 * se_random)

    # z 检验
    z = log_hr_pooled_random / se_random if se_random > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    studies = []
    for i in range(k):
        studies.append({
            "study": study_names[i],
            "hr": hr_values[i],
            "ci_lower": hr_ci_lower[i],
            "ci_upper": hr_ci_upper[i],
            "log_hr": log_hr[i],
            "se": se[i],
            "weight": weights_fixed[i] / w_sum * 100,
        })

    return {
        "k": k,
        "pooled_hr": hr_pooled,
        "ci": (hr_ci_lo, hr_ci_hi),
        "z": z,
        "p_value": p_value,
        "Q": Q,
        "df_Q": df_Q,
        "p_heterogeneity": p_heterogeneity,
        "I2": I2,
        "tau2": tau2,
        "studies": studies,
        "model": "random" if random_effects else "fixed",
    }


# ============================================================
# 发表偏倚检验
# ============================================================

def eggers_test(effect_sizes: np.ndarray, se: np.ndarray) -> Dict:
    """
    Egger's test — 回归法检验发表偏倚

    原理: 以效应量/SE 为因变量，1/SE 为自变量做线性回归
    截距显著偏离 0 → 存在发表偏倚
    """
    valid = ~np.isnan(effect_sizes) & ~np.isnan(se) & (se > 0)
    if valid.sum() < 3:
        return {"error": "数据不足（至少需要3个研究）"}

    y = effect_sizes[valid] / se[valid]
    x = 1.0 / se[valid]

    # 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # 截距的 t 检验
    t_stat = intercept / std_err if std_err > 0 else 0
    df = len(x) - 2
    p_intercept = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    return {
        "intercept": intercept,
        "intercept_se": std_err,
        "t_stat": t_stat,
        "p_value": p_intercept,
        "bias_detected": p_intercept < 0.10,
        "n_studies": len(x),
    }


def beggs_test(effect_sizes: np.ndarray, se: np.ndarray) -> Dict:
    """
    Begg's test — 秩相关法检验发表偏倚
    """
    valid = ~np.isnan(effect_sizes) & ~np.isnan(se) & (se > 0)
    if valid.sum() < 4:
        return {"error": "数据不足（至少需要4个研究）"}

    # 标准化效应量
    standardized = effect_sizes[valid] / se[valid]
    variances = se[valid] ** 2

    # Kendall's tau
    tau, p_value = stats.kendalltau(standardized, variances)

    return {
        "tau": tau,
        "p_value": p_value,
        "bias_detected": p_value < 0.10,
        "n_studies": len(standardized),
    }


# ============================================================
# 敏感性分析 (Leave-One-Out)
# ============================================================

def leave_one_out_sensitivity(
    proportions: List[float],
    sample_sizes: List[int],
    study_names: List[str] = None,
) -> List[Dict]:
    """
    逐一剔除法敏感性分析

    每次剔除一个研究，重新计算合并效应量，
    观察结果是否稳定。
    """
    k = len(proportions)
    if k < 3:
        return []

    if study_names is None:
        study_names = [f"Study {i+1}" for i in range(k)]

    results = []
    for i in range(k):
        remaining_p = [p for j, p in enumerate(proportions) if j != i]
        remaining_n = [n for j, n in enumerate(sample_sizes) if j != i]
        remaining_names = [s for j, s in enumerate(study_names) if j != i]

        result = proportion_meta_analysis(
            remaining_p, remaining_n, remaining_names, random_effects=True
        )

        results.append({
            "excluded_study": study_names[i],
            "pooled_proportion": result["pooled_proportion_random"],
            "ci": result["ci_random"],
            "I2": result["I2"],
            "k_remaining": k - 1,
        })

    return results


# ============================================================
# 亚组分析
# ============================================================

def subgroup_analysis(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    sample_col: str = "sample_size",
    name_col: str = "first_author",
) -> Dict:
    """
    按指定列进行亚组分析

    Parameters
    ----------
    df : DataFrame
    group_col : str
        分组列名（如 "lymphoma_subtypes", "drugs"）
    value_col : str
        效应量列名（如 "orr", "cr_rate"）
    sample_col : str
        样本量列名
    name_col : str
        研究名称列名

    Returns
    -------
    dict : 各亚组的合并结果
    """
    # 过滤有效数据
    valid = df[[value_col, sample_col]].dropna()
    df_valid = df.loc[valid.index].copy()

    # 展开分组的多个值（如 "DLBCL; FL" → 两条记录）
    expanded_rows = []
    for idx, row in df_valid.iterrows():
        groups = str(row[group_col]).split("; ")
        for g in groups:
            g = g.strip()
            if g and g != "Unspecified" and g != "Unknown":
                expanded_rows.append({
                    "group": g,
                    "value": row[value_col],
                    "sample": row[sample_col],
                    "name": f"{row.get(name_col, '')} {row.get('pub_year', '')}",
                })

    if not expanded_rows:
        return {"error": "无有效分组数据"}

    exp_df = pd.DataFrame(expanded_rows)

    # 按组分析
    subgroups = {}
    for group_name, group_df in exp_df.groupby("group"):
        if len(group_df) < 1:
            continue

        proportions = group_df["value"].tolist()
        sample_sizes = group_df["sample"].astype(int).tolist()
        names = group_df["name"].tolist()

        result = proportion_meta_analysis(proportions, sample_sizes, names)
        subgroups[group_name] = {
            "k": len(group_df),
            "pooled_proportion": result["pooled_proportion_random"],
            "ci": result["ci_random"],
            "I2": result["I2"],
            "p_heterogeneity": result["p_heterogeneity"],
        }

    return subgroups


# ============================================================
# 描述性统计
# ============================================================

def descriptive_statistics(df: pd.DataFrame) -> Dict:
    """生成描述性统计表"""
    stats = {}

    # 总文献数
    stats["total_studies"] = len(df)

    # 年份分布
    if "pub_year" in df.columns:
        years = pd.to_numeric(df["pub_year"], errors="coerce").dropna()
        stats["year_range"] = f"{int(years.min())}–{int(years.max())}"
        stats["year_distribution"] = years.value_counts().sort_index().to_dict()

    # 研究类型分布
    if "study_type" in df.columns:
        all_types = []
        for st in df["study_type"].dropna():
            all_types.extend([t.strip() for t in st.split(";") if t.strip()])
        stats["study_types"] = pd.Series(all_types).value_counts().to_dict()

    # 淋巴瘤亚型分布
    if "lymphoma_subtypes" in df.columns:
        all_subtypes = []
        for sub in df["lymphoma_subtypes"].dropna():
            all_subtypes.extend([s.strip() for s in sub.split(";") if s.strip()])
        stats["lymphoma_subtypes"] = pd.Series(all_subtypes).value_counts().to_dict()

    # 药物分布
    if "drugs" in df.columns:
        all_drugs = []
        for d in df["drugs"].dropna():
            all_drugs.extend([x.strip() for x in d.split(";") if x.strip()])
        stats["drugs"] = pd.Series(all_drugs).value_counts().to_dict()

    # 样本量统计
    if "sample_size" in df.columns:
        ss = pd.to_numeric(df["sample_size"], errors="coerce").dropna()
        stats["sample_size"] = {
            "median": float(ss.median()),
            "mean": float(ss.mean()),
            "min": int(ss.min()),
            "max": int(ss.max()),
            "total_patients": int(ss.sum()),
        }

    # 疗效指标统计
    for col in ["orr", "cr_rate", "pr_rate", "pfs_median", "os_median"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) > 0:
                stats[col] = {
                    "n_studies": len(vals),
                    "median": float(vals.median()),
                    "mean": float(vals.mean()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                }

    return stats


# ============================================================
# 完整 Meta 分析流程
# ============================================================

class MetaAnalyzer:
    """完整的 Meta 分析流程"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}
        self.output_dir = _get_output_dir()
        os.makedirs(self.output_dir, exist_ok=True)

    def run_full_analysis(self) -> Dict:
        """运行完整 meta 分析"""
        print(f"\n{'='*60}")
        print("Meta 分析")
        print(f"{'='*60}")
        print(f"纳入文献: {len(self.df)} 篇")

        results = {}

        # 1. 描述性统计
        print("\n--- 1. 描述性统计 ---")
        results["descriptive"] = descriptive_statistics(self.df)
        self._print_descriptive(results["descriptive"])

        # 2. ORR 合并分析
        print("\n--- 2. ORR (总缓解率) 合并分析 ---")
        results["orr_meta"] = self._run_proportion_meta("orr")
        self._print_proportion_meta(results["orr_meta"], "ORR")

        # 3. CR 合并分析
        print("\n--- 3. CR (完全缓解率) 合并分析 ---")
        results["cr_meta"] = self._run_proportion_meta("cr_rate")
        self._print_proportion_meta(results["cr_meta"], "CR")

        # 4. HR 分析 (PFS)
        print("\n--- 4. PFS 风险比 (HR) 分析 ---")
        results["hr_pfs_meta"] = self._run_hr_meta("hr_pfs", "hr_pfs_ci_lower", "hr_pfs_ci_upper")
        if "error" not in results["hr_pfs_meta"]:
            self._print_hr_meta(results["hr_pfs_meta"], "PFS")

        # 5. HR 分析 (OS)
        print("\n--- 5. OS 风险比 (HR) 分析 ---")
        results["hr_os_meta"] = self._run_hr_meta("hr_os", "hr_os_ci_lower", "hr_os_ci_upper")
        if "error" not in results["hr_os_meta"]:
            self._print_hr_meta(results["hr_os_meta"], "OS")

        # 6. 亚组分析 — 按淋巴瘤亚型
        print("\n--- 6. 亚组分析: 按淋巴瘤亚型 ---")
        results["subgroup_lymphoma"] = subgroup_analysis(
            self.df, "lymphoma_subtypes", "orr"
        )
        self._print_subgroups(results["subgroup_lymphoma"], "淋巴瘤亚型")

        # 7. 亚组分析 — 按药物
        print("\n--- 7. 亚组分析: 按药物/疗法 ---")
        results["subgroup_drug"] = subgroup_analysis(
            self.df, "drugs", "orr"
        )
        self._print_subgroups(results["subgroup_drug"], "药物/疗法")

        # 8. 发表偏倚检验
        print("\n--- 8. 发表偏倚检验 ---")
        results["publication_bias"] = self._test_publication_bias()
        self._print_bias(results["publication_bias"])

        # 9. 敏感性分析
        print("\n--- 9. 敏感性分析 (Leave-One-Out) ---")
        results["sensitivity"] = self._run_sensitivity("orr")
        self._print_sensitivity(results["sensitivity"])

        self.results = results

        # 保存结果
        self._save_results(results)

        return results

    def _run_proportion_meta(self, col: str) -> Dict:
        """运行比例 meta 分析"""
        valid = self.df[[col, "sample_size"]].dropna()
        if len(valid) < 2:
            return {"error": f"有效数据不足（{len(valid)} 篇）"}

        proportions = valid[col].astype(float).tolist()
        sample_sizes = valid["sample_size"].astype(int).tolist()
        names = [
            f"{r.get('first_author', 'Unknown')} {r.get('pub_year', '')}"
            for _, r in valid.iterrows()
        ]

        return proportion_meta_analysis(proportions, sample_sizes, names)

    def _run_hr_meta(self, hr_col: str, lo_col: str, hi_col: str) -> Dict:
        """运行 HR meta 分析"""
        cols = [hr_col, lo_col, hi_col]
        valid = self.df[cols].dropna()
        if len(valid) < 2:
            return {"error": f"有效 HR 数据不足（{len(valid)} 篇）"}

        hr_values = valid[hr_col].astype(float).tolist()
        hr_lo = valid[lo_col].astype(float).tolist()
        hr_hi = valid[hi_col].astype(float).tolist()
        names = [
            f"{r.get('first_author', 'Unknown')} {r.get('pub_year', '')}"
            for _, r in valid.iterrows()
        ]

        return hr_meta_analysis(hr_values, hr_lo, hr_hi, names)

    def _test_publication_bias(self) -> Dict:
        """发表偏倚检验"""
        valid = self.df[["orr", "sample_size"]].dropna()
        if len(valid) < 4:
            return {"error": "数据不足（至少需要4个研究）"}

        proportions = valid["orr"].astype(float).values
        n = valid["sample_size"].astype(int).values

        # Freeman-Tukey 变换后的效应量
        effect_sizes = np.array([ft_transform(p, ni) for p, ni in zip(proportions, n)])
        se = np.sqrt(1.0 / (n + 1))

        egger = eggers_test(effect_sizes, se)
        begg = beggs_test(effect_sizes, se)

        return {
            "egger": egger,
            "begg": begg,
            "n_studies": len(valid),
        }

    def _run_sensitivity(self, col: str) -> List[Dict]:
        """敏感性分析"""
        valid = self.df[[col, "sample_size"]].dropna()
        if len(valid) < 3:
            return []

        proportions = valid[col].astype(float).tolist()
        sample_sizes = valid["sample_size"].astype(int).tolist()
        names = [
            f"{r.get('first_author', 'Unknown')} {r.get('pub_year', '')}"
            for _, r in valid.iterrows()
        ]

        return leave_one_out_sensitivity(proportions, sample_sizes, names)

    # --------------------------------------------------------
    # 打印函数
    # --------------------------------------------------------
    def _print_descriptive(self, stats: Dict):
        print(f"  总文献数: {stats.get('total_studies', 0)}")
        if "year_range" in stats:
            print(f"  年份范围: {stats['year_range']}")
        if "sample_size" in stats:
            ss = stats["sample_size"]
            print(f"  样本量: 中位数={ss['median']:.0f}, 范围={ss['min']}–{ss['max']}, 总计={ss['total_patients']}")
        if "lymphoma_subtypes" in stats:
            print(f"  淋巴瘤亚型:")
            for k, v in list(stats["lymphoma_subtypes"].items())[:8]:
                print(f"    {k}: {v}")
        if "drugs" in stats:
            print(f"  药物/疗法:")
            for k, v in list(stats["drugs"].items())[:8]:
                print(f"    {k}: {v}")

    def _print_proportion_meta(self, result: Dict, label: str):
        if "error" in result:
            print(f"  ⚠ {result['error']}")
            return
        print(f"  纳入研究数: {result['k']}")
        print(f"  合并{label} (随机效应): {result['pooled_proportion_random']*100:.1f}% "
              f"[95% CI: {result['ci_random'][0]*100:.1f}%–{result['ci_random'][1]*100:.1f}%]")
        print(f"  合并{label} (固定效应): {result['pooled_proportion_fixed']*100:.1f}% "
              f"[95% CI: {result['ci_fixed'][0]*100:.1f}%–{result['ci_fixed'][1]*100:.1f}%]")
        print(f"  异质性: Q={result['Q']:.2f}, df={result['df_Q']}, "
              f"I²={result['I2']:.1f}%, p={result['p_heterogeneity']:.4f}")
        if result["I2"] < 25:
            print(f"  → 异质性低 (I²<25%)")
        elif result["I2"] < 50:
            print(f"  → 异质性中等 (25%≤I²<50%)")
        elif result["I2"] < 75:
            print(f"  → 异质性较大 (50%≤I²<75%)")
        else:
            print(f"  → 异质性很大 (I²≥75%)")

    def _print_hr_meta(self, result: Dict, label: str):
        print(f"  纳入研究数: {result['k']}")
        print(f"  合并HR ({label}): {result['pooled_hr']:.3f} "
              f"[95% CI: {result['ci'][0]:.3f}–{result['ci'][1]:.3f}]")
        print(f"  z = {result['z']:.3f}, p = {result['p_value']:.4f}")
        print(f"  异质性: Q={result['Q']:.2f}, I²={result['I2']:.1f}%, "
              f"p_het={result['p_heterogeneity']:.4f}")
        if result['p_value'] < 0.05:
            if result['pooled_hr'] < 1:
                print(f"  → 试验组{label}显著优于对照组 (HR<1, p<0.05)")
            else:
                print(f"  → 试验组{label}显著劣于对照组 (HR>1, p<0.05)")
        else:
            print(f"  → 两组{label}差异无统计学意义 (p≥0.05)")

    def _print_subgroups(self, subgroups: Dict, label: str):
        if "error" in subgroups:
            print(f"  ⚠ {subgroups['error']}")
            return
        for name, sg in sorted(subgroups.items(), key=lambda x: -x[1].get("k", 0)):
            ci = sg.get("ci", (0, 0))
            print(f"  {name}: k={sg['k']}, "
                  f"合并ORR={sg['pooled_proportion']*100:.1f}% "
                  f"[{ci[0]*100:.1f}%–{ci[1]*100:.1f}%], "
                  f"I²={sg['I2']:.1f}%")

    def _print_bias(self, bias: Dict):
        if "error" in bias:
            print(f"  ⚠ {bias['error']}")
            return
        egger = bias.get("egger", {})
        begg = bias.get("begg", {})
        if "error" not in egger:
            print(f"  Egger's test: 截距={egger.get('intercept', 0):.3f}, "
                  f"p={egger.get('p_value', 1):.4f} "
                  f"{'→ 存在发表偏倚' if egger.get('bias_detected') else '→ 无显著发表偏倚'}")
        if "error" not in begg:
            print(f"  Begg's test: tau={begg.get('tau', 0):.3f}, "
                  f"p={begg.get('p_value', 1):.4f} "
                  f"{'→ 存在发表偏倚' if begg.get('bias_detected') else '→ 无显著发表偏倚'}")

    def _print_sensitivity(self, results: List[Dict]):
        if not results:
            print("  ⚠ 数据不足")
            return
        print(f"  逐一剔除后合并ORR范围:")
        pooled_vals = [r["pooled_proportion"] for r in results]
        print(f"    最小: {min(pooled_vals)*100:.1f}% (剔除 {min(results, key=lambda x: x['pooled_proportion'])['excluded_study']})")
        print(f"    最大: {max(pooled_vals)*100:.1f}% (剔除 {max(results, key=lambda x: x['pooled_proportion'])['excluded_study']})")
        print(f"    → 结果{'稳定' if max(pooled_vals) - min(pooled_vals) < 0.1 else '不稳定'}")

    def _save_results(self, results: Dict):
        """保存分析结果"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"meta_analysis_results_{timestamp}.json")

        # 序列化（处理不可JSON序列化的类型）
        def default_serializer(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=default_serializer)
        print(f"\n✓ 分析结果已保存: {path}")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    # 加载清洗后的数据
    results_dir = _get_output_dir()
    csv_files = [f for f in os.listdir(results_dir) if f.startswith("cleaned_data") and f.endswith(".csv")]

    if not csv_files:
        print("✗ 未找到清洗数据，请先运行 data_cleaner.py")
        exit(1)

    latest = sorted(csv_files)[-1]
    csv_path = os.path.join(results_dir, latest)
    print(f"加载: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"数据量: {len(df)} 篇文献")

    analyzer = MetaAnalyzer(df)
    results = analyzer.run_full_analysis()

    print(f"\n{'='*60}")
    print("Meta 分析完成！")
    print(f"{'='*60}")
