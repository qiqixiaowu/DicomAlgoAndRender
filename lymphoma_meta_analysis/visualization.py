#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化图表模块
==============
生成 Meta 分析的标准图表：

1. 森林图 (Forest Plot) — 比例 meta 分析
2. 森林图 — HR meta 分析
3. 漏斗图 (Funnel Plot) — 发表偏倚
4. 亚组分析图
5. 年份/期刊分布图
6. 药物-疗效热力图
7. 敏感性分析图
"""

import os
import sys
import math
import json
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
from matplotlib.font_manager import FontProperties, fontManager

# === PyInstaller 资源路径 ===
def _resource_path(relative_path: str) -> str:
    """获取资源文件的绝对路径（兼容 PyInstaller 打包环境）"""
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller 打包环境
        return os.path.join(sys._MEIPASS, relative_path)
    # 开发环境
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

# === 中文字体设置 ===
# 尝试加载打包的 SimHei 字体，否则使用系统字体
_font_loaded = False
_packed_font_path = _resource_path(os.path.join('fonts', 'simhei.ttf'))
if os.path.exists(_packed_font_path):
    try:
        fontManager.addfont(_packed_font_path)
        _font_loaded = True
    except Exception:
        pass

if _font_loaded:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
else:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"


# ============================================================
# 森林图 — 比例 Meta 分析
# ============================================================

def forest_plot_proportion(meta_result: Dict, label: str = "ORR", output_path: str = None):
    """
    绘制比例 meta 分析的森林图

    每行一个研究，显示：
      - 研究名称
      - 比例 + 95% CI
      - 权重
      - 图形化的效应量和CI
    """
    if "error" in meta_result:
        print(f"  ⚠ 无法绘制 {label} 森林图: {meta_result['error']}")
        return

    studies = meta_result["studies"]
    k = len(studies)

    if k == 0:
        return

    # 图形布局
    fig_height = max(4, 0.4 * k + 3)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Y 轴位置（从上到下）
    y_positions = list(range(k, 0, -1))

    # 绘制每个研究
    for i, study in enumerate(studies):
        y = y_positions[i]
        prop = study["proportion"] * 100
        ci_lo = study["ci_lower"] * 100
        ci_hi = study["ci_upper"] * 100
        weight = study.get("weight_random", study.get("weight_fixed", 0))

        # 方块（大小代表权重）
        box_size = 0.3 + weight / 100 * 0.7
        ax.scatter(prop, y, s=box_size * 200, marker="s",
                   color="#4472C4", edgecolors="navy", zorder=3)

        # CI 线
        ax.plot([ci_lo, ci_hi], [y, y], color="#4472C4", linewidth=1.5, zorder=2)
        # CI 端线
        ax.plot([ci_lo, ci_lo], [y - 0.15, y + 0.15], color="#4472C4", linewidth=1.5)
        ax.plot([ci_hi, ci_hi], [y - 0.15, y + 0.15], color="#4472C4", linewidth=1.5)

    # 合并效应量
    pooled = meta_result["pooled_proportion_random"] * 100
    pooled_lo = meta_result["ci_random"][0] * 100
    pooled_hi = meta_result["ci_random"][1] * 100

    # 菱形
    diamond_y = 0
    diamond = mpatches.Polygon(
        [(pooled_lo, diamond_y), (pooled, diamond_y + 0.3),
         (pooled_hi, diamond_y), (pooled, diamond_y - 0.3)],
        closed=True, facecolor="#ED7D31", edgecolor="darkred", linewidth=1.5, zorder=4
    )
    ax.add_patch(diamond)

    # 合并效应量虚线
    ax.axvline(pooled, color="#ED7D31", linestyle="--", linewidth=1, alpha=0.7, zorder=1)

    # 研究名称（右侧）
    for i, study in enumerate(studies):
        y = y_positions[i]
        prop = study["proportion"] * 100
        ci_lo = study["ci_lower"] * 100
        ci_hi = study["ci_upper"] * 100
        weight = study.get("weight_random", study.get("weight_fixed", 0))
        text = f"{prop:.1f}% [{ci_lo:.1f}–{ci_hi:.1f}]  ({weight:.1f}%)"
        ax.text(105, y, text, va="center", fontsize=7, family="monospace")

    # 合并效应量标注
    ax.text(105, 0, f"{pooled:.1f}% [{pooled_lo:.1f}–{pooled_hi:.1f}]",
            va="center", fontsize=9, fontweight="bold", family="monospace", color="#ED7D31")

    # 轴设置
    ax.set_xlim(0, 130)
    ax.set_ylim(-1, k + 1)
    ax.set_xlabel(f"{label} (%)", fontsize=11)
    ax.set_yticks([])
    ax.set_title(f"森林图: {label} 合并分析 (随机效应模型)\n"
                 f"I²={meta_result['I2']:.1f}%, "
                 f"Q={meta_result['Q']:.2f}, "
                 f"p_het={meta_result['p_heterogeneity']:.4f}",
                 fontsize=12, fontweight="bold")

    # 网格
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.grid(axis="x", alpha=0.3)

    # 分隔线
    ax.axhline(0.5, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"  ✓ 森林图已保存: {output_path}")
    plt.close()


# ============================================================
# 棑林图 — HR Meta 分析
# ============================================================

def forest_plot_hr(meta_result: Dict, label: str = "PFS", output_path: str = None):
    """绘制 HR meta 分析的森林图"""
    if "error" in meta_result:
        print(f"  ⚠ 无法绘制 {label} HR 森林图: {meta_result['error']}")
        return

    studies = meta_result["studies"]
    k = len(studies)

    fig_height = max(4, 0.4 * k + 3)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y_positions = list(range(k, 0, -1))

    for i, study in enumerate(studies):
        y = y_positions[i]
        hr = study["hr"]
        ci_lo = study["ci_lower"]
        ci_hi = study["ci_upper"]
        weight = study.get("weight", 0)

        box_size = 0.3 + weight / 100 * 0.7
        ax.scatter(hr, y, s=box_size * 200, marker="s",
                   color="#4472C4", edgecolors="navy", zorder=3)

        ax.plot([ci_lo, ci_hi], [y, y], color="#4472C4", linewidth=1.5, zorder=2)
        ax.plot([ci_lo, ci_lo], [y - 0.15, y + 0.15], color="#4472C4", linewidth=1.5)
        ax.plot([ci_hi, ci_hi], [y - 0.15, y + 0.15], color="#4472C4", linewidth=1.5)

    # 合并 HR
    pooled = meta_result["pooled_hr"]
    pooled_lo = meta_result["ci"][0]
    pooled_hi = meta_result["ci"][1]

    diamond = mpatches.Polygon(
        [(pooled_lo, 0), (pooled, 0.3), (pooled_hi, 0), (pooled, -0.3)],
        closed=True, facecolor="#ED7D31", edgecolor="darkred", linewidth=1.5, zorder=4
    )
    ax.add_patch(diamond)
    ax.axvline(pooled, color="#ED7D31", linestyle="--", linewidth=1, alpha=0.7)

    # 无效线 HR=1
    ax.axvline(1.0, color="black", linestyle="-", linewidth=1, alpha=0.5)

    # 右侧标注
    for i, study in enumerate(studies):
        y = y_positions[i]
        hr = study["hr"]
        ci_lo = study["ci_lower"]
        ci_hi = study["ci_upper"]
        weight = study.get("weight", 0)
        text = f"{hr:.2f} [{ci_lo:.2f}–{ci_hi:.2f}]  ({weight:.1f}%)"
        ax.text(max(pooled_hi, ci_hi) * 1.3, y, text, va="center", fontsize=7, family="monospace")

    ax.text(max(pooled_hi, pooled_hi) * 1.3, 0,
            f"{pooled:.2f} [{pooled_lo:.2f}–{pooled_hi:.2f}]",
            va="center", fontsize=9, fontweight="bold", family="monospace", color="#ED7D31")

    # 对数刻度
    ax.set_xscale("log")
    x_min = max(0.1, min(s["ci_lower"] for s in studies) * 0.8)
    x_max = min(10, max(s["ci_upper"] for s in studies) * 1.2)
    ax.set_xlim(x_min, x_max * 2.5)

    ax.set_ylim(-1, k + 1)
    ax.set_xlabel(f"Hazard Ratio ({label}, log scale)", fontsize=11)
    ax.set_yticks([])
    ax.set_title(f"森林图: {label} HR 合并分析\n"
                 f"合并HR={pooled:.3f} [{pooled_lo:.3f}–{pooled_hi:.3f}], "
                 f"p={meta_result['p_value']:.4f}, "
                 f"I²={meta_result['I2']:.1f}%",
                 fontsize=12, fontweight="bold")

    ax.grid(axis="x", alpha=0.3, which="both")
    ax.axhline(0.5, color="gray", linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"  ✓ HR 森林图已保存: {output_path}")
    plt.close()


# ============================================================
# 漏斗图
# ============================================================

def funnel_plot(meta_result: Dict, effect_sizes: np.ndarray = None,
                se: np.ndarray = None, label: str = "ORR", output_path: str = None):
    """
    绘制漏斗图检验发表偏倚

    X 轴: 效应量
    Y 轴: 标准误 (SE) — 越往上 SE 越小（大样本研究）
    对称 → 无发表偏倚
    """
    if "error" in meta_result:
        print(f"  ⚠ 无法绘制漏斗图: {meta_result['error']}")
        return

    studies = meta_result["studies"]
    k = len(studies)

    # 效应量和 SE
    if effect_sizes is None or se is None:
        # 从研究数据重建
        props = np.array([s["proportion"] for s in studies])
        ns = np.array([s["sample_size"] for s in studies])
        effect_sizes = np.array([
            math.asin(math.sqrt(min(max(p * n / (n + 1), 0.0), 1.0))) +
            math.asin(math.sqrt(min(max((p * n + 1) / (n + 1), 0.0), 1.0)))
            for p, n in zip(props, ns)
        ])
        se = np.sqrt(1.0 / (ns + 1))

    pooled = meta_result["pooled_proportion_random"]

    fig, ax = plt.subplots(figsize=(8, 6))

    # 散点
    ax.scatter(effect_sizes, se, s=60, color="#4472C4",
               edgecolors="navy", alpha=0.7, zorder=3)

    # 合并效应量竖线
    pooled_t = math.asin(math.sqrt(min(max(pooled * 100 / 101, 0.0), 1.0))) + \
               math.asin(math.sqrt(min(max((pooled * 100 + 1) / 101, 0.0), 1.0)))
    ax.axvline(pooled_t, color="#ED7D31", linestyle="--", linewidth=1.5, label="合并效应量")

    # 漏斗边界 (95% CI)
    se_range = np.linspace(0, se.max() * 1.2, 100)
    ax.plot(pooled_t - 1.96 * se_range, se_range, "k--", linewidth=1, alpha=0.5, label="95% CI")
    ax.plot(pooled_t + 1.96 * se_range, se_range, "k--", linewidth=1, alpha=0.5)

    ax.set_xlabel("效应量 (Freeman-Tukey 变换)", fontsize=11)
    ax.set_ylabel("标准误 (SE)", fontsize=11)
    ax.set_title(f"漏斗图: {label} 发表偏倚检验\n"
                 f"(对称→无偏倚, 不对称→可能存在发表偏倚)",
                 fontsize=12, fontweight="bold")
    ax.invert_yaxis()  # SE 小的在上方
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"  ✓ 漏斗图已保存: {output_path}")
    plt.close()


# ============================================================
# 亚组分析图
# ============================================================

def subgroup_bar_chart(subgroups: Dict, label: str = "亚组", value_label: str = "ORR",
                       output_path: str = None):
    """亚组分析柱状图"""
    if "error" in subgroups or not subgroups:
        print(f"  ⚠ 无法绘制亚组图: {subgroups.get('error', '无数据')}")
        return

    # 按合并比例排序
    items = sorted(subgroups.items(), key=lambda x: -x[1].get("pooled_proportion", 0))
    names = [x[0] for x in items]
    values = [x[1]["pooled_proportion"] * 100 for x in items]
    ci_los = [x[1]["ci"][0] * 100 for x in items]
    ci_his = [x[1]["ci"][1] * 100 for x in items]
    ks = [x[1]["k"] for x in items]

    fig_height = max(4, 0.5 * len(names) + 2)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    y_pos = range(len(names))

    bars = ax.barh(y_pos, values, color=colors, edgecolor="navy", linewidth=0.5, height=0.6)

    # CI 误差线
    for i, (v, lo, hi) in enumerate(zip(values, ci_los, ci_his)):
        ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="none",
                    color="black", capsize=3, linewidth=1)

    # 标注
    for i, (v, k) in enumerate(zip(values, ks)):
        ax.text(v + 2, i, f"{v:.1f}% (k={k})", va="center", fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(f"合并 {value_label} (%)", fontsize=11)
    ax.set_title(f"亚组分析: {label} — 合并{value_label}", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(ci_his) * 1.2)
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"  ✓ 亚组分析图已保存: {output_path}")
    plt.close()


# ============================================================
# 年份分布图
# ============================================================

def year_distribution_plot(df: pd.DataFrame, output_path: str = None):
    """文献年份分布图"""
    years = pd.to_numeric(df["pub_year"], errors="coerce").dropna().astype(int)

    fig, ax = plt.subplots(figsize=(10, 5))

    year_counts = years.value_counts().sort_index()
    bars = ax.bar(year_counts.index, year_counts.values, color="#4472C4",
                  edgecolor="navy", linewidth=0.5)

    # 趋势线
    if len(year_counts) > 2:
        z = np.polyfit(year_counts.index, year_counts.values, 1)
        p = np.poly1d(z)
        ax.plot(year_counts.index, p(year_counts.index), "r--", linewidth=1.5, alpha=0.7, label="趋势")

    ax.set_xlabel("发表年份", fontsize=11)
    ax.set_ylabel("文献数量", fontsize=11)
    ax.set_title("文献年份分布", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1, str(int(h)),
                ha="center", fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"  ✓ 年份分布图已保存: {output_path}")
    plt.close()


# ============================================================
# 药物-疗效热力图
# ============================================================

def drug_efficacy_heatmap(df: pd.DataFrame, output_path: str = None):
    """药物 × 疗效指标 热力图"""
    # 展开药物
    rows = []
    for _, r in df.iterrows():
        drugs = str(r.get("drugs", "")).split("; ")
        for d in drugs:
            d = d.strip()
            if d and d != "Unspecified":
                rows.append({
                    "drug": d,
                    "orr": r.get("orr"),
                    "cr_rate": r.get("cr_rate"),
                    "pr_rate": r.get("pr_rate"),
                    "pfs_median": r.get("pfs_median"),
                    "os_median": r.get("os_median"),
                })

    if not rows:
        return

    heat_df = pd.DataFrame(rows)
    heat_df = heat_df.groupby("drug").mean(numeric_only=True)
    heat_df = heat_df.round(1)

    # 只保留有足够数据的药物
    heat_df = heat_df.dropna(how="all")
    if len(heat_df) == 0:
        return

    # 排序
    heat_df = heat_df.sort_values("orr", ascending=False)

    fig_height = max(4, 0.5 * len(heat_df) + 2)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # 分开百分比和月数，分别归一化
    pct_cols = ["orr", "cr_rate", "pr_rate"]
    month_cols = ["pfs_median", "os_median"]

    display_df = heat_df.copy()
    # 百分比列转为 0-100
    for c in pct_cols:
        if c in display_df.columns:
            display_df[c] = display_df[c] * 100

    # 确保所有 5 列都存在，缺失的补 NaN
    all_cols = ["orr", "cr_rate", "pr_rate", "pfs_median", "os_median"]
    for c in all_cols:
        if c not in display_df.columns:
            display_df[c] = float("nan")
    display_df = display_df[all_cols]

    col_labels = ["ORR(%)", "CR(%)", "PR(%)", "PFS(月)", "OS(月)"]

    im = ax.imshow(display_df.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(display_df)))
    ax.set_yticklabels(display_df.index, fontsize=9)

    # 数值标注
    for i in range(len(display_df)):
        for j in range(len(display_df.columns)):
            val = display_df.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color="black" if val < display_df.values.max() * 0.6 else "white")

    ax.set_title("药物 × 疗效指标 热力图", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"  ✓ 热力图已保存: {output_path}")
    plt.close()


# ============================================================
# 敏感性分析图
# ============================================================

def sensitivity_plot(sensitivity_results: List[Dict], output_path: str = None):
    """敏感性分析图（逐一剔除后合并效应量变化）"""
    if not sensitivity_results:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(sensitivity_results) + 2)))

    names = [r["excluded_study"] for r in sensitivity_results]
    pooled = [r["pooled_proportion"] * 100 for r in sensitivity_results]
    ci_los = [r["ci"][0] * 100 for r in sensitivity_results]
    ci_his = [r["ci"][1] * 100 for r in sensitivity_results]

    y_pos = range(len(names))

    ax.errorbar(pooled, y_pos, xerr=[[p - lo for p, lo in zip(pooled, ci_los)],
                                      [hi - p for p, hi in zip(pooled, ci_his)]],
                fmt="o", color="#4472C4", capsize=3, markersize=6, linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("剔除该研究后的合并 ORR (%)", fontsize=11)
    ax.set_title("敏感性分析 (Leave-One-Out)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"  ✓ 敏感性分析图已保存: {output_path}")
    plt.close()


# ============================================================
# 综合可视化
# ============================================================

class Visualizer:
    """生成所有可视化图表"""

    def __init__(self, df: pd.DataFrame, results: Dict = None):
        self.df = df
        self.results = results or {}
        # PyInstaller 打包后，输出到 exe 所在目录
        if getattr(sys, 'frozen', False):
            base = os.path.dirname(sys.executable)
        else:
            base = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(base, "results", "figures")
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_all(self, results: Dict = None):
        """生成所有图表"""
        if results:
            self.results = results

        print(f"\n{'='*60}")
        print("生成可视化图表")
        print(f"{'='*60}")

        # 1. ORR 森林图
        if "orr_meta" in self.results and "error" not in self.results["orr_meta"]:
            print("\n[1/8] ORR 森林图...")
            forest_plot_proportion(
                self.results["orr_meta"], "ORR",
                os.path.join(self.output_dir, "forest_orr.png")
            )

        # 2. CR 森林图
        if "cr_meta" in self.results and "error" not in self.results["cr_meta"]:
            print("[2/8] CR 森林图...")
            forest_plot_proportion(
                self.results["cr_meta"], "CR",
                os.path.join(self.output_dir, "forest_cr.png")
            )

        # 3. PFS HR 森林图
        if "hr_pfs_meta" in self.results and "error" not in self.results["hr_pfs_meta"]:
            print("[3/8] PFS HR 森林图...")
            forest_plot_hr(
                self.results["hr_pfs_meta"], "PFS",
                os.path.join(self.output_dir, "forest_hr_pfs.png")
            )

        # 4. OS HR 森林图
        if "hr_os_meta" in self.results and "error" not in self.results["hr_os_meta"]:
            print("[4/8] OS HR 森林图...")
            forest_plot_hr(
                self.results["hr_os_meta"], "OS",
                os.path.join(self.output_dir, "forest_hr_os.png")
            )

        # 5. 漏斗图
        if "orr_meta" in self.results and "error" not in self.results["orr_meta"]:
            print("[5/8] 漏斗图...")
            funnel_plot(
                self.results["orr_meta"],
                label="ORR",
                output_path=os.path.join(self.output_dir, "funnel_orr.png")
            )

        # 6. 亚组分析图 — 淋巴瘤亚型
        if "subgroup_lymphoma" in self.results and "error" not in self.results["subgroup_lymphoma"]:
            print("[6/8] 亚组分析图 (淋巴瘤亚型)...")
            subgroup_bar_chart(
                self.results["subgroup_lymphoma"], "淋巴瘤亚型", "ORR",
                os.path.join(self.output_dir, "subgroup_lymphoma.png")
            )

        # 7. 亚组分析图 — 药物
        if "subgroup_drug" in self.results and "error" not in self.results["subgroup_drug"]:
            print("[7/8] 亚组分析图 (药物)...")
            subgroup_bar_chart(
                self.results["subgroup_drug"], "药物/疗法", "ORR",
                os.path.join(self.output_dir, "subgroup_drug.png")
            )

        # 8. 年份分布 + 热力图 + 敏感性
        print("[8/8] 其他图表...")
        year_distribution_plot(self.df, os.path.join(self.output_dir, "year_distribution.png"))
        drug_efficacy_heatmap(self.df, os.path.join(self.output_dir, "drug_efficacy_heatmap.png"))

        if "sensitivity" in self.results and self.results["sensitivity"]:
            sensitivity_plot(self.results["sensitivity"],
                             os.path.join(self.output_dir, "sensitivity.png"))

        print(f"\n✓ 所有图表已保存到: {self.output_dir}")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    if getattr(sys, 'frozen', False):
        _base = os.path.dirname(sys.executable)
    else:
        _base = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_base, "results")

    # 加载数据
    csv_files = [f for f in os.listdir(results_dir) if f.startswith("cleaned_data") and f.endswith(".csv")]
    if not csv_files:
        print("✗ 未找到清洗数据")
        exit(1)

    df = pd.read_csv(os.path.join(results_dir, sorted(csv_files)[-1]))

    # 加载分析结果
    json_files = [f for f in os.listdir(results_dir) if f.startswith("meta_analysis_results") and f.endswith(".json")]
    results = {}
    if json_files:
        with open(os.path.join(results_dir, sorted(json_files)[-1]), "r", encoding="utf-8") as f:
            results = json.load(f)

    viz = Visualizer(df, results)
    viz.generate_all()
