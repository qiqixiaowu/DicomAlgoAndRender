#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
淋巴瘤用药 Meta 分析 — 主运行脚本
===================================
一键完成完整流程：

  1. PubMed 爬取文献
  2. 数据清洗与信息提取
  3. Meta 分析统计
  4. 可视化图表生成
  5. 生成分析报告

用法:
    python main.py                    # 使用默认参数运行完整流程
    python main.py --max 300          # 限制最大文献数
    python main.py --skip-crawl       # 跳过爬取，使用已有数据
    python main.py --custom-query     # 交互式自定义检索
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Windows 控制台 UTF-8 输出（避免 GBK 编码错误）
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pubmed_crawler import PubMedCrawler, PubMedConfig, build_lymphoma_drug_query
from data_cleaner import DataCleaner
from meta_analysis import MetaAnalyzer
from visualization import Visualizer


# ============================================================
# 报告生成
# ============================================================

def generate_report(df, results, output_dir):
    """生成 Markdown 格式的 Meta 分析报告"""
    report_path = os.path.join(output_dir, "meta_analysis_report.md")

    desc = results.get("descriptive", {})
    orr_meta = results.get("orr_meta", {})
    cr_meta = results.get("cr_meta", {})
    hr_pfs = results.get("hr_pfs_meta", {})
    hr_os = results.get("hr_os_meta", {})
    bias = results.get("publication_bias", {})
    subgroup_lymphoma = results.get("subgroup_lymphoma", {})
    subgroup_drug = results.get("subgroup_drug", {})
    sensitivity = results.get("sensitivity", [])

    report = f"""# 淋巴瘤用药 Meta 分析报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. 研究概况

| 指标 | 数值 |
|------|------|
| 纳入文献数 | {len(df)} 篇 |
| 年份范围 | {desc.get('year_range', 'N/A')} |
| 总患者数 | {desc.get('sample_size', {}).get('total_patients', 'N/A')} |
| 样本量中位数 | {desc.get('sample_size', {}).get('median', 'N/A')} |

## 2. 文献特征

### 2.1 淋巴瘤亚型分布

| 亚型 | 文献数 |
|------|--------|
"""
    for k, v in list(desc.get("lymphoma_subtypes", {}).items())[:10]:
        report += f"| {k} | {v} |\n"

    report += """
### 2.2 药物/疗法分布

| 药物/疗法 | 文献数 |
|-----------|--------|
"""
    for k, v in list(desc.get("drugs", {}).items())[:15]:
        report += f"| {k} | {v} |\n"

    report += """
### 2.3 研究类型分布

| 研究类型 | 文献数 |
|----------|--------|
"""
    for k, v in list(desc.get("study_types", {}).items())[:10]:
        report += f"| {k} | {v} |\n"

    # ORR
    report += f"""
## 3. 疗效合并分析

### 3.1 ORR (总缓解率)

| 指标 | 数值 |
|------|------|
| 纳入研究数 | {orr_meta.get('k', 'N/A')} |
| 合并 ORR (随机效应) | {orr_meta.get('pooled_proportion_random', 0)*100:.1f}% [95% CI: {orr_meta.get('ci_random', (0,0))[0]*100:.1f}%–{orr_meta.get('ci_random', (0,0))[1]*100:.1f}%] |
| 合并 ORR (固定效应) | {orr_meta.get('pooled_proportion_fixed', 0)*100:.1f}% [95% CI: {orr_meta.get('ci_fixed', (0,0))[0]*100:.1f}%–{orr_meta.get('ci_fixed', (0,0))[1]*100:.1f}%] |
| 异质性 I² | {orr_meta.get('I2', 0):.1f}% |
| Q 统计量 | {orr_meta.get('Q', 0):.2f} (df={orr_meta.get('df_Q', 0)}) |
| p (异质性) | {orr_meta.get('p_heterogeneity', 1):.4f} |

![ORR森林图](results/figures/forest_orr.png)

### 3.2 CR (完全缓解率)

| 指标 | 数值 |
|------|------|
| 纳入研究数 | {cr_meta.get('k', 'N/A')} |
| 合并 CR (随机效应) | {cr_meta.get('pooled_proportion_random', 0)*100:.1f}% [95% CI: {cr_meta.get('ci_random', (0,0))[0]*100:.1f}%–{cr_meta.get('ci_random', (0,0))[1]*100:.1f}%] |
| 异质性 I² | {cr_meta.get('I2', 0):.1f}% |

![CR森林图](results/figures/forest_cr.png)
"""

    # HR
    if "error" not in hr_pfs:
        report += f"""
### 3.3 PFS 风险比 (HR)

| 指标 | 数值 |
|------|------|
| 纳入研究数 | {hr_pfs.get('k', 'N/A')} |
| 合并 HR | {hr_pfs.get('pooled_hr', 0):.3f} [95% CI: {hr_pfs.get('ci', (0,0))[0]:.3f}–{hr_pfs.get('ci', (0,0))[1]:.3f}] |
| z 检验 p 值 | {hr_pfs.get('p_value', 1):.4f} |
| 异质性 I² | {hr_pfs.get('I2', 0):.1f}% |

![PFS HR森林图](results/figures/forest_hr_pfs.png)
"""

    if "error" not in hr_os:
        report += f"""
### 3.4 OS 风险比 (HR)

| 指标 | 数值 |
|------|------|
| 纳入研究数 | {hr_os.get('k', 'N/A')} |
| 合并 HR | {hr_os.get('pooled_hr', 0):.3f} [95% CI: {hr_os.get('ci', (0,0))[0]:.3f}–{hr_os.get('ci', (0,0))[1]:.3f}] |
| z 检验 p 值 | {hr_os.get('p_value', 1):.4f} |
| 异质性 I² | {hr_os.get('I2', 0):.1f}% |

![OS HR森林图](results/figures/forest_hr_os.png)
"""

    # 亚组分析
    report += """
## 4. 亚组分析

### 4.1 按淋巴瘤亚型

| 亚型 | 研究数 | 合并 ORR | 95% CI | I² |
|------|--------|----------|--------|-----|
"""
    if "error" not in subgroup_lymphoma:
        for name, sg in sorted(subgroup_lymphoma.items(), key=lambda x: -x[1].get("k", 0)):
            ci = sg.get("ci", (0, 0))
            report += f"| {name} | {sg['k']} | {sg['pooled_proportion']*100:.1f}% | [{ci[0]*100:.1f}%–{ci[1]*100:.1f}%] | {sg['I2']:.1f}% |\n"

    report += """
### 4.2 按药物/疗法

| 药物/疗法 | 研究数 | 合并 ORR | 95% CI | I² |
|-----------|--------|----------|--------|-----|
"""
    if "error" not in subgroup_drug:
        for name, sg in sorted(subgroup_drug.items(), key=lambda x: -x[1].get("k", 0)):
            ci = sg.get("ci", (0, 0))
            report += f"| {name} | {sg['k']} | {sg['pooled_proportion']*100:.1f}% | [{ci[0]*100:.1f}%–{ci[1]*100:.1f}%] | {sg['I2']:.1f}% |\n"

    report += """
![亚组分析-淋巴瘤](results/figures/subgroup_lymphoma.png)
![亚组分析-药物](results/figures/subgroup_drug.png)
![药物疗效热力图](results/figures/drug_efficacy_heatmap.png)
"""

    # 发表偏倚
    report += """
## 5. 发表偏倚检验

"""
    if "error" not in bias:
        egger = bias.get("egger", {})
        begg = bias.get("begg", {})
        if "error" not in egger:
            report += f"""| 检验方法 | 统计量 | p 值 | 结论 |
|----------|--------|------|------|
| Egger's test | 截距={egger.get('intercept', 0):.3f} | {egger.get('p_value', 1):.4f} | {'存在发表偏倚' if egger.get('bias_detected') else '无显著偏倚'} |
"""
        if "error" not in begg:
            report += f"| Begg's test | tau={begg.get('tau', 0):.3f} | {begg.get('p_value', 1):.4f} | {'存在发表偏倚' if begg.get('bias_detected') else '无显著偏倚'} |\n"

        report += "\n![漏斗图](results/figures/funnel_orr.png)\n"

    # 敏感性
    if sensitivity:
        pooled_vals = [r["pooled_proportion"] for r in sensitivity]
        report += f"""
## 6. 敏感性分析

逐一剔除法 (Leave-One-Out) 结果：

- 剔除后合并 ORR 范围: {min(pooled_vals)*100:.1f}% – {max(pooled_vals)*100:.1f}%
- 结果{'稳定' if max(pooled_vals) - min(pooled_vals) < 0.1 else '不稳定'}（阈值 10%）

![敏感性分析](results/figures/sensitivity.png)
"""

    # 结论
    orr_pooled = orr_meta.get("pooled_proportion_random", 0) * 100
    cr_pooled = cr_meta.get("pooled_proportion_random", 0) * 100
    report += f"""
## 7. 结论

基于 {len(df)} 篇文献的 Meta 分析：

1. **总缓解率 (ORR)**: 合并 ORR 为 **{orr_pooled:.1f}%**，表明淋巴瘤药物治疗整体疗效显著。
2. **完全缓解率 (CR)**: 合并 CR 为 **{cr_pooled:.1f}%**，反映了深度缓解的比例。
3. **异质性**: ORR 分析的 I² = {orr_meta.get('I2', 0):.1f}%，{'异质性较低，结果较可靠' if orr_meta.get('I2', 100) < 50 else '异质性较高，需谨慎解读'}。
4. **发表偏倚**: {'存在' if bias.get('egger', {}).get('bias_detected') else '无明显'}发表偏倚。

### 局限性

- 数据来源于 PubMed 摘要的自动提取，可能存在信息遗漏
- 部分研究未报告完整的疗效数据
- 不同淋巴瘤亚型和治疗方案存在临床异质性
- 未纳入灰色文献（会议摘要、未发表研究）

### 建议

- 结合临床专业知识解读结果
- 对高异质性的分析应进行亚组分析探索来源
- 考虑手动核对关键研究的全文数据

---

*本报告由淋巴瘤用药 Meta 分析工具自动生成，仅供研究参考。*
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✓ 分析报告已保存: {report_path}")
    return report_path


# ============================================================
# 主流程
# ============================================================

def run_full_pipeline(
    max_results: int = 200,
    skip_crawl: bool = False,
    date_range: str = "2015/01/01:2025/12/31",
    custom_lymphoma_types=None,
    custom_drug_names=None,
):
    """运行完整流程"""
    # PyInstaller 打包后，结果应写到 exe 所在目录，而非临时解压目录
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("  淋巴瘤用药 Meta 分析 — 完整流程")
    print("=" * 60)
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  最大文献数: {max_results}")
    print(f"  日期范围: {date_range}")
    print(f"  跳过爬取: {skip_crawl}")
    print("=" * 60)

    # === Step 1: PubMed 爬取 ===
    articles = None
    if not skip_crawl:
        print("\n" + "▶ " * 30)
        print("  STEP 1: PubMed 文献爬取")
        print("▶ " * 30)

        crawler = PubMedCrawler()
        articles = crawler.run(
            max_results=max_results,
            date_range=date_range,
            custom_lymphoma_types=custom_lymphoma_types,
            custom_drug_names=custom_drug_names,
        )

        if not articles:
            print("✗ 爬取失败或无结果，尝试使用已有数据...")
            skip_crawl = True

    if skip_crawl or articles is None:
        # 加载已有数据 — 先从 exe 旁边的 results 目录找
        json_files = [f for f in os.listdir(results_dir)
                      if f.startswith("pubmed_lymphoma") and f.endswith(".json")]
        
        # 如果 exe 旁边没有，尝试从打包资源中找
        if not json_files and hasattr(sys, '_MEIPASS'):
            packed_results = os.path.join(sys._MEIPASS, 'results')
            if os.path.isdir(packed_results):
                json_files = [f for f in os.listdir(packed_results)
                              if f.startswith("pubmed_lymphoma") and f.endswith(".json")]
                if json_files:
                    # 复制到 exe 旁边的 results 目录
                    import shutil
                    for jf in json_files:
                        shutil.copy2(os.path.join(packed_results, jf),
                                     os.path.join(results_dir, jf))
        
        if not json_files:
            print("✗ 无已有爬取数据，请先运行爬取（不加 --skip-crawl）")
            return
        latest = sorted(json_files)[-1]
        with open(os.path.join(results_dir, latest), "r", encoding="utf-8") as f:
            articles = json.load(f)
        print(f"  使用已有数据: {latest} ({len(articles)} 篇)")

    # === Step 2: 数据清洗 ===
    print("\n" + "▶ " * 30)
    print("  STEP 2: 数据清洗与信息提取")
    print("▶ " * 30)

    cleaner = DataCleaner()
    df = cleaner.process_articles(articles)
    cleaner.save(df)

    if len(df) == 0:
        print("✗ 无有效数据，流程终止")
        return

    # === Step 3: Meta 分析 ===
    print("\n" + "▶ " * 30)
    print("  STEP 3: Meta 分析")
    print("▶ " * 30)

    analyzer = MetaAnalyzer(df)
    results = analyzer.run_full_analysis()

    # === Step 4: 可视化 ===
    print("\n" + "▶ " * 30)
    print("  STEP 4: 可视化图表生成")
    print("▶ " * 30)

    viz = Visualizer(df, results)
    viz.generate_all()

    # === Step 5: 报告 ===
    print("\n" + "▶ " * 30)
    print("  STEP 5: 生成分析报告")
    print("▶ " * 30)

    report_path = generate_report(df, results, base_dir)

    # === 完成 ===
    print("\n" + "=" * 60)
    print("  ✅ 全流程完成！")
    print("=" * 60)
    print(f"  结果目录: {results_dir}")
    print(f"  图表目录: {os.path.join(results_dir, 'figures')}")
    print(f"  分析报告: {report_path}")
    print("=" * 60)


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="淋巴瘤用药 Meta 分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                         # 默认运行完整流程
  python main.py --max 300               # 限制最大文献数
  python main.py --skip-crawl            # 跳过爬取，使用已有数据
  python main.py --date 2018/01/01:2025/12/31  # 指定日期范围
        """
    )
    parser.add_argument("--max", type=int, default=200,
                        help="最大爬取文献数 (默认: 200)")
    parser.add_argument("--skip-crawl", action="store_true",
                        help="跳过 PubMed 爬取，使用已有数据")
    parser.add_argument("--date", type=str, default="2015/01/01:2025/12/31",
                        help="日期范围 (格式: YYYY/MM/DD:YYYY/MM/DD)")
    parser.add_argument("--api-key", type=str, default="",
                        help="NCBI API key (提高请求速率)")

    args = parser.parse_args()

    # 设置 API key
    if args.api_key:
        PubMedConfig.API_KEY = args.api_key
        PubMedConfig.RATE_LIMIT_DELAY = 0.1

    run_full_pipeline(
        max_results=args.max,
        skip_crawl=args.skip_crawl,
        date_range=args.date,
    )


if __name__ == "__main__":
    main()
