# 淋巴瘤用药 Meta 分析工具

> 自动从 PubMed 检索淋巴瘤用药相关文献，提取疗效数据，进行 Meta 分析并生成可视化图表和报告。

## 功能概览

```
PubMed 爬取 → 数据清洗 → Meta 分析 → 可视化 → 报告
```

| 模块 | 功能 |
|------|------|
| `pubmed_crawler.py` | 通过 NCBI E-utilities API 检索淋巴瘤用药文献 |
| `data_cleaner.py` | 从摘要中自动提取疗效指标（ORR/CR/PFS/OS/HR等） |
| `meta_analysis.py` | 比例/HR meta 分析、亚组分析、异质性检验、发表偏倚、敏感性分析 |
| `visualization.py` | 森林图、漏斗图、亚组图、热力图等 |
| `main.py` | 一键运行完整流程 |

## 快速开始

### 1. 安装依赖

```bash
pip install requests beautifulsoup4 pandas matplotlib seaborn lxml tqdm numpy scipy
```

### 2. 运行完整流程

```bash
python main.py
```

### 3. 常用选项

```bash
# 限制最大文献数
python main.py --max 300

# 指定日期范围
python main.py --date 2018/01/01:2025/12/31

# 跳过爬取（使用已有数据重新分析）
python main.py --skip-crawl

# 使用 NCBI API key（提高速率，需在 https://www.ncbi.nlm.nih.gov/account/ 申请）
python main.py --api-key YOUR_API_KEY
```

## 检索范围

### 淋巴瘤类型
- 非霍奇金淋巴瘤 (NHL)
- 霍奇金淋巴瘤 (HL)
- 弥漫大B细胞淋巴瘤 (DLBCL)
- 滤泡性淋巴瘤 (FL)
- 套细胞淋巴瘤 (MCL)
- T细胞淋巴瘤

### 药物/疗法
- 利妥昔单抗 (Rituximab) / R-CHOP
- 维布妥昔单抗 (Brentuximab vedotin)
- CAR-T 细胞疗法
- PD-1 抑制剂 (Pembrolizumab, Nivolumab)
- 来那度胺 (Lenalidomide)
- BTK 抑制剂 (Ibrutinib, Acalabrutinib, Zanubrutinib)
- BCL-2 抑制剂 (Venetoclax)
- ADC 药物 (Polatuzumab vedotin)

### 研究类型
- 随机对照试验 (RCT)
- 临床试验
- Meta 分析

## Meta 分析方法

| 分析内容 | 方法 |
|----------|------|
| 比例合并 (ORR/CR) | Freeman-Tukey 双反正弦变换 |
| 效应量合并 (HR) | 对数变换 + DerSimonian-Laird |
| 异质性检验 | Cochran's Q + I² |
| 发表偏倚 | Egger's test + Begg's test |
| 敏感性分析 | Leave-One-Out 逐一剔除法 |
| 亚组分析 | 按淋巴瘤亚型 / 药物 / 研究类型 |

## 输出文件

```
lymphoma_meta_analysis/
├── results/
│   ├── pubmed_lymphoma_*.json     # 原始爬取数据
│   ├── pubmed_lymphoma_*.csv
│   ├── cleaned_data_*.csv         # 清洗后结构化数据
│   ├── cleaned_data_*.json
│   ├── meta_analysis_results_*.json  # 分析结果
│   ├── meta_analysis_report.md    # Markdown 报告
│   └── figures/                   # 可视化图表
│       ├── forest_orr.png         # ORR 森林图
│       ├── forest_cr.png          # CR 森林图
│       ├── forest_hr_pfs.png      # PFS HR 森林图
│       ├── forest_hr_os.png       # OS HR 森林图
│       ├── funnel_orr.png         # 漏斗图
│       ├── subgroup_lymphoma.png  # 亚组分析(淋巴瘤)
│       ├── subgroup_drug.png      # 亚组分析(药物)
│       ├── drug_efficacy_heatmap.png  # 药物疗效热力图
│       ├── year_distribution.png  # 年份分布
│       └── sensitivity.png        # 敏感性分析
```

## 分步运行

也可以单独运行各模块：

```bash
# 1. 仅爬取
python pubmed_crawler.py

# 2. 仅清洗
python data_cleaner.py

# 3. 仅分析
python meta_analysis.py

# 4. 仅可视化
python visualization.py
```

## 自定义检索

编辑 `pubmed_crawler.py` 中的 `build_lymphoma_drug_query()` 或在 `main.py` 中传入参数：

```python
from pubmed_crawler import PubMedCrawler

crawler = PubMedCrawler()
articles = crawler.run(
    max_results=500,
    custom_lymphoma_types=["Lymphoma, Large B-Cell, Diffuse"],
    custom_drug_names=["rituximab", "R-CHOP", "CAR-T cell therapy"],
    date_range="2018/01/01:2025/12/31",
)
```

## 注意事项

1. **NCBI 速率限制**: 无 API key 时每秒最多 3 次请求，有 key 时 10 次/秒
2. **数据来源**: 仅从 PubMed 摘要自动提取，可能存在遗漏，关键研究建议核对全文
3. **临床异质性**: 不同淋巴瘤亚型和治疗方案存在显著临床异质性，合并结果需谨慎解读
4. **仅供研究**: 本工具生成的分析结果仅供学术研究参考，不构成临床建议

## 技术栈

- Python 3.10+
- NCBI E-utilities API (esearch + efetch)
- pandas / numpy / scipy (数据处理与统计)
- matplotlib / seaborn (可视化)
