#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据清洗与信息提取模块
====================
从 PubMed 爬取的文献中提取：
  - 研究类型（RCT / 队列 / 病例报告等）
  - 淋巴瘤亚型
  - 药物/疗法名称
  - 样本量
  - 疗效指标（ORR / CR / PFS / OS 等）
  - 不良事件信息
  - 偏倚风险评估信息

支持从摘要中用正则 + 关键词匹配自动提取结构化数据。
"""

import re
import json
import os
import sys
from typing import List, Dict, Optional, Tuple
import pandas as pd
from tqdm import tqdm


def _get_output_dir() -> str:
    """获取输出目录 — PyInstaller 打包后使用 exe 所在目录"""
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "results")


# ============================================================
# 关键词词典
# ============================================================

# 淋巴瘤亚型关键词
LYMPHOMA_SUBTYPES = {
    "DLBCL": [
        r"diffuse large B.?cell lymphoma",
        r"DLBCL",
    ],
    "FL": [
        r"follicular lymphoma",
        r"\bFL\b",
    ],
    "MCL": [
        r"mantle.?cell lymphoma",
        r"\bMCL\b",
    ],
    "HL": [
        r"Hodgkin lymphoma",
        r"Hodgkin'?s? disease",
        r"\bHL\b(?!\w)",
    ],
    "NHL": [
        r"non.?Hodgkin lymphoma",
        r"\bNHL\b",
    ],
    "CLL/SLL": [
        r"chronic lymphocytic leukemia",
        r"small lymphocytic lymphoma",
        r"\bCLL\b",
        r"\bSLL\b",
    ],
    "MZL": [
        r"marginal zone lymphoma",
        r"\bMZL\b",
    ],
    "PTCL": [
        r"peripheral T.?cell lymphoma",
        r"\bPTCL\b",
    ],
    "CTCL": [
        r"cutaneous T.?cell lymphoma",
        r"\bCTCL\b",
    ],
    "BL": [
        r"Burkitt lymphoma",
        r"\bBL\b(?!\w)",
    ],
    "WM": [
        r"Waldenström",
        r"Waldenstrom",
        r"\bWM\b(?!\w)",
    ],
}

# 药物/疗法关键词
DRUG_THERAPIES = {
    "Rituximab": [r"rituximab", r"\bRTX\b"],
    "R-CHOP": [r"R.?CHOP"],
    "CHOP": [r"\bCHOP\b"],
    "Brentuximab vedotin": [r"brentuximab", r"\bBV\b", r"SGN.?35"],
    "CAR-T": [r"CAR.?T", r"chimeric antigen receptor"],
    "Axicabtagene ciloleucel": [r"axicabtagene", r"axi.?cel", r"Yescarta"],
    "Tisagenlecleucel": [r"tisagenlecleucel", r"tisa.?cel", r"Kymriah"],
    "Pembrolizumab": [r"pembrolizumab", r"Keytruda"],
    "Nivolumab": [r"nivolumab", r"Opdivo"],
    "Lenalidomide": [r"lenalidomide", r"Revlimid"],
    "Ibrutinib": [r"ibrutinib", r"Imbruvica"],
    "Venetoclax": [r"venetoclax", r"Venclexta"],
    "Polatuzumab vedotin": [r"polatuzumab", r"Polivy"],
    "Obinutuzumab": [r"obinutuzumab", r"Gazyva"],
    "Acalabrutinib": [r"acalabrutinib", r"Calquence"],
    "Zanubrutinib": [r"zanubrutinib", r"Brukinsa"],
    "Copanlisib": [r"copanlisib", r"Aliqopa"],
    "Duvelisib": [r"duvelisib", r"Copiktra"],
    "Idelalisib": [r"idelalisib", r"Zydelig"],
    "Autologous transplant": [r"autologous (stem cell )?transplant", r"ASCT"],
    "Allogeneic transplant": [r"allogeneic (stem cell )?transplant"],
}

# 疗效指标关键词
EFFICACY_OUTCOMES = {
    "ORR": [r"overall response rate", r"\bORR\b"],
    "CR": [r"complete response", r"complete remission", r"\bCR\b(?!\w)"],
    "PR": [r"partial response", r"partial remission", r"\bPR\b(?!\w)"],
    "PFS": [r"progression.?free survival", r"\bPFS\b"],
    "OS": [r"overall survival", r"\bOS\b(?!\w)"],
    "DOR": [r"duration of response", r"\bDOR\b"],
    "EFS": [r"event.?free survival", r"\bEFS\b"],
    "DFS": [r"disease.?free survival", r"\bDFS\b"],
    "TTNT": [r"time to next treatment", r"\bTTNT\b"],
}

# 不良事件关键词
ADVERSE_EVENTS = {
    "Neutropenia": [r"neutropenia", r"neutropaenia"],
    "Anemia": [r"anemia", r"anaemia"],
    "Thrombocytopenia": [r"thrombocytopenia", r"thrombocytopaenia"],
    "Infection": [r"infection", r"sepsis", r"febrile neutropenia"],
    "Fatigue": [r"fatigue", r"asthenia"],
    "Nausea": [r"nausea", r"vomiting"],
    "Diarrhea": [r"diarrhea", r"diarrhoea"],
    "CRS": [r"cytokine release syndrome", r"\bCRS\b"],
    "Neurotoxicity": [r"neurotoxicity", r"ICANS", r"immune effector cell.?associated neurotoxicity"],
    "Skin": [r"rash", r"pruritus", r"skin toxicity"],
    "Hepatotoxicity": [r"hepatotoxicity", r"elevated transaminase", r"ALT elevation"],
    "Cardiac": [r"cardiac toxicity", r"arrhythmia", r"atrial fibrillation"],
}

# 研究类型关键词
STUDY_TYPES = {
    "RCT": [r"randomized", r"randomised", r"randomly assigned"],
    "Phase III": [r"phase\s*(?:III|3)\b", r"phase\s*3"],
    "Phase II": [r"phase\s*(?:II|2)\b", r"phase\s*2"],
    "Phase I": [r"phase\s*(?:I|1)\b", r"phase\s*1"],
    "Meta-Analysis": [r"meta.?analysis", r"systematic review"],
    "Retrospective": [r"retrospective"],
    "Prospective": [r"prospective"],
    "Single-arm": [r"single.?arm", r"open.?label"],
}


# ============================================================
# 文本提取工具
# ============================================================

def _search_keywords(text: str, patterns: List[str]) -> bool:
    """检查文本中是否包含任一模式"""
    text_lower = text.lower()
    for pat in patterns:
        if re.search(pat, text_lower, re.IGNORECASE):
            return True
    return False


def _extract_number(text: str, pattern: str) -> Optional[float]:
    """从文本中提取数字"""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            return None
    return None


def _extract_percentage(text: str, keyword_patterns: List[str]) -> Optional[float]:
    """
    从摘要中提取与关键词相关的百分比数值（返回 0-100）。
    例如: "ORR was 85%" → 85.0
          "complete response rate of 72.3%" → 72.3

    改进：
    - 限制搜索窗口在关键词前后 150 字符内，避免跨句误匹配
    - 要求百分比附近有 rate/response/remission 等疗效语境词
    - 排除 "6-12 months"、"range 50-80" 等非疗效数字
    - 多个匹配时取最接近关键词的那个
    """
    text_lower = text.lower()
    best_val = None
    best_dist = 9999

    for kw_pat in keyword_patterns:
        for kw_match in re.finditer(kw_pat, text_lower, re.IGNORECASE):
            kw_pos = kw_match.start()
            # 关键词前后 150 字符的上下文窗口
            ctx_start = max(0, kw_pos - 150)
            ctx_end = min(len(text_lower), kw_pos + 150)
            context = text_lower[ctx_start:ctx_end]

            # 在上下文中搜索百分比
            # 模式1: "关键词 ... XX%" (关键词在前)
            after = text_lower[kw_pos:ctx_end]
            # 模式2: "XX% ... 关键词" (关键词在后)
            before = text_lower[ctx_start:kw_pos]

            pct_patterns = [
                # "rate was 85%", "rate of 72.3%", "ORR 85%"
                rf"(?:rate|response|remission|ORR|CR|PR)\s*(?:was|of|reached|=|achieved)?\s*(\d+\.?\d*)\s*%",
                # "85% ORR", "85% complete response"
                rf"(\d+\.?\d*)\s*%\s*(?:rate|response|remission|ORR|CR|PR)",
                # "ORR was 85%" — 关键词紧跟
                rf"{kw_pat}\s*(?:was|of|reached|=|achieved|\(\s*)?(\d+\.?\d*)\s*%",
                # "XX% (95% CI ...)" — 带置信区间的疗效值
                rf"(?:rate|response|remission)\s*(?:was|of|=)?\s*(\d+\.?\d*)\s*%\s*\(\s*95",
            ]

            for p in pct_patterns:
                for m in re.finditer(p, context, re.IGNORECASE):
                    try:
                        val = float(m.group(1))
                    except (ValueError, IndexError):
                        continue
                    # 合理范围检查：疗效百分比应在 0-100
                    if 0 <= val <= 100:
                        # 计算与关键词的距离
                        dist = abs(ctx_start + m.start() - kw_pos)
                        if dist < best_dist:
                            best_dist = dist
                            best_val = val

    return best_val


def _extract_median_value(text: str, keyword_patterns: List[str], unit: str = "months") -> Optional[float]:
    """
    提取中位值（如中位 PFS = 12.3 个月）
    """
    text_lower = text.lower()
    for kw_pat in keyword_patterns:
        patterns = [
            rf"median[^.]*?{kw_pat}[^.]*?(\d+\.?\d*)\s*{unit}",
            rf"{kw_pat}[^.]*?median[^.]*?(\d+\.?\d*)\s*{unit}",
            rf"median[^.]*?(\d+\.?\d*)\s*{unit}[^.]*?{kw_pat}",
        ]
        for p in patterns:
            match = re.search(p, text_lower, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    continue
    return None


def _extract_sample_size(text: str) -> Optional[int]:
    """提取样本量 — 多模式匹配，取最大合理值"""
    patterns = [
        # "42 patients were enrolled/treated/evaluable"
        r"(\d+)\s*patients?\s*(?:were|enrolled|included|treated|evaluable|analyzed|assessed|studied)",
        # "42 were enrolled/treated as patients"
        r"(\d+)\s*(?:were|enrolled|included|treated|evaluable|analyzed|assessed)\s*patients?",
        # "total of 42 patients"
        r"total\s*of\s*(\d+)\s*patients?",
        # "42 subjects were enrolled"
        r"(\d+)\s*subjects?\s*(?:were|enrolled|included|treated|evaluable)",
        # "cohort of 42 patients"
        r"cohort\s*of\s*(\d+)\s*patients?",
        # "42 patients with"
        r"(\d+)\s*patients?\s*with",
        # "enrolled 42 patients"
        r"(?:enrolled|treated|included|analyzed|recruited)\s*(\d+)\s*patients?",
        # "n = 42" or "N=42"
        r"[nN]\s*=\s*(\d+)",
        # "sample size of 42" or "sample size was 42"
        r"sample\s*size\s*(?:of|was|=)\s*(\d+)",
        # "42 cases" (only if near lymphoma/study context)
        r"(\d+)\s*cases?\s*(?:were|of|with|had)",
        # "42 participants"
        r"(\d+)\s*participants?\s*(?:were|enrolled|included)",
        # "20 of 42 patients" — extract denominator
        r"\d+\s*of\s*(\d+)\s*patients?",
        # "42 evaluable patients"
        r"(\d+)\s*evaluable\s*patients?",
        # "42 patients received"
        r"(\d+)\s*patients?\s*received",
    ]

    candidates = []
    for p in patterns:
        for match in re.finditer(p, text, re.IGNORECASE):
            try:
                n = int(match.group(1))
            except (ValueError, IndexError):
                continue
            if 5 <= n <= 100000:
                candidates.append(n)

    if candidates:
        # 取中位数，避免极端值干扰
        candidates.sort()
        return candidates[len(candidates) // 2]

    return None


def _estimate_sample_size_from_ci(text: str, pct_val: Optional[float]) -> Optional[int]:
    """
    当无法直接提取样本量时，尝试从百分比置信区间宽度反推 n。
    对于比例 p，Wilson CI 宽度 ≈ 2 * z * sqrt(p*(1-p)/n)
    z=1.96 (95% CI)，所以 n ≈ (1.96^2 * p * (1-p)) / (half_width / 100)^2
    pct_val 是 0-100 的百分比。
    """
    if pct_val is None or pct_val <= 0 or pct_val >= 100:
        return None

    text_lower = text.lower()

    # 搜索 "XX% (95% CI YY-ZZ)" 模式
    ci_patterns = [
        r"(\d+\.?\d*)\s*%\s*\(\s*95%?\s*ci\s*(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*\)",
        r"(\d+\.?\d*)\s*%\s*\(\s*95%?\s*ci\s*(\d+\.?\d*)\s*to\s*(\d+\.?\d*)\s*\)",
        r"(\d+\.?\d*)\s*%\s*\(\s*95%?\s*CI\s*(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*\)",
    ]

    for p in ci_patterns:
        for m in re.finditer(p, text_lower, re.IGNORECASE):
            try:
                point = float(m.group(1))
                lo = float(m.group(2))
                hi = float(m.group(3))
            except (ValueError, IndexError):
                continue

            # 检查 point 是否接近 pct_val（±5%）
            if abs(point - pct_val) > 5:
                continue

            half_width = (hi - lo) / 2.0  # CI 半宽（百分比单位）
            if half_width < 0.5:
                continue  # 太窄，不可靠

            p_prop = pct_val / 100.0
            # n ≈ z^2 * p * (1-p) / (half_width/100)^2
            n_est = (1.96**2 * p_prop * (1 - p_prop)) / ((half_width / 100.0) ** 2)
            n_est = int(round(n_est))

            if 5 <= n_est <= 100000:
                return n_est

    return None


def _extract_hazard_ratio(text: str, keyword_patterns: List[str]) -> Optional[Tuple[float, float, float]]:
    """
    提取风险比 HR (point estimate, lower CI, upper CI)
    格式: HR = 0.65 (95% CI 0.50-0.85)
    """
    text_lower = text.lower()
    for kw_pat in keyword_patterns:
        # 先找到关键词附近的文本
        kw_match = re.search(kw_pat, text_lower, re.IGNORECASE)
        if not kw_match:
            continue

        # 在关键词后 300 字符内搜索 HR
        start = max(0, kw_match.start() - 200)
        end = min(len(text_lower), kw_match.end() + 300)
        context = text_lower[start:end]

        hr_patterns = [
            r"hr\s*=?\s*(\d+\.?\d*)\s*\(?\s*95%?\s*ci\s*(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)",
            r"hazard\s*ratio\s*=?\s*(\d+\.?\d*)\s*\(?\s*95%?\s*ci\s*(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)",
            r"hr\s*=?\s*(\d+\.?\d*)\s*\(?\s*(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)",
        ]
        for hp in hr_patterns:
            match = re.search(hp, context, re.IGNORECASE)
            if match:
                try:
                    hr = float(match.group(1))
                    lo = float(match.group(2))
                    hi = float(match.group(3))
                    if 0 < hr < 10 and 0 < lo < hi < 10:
                        return (hr, lo, hi)
                except (ValueError, IndexError):
                    continue
    return None


# ============================================================
# 数据清洗器
# ============================================================

class DataCleaner:
    """从 PubMed 文献中提取结构化 meta 分析数据"""

    def __init__(self):
        self.results = []

    def process_articles(self, articles: List[Dict]) -> pd.DataFrame:
        """
        处理所有文献，提取结构化数据。

        Returns
        -------
        pd.DataFrame : 结构化数据
        """
        print(f"\n{'='*60}")
        print("数据清洗与信息提取")
        print(f"{'='*60}")
        print(f"待处理文献: {len(articles)} 篇")

        results = []
        for article in tqdm(articles, desc="提取数据", unit="篇"):
            extracted = self._process_single(article)
            if extracted:
                results.append(extracted)

        df = pd.DataFrame(results)
        print(f"\n成功提取: {len(df)} / {len(articles)} 篇")

        # 过滤：只保留包含疗效数据的文献
        efficacy_cols = ["orr", "cr_rate", "pr_rate", "pfs_median", "os_median", "hr_pfs", "hr_os"]
        has_efficacy = df[efficacy_cols].notna().any(axis=1)
        df_filtered = df[has_efficacy].copy()
        print(f"含疗效数据: {len(df_filtered)} 篇")

        return df_filtered

    def _process_single(self, article: Dict) -> Optional[Dict]:
        """处理单篇文献"""
        title = article.get("title", "")
        abstract = article.get("abstract", "")
        full_text = f"{title}. {abstract}"

        if not abstract:
            return None

        result = {
            "pmid": article.get("pmid", ""),
            "title": title,
            "abstract": abstract,
            "authors": article.get("authors", ""),
            "first_author": article.get("first_author", ""),
            "journal": article.get("journal", ""),
            "pub_year": article.get("pub_year", ""),
            "doi": article.get("doi", ""),
            "publication_types": article.get("publication_types", ""),
            "mesh_terms": article.get("mesh_terms", ""),
            "keywords": article.get("keywords", ""),
        }

        # === 提取淋巴瘤亚型 ===
        subtypes_found = []
        for subtype, patterns in LYMPHOMA_SUBTYPES.items():
            if _search_keywords(full_text, patterns):
                subtypes_found.append(subtype)
        result["lymphoma_subtypes"] = "; ".join(subtypes_found) if subtypes_found else "Unspecified"

        # === 提取药物/疗法 ===
        drugs_found = []
        for drug, patterns in DRUG_THERAPIES.items():
            if _search_keywords(full_text, patterns):
                drugs_found.append(drug)
        result["drugs"] = "; ".join(drugs_found) if drugs_found else "Unspecified"

        # === 提取研究类型 ===
        study_types_found = []
        for stype, patterns in STUDY_TYPES.items():
            if _search_keywords(full_text, patterns):
                study_types_found.append(stype)
        result["study_type"] = "; ".join(study_types_found) if study_types_found else "Unknown"

        # === 提取样本量 ===
        sample_size = _extract_sample_size(full_text)
        result["sample_size"] = sample_size

        # === 提取疗效指标 ===
        # 注意：_extract_percentage 返回 0-100 的百分数值
        # meta_analysis.py 的 proportion_meta_analysis 期望 0-1 的小数
        # 因此这里除以 100 转换

        # ORR
        orr_pct = _extract_percentage(full_text, EFFICACY_OUTCOMES["ORR"])
        result["orr"] = orr_pct / 100.0 if orr_pct is not None else None

        # CR rate
        cr_pct = _extract_percentage(full_text, EFFICACY_OUTCOMES["CR"])
        result["cr_rate"] = cr_pct / 100.0 if cr_pct is not None else None

        # PR rate
        pr_pct = _extract_percentage(full_text, EFFICACY_OUTCOMES["PR"])
        result["pr_rate"] = pr_pct / 100.0 if pr_pct is not None else None

        # === 样本量回退估算 ===
        # 如果直接提取失败，尝试从 ORR 或 CR 的置信区间宽度反推 n
        if sample_size is None:
            for pct in [orr_pct, cr_pct, pr_pct]:
                est_n = _estimate_sample_size_from_ci(full_text, pct)
                if est_n is not None:
                    result["sample_size"] = est_n
                    break

        # 中位 PFS
        result["pfs_median"] = _extract_median_value(full_text, EFFICACY_OUTCOMES["PFS"])

        # 中位 OS
        result["os_median"] = _extract_median_value(full_text, EFFICACY_OUTCOMES["OS"])

        # HR for PFS
        hr_pfs = _extract_hazard_ratio(full_text, EFFICACY_OUTCOMES["PFS"])
        if hr_pfs:
            result["hr_pfs"] = hr_pfs[0]
            result["hr_pfs_ci_lower"] = hr_pfs[1]
            result["hr_pfs_ci_upper"] = hr_pfs[2]
        else:
            result["hr_pfs"] = None
            result["hr_pfs_ci_lower"] = None
            result["hr_pfs_ci_upper"] = None

        # HR for OS
        hr_os = _extract_hazard_ratio(full_text, EFFICACY_OUTCOMES["OS"])
        if hr_os:
            result["hr_os"] = hr_os[0]
            result["hr_os_ci_lower"] = hr_os[1]
            result["hr_os_ci_upper"] = hr_os[2]
        else:
            result["hr_os"] = None
            result["hr_os_ci_lower"] = None
            result["hr_os_ci_upper"] = None

        # === 提取不良事件 ===
        ae_found = []
        for ae, patterns in ADVERSE_EVENTS.items():
            if _search_keywords(full_text, patterns):
                ae_found.append(ae)
        result["adverse_events"] = "; ".join(ae_found) if ae_found else "Not reported"

        # === 偏倚风险评估辅助信息 ===
        result["is_rct"] = "RCT" in result["study_type"]
        result["is_meta"] = "Meta-Analysis" in result["study_type"]
        result["has_randomization"] = _search_keywords(full_text, [r"random(ly|ized|ised)"])
        result["has_blinding"] = _search_keywords(full_text, [r"blind", r"double.?blind", r"masked"])
        result["has_control_group"] = _search_keywords(full_text, [
            r"control\s*arm", r"control\s*group", r"placebo",
            r"standard\s*of\s*care", r"comparator"
        ])

        return result

    def save(self, df: pd.DataFrame, filename: str = None) -> str:
        """保存清洗后的数据"""
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cleaned_data_{timestamp}"

        output_dir = _get_output_dir()
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, f"{filename}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"✓ 清洗数据已保存: {csv_path}")

        # 同时保存 JSON
        json_path = os.path.join(output_dir, f"{filename}.json")
        df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        print(f"✓ JSON 数据已保存: {json_path}")

        return csv_path


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    # 加载爬取的数据
    results_dir = _get_output_dir()
    json_files = [f for f in os.listdir(results_dir) if f.startswith("pubmed_lymphoma") and f.endswith(".json")]

    if not json_files:
        print("✗ 未找到爬取数据，请先运行 pubmed_crawler.py")
        exit(1)

    latest = sorted(json_files)[-1]
    json_path = os.path.join(results_dir, latest)
    print(f"加载: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    cleaner = DataCleaner()
    df = cleaner.process_articles(articles)
    cleaner.save(df)

    print(f"\n✓ 数据清洗完成！共 {len(df)} 篇含疗效数据")
    print(f"\n数据预览:")
    display_cols = ["pmid", "first_author", "pub_year", "lymphoma_subtypes", "drugs",
                    "study_type", "sample_size", "orr", "cr_rate", "pfs_median", "os_median"]
    print(df[display_cols].head(10).to_string())
