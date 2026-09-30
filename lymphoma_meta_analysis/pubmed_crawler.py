#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PubMed 爬虫模块
====================
通过 NCBI E-utilities API 检索淋巴瘤用药相关文献，
抓取标题、摘要、作者、期刊、发表日期、MeSH词等元数据。

使用 NCBI 官方 API（非网页爬虫），遵守 NCBI 使用政策：
  - 每秒最多 3 次请求（无 API key 时）
  - 有 API key 时每秒最多 10 次
"""

import time
import json
import re
import os
from datetime import datetime
from typing import List, Dict, Optional

import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm


# ============================================================
# 配置
# ============================================================
class PubMedConfig:
    """PubMed E-utilities 配置"""

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    # 如果你有 NCBI API key，填入此处可提高速率限制（10 req/s）
    API_KEY = ""  # 例如: "a1b2c3d4e5f6g7h8i9j0"

    # 无 API key 时，每秒最多 3 次请求
    RATE_LIMIT_DELAY = 0.34 if not API_KEY else 0.1

    # 每批 efetch 的 PMID 数量（NCBI 建议不超过 200）
    BATCH_SIZE = 100

    # 检索超时（秒）— EFetch 批量获取可能较慢，设为 120 秒
    TIMEOUT = 120

    # 结果目录
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ============================================================
# 查询构建器
# ============================================================
def build_lymphoma_drug_query(
    lymphoma_types: Optional[List[str]] = None,
    drug_names: Optional[List[str]] = None,
    study_types: Optional[List[str]] = None,
    date_range: Optional[str] = None,
    max_results: int = 500,
) -> str:
    """
    构建淋巴瘤用药的 PubMed 检索式。

    Parameters
    ----------
    lymphoma_types : list
        淋巴瘤类型，如 ["Hodgkin Lymphoma", "Diffuse large B-cell lymphoma"]
    drug_names : list
        药物名称，如 ["rituximab", "brentuximab vedotin", "CAR-T"]
    study_types : list
        研究类型筛选，如 ["Clinical Trial", "Meta-Analysis", "Randomized Controlled Trial"]
    date_range : str
        日期范围，如 "2015/01/01:2025/12/31"
    max_results : int
        最大返回结果数

    Returns
    -------
    str : 完整的 PubMed 检索式
    """
    # 默认淋巴瘤类型
    if lymphoma_types is None:
        lymphoma_types = [
            "Lymphoma, Non-Hodgkin",      # 非霍奇金淋巴瘤
            "Hodgkin Disease",             # 霍奇金淋巴瘤
            "Lymphoma, Large B-Cell, Diffuse",  # 弥漫大B细胞淋巴瘤
            "Lymphoma, Follicular",        # 滤泡性淋巴瘤
            "Mantle-Cell Lymphoma",        # 套细胞淋巴瘤
            "Lymphoma, T-Cell",            # T细胞淋巴瘤
        ]

    # 默认药物/疗法
    if drug_names is None:
        drug_names = [
            "rituximab",                   # 利妥昔单抗
            "brentuximab vedotin",         # 维布妥昔单抗
            "CAR-T cell therapy",          # CAR-T 细胞疗法
            "pembrolizumab",               # 帕博利珠单抗
            "nivolumab",                   # 纳武利尤单抗
            "lenalidomide",                # 来那度胺
            "ibrutinib",                   # 伊布替尼
            "venetoclax",                  # 维奈托克
            "polatuzumab vedotin",         # 波拉妥珠单抗
            "CHOP",                        # CHOP 方案
            "R-CHOP",                      # R-CHOP 方案
        ]

    # 默认研究类型（用 Title/Abstract 搜索，比 Publication Type 更宽松）
    if study_types is None:
        study_types = [
            "randomized",
            "clinical trial",
            "meta-analysis",
            "phase 2",
            "phase 3",
        ]

    # 构建淋巴瘤部分 — 用 MeSH + Title/Abstract 双重匹配
    lymphoma_terms = " OR ".join(f'"{t}"[MeSH]' for t in lymphoma_types)

    # 构建药物部分
    drug_terms = " OR ".join(f'"{d}"[Title/Abstract]' for d in drug_names)

    # 构建研究类型部分 — 用 Title/Abstract 搜索（更宽松）
    study_terms = " OR ".join(f'"{s}"[Title/Abstract]' for s in study_types)

    # 组合检索式：淋巴瘤 AND 药物 AND (研究类型 OR 临床试验过滤)
    # 使用 [Filter] 做临床试验过滤，比 [Publication Type] 更全面
    query = f"({lymphoma_terms}) AND ({drug_terms})"
    query += f" AND (({study_terms}) OR "
    query += '"clinical trial"[Filter] OR "randomized controlled trial"[Filter] OR "meta-analysis"[Filter])'

    # 添加日期范围（注意：日期值不能加引号，否则 PubMed 返回 0 结果）
    if date_range:
        query += f' AND {date_range}[Date - Publication]'

    # 添加英文限制
    query += ' AND "English"[Language]'

    return query


# ============================================================
# PubMed 爬虫
# ============================================================
class PubMedCrawler:
    """PubMed E-utilities 爬虫"""

    def __init__(self, config: PubMedConfig = None):
        self.config = config or PubMedConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "LymphomaMetaAnalysis/1.0 (research use)",
            "Accept": "application/xml",
        })
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

    def _get_params(self, extra: dict = None) -> dict:
        """构建基础参数（含 API key）"""
        params = {"db": "pubmed", "retmode": "json"}
        if self.config.API_KEY:
            params["api_key"] = self.config.API_KEY
        if extra:
            params.update(extra)
        return params

    def _rate_limit(self):
        """速率限制"""
        time.sleep(self.config.RATE_LIMIT_DELAY)

    # --------------------------------------------------------
    # Step 1: ESearch — 获取 PMID 列表
    # --------------------------------------------------------
    def search(self, query: str, max_results: int = 500) -> List[str]:
        """
        执行 ESearch，返回 PMID 列表。

        Parameters
        ----------
        query : str
            PubMed 检索式
        max_results : int
            最大返回数量

        Returns
        -------
        list[str] : PMID 列表
        """
        print(f"\n{'='*60}")
        print("PubMed ESearch 检索")
        print(f"{'='*60}")
        print(f"检索式: {query[:200]}...")
        print(f"最大结果数: {max_results}")

        params = self._get_params({
            "term": query,
            "retmax": max_results,
            "sort": "date",
            "field": "all",
        })

        try:
            resp = self.session.get(
                self.config.ESEARCH_URL,
                params=params,
                timeout=self.config.TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            result = data.get("esearchresult", {})
            total_count = int(result.get("count", 0))
            pmids = result.get("idlist", [])

            print(f"总匹配数: {total_count}")
            print(f"获取 PMID 数: {len(pmids)}")

            if not pmids:
                print("⚠ 未找到任何结果，请检查检索式")
                return []

            return pmids

        except requests.RequestException as e:
            print(f"✗ ESearch 请求失败: {e}")
            return []

    # --------------------------------------------------------
    # Step 2: EFetch — 批量获取文献全文摘要
    # --------------------------------------------------------
    def fetch_details(self, pmids: List[str]) -> List[Dict]:
        """
        批量 EFetch，获取每篇文献的详细信息。

        Returns
        -------
        list[dict] : 文献信息列表
        """
        print(f"\n{'='*60}")
        print("PubMed EFetch 批量获取文献详情")
        print(f"{'='*60}")
        print(f"总 PMID 数: {len(pmids)}")
        print(f"每批大小: {self.config.BATCH_SIZE}")

        all_articles = []

        # 分批获取
        batches = [
            pmids[i:i + self.config.BATCH_SIZE]
            for i in range(0, len(pmids), self.config.BATCH_SIZE)
        ]

        for batch_idx, batch in enumerate(tqdm(batches, desc="批量获取", unit="batch")):
            articles = self._fetch_batch(batch)
            all_articles.extend(articles)
            self._rate_limit()

        print(f"\n成功获取: {len(all_articles)} / {len(pmids)} 篇")
        return all_articles

    def _fetch_batch(self, pmid_list: List[str]) -> List[Dict]:
        """获取一批 PMID 的详细信息"""
        pmid_str = ",".join(pmid_list)

        params = {
            "db": "pubmed",
            "id": pmid_str,
            "retmode": "xml",
            "rettype": "abstract",
        }
        if self.config.API_KEY:
            params["api_key"] = self.config.API_KEY

        try:
            resp = self.session.get(
                self.config.EFETCH_URL,
                params=params,
                timeout=self.config.TIMEOUT,
            )
            resp.raise_for_status()
            return self._parse_xml(resp.text)

        except requests.RequestException as e:
            print(f"✗ EFetch 批次失败: {e}")
            return []

    # --------------------------------------------------------
    # Step 3: XML 解析
    # --------------------------------------------------------
    def _parse_xml(self, xml_text: str) -> List[Dict]:
        """解析 PubMed XML，提取结构化信息"""
        articles = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            print(f"✗ XML 解析失败: {e}")
            return articles

        for article_elem in root.findall(".//PubmedArticle"):
            try:
                article = self._parse_single_article(article_elem)
                if article:
                    articles.append(article)
            except Exception as e:
                # 跳过解析失败的条目
                continue

        return articles

    def _parse_single_article(self, elem: ET.Element) -> Optional[Dict]:
        """解析单篇文献"""
        article = {}

        # PMID
        pmid_elem = elem.find(".//PMID")
        article["pmid"] = pmid_elem.text if pmid_elem is not None else ""

        # 标题
        title_elem = elem.find(".//ArticleTitle")
        article["title"] = self._get_text_with_children(title_elem) if title_elem is not None else ""

        # 摘要
        abstract_parts = []
        for abs_elem in elem.findall(".//AbstractText"):
            label = abs_elem.get("Label", "")
            text = self._get_text_with_children(abs_elem)
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        article["abstract"] = " ".join(abstract_parts)

        # 作者
        authors = []
        for author in elem.findall(".//Author"):
            last = author.findtext("LastName", "")
            init = author.findtext("Initials", "")
            if last:
                authors.append(f"{last} {init}".strip())
        article["authors"] = "; ".join(authors)
        article["first_author"] = authors[0] if authors else ""
        article["n_authors"] = len(authors)

        # 期刊
        journal_elem = elem.find(".//Journal")
        if journal_elem is not None:
            article["journal"] = journal_elem.findtext("Title", "")
            iso_elem = journal_elem.find(".//ISOAbbreviation")
            article["journal_iso"] = iso_elem.text if iso_elem is not None else ""
        else:
            article["journal"] = ""
            article["journal_iso"] = ""

        # 发表日期
        pub_date = elem.find(".//PubDate")
        if pub_date is not None:
            year = pub_date.findtext("Year", "")
            month = pub_date.findtext("Month", "01")
            day = pub_date.findtext("Day", "01")
            article["pub_year"] = year
            article["pub_date"] = f"{year} {month} {day}".strip()
        else:
            article["pub_year"] = ""
            article["pub_date"] = ""

        # DOI
        doi_elem = elem.find(".//ArticleId[@IdType='doi']")
        article["doi"] = doi_elem.text if doi_elem is not None else ""

        # 文章类型
        pub_types = []
        for pt in elem.findall(".//PublicationType"):
            if pt.text:
                pub_types.append(pt.text)
        article["publication_types"] = "; ".join(pub_types)

        # MeSH 词
        mesh_terms = []
        for mesh in elem.findall(".//MeshHeading/DescriptorName"):
            if mesh.text:
                mesh_terms.append(mesh.text)
        article["mesh_terms"] = "; ".join(mesh_terms)

        # 关键词
        keywords = []
        for kw in elem.findall(".//Keyword"):
            if kw.text:
                keywords.append(kw.text)
        article["keywords"] = "; ".join(keywords)

        # 语言
        article["language"] = elem.findtext(".//Language", "")

        # 国家
        article["country"] = elem.findtext(".//Country", "")

        return article

    def _get_text_with_children(self, elem: ET.Element) -> str:
        """获取元素及其所有子元素的文本（处理 <i>, <b> 等标签）"""
        if elem is None:
            return ""
        text = elem.text or ""
        for child in elem:
            text += child.text or ""
            if child.tail:
                text += child.tail
        return text.strip()

    # --------------------------------------------------------
    # 保存结果
    # --------------------------------------------------------
    def save_results(self, articles: List[Dict], filename: str = None) -> str:
        """保存结果到 JSON 和 CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pubmed_lymphoma_{timestamp}"

        # JSON
        json_path = os.path.join(self.config.OUTPUT_DIR, f"{filename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"✓ JSON 已保存: {json_path}")

        # CSV
        import pandas as pd
        csv_path = os.path.join(self.config.OUTPUT_DIR, f"{filename}.csv")
        df = pd.DataFrame(articles)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"✓ CSV 已保存: {csv_path}")

        return json_path

    # --------------------------------------------------------
    # 完整流程
    # --------------------------------------------------------
    def run(
        self,
        query: str = None,
        max_results: int = 500,
        custom_lymphoma_types: List[str] = None,
        custom_drug_names: List[str] = None,
        date_range: str = None,
    ) -> List[Dict]:
        """
        完整爬取流程：检索 → 获取详情 → 保存

        Parameters
        ----------
        query : str
            自定义检索式（如果为 None，则自动构建）
        max_results : int
            最大结果数
        custom_lymphoma_types : list
            自定义淋巴瘤类型
        custom_drug_names : list
            自定义药物名称
        date_range : str
            日期范围

        Returns
        -------
        list[dict] : 文献信息列表
        """
        if query is None:
            query = build_lymphoma_drug_query(
                lymphoma_types=custom_lymphoma_types,
                drug_names=custom_drug_names,
                date_range=date_range,
                max_results=max_results,
            )

        # Step 1: 搜索
        pmids = self.search(query, max_results=max_results)
        if not pmids:
            return []

        # Step 2: 获取详情
        articles = self.fetch_details(pmids)
        if not articles:
            print("✗ 未获取到任何文献详情")
            return []

        # Step 3: 保存
        self.save_results(articles)

        # 打印摘要统计
        self._print_summary(articles)

        return articles

    def _print_summary(self, articles: List[Dict]):
        """打印结果摘要"""
        print(f"\n{'='*60}")
        print("检索结果摘要")
        print(f"{'='*60}")
        print(f"总文献数: {len(articles)}")

        # 按年份统计
        years = {}
        for a in articles:
            y = a.get("pub_year", "未知")
            years[y] = years.get(y, 0) + 1
        print("\n按年份分布:")
        for y in sorted(years.keys(), reverse=True)[:10]:
            print(f"  {y}: {years[y]} 篇")

        # 按期刊统计
        journals = {}
        for a in articles:
            j = a.get("journal", "未知")
            journals[j] = journals.get(j, 0) + 1
        print("\nTop 10 期刊:")
        for j, c in sorted(journals.items(), key=lambda x: -x[1])[:10]:
            print(f"  {j}: {c} 篇")

        # 按文章类型统计
        types = {}
        for a in articles:
            for pt in a.get("publication_types", "").split("; "):
                if pt:
                    types[pt] = types.get(pt, 0) + 1
        print("\n文章类型分布:")
        for t, c in sorted(types.items(), key=lambda x: -x[1])[:10]:
            print(f"  {t}: {c} 篇")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    crawler = PubMedCrawler()

    # 方式1: 使用默认检索式
    articles = crawler.run(max_results=200, date_range="2015/01/01:2025/12/31")

    # 方式2: 自定义检索（取消注释使用）
    # articles = crawler.run(
    #     max_results=300,
    #     custom_lymphoma_types=["Lymphoma, Large B-Cell, Diffuse"],
    #     custom_drug_names=["rituximab", "R-CHOP", "CAR-T cell therapy"],
    #     date_range="2018/01/01:2025/12/31",
    # )

    # 方式3: 完全自定义检索式
    # articles = crawler.run(
    #     query='("lymphoma"[MeSH]) AND ("rituximab"[Title/Abstract]) AND ("Clinical Trial"[Publication Type])',
    #     max_results=500,
    # )

    print(f"\n✓ 完成！共获取 {len(articles)} 篇文献")
