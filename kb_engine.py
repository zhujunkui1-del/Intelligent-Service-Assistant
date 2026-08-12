# -*- coding: utf-8 -*-
"""氢璞创能知识库引擎。

本模块不依赖 Streamlit，便于 app.py 调用和 test_kb.py 独立测试。
"""
from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

PROJECT_DIR = Path(__file__).parent
WEBSITE_URL = "https://www.nowogen.com"
NEWS_LIST_URL = WEBSITE_URL + "/h-col-104.html"
NEWS_ARTICLE_IDS = [
    4, 12, 13, 14, 15, 16, 17, 18, 20, 21,
    22, 23, 25, 26, 27, 28, 29, 30, 31, 32,
]

MODEL_RE = re.compile(r"(?:ST|OCEAN|CESP|E200)[A-Z0-9/]*", re.IGNORECASE)
YEAR_RE = re.compile(r"(?:19|20)\d{2}")
DATE_RE = re.compile(
    r"(20\d{2})[年\-/.](\d{1,2})[月\-/.](\d{1,2})"
)

SECTION_ALIASES = {
    "第四代": "第四代碳复合板电堆",
    "第五代": "第五代碳复合板电堆",
    "第六代": "第六代碳复合板电堆",
    "第七代": "第七代碳复合板电堆",
    "空冷": "阴极封闭式空冷电堆",
    "金属电堆": "金属电堆",
    "船用": "船用燃料电池系统",
    "PEM": "PEM制氢设备",
    "制氢": "PEM制氢设备",
    "合作伙伴": "合作伙伴",
    "大事记": "企业大事记",
    "储能": "氢璞数字化平台与储能解决方案",
    "交通": "氢能源交通解决方案",
    "优势": "核心能力",
}

PARTNER_TERMS = ("合作伙伴", "合作企业", "伙伴", "客户", "整车厂", "系统集成商")
NEWS_TERMS = ("新闻", "发布", "哪一年", "什么时候", "何时", "报道", "官网")


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _clean_bullet(s: str) -> str:
    """清理标题/列表里的 Markdown 强调符号，保留产品尺寸中的 *。"""
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"^\s*[-*]\s+", "", s)
    return _clean(s)


def _split_sections(text: str) -> List[tuple[str, str]]:
    parts = re.split(r"\n(?=## )", text)
    out: List[tuple[str, str]] = []
    for part in parts:
        lines = part.splitlines()
        if not lines:
            continue
        title = lines[0].strip().lstrip("#").strip()
        if title:
            out.append((title, part))
    return out


def _parse_tables(section: str) -> tuple[List[List[List[str]]], str]:
    tables: List[List[List[str]]] = []
    lines = section.splitlines()
    rest: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|") and i + 1 < len(lines) and re.match(
            r"^\s*\|?[\s:\-|]+\|?\s*$", lines[i + 1]
        ):
            j = i
            rows: List[List[str]] = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                cells = [_clean(c) for c in lines[j].strip().strip("|").split("|")]
                rows.append(cells)
                j += 1
            if rows:
                tables.append(rows)
            i = j
        else:
            rest.append(lines[i])
            i += 1
    return tables, "\n".join(rest)


def _is_separator(cells: List[str]) -> bool:
    return all(set(c) <= set("-:| ") for c in cells)


def _table_chunks(section_title: str, rows: List[List[str]]) -> List[str]:
    if not rows:
        return []
    header = rows[0]
    data = [r for r in rows[1:] if not _is_separator(r)]
    if not data:
        return []

    chunks: List[str] = []

    # 键值表：每一行参数一个 chunk。
    if len(header) <= 2:
        for row in data:
            parts: List[str] = []
            for k in range(max(len(header), len(row))):
                h = _clean(header[k]) if k < len(header) else f"参数{k}"
                v = _clean(row[k]) if k < len(row) else ""
                if v:
                    parts.append(f"{h}: {v}")
            if parts:
                chunks.append(_clean(section_title + " | " + "; ".join(parts)))
        return chunks

    # 型号宽表：转置成“每个型号一个完整参数 chunk”。
    if "型号" in header[0] or header[0].lower() == "model":
        for col in range(1, len(header)):
            model = _clean(header[col])
            if not model:
                continue
            parts: List[str] = []
            for row in data:
                if col >= len(row) or not _clean(row[col]):
                    continue
                label = _clean(row[0])
                parts.append(f"{label}: {_clean(row[col])}")
            if parts:
                chunks.append(
                    _clean(section_title + " | " + model + " | " + "; ".join(parts))
                )
        return chunks

    # 普通宽表：每一行一个 chunk。
    for row in data:
        parts: List[str] = []
        for k in range(len(header)):
            if k < len(row) and _clean(row[k]):
                parts.append(f"{_clean(header[k])}: {_clean(row[k])}")
        if parts:
            chunks.append(_clean(section_title + " | " + "; ".join(parts)))
    return chunks


def _model_list_chunk(section_title: str, rows: List[List[str]]) -> Optional[str]:
    """为每个产品分节生成“型号列表 + 关键功率”，保证问型号能一次命中。"""
    if not rows:
        return None
    header = rows[0]
    if "型号" not in header[0] and header[0].lower() != "model":
        return None
    data = [r for r in rows[1:] if not _is_separator(r)]
    if not data or len(header) <= 1:
        return None

    power_row = None
    for row in data:
        label = _clean(row[0]) if row else ""
        if "功率" in label and ("额定" in label or "输出" in label or "高效" in label):
            power_row = row
            break
    if power_row is None:
        power_row = next((r for r in data if "功率" in _clean(r[0])), None)

    models: List[str] = []
    for col in range(1, len(header)):
        model = _clean(header[col])
        if not model:
            continue
        power = _clean(power_row[col]) if power_row and col < len(power_row) else ""
        models.append(f"{model}: {power}" if power else model)

    if not models:
        return None
    return _clean(section_title + " 型号列表 | " + "; ".join(models))


def _bullet_chunks(section_title: str, rest_text: str) -> List[str]:
    chunks: List[str] = []
    for line in rest_text.splitlines():
        m = re.match(r"^\s*(?:[-*]|\d+[.)])\s+(.*)", line)
        if not m:
            continue
        bullet = _clean_bullet(m.group(1))
        if len(bullet) < 4:
            continue

        is_partner = "合作伙伴" in section_title or "伙伴" in section_title
        if is_partner and ("、" in bullet or "，" in bullet):
            cat = ""
            rest = bullet
            if "：" in bullet or ":" in bullet:
                cat, rest = re.split(r"[:：]", bullet, 1)
                cat = _clean(cat)
                rest = _clean(rest)
            for part in re.split(r"[、，]", rest):
                part = _clean(part)
                if part:
                    prefix = f"{cat} | " if cat else ""
                    chunks.append(_clean(section_title + " | " + prefix + part))
        else:
            chunks.append(_clean(section_title + " | " + bullet))
    return chunks


def _sentence_chunks(section_title: str, rest_text: str) -> List[str]:
    lines: List[str] = []
    for line in rest_text.splitlines():
        ls = line.strip()
        if not ls or ls.startswith("#") or ls.startswith("|"):
            continue
        if re.match(r"^\s*(?:[-*]|\d+[.)])\s+", line):
            continue
        lines.append(_clean(ls))
    paragraph = " ".join(lines)
    chunks: List[str] = []
    for sentence in re.split(r"(?<=[。；！？])", paragraph):
        sentence = _clean(sentence)
        if len(sentence) >= 20:
            chunks.append(_clean(section_title + " | " + sentence))
    return chunks


def parse_markdown_docs(project_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """解析目录中的知识库 md，返回结构化文档 chunk。"""
    project_dir = project_dir or PROJECT_DIR
    raw_chunks: List[tuple[str, str, str, int, str]] = []

    for file in sorted(project_dir.glob("*.md")):
        if file.name.upper() == "README.MD":
            continue
        try:
            text = file.read_text(encoding="utf-8")
        except Exception:
            continue

        for section_idx, (section_title, section_text) in enumerate(_split_sections(text)):
            body = section_text.strip()
            if len(body) > 60:
                raw_chunks.append(
                    ("overview", file.name, _clean(section_title + ": " + body[:1400]),
                     section_idx + 1, section_title)
                )

            tables, rest_text = _parse_tables(section_text)
            for rows in tables:
                model_list = _model_list_chunk(section_title, rows)
                if model_list:
                    raw_chunks.append(
                        ("model_list", file.name, model_list, section_idx + 1, section_title)
                    )
                for chunk in _table_chunks(section_title, rows):
                    raw_chunks.append(
                        ("table", file.name, chunk, section_idx + 1, section_title)
                    )
            for chunk in _bullet_chunks(section_title, rest_text):
                raw_chunks.append(
                    ("bullet", file.name, chunk, section_idx + 1, section_title)
                )
            for chunk in _sentence_chunks(section_title, rest_text):
                raw_chunks.append(
                    ("sentence", file.name, chunk, section_idx + 1, section_title)
                )

    seen: set[str] = set()
    docs: List[Dict[str, Any]] = []
    for kind, source, text, page, section in raw_chunks:
        text = _clean(text)
        if len(text) < 12 or text in seen:
            continue
        seen.add(text)
        docs.append(
            {
                "text": text,
                "source": source,
                "page": page,
                "section": section,
                "kind": kind,
                "url": "",
                "date": "",
            }
        )
    return docs


def _local_chunks(docs: List[Dict[str, Any]]) -> tuple[List[str], List[Dict[str, Any]]]:
    return [d["text"] for d in docs], [
        {
            "source": d["source"],
            "page": d["page"],
            "section": d["section"],
            "kind": d["kind"],
            "url": d.get("url", ""),
            "date": d.get("date", ""),
        }
        for d in docs
    ]


def build_local_kb(project_dir: Optional[Path] = None) -> Dict[str, Any]:
    docs = parse_markdown_docs(project_dir)
    chunks, metadatas = _local_chunks(docs)
    kb: Dict[str, Any] = {
        "chunks": chunks,
        "metadatas": metadatas,
        "vectorizer": None,
        "matrix": None,
        "web_count": 0,
    }
    if not chunks:
        return kb

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 3),
        max_features=8000,
        sublinear_tf=True,
    )
    kb["vectorizer"] = vectorizer
    kb["matrix"] = vectorizer.fit_transform(chunks)
    return kb


def build_knowledge_base(project_dir: Optional[Path] = None) -> Dict[str, Any]:
    """构建本地 KB 并启动后台官网抓取。"""
    kb = build_local_kb(project_dir)
    threading.Thread(target=crawl_website, args=(kb,), daemon=True).start()
    return kb


def _extract_date(text: str) -> str:
    for m in DATE_RE.finditer(text):
        year, month, day = m.group(1), m.group(2), m.group(3)
        try:
            return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
        except ValueError:
            continue
    return ""


def _parse_web_page(url: str) -> Optional[Dict[str, str]]:
    try:
        resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
    except Exception:
        return None
    if resp.status_code != 200:
        return None

    try:
        resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        if "页面未找到" in soup.get_text(" ", strip=True)[:80]:
            return None

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        title = soup.title.get_text(strip=True) if soup.title else url
        meta_desc = ""
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            meta_desc = meta["content"].strip()

        body = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n", strip=True))
        body = _clean(body)[:6000]
        if len(body) < 100:
            return None

        date = _extract_date(title + " " + meta_desc + " " + body)
        return {"title": _clean(title), "desc": _clean(meta_desc), "body": body, "date": date, "url": url}
    except Exception:
        return None


def _web_page_chunks(page: Dict[str, str], index: int) -> List[Dict[str, Any]]:
    title = page["title"]
    date = page.get("date", "")
    url = page.get("url", "")
    source = f"官网: {title}"
    section = "官网新闻"
    date_text = f"发布日期: {date}" if date else "发布日期: 未知"

    chunks: List[Dict[str, Any]] = []
    overview = _clean(f"{section} | {title} | {date_text} | 来源: {url} | {page.get('desc', '')}")
    if len(overview) >= 12:
        chunks.append(
            {
                "text": overview,
                "source": source,
                "page": index,
                "section": section,
                "kind": "web_overview",
                "url": url,
                "date": date,
            }
        )

    body = page.get("body", "")
    for sentence in re.split(r"(?<=[。！？])|\n+", body):
        sentence = _clean(sentence)
        if len(sentence) < 24:
            continue
        chunks.append(
            {
                "text": _clean(f"{section} | {title} | {date_text} | 来源: {url} | {sentence}"),
                "source": source,
                "page": index,
                "section": section,
                "kind": "web_sentence",
                "url": url,
                "date": date,
            }
        )
    return chunks[:20]


def crawl_website(kb: Dict[str, Any]) -> Dict[str, Any]:
    """抓取官网并追加到 KB。失败静默跳过，绝不阻塞主页面。"""
    urls = [WEBSITE_URL, NEWS_LIST_URL] + [
        WEBSITE_URL + f"/h-nd-{i}.html" for i in NEWS_ARTICLE_IDS
    ]

    pages: List[Dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_parse_web_page, url): url for url in urls}
        for future in as_completed(futures):
            page = future.result()
            if page:
                pages.append(page)

    if not pages:
        return kb

    web_docs: List[Dict[str, Any]] = []
    for idx, page in enumerate(pages, start=1):
        web_docs.extend(_web_page_chunks(page, idx))

    if not web_docs:
        return kb

    seen_texts = set(kb["chunks"])
    for doc in web_docs:
        text = _clean(doc["text"])
        if len(text) < 12 or text in seen_texts:
            continue
        seen_texts.add(text)
        doc["text"] = text
        kb["chunks"].append(text)
        kb["metadatas"].append(
            {
                "source": doc["source"],
                "page": doc["page"],
                "section": doc["section"],
                "kind": doc["kind"],
                "url": doc["url"],
                "date": doc["date"],
            }
        )
        kb["web_count"] = kb.get("web_count", 0) + 1

    if kb["vectorizer"] is not None and kb["chunks"]:
        kb["matrix"] = kb["vectorizer"].transform(kb["chunks"])
    return kb


def _boost_score(meta: Dict[str, Any], text: str, query: str) -> float:
    bonus = 0.0
    query_upper = query.upper()
    text_upper = text.upper()

    models = set(MODEL_RE.findall(query_upper))
    if models:
        chunk_models = set(MODEL_RE.findall(text_upper))
        if models & chunk_models:
            bonus += 3.0

    years = set(YEAR_RE.findall(query))
    if years:
        chunk_years = set(YEAR_RE.findall(text))
        if years & chunk_years:
            bonus += 1.5

    section = str(meta.get("section", ""))
    for alias, canonical in SECTION_ALIASES.items():
        if alias.lower() in query.lower():
            if canonical in section or canonical in text:
                bonus += 0.8
                break

    if any(term in query for term in PARTNER_TERMS) and "合作伙伴" in section:
        bonus += 1.5
    if any(term in query for term in NEWS_TERMS) and ("官网新闻" in section or "官网:" in str(meta.get("source", ""))):
        bonus += 1.5
    return bonus


def search_kb(kb: Dict[str, Any], query: str, k: int = 12) -> List[Dict[str, Any]]:
    if kb.get("vectorizer") is None or not kb.get("chunks"):
        return []

    qvec = kb["vectorizer"].transform([query])
    scores = cosine_similarity(qvec, kb["matrix"]).flatten()

    boosted = np.array(
        [
            float(score) + _boost_score(kb["metadatas"][i], kb["chunks"][i], query)
            for i, score in enumerate(scores)
        ]
    )
    top = np.argsort(boosted)[-k:][::-1]

    results: List[Dict[str, Any]] = []
    for i in top:
        if scores[i] <= 0 and boosted[i] <= 0:
            continue
        meta = kb["metadatas"][i]
        results.append(
            {
                "text": kb["chunks"][i],
                "source": meta.get("source", ""),
                "page": meta.get("page", 0),
                "section": meta.get("section", ""),
                "kind": meta.get("kind", ""),
                "url": meta.get("url", ""),
                "date": meta.get("date", ""),
                "score": float(scores[i]),
            }
        )
    return results
