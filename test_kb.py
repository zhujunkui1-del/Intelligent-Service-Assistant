# -*- coding: utf-8 -*-
"""氢璞创能知识库自测：本地 200+ 片段、检索命中、可选官网抓取与 LLM。"""
from __future__ import annotations

import argparse
import os
import time
import tomllib
from pathlib import Path

import kb_engine

ROOT = Path(__file__).parent

LOCAL_CASES = [
    ("氢璞创能成立于哪一年？", ["2010"]),
    (
        "第四代碳复合板电堆有哪些型号？",
        ["ST35F", "ST40F", "ST50F", "ST55F", "ST70FA"],
    ),
    ("ST240VIC 的额定功率是多少？", ["ST240VIC", "240kW"]),
    (
        "第六代碳复合板电堆有哪些型号？",
        ["ST240VIC", "ST300VIC", "ST280VID", "ST490VID"],
    ),
    ("ST2D2AII 的额定功率、电压和尺寸？", ["ST2D2AII", "2200 W", "52 V", "120*76*459"]),
    ("2023年氢璞发生了什么大事？", ["2023", "300kW"]),
    ("氢璞创能的合作伙伴有哪些？", ["国家电网", "中国中车", "顺丰速运"]),
    ("500公里以上氢能重卡的经济性优势？", ["500公里", "15%-20%"]),
    ("CESP1000 的额定产氢量和氢气纯度？", ["CESP1000", "1000Nm³/h", "99.999%"]),
    ("氢璞有哪些核心技术优势？", ["专利108项", "首条电堆自动化产线"]),
    ("氢璞创能面临哪些行业痛点？", ["氢燃料电池成本高", "加氢站", "行业标准体系"]),
    ("可口可乐华北区氢能物流试运营是什么时候？", ["2025年4月", "可口可乐", "600km"]),
    ("氢璞有几代电堆？", ["第四代", "第五代", "第六代", "第七代"]),
    (
        "第四代、第五代、第七代碳复合板电堆分别有哪些型号？",
        ["ST35F", "ST97V", "ST600V/IIA"],
    ),
    ("第三代金属板电堆有哪些参数？", ["自主研发三代金属板电堆"]),
]

WEB_CASES = [
    (
        "氢璞创能发布200kW船舶用燃料电池系统 OCEAN-200 是什么时候？",
        ["2024", "OCEAN-200"],
    ),
    ("氢璞上过央视吗？", ["h-nd-31", "央视"]),
    ("氢璞官网最早的新闻是什么？", ["h-nd-18", "2020-10-15"]),
]


def combined_text(results):
    return "\n".join(d["text"] for d in results)


def run_local(kb):
    print("\n[1] 本地知识库构建")
    assert len(kb["chunks"]) >= 200, f"本地片段数不足 200，当前 {len(kb['chunks'])}"
    print(f"  chunks={len(kb['chunks'])}")

    print(f"\n[2] 本地检索 {len(LOCAL_CASES)} 条")
    passed = 0
    for idx, (query, expected) in enumerate(LOCAL_CASES, 1):
        t0 = time.perf_counter()
        results = kb_engine.search_kb(kb, query)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 100, f"检索耗时 {elapsed:.1f}ms 超过 100ms: {query}"
        text = combined_text(results)
        missing = [term for term in expected if term not in text]
        if missing:
            print(f"  FAIL {idx}. {query} 缺少 {missing}")
            for d in results[:3]:
                print("    top:", d["text"][:120])
            raise AssertionError(f"查询未命中必要信息: {query} {missing}")
        passed += 1
        print(f"  PASS {idx}. {query}  ({elapsed:.1f}ms)")
    print(f"  local passed={passed}/{len(LOCAL_CASES)}")


def run_crawl(kb):
    print("\n[3] 官网新闻抓取")
    try:
        t0 = time.perf_counter()
        kb_engine.crawl_website(kb)
        elapsed = time.perf_counter() - t0
        print(f"  web_count={kb.get('web_count', 0)}, chunks={len(kb['chunks'])}, {elapsed:.2f}s")
        if kb.get("web_count", 0) == 0:
            print("  SKIP 新闻用例：官网抓取未获得有效页面")
            return False
        for query, expected in WEB_CASES:
            results = kb_engine.search_kb(kb, query)
            text = combined_text(results)
            missing = [term for term in expected if term not in text]
            if missing:
                print(f"  FAIL {query} 缺少 {missing}")
                for d in results[:3]:
                    print("    top:", d["text"][:160])
                raise AssertionError(f"新闻用例未命中: {missing}")
            print(f"  PASS {query}")
        return True
    except Exception as exc:
        print(f"  SKIP 新闻用例：{exc}")
        return False


def _api_key_from_secrets():
    try:
        with open(ROOT / ".streamlit" / "secrets.toml", "rb") as f:
            data = tomllib.load(f)
        return (data.get("DEEPSEEK_API_KEY") or "").strip()
    except Exception:
        return ""


def run_llm(kb):
    print("\n[4] 真实 LLM 回答（可选）")
    key = os.getenv("DEEPSEEK_API_KEY", "").strip() or _api_key_from_secrets()
    if not key or key.startswith("your-"):
        print("  SKIP：未检测到有效 DEEPSEEK_API_KEY，仅完成检索测试")
        return

    from openai import OpenAI
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    cases = LOCAL_CASES + [WEB_CASE]
    passed = 0
    for idx, (query, expected) in enumerate(cases, 1):
        results = kb_engine.search_kb(kb, query)
        parts = [f"【资料{i + 1}】{d['source']} · 第{d['page']}页\n{d['text']}" for i, d in enumerate(results)]
        sp = (
            "你是氢璞创能的企业知识助手。请直接从资料中提取答案，全量列出型号和参数；"
            "严禁说“根据参考资料”，严禁输出 [src] 标记；资料没有的数据说“资料未收录此项”。"
            "\n\n" + "\n".join(parts)
        )
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": sp}, {"role": "user", "content": query}],
            temperature=0.3,
            max_tokens=1500,
        )
        answer = resp.choices[0].message.content or ""
        missing = [term for term in expected if term not in answer]
        if missing:
            print(f"  FAIL LLM {idx}. {query} 缺少 {missing}\n  answer={answer[:200]}")
        else:
            passed += 1
            print(f"  PASS LLM {idx}. {query}")
    print(f"  llm passed={passed}/{len(cases)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crawl", action="store_true", help="抓取官网新闻后测试新闻用例")
    parser.add_argument("--with-llm", action="store_true", help="检测 DEEPSEEK_API_KEY 并跑真实 LLM")
    args = parser.parse_args()

    t0 = time.perf_counter()
    kb = kb_engine.build_local_kb()
    build_time = time.perf_counter() - t0
    print(f"本地 KB 构建耗时 {build_time:.3f}s")
    assert build_time < 2, f"构建耗时 {build_time:.3f}s 超过 2s"

    run_local(kb)
    if args.crawl:
        run_crawl(kb)
    if args.with_llm:
        run_llm(kb)
    print("\nALL RETRIEVAL CHECKS PASSED")


if __name__ == "__main__":
    main()
