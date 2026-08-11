import streamlit as st
import os
import re
import threading
from pathlib import Path
import pdfplumber
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

st.set_page_config(
    page_title="氢璞创能 · 智能知识助手",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- CSS + Dark Mode JS ----
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F0F4F8 0%, #E2E8F0 100%);
        border-right: 1px solid #CBD5E1;
    }
    section[data-testid="stSidebar"] h3 { color: #0F4C81; font-size: 0.95rem; font-weight: 700; margin-top: 1rem; }
    section[data-testid="stSidebar"] hr { border-color: #CBD5E1; margin: 0.4rem 0; }

    html[data-theme="dark"] [data-testid="stSidebar"],
    html[data-theme="dark"] [data-testid="stSidebar"] > div:first-child,
    html[data-theme="dark"] [data-testid="stSidebarContent"] {
        background: #111827 !important;
        background-color: #111827 !important;
        background-image: none !important;
    }
    html[data-theme="dark"] [data-testid="stSidebar"] { border-right: 1px solid #1F2937 !important; }
    html[data-theme="dark"] [data-testid="stSidebar"] h3 { color: #93C5FD !important; }
    html[data-theme="dark"] [data-testid="stSidebar"] hr { border-color: #374151 !important; }
    html[data-theme="dark"] .status-ok { background: #064E3B !important; border-color: #059669 !important; color: #D1FAE5 !important; }
    html[data-theme="dark"] .status-warn { background: #78350F !important; border-color: #D97706 !important; color: #FEF3C7 !important; }
    html[data-theme="dark"] .status-info { background: #1E3A5F !important; border-color: #3B82F6 !important; color: #DBEAFE !important; }
    html[data-theme="dark"] .app-title { color: #93C5FD !important; border-bottom-color: #22D3EE !important; }
    html[data-theme="dark"] [data-testid="stChatInput"] { background: #0E1117 !important; }
    html[data-theme="dark"] [data-testid="stSidebar"] .stButton > button { background: #1F2937 !important; border-color: #374151 !important; color: #D1D5DB !important; }
    html[data-theme="dark"] [data-testid="stSidebar"] input { background: #1F2937 !important; border-color: #374151 !important; color: #F9FAFB !important; }

    section[data-testid="stSidebar"] button[data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] { display: none !important; visibility: hidden !important; pointer-events: none !important; }

    .app-title { color: #0F4C81; font-size: 1.8rem; font-weight: 800; border-bottom: 3px solid #00B4D8; padding-bottom: 0.4rem; margin-bottom: 0; }
    .status-box { border-radius: 8px; padding: 0.4rem 0.65rem; margin: 0.3rem 0; font-size: 0.82rem; }
    .status-ok { background: #ECFDF5; border: 1px solid #6EE7B7; color: #065F46; }
    .status-warn { background: #FFFBEB; border: 1px solid #FCD34D; color: #92400E; }
    .status-info { background: #EFF6FF; border: 1px solid #93C5FD; color: #1E40AF; }
    [data-testid="stChatMessage"] { border-radius: 10px; padding: 0.4rem 0.8rem; }
    .stChatMessage:first-of-type { margin-top: 5px; }
    section[data-testid="stSidebar"] .stButton > button { border-radius: 6px; font-size: 0.82rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    [data-testid="stChatInput"] { position: sticky; bottom: 0; z-index: 100; background: inherit; padding-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ---- Config ----
PROJECT_DIR = Path(__file__).parent
WEBSITE_URL = "https://www.nowogen.com"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

PROVIDERS = {
    "DeepSeek": {"base": "https://api.deepseek.com", "model": "deepseek-chat"},
    "OpenAI":   {"base": "https://api.openai.com/v1",    "model": "gpt-4o-mini"},
}

# ---- Session State Init ----
def init_session():
    defaults = {
        "api_key": "",
        "openai_key": "",
        "provider": "DeepSeek",
        "messages": [],
        "pending_question": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if not st.session_state.messages:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "您好! 我是氢璞创能的企业知识与智能服务助手，请随时向我提问。"
        }]

init_session()

def get_api_config():
    p = st.session_state.provider
    cfg = PROVIDERS[p]
    k = st.session_state.api_key if p == "DeepSeek" else st.session_state.openai_key
    return k, cfg["base"], cfg["model"]

# ---- Knowledge Base ----
@st.cache_resource
def build_knowledge_base():
    docs = _parse_pdfs()
    kb = {"chunks": [], "metadatas": [], "vectorizer": None, "matrix": None, "web_count": 0}
    if docs:
        chunks, metas = _chunk_docs(docs)
        kb["chunks"] = chunks
        kb["metadatas"] = metas
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), max_features=3000)
        kb["matrix"] = vec.fit_transform(chunks)
        kb["vectorizer"] = vec
    threading.Thread(target=_scrape_and_append, args=(kb,), daemon=True).start()
    return kb

def _parse_pdfs():
    docs = []
    for p in sorted(PROJECT_DIR.glob("*.pdf")):
        if p.name.startswith("00_"): continue
        try:
            with pdfplumber.open(p) as pdf:
                for i, page in enumerate(pdf.pages):
                    t = (page.extract_text() or "").strip()
                    if len(t) > 20:
                        docs.append({"source": p.name, "page": i + 1, "text": t})
        except: pass
    return docs

def _chunk_docs(docs):
    chunks, metas = [], []
    for d in docs:
        text = d["text"]
        start = 0; ci = 0
        while start < len(text):
            c = text[start:start + CHUNK_SIZE].strip()
            if c:
                chunks.append(c)
                metas.append({"source": d["source"], "page": d["page"], "ci": ci})
                ci += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks, metas

def _scrape_and_append(kb):
    try:
        resp = requests.get(WEBSITE_URL, timeout=3, headers={"User-Agent": "Mozilla/5.0"})
        resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]): tag.decompose()
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n", strip=True))
        if len(text) > 100 and kb["vectorizer"] is not None:
            wc, wm = _chunk_docs([{"source": "官网: www.nowogen.com", "page": 1, "text": text}])
            kb["chunks"].extend(wc); kb["metadatas"].extend(wm)
            kb["matrix"] = kb["vectorizer"].transform(kb["chunks"])
            kb["web_count"] = len(wc)
    except: pass

def search_kb(kb, query, k=5):
    if kb["vectorizer"] is None or not kb["chunks"]: return []
    qvec = kb["vectorizer"].transform([query])
    scores = cosine_similarity(qvec, kb["matrix"]).flatten()
    top = np.argsort(scores)[-k:][::-1]
    return [{"text": kb["chunks"][i], "source": kb["metadatas"][i]["source"],
             "page": kb["metadatas"][i]["page"], "score": float(scores[i])}
            for i in top if scores[i] > 0]

# ---- Chat ----
def generate_answer(query, ctx_docs, history):
    api_key, base_url, model = get_api_config()
    if not api_key: return None, "API Key not configured"
    parts = [f"[src {i+1}] {d['source']} p.{d['page']}:\n{d['text']}" for i, d in enumerate(ctx_docs)]
    sp = f"""You are an enterprise knowledge assistant for Beijing Hydrotrans Creative Energy Technology Co., Ltd.
Answer user questions based on the provided reference materials.

=== References ===
{chr(10).join(parts)}
=== End ===

Rules:
1. Answer strictly based on references - do not fabricate
2. If not covered, clearly state so
3. Cite sources like [src 1], [src 2]
4. Be concise, professional, well-organized
5. Respond in Chinese"""
    try:
        cli = OpenAI(api_key=api_key, base_url=base_url)
        msgs = [{"role": "system", "content": sp}]
        msgs.extend(history[-6:])
        msgs.append({"role": "user", "content": query})
        resp = cli.chat.completions.create(model=model, messages=msgs, temperature=0.3, max_tokens=1500)
        return resp.choices[0].message.content, None
    except Exception as e:
        return None, f"API Error: {e}"

kb = build_knowledge_base()

# ---- UI: Title ----
st.markdown(
    '<h1 class="app-title"><i class="fa-solid fa-bolt" style="color: var(--accent);"></i> '
    '氢璞创能 · 企业知识与智能服务助手</h1>',
    unsafe_allow_html=True
)

# ---- Sidebar ----
with st.sidebar:
    st.markdown('<h3><i class="fa-solid fa-gear"></i> 设置</h3>', unsafe_allow_html=True)
    st.session_state.provider = st.selectbox(
        "LLM 提供商", ["DeepSeek", "OpenAI"],
        index=0 if st.session_state.provider == "DeepSeek" else 1
    )
    if st.session_state.provider == "DeepSeek":
        st.session_state.api_key = st.text_input(
            "DeepSeek API Key", type="password", value=st.session_state.api_key,
            placeholder="sk-...", key="ds_key_input"
        )
    else:
        st.session_state.openai_key = st.text_input(
            "OpenAI API Key", type="password", value=st.session_state.openai_key,
            placeholder="sk-...", key="oai_key_input"
        )
    current_key, _, _ = get_api_config()
    if current_key:
        st.markdown(
            f'<div class="status-box status-ok">'
            f'<i class="fa-solid fa-circle-check"></i> {st.session_state.provider} Key 已配置</div>',
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<h3><i class="fa-solid fa-database"></i> 知识库</h3>', unsafe_allow_html=True)
    total = len(kb["chunks"])
    if total > 0:
        web_info = f' <span style="font-size:0.75rem;">(+{kb["web_count"]} 官网)</span>' if kb["web_count"] else ""
        st.markdown(
            f'<div class="status-box status-ok">'
            f'<i class="fa-solid fa-check-circle"></i> {total} 个知识片段{web_info}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-box status-warn">'
            '<i class="fa-solid fa-triangle-exclamation"></i> 未找到 PDF 文件</div>',
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<h3><i class="fa-regular fa-lightbulb"></i> 推荐问题</h3>', unsafe_allow_html=True)
    for q in [
        "氢璞创能的主要产品有哪些?",
        "氢璞的电堆功率范围是多少?",
        "氢璞在氢燃料电池领域有哪些技术优势?",
        "请介绍一下氢璞的核心技术路线",
        "氢璞的产品应用在哪些场景?",
    ]:
        if st.button(q, key=f"s_{abs(hash(q))}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

# ---- Chat ----
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

query = st.session_state.pop("pending_question", None)
if query is None:
    query = st.chat_input("请输入您的问题...")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    api_key, base_url, model = get_api_config()
    if not api_key:
        st.toast(f"请先配置 {st.session_state.provider} API Key", icon=":material/warning:")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"请在侧边栏配置 {st.session_state.provider} API Key 后再提问。"
        })
    elif not kb["chunks"]:
        st.toast("知识库为空", icon=":material/error:")
        st.session_state.messages.append({
            "role": "assistant", "content": "知识库为空，请检查 PDF 文件是否存在。"
        })
    else:
        with st.spinner("正在检索..."):
            docs = search_kb(kb, query)
        if docs:
            with st.spinner("生成回答..."):
                hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                ans, err = generate_answer(query, docs, hist)
            if err:
                st.toast(err, icon=":material/error:")
                ans = f"API 调用失败: {err}"
            with st.chat_message("assistant"):
                st.markdown(ans)
                with st.expander("参考来源"):
                    for i, d in enumerate(docs):
                        st.markdown(
                            f'<p style="margin:0.2rem 0;font-size:0.85rem;">'
                            f'<i class="fa-solid fa-link"></i> '
                            f'<strong>[来源{i+1}]</strong> {d["source"]} · 第{d["page"]}页</p>',
                            unsafe_allow_html=True
                        )
                        st.text(d["text"][:250] + ("..." if len(d["text"]) > 250 else ""))
            st.session_state.messages.append({"role": "assistant", "content": ans})
        else:
            st.session_state.messages.append({
                "role": "assistant", "content": "未找到相关内容，请尝试换个问法。"
            })
