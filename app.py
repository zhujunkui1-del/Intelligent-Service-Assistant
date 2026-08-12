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
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- CSS & Dark Mode JS ----
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
    /* ========== 主题变量 ========== */
    :root {
        --sidebar-border: #CBD5E1;
        --accent-color: #00B4D8;
        --status-ok-bg: #ECFDF5;
        --status-ok-border: #6EE7B7;
        --status-ok-text: #065F46;
        --status-warn-bg: #FFFBEB;
        --status-warn-border: #FCD34D;
        --status-warn-text: #92400E;
        --status-info-bg: #EFF6FF;
        --status-info-border: #93C5FD;
        --status-info-text: #1E40AF;
    }
    
    /* ========== 侧边栏 - 背景由 JS 完全控制 ========== */
    section[data-testid="stSidebar"] {
        border-right: none !important;
        box-shadow: 2px 0 12px rgba(0, 0, 0, 0.08);
        /* 不设置 background，全部交给 JS */
    }
    
    /* 侧边栏内部所有容器 - 设为透明，让外层颜色透出来 */
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div > div,
    section[data-testid="stSidebar"] > div > div > div,
    section[data-testid="stSidebar"] > div > div > div > div,
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarContent"] > div,
    [data-testid="stSidebarContent"] > div > div,
    [data-testid="stSidebarContent"] > div > div > div {
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
    }
    
    /* ========== 标题（颜色由 JS 控制） ========== */
    .app-title {
        position: sticky;
        top: 0;
        z-index: 50;
        background: inherit;
        border-bottom: 3px solid var(--accent-color) !important;
        padding-bottom: 0.4rem;
        margin-bottom: 0.5rem;
        font-size: 1.5rem;
        font-weight: 800;
    }
    
    .app-title i {
        color: var(--accent-color) !important;
    }
    
    /* ========== 侧边栏标题（颜色由 JS 控制） ========== */
    section[data-testid="stSidebar"] h3 {
        font-size: 0.95rem;
        font-weight: 700;
        margin-top: 1rem;
    }
    section[data-testid="stSidebar"] h3:first-of-type { margin-top: 0.4rem; }
    section[data-testid="stSidebar"] label { margin-bottom: 0; padding-bottom: 0; }
    
    section[data-testid="stSidebar"] hr {
        border-color: var(--sidebar-border) !important;
        margin: 0.4rem 0;
    }
    
    /* ========== 状态指示器 ========== */
    .status-box {
        border-radius: 8px;
        padding: 0.4rem 0.65rem;
        margin: 0.3rem 0;
        font-size: 0.82rem;
    }
    .status-ok {
        background: var(--status-ok-bg) !important;
        border: 1px solid var(--status-ok-border) !important;
        color: var(--status-ok-text) !important;
    }
    .status-warn {
        background: var(--status-warn-bg) !important;
        border: 1px solid var(--status-warn-border) !important;
        color: var(--status-warn-text) !important;
    }
    .status-info {
        background: var(--status-info-bg) !important;
        border: 1px solid var(--status-info-border) !important;
        color: var(--status-info-text) !important;
    }
    
    /* ========== 聊天组件 ========== */
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }
    [data-testid="stChatMessage"] {
        border-radius: 10px;
        padding: 0.4rem 0.8rem;
    }
    .stChatMessage:first-of-type {
        margin-top: 5px;
    }
    section[data-testid="stSidebar"] .stButton > button {
        border-radius: 6px;
        font-size: 0.82rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    [data-testid="stChatInput"] {
        position: sticky;
        bottom: 0;
        z-index: 100;
        background: inherit;
        padding-top: 0.5rem;
    }
    
    /* 深色模式聊天输入框 */
    [data-theme="dark"] [data-testid="stChatInput"],
    [data-theme="dark"] [data-testid="stChatInput"]:focus,
    [data-theme="dark"] [data-testid="stChatInput"]:hover {
        background: #0E1117 !important;
    }
</style>

<script>
(function() {
    function fixSidebar() {
        var isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        var sidebar = document.querySelector('section[data-testid="stSidebar"]');
        if (!sidebar) return;
        
        // ---- 改背景（白天和深色都显式设置） ----
        if (isDark) {
            // 深色模式
            sidebar.style.setProperty('background', '#2b3347', 'important');
            sidebar.style.setProperty('background-color', '#2b3347', 'important');
            sidebar.style.setProperty('background-image', 'none', 'important');
        } else {
            // 白天模式
            sidebar.style.setProperty('background', '#e2e9f1', 'important');
            sidebar.style.setProperty('background-color', '#e2e9f1', 'important');
            sidebar.style.setProperty('background-image', 'none', 'important');
        }
        
        // 让所有子容器透明
        var children = sidebar.querySelectorAll('div');
        for (var i = 0; i < children.length; i++) {
            children[i].style.setProperty('background', 'transparent', 'important');
            children[i].style.setProperty('background-color', 'transparent', 'important');
            children[i].style.setProperty('background-image', 'none', 'important');
        }
        
        var sc = document.querySelector('[data-testid="stSidebarContent"]');
        if (sc) {
            sc.style.setProperty('background', 'transparent', 'important');
            sc.style.setProperty('background-color', 'transparent', 'important');
            sc.style.setProperty('background-image', 'none', 'important');
        }
        
        // ---- 改文字颜色 ----
        var title = document.querySelector('.app-title');
        var headings = document.querySelectorAll('section[data-testid="stSidebar"] h3');
        
        if (isDark) {
            // 夜间模式：纯白色
            if (title) title.style.setProperty('color', '#FFFFFF', 'important');
            for (var i = 0; i < headings.length; i++) {
                headings[i].style.setProperty('color', '#FFFFFF', 'important');
            }
        } else {
            // 白天模式：深蓝色 #0F4C82
            if (title) title.style.setProperty('color', '#0F4C82', 'important');
            for (var i = 0; i < headings.length; i++) {
                headings[i].style.setProperty('color', '#0F4C82', 'important');
            }
        }
    }
    
    // 监听 data-theme 变化（事件驱动，不轮询）
    var observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(m) {
            if (m.attributeName === 'data-theme') {
                fixSidebar();
            }
        });
    });
    observer.observe(document.documentElement, { attributes: true });
    
    // DOM加载完成后执行
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', fixSidebar);
    } else {
        fixSidebar();
    }
})();
</script>
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
            "content": "您好！我是氢璞创能的企业知识与智能服务助手，请随时向我提问。"
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
    for p in sorted(PROJECT_DIR.glob("*.md")):
        if p.name.upper() == "README.MD": continue
        try:
            text = p.read_text(encoding="utf-8")
            sections = re.split(r"\n(?=## )", text)
            for si, section in enumerate(sections):
                s = section.strip()
                if len(s) > 30:
                    docs.append({"source": p.name, "page": si + 1, "text": s})
        except:
            pass
    return docs

def _chunk_docs(docs):
    chunks, metas = [], []
    for d in docs:
        text = d["text"]
        start = 0
        ci = 0
        while start < len(text):
            c = text[start:start + CHUNK_SIZE].strip()
            if c:
                chunks.append(c)
                metas.append({"source": d["source"], "page": d["page"], "ci": ci})
                ci += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks, metas

def _scrape_and_append(kb):
    import urllib.parse
    try:
        visited = set()
        to_visit = [
            WEBSITE_URL,
            WEBSITE_URL.rstrip("/") + "/h-col-104.html",  # 新闻中心
        ]
        all_texts = []
        
        while to_visit and len(visited) < 20:
            url = to_visit.pop(0)
            if url in visited:
                continue
            visited.add(url)
            try:
                resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
                resp.encoding = resp.apparent_encoding or "utf-8"
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Collect internal links
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    full = urllib.parse.urljoin(url, href)
                    base_domain = urllib.parse.urlparse(WEBSITE_URL).netloc
                    if urllib.parse.urlparse(full).netloc == base_domain and full not in visited:
                        to_visit.append(full)
                
                # Extract text
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n", strip=True))
                if len(text) > 100:
                    title_tag = soup.find("title")
                    page_title = title_tag.get_text(strip=True) if title_tag else url
                    all_texts.append({"source": f"官网: {page_title}", "page": len(all_texts)+1, "text": text})
            except:
                continue
        
        if all_texts and kb["vectorizer"] is not None:
            wc, wm = _chunk_docs(all_texts)
            kb["chunks"].extend(wc)
            kb["metadatas"].extend(wm)
            kb["matrix"] = kb["vectorizer"].transform(kb["chunks"])
            kb["web_count"] = len(wc)
    except:
        pass

def search_kb(kb, query, k=5):
    if kb["vectorizer"] is None or not kb["chunks"]:
        return []
    qvec = kb["vectorizer"].transform([query])
    scores = cosine_similarity(qvec, kb["matrix"]).flatten()
    top = np.argsort(scores)[-k:][::-1]
    return [{"text": kb["chunks"][i], "source": kb["metadatas"][i]["source"],
             "page": kb["metadatas"][i]["page"], "score": float(scores[i])}
            for i in top if scores[i] > 0]

# ---- Chat ----
def generate_answer(query, ctx_docs, history):
    api_key, base_url, model = get_api_config()
    if not api_key:
        return None, "API Key not configured"
    parts = [f"{d['source']} p.{d['page']}:\n{d['text']}" for i, d in enumerate(ctx_docs)]
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
5. If multiple questions are asked, answer each one separately with clear numbering
3. Do NOT preface answers with phrases like "根据参考资料", "根据提供的材料" - answer directly
4. Be concise, professional, well-organized
5. If multiple questions are asked, answer each one separately with clear numbering
6. Respond in Chinese"""
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
    '<h1 class="app-title"><i class="fa-solid fa-bolt"></i> '
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
    key_valid = False
    if current_key:
        if st.session_state.get('_key_validated_key') == current_key and st.session_state.get('_key_valid', False):
            key_valid = True
        else:
            with st.spinner('正在验证 API Key...'):
                try:
                    _, test_base, _ = get_api_config()
                    test_cli = OpenAI(api_key=current_key, base_url=test_base)
                    test_cli.models.list()
                    st.session_state['_key_valid'] = True
                    st.session_state['_key_validated_key'] = current_key
                    key_valid = True
                except Exception:
                    st.session_state['_key_valid'] = False
                    st.session_state['_key_validated_key'] = ''
    if key_valid:
        st.markdown(
            f'<div class="status-box status-ok">'
            f'<i class="fa-solid fa-circle-check"></i> {st.session_state.provider} Key 已验证</div>',
            unsafe_allow_html=True
        )
    elif current_key:
        st.markdown(
            f'<div class="status-box status-warn">'
            f'<i class="fa-solid fa-triangle-exclamation"></i> API Key 无效</div>',
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


# ---- Chat ----
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

chat_val = st.chat_input("请输入您的问题...")
pending = st.session_state.pop("pending_question", None)
query = pending or chat_val

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
            "role": "assistant",
            "content": "知识库为空，请检查 PDF 文件是否存在。"
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
                "role": "assistant",
                "content": "未找到相关内容，请尝试换个问法。"
            })
    st.rerun()
