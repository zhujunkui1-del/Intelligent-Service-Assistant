import base64
import re
from pathlib import Path

import streamlit as st
from openai import OpenAI
from kb_engine import build_knowledge_base, search_kb

st.set_page_config(
    page_title="氢璞创能 · 智能知识助手",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- CSS (light-only) ----
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
    :root {
        --app-bg: #ffffff;
        --sidebar-bg: #e2e9f1;
        --sidebar-border: #CBD5E1;
        --accent-color: #00B4D8;
        --title-color: #0F4C82;
        --app-title-color: #000000;
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

    /* Light-only appearance for Chrome/Edge consistency */
    html, body {
        color-scheme: light only;
    }

    /* Remove Streamlit default toolbar (Deploy / menu) */
    header[data-testid="stHeader"],
    .stAppHeader {
        display: none !important;
        visibility: hidden !important;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background: var(--app-bg) !important;
    }

    /* ========== Sidebar ========== */
    section[data-testid="stSidebar"] {
        background: var(--sidebar-bg) !important;
        background-color: var(--sidebar-bg) !important;
        background-image: none !important;
        border-right: none !important;
        box-shadow: 2px 0 12px rgba(0, 0, 0, 0.08);
    }

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

    section[data-testid="stSidebar"] h3 {
        color: var(--title-color) !important;
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

    /* ========== Sticky title ========== */
    [data-testid="stMainBlockContainer"] {
        padding-top: 0.2rem !important;
    }

    .app-title {
        position: -webkit-sticky;
        position: sticky;
        top: 0;
        z-index: 60;
        background: var(--app-bg);
        color: var(--app-title-color) !important;
        margin: 0 !important;
        padding-top: 0.2rem;
        border-bottom: 3px solid var(--accent-color) !important;
        padding-bottom: 0.4rem;
        font-size: 1.5rem;
        font-weight: 800;
    }

    .app-logo {
        height: 1em;
        width: auto;
        vertical-align: -0.12em;
        margin-right: 0.5rem;
    }

    .app-title i {
        color: var(--accent-color) !important;
    }

    /* Keep logo fixed; shift title text down 2px */
    .app-title span {
        position: relative;
        top: 5px;
    }

    /* ========== Status indicators ========== */
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

    /* ========== Chat components ========== */
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }
    [data-testid="stChatMessage"] {
        border-radius: 10px;
        padding: 0.4rem 0.8rem;
        color: #000000 !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {
        color: #000000 !important;
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
        background: var(--app-bg) !important;
        padding-top: 0.5rem;
    }

    /* Provider logos in selectbox */
    .provider-logo {
        height: 1em;
        width: auto;
        vertical-align: -0.1em;
        margin-right: 0.35rem;
        display: inline-block;
    }
    [role="option"] {
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    [data-testid="stSelectbox"] [data-provider-logo-added] {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }
</style>
""", unsafe_allow_html=True)

def _provider_logo_uri(filename):
    try:
        data = (Path(__file__).parent / "素材" / filename).read_bytes()
        mime = "image/jpeg" if data[:3] == b"\xff\xd8\xff" else "image/png"
        return "data:" + mime + ";base64," + base64.b64encode(data).decode("ascii")
    except Exception:
        return ""


deepseek_logo_uri = _provider_logo_uri("DeepSeek Icon - Colored.png")
openai_logo_uri = _provider_logo_uri("ChatGPT Logo - Black.png")

_provider_logo_css = """
<style>
    /* Selected provider value shown in the closed selectbox */
    [data-testid="stSelectbox"] input[value="DeepSeek"] {
        background-image: url("__DEEPSEEK_LOGO__");
        background-repeat: no-repeat !important;
        background-position: 0.4em 50% !important;
        background-size: 1em auto !important;
        padding-left: 1.6em !important;
    }
    [data-testid="stSelectbox"] input[value="OpenAI"] {
        background-image: url("__OPENAI_LOGO__");
        background-repeat: no-repeat !important;
        background-position: 0.4em 50% !important;
        background-size: 1em auto !important;
        padding-left: 1.6em !important;
    }

    /* Provider options in the open dropdown */
    [data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-posinset="1"] {
        background-image: url("__DEEPSEEK_LOGO__");
        background-repeat: no-repeat !important;
        background-position: 0.4em 50% !important;
        background-size: 1em auto !important;
        padding-left: 1.6em !important;
    }
    [data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-posinset="2"] {
        background-image: url("__OPENAI_LOGO__");
        background-repeat: no-repeat !important;
        background-position: 0.4em 50% !important;
        background-size: 1em auto !important;
        padding-left: 1.6em !important;
    }
        /* ========== 聊天输入框：仿豆包光影样式 ========== */
    [data-testid="stChatInput"] {
        position: sticky;
        bottom: 0;
        z-index: 100;
        background: var(--app-bg) !important;
        padding-top: 0.5rem;
    }
    /* 输入框外层容器 */
    [data-testid="stChatInput"] > div {
        border: none !important;
        border-radius: 24px !important;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08) !important;
        transition: box-shadow 0.22s ease-out !important;
        background: #ffffff !important;
    }
    /* 激活/聚焦状态：蓝色向外渐变光晕 */
    [data-testid="stChatInput"] > div:focus-within {
        box-shadow: 0 2px 14px rgba(0, 0, 0, 0.10), 0 0 0 2px rgba(0,180,216,0.25), 0 0 18px 6px rgba(0,180,216,0.18) !important;
    }
    /* 清除原生自带边框 */
    [data-testid="stChatInput"] textarea {
        border: 0 !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        outline: none !important;
        box-shadow: none !important;
        border: none !important;
    }
</style>
"""

_provider_logo_css = _provider_logo_css.replace("__DEEPSEEK_LOGO__", deepseek_logo_uri)
_provider_logo_css = _provider_logo_css.replace("__OPENAI_LOGO__", openai_logo_uri)
st.markdown(_provider_logo_css, unsafe_allow_html=True)

PROVIDERS = {
    "DeepSeek": {"base": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
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
        "busy": False,
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
    return (k or "").strip(), cfg["base"], cfg["model"]

# ---- Knowledge Base ----
@st.cache_resource(show_spinner=False)
def load_kb():
    return build_knowledge_base()

# ---- Chat ----
_URL_RE = re.compile(r"(?<![\w\[(<])(https?://[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+)")


def _fix_links(text: str) -> str:
    """把裸网址包成 <url> 并在其后补空格，避免 Streamlit 把网址后的中文/标点吞进链接。"""
    if not text:
        return text
    out = []
    pos = 0
    for m in _URL_RE.finditer(text):
        url = m.group(1).rstrip(".,;:!?")
        out.append(text[pos:m.start()])
        out.append(f"<{url}>")
        if m.end() < len(text) and not text[m.end()].isspace() and text[m.end()] not in ".,;:!?。，；：！？、)）】]":
            out.append(" ")
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


def generate_answer(query, ctx_docs, history):
    api_key, base_url, model = get_api_config()
    if not api_key:
        return None, "API Key not configured"

    parts = []
    for i, d in enumerate(ctx_docs):
        meta = f"{d['source']} · 第{d['page']}页"
        if d.get("section"):
            meta += f" · {d['section']}"
        if d.get("date"):
            meta += f" · 日期:{d['date']}"
        if d.get("url"):
            meta += f" · 来源:{d['url']}"
        parts.append(f"【资料{i + 1}】{meta}\n{d['text']}")

    sp = f"""你是北京氢璞创能科技有限公司的企业知识助手。

以下是公司产品参数、企业资料和官网新闻，请直接从其中提取答案：

=== 资料开始 ===
{chr(10).join(parts)}
=== 资料结束 ===

回答要求：
1. 涉及型号或参数时，必须全量列出所有型号，参数用表格或列表呈现，不得省略任何型号。
2. 涉及时间或新闻时，明确给出日期、新闻标题和来源网址。
3. 严禁“根据”、“资料”这两个词出现在回答中，禁止使用“资料显示”、“根据资料”、“资料提到...”等措辞。
4. 严禁输出 [src]、[1]、<cite> 之类的引用标记。
5. 多个问题时逐条回答。
6. 资料中没有的数据，尝试在"https://www.nowogen.com/h-col-104.html"下的新闻页查找，仍旧找不到，直接说“很抱歉，我目前尚不了解...，您可前往官网或联系人工客服获取更多信息”，严禁猜测或编造。
7. 问优势时，列出资料中的技术、制造、市场、服务优势；问不足时，不得编造官方缺点，可基于资料中的参数差异做客观对比，并说明这是参数对比。
8. 用中文回答。
9. 禁止决策与报价：绝不提供具体折扣、成交价或代替用户做最终决定。
10. 禁止竞品攻击：若被问及竞品对比，统一回复：“我们更关注自身产品的迭代，详情请参考官方说明书。”
11. 安全过滤：拒绝回答涉政、涉黄、歧视及与业务无关的话题。
12. 转人工条件：当用户情绪激动或连续3次提问未命中知识库时，主动提供转人工入口。
13. 语气控制：使用专业、客观、平实的书面语，禁用表情包、网络流行语和夸张修辞。
16. 输出网址时，网址末尾与后续文字之间必须保留一个空格，避免网址后面的中文或标点被吞进链接。
14. 用户问第X代电堆型号时，必须完整列出该代全部型号后再展开参数；问“有几代/哪几代电堆”时，按碳复合板电堆第四至第七代分代列出。不得说“具体参数未列出”等消极、不专业的话。
15. 关于金属板电堆：资料仅写明“自主研发三代金属板电堆”，未收录任何一代金属板电堆的具体型号与参数；禁止编造“第三代金属板电堆”或其他代次的参数。
16. 输出url网址时，网址末尾与后续文字之间必须保留一个空格，避免网址后面的中文或标点被吞进链接。
17. 严禁说出类似“官网没有...的资料”或“官网未列出...数据”之类的表述，绝对不能说官网没有相关的资料。
18. 在知识库查不到的资料，回答“很抱歉，我目前尚不了解...，您可前往官网或联系人工客服获取更多信息”；资料不全的情况下，回答“更多详情，您可前往官网或联系人工客服获取更多信息”；不得说“具体参数/资料未列出”等消极、不专业的话。
"""

    try:
        cli = OpenAI(api_key=api_key, base_url=base_url)
        msgs = [{"role": "system", "content": sp}]
        msgs.extend(history[-6:])
        msgs.append({"role": "user", "content": query})
        resp = cli.chat.completions.create(model=model, messages=msgs, temperature=0.3, max_tokens=1500)
        return _fix_links(resp.choices[0].message.content or ""), None
    except Exception as e:
        return None, f"API Error: {e}"

kb = load_kb()

# ---- UI: Title ----
def _logo_data_uri() -> str:
    try:
        logo = Path(__file__).parent / "素材" / "logo.png"
        data = logo.read_bytes()
        mime = "image/jpeg" if data[:3] == b"\xff\xd8\xff" else "image/png"
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return ""


logo_uri = _logo_data_uri()
if logo_uri:
    st.markdown(
        f'<h1 class="app-title"><img class="app-logo" src="{logo_uri}" alt="氢璞创能 logo"> '
        '<span>氢璞创能 · 企业知识与智能服务助手</span></h1>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<h1 class="app-title"><i class="fa-solid fa-bolt"></i> '
        '<span>氢璞创能 · 企业知识与智能服务助手</span></h1>',
        unsafe_allow_html=True
    )

# ---- Sidebar ----
with st.sidebar:
    st.markdown('<h3><i class="fa-solid fa-gear"></i> 设置</h3>', unsafe_allow_html=True)
    # 正确双向绑定写法
    provider_options = ["DeepSeek", "OpenAI"]
    # 根据当前session值找到正确索引
    selected_idx = provider_options.index(st.session_state.provider)
    new_provider = st.selectbox(
        "LLM 提供商",
        provider_options,
        index=selected_idx
    )
    # 只有值发生变化才写入session，避免反复刷新
    if new_provider != st.session_state.provider:
        st.session_state.provider = new_provider
        st.rerun()

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
                    _, test_base, test_model = get_api_config()
                    test_cli = OpenAI(api_key=current_key, base_url=test_base)
                    test_cli.chat.completions.create(
                        model=test_model,
                        messages=[{"role": "user", "content": "ping"}],
                        max_tokens=1,
                        temperature=0,
                    )
                    st.session_state['_key_valid'] = True
                    st.session_state['_key_validated_key'] = current_key
                    key_valid = True
                except Exception as _val_err:
                    st.session_state['_key_valid'] = False
                    st.session_state['_key_validated_key'] = ''
                    st.session_state['_key_validate_error'] = str(_val_err)
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
            '<i class="fa-solid fa-triangle-exclamation"></i> 未找到知识库文件</div>',
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
        if st.button(q, key=f"s_{abs(hash(q))}", use_container_width=True, disabled=st.session_state.busy):
            if st.session_state.pending_question is None:
                st.session_state.pending_question = q


# ---- Chat ----
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

chat_val = st.chat_input("请输入您的问题...", disabled=st.session_state.busy)
pending = st.session_state.pop("pending_question", None)
query = pending or chat_val

if query and not st.session_state.busy:
    st.session_state.busy = True
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
            "content": "知识库为空，请检查知识库文件是否存在。"
        })
    else:
        with st.spinner("正在检索..."):
            docs = search_kb(kb, query)
        if docs:
            with st.spinner("生成回答..."):
                hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-3:-1]]
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
                            f'<strong>来源 {i+1}</strong> {d["source"]} · 第{d["page"]}页</p>',
                            unsafe_allow_html=True
                        )
                        st.text(d["text"][:250] + ("..." if len(d["text"]) > 250 else ""))
            st.session_state.messages.append({"role": "assistant", "content": ans})
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "未找到相关内容，请尝试换个问法。"
            })
    st.session_state.busy = False
    st.rerun()
