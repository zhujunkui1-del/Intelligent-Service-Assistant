import streamlit as st
from openai import OpenAI
from kb_engine import build_knowledge_base, search_kb

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
3. 严禁说“根据参考资料”“根据提供的资料”等任何“根据...资料”的话，直接回答问题。
4. 严禁输出 [src]、[1]、<cite> 之类的引用标记。
5. 多个问题时逐条回答。
6. 资料中没有的数据，直接说“资料未收录此项”，不得猜测或编造。
7. 问优势时，列出资料中的技术、制造、市场、服务优势；问不足时，不得编造官方缺点，可基于资料中的参数差异做客观对比，并说明这是参数对比。
8. 用中文回答。"""

    try:
        cli = OpenAI(api_key=api_key, base_url=base_url)
        msgs = [{"role": "system", "content": sp}]
        msgs.extend(history[-6:])
        msgs.append({"role": "user", "content": query})
        resp = cli.chat.completions.create(model=model, messages=msgs, temperature=0.3, max_tokens=1500)
        return resp.choices[0].message.content, None
    except Exception as e:
        return None, f"API Error: {e}"

kb = load_kb()

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
