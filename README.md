# 氢璞创能 · 企业知识与智能服务助手🤖

基于本地稀疏检索与 LLM 的 RAG 企业知识库问答系统。知识来源包括氢璞创能的产品资料、企业宣传册和官网新闻。

## 功能💻

- 启动时自动读取三份 Markdown 资料并构建本地知识库。
- 后台抓取官网新闻，新闻片段附带标题、日期和来源 URL。
- 支持 DeepSeek 与 OpenAI 双提供商切换。
- 型号、参数、大事记、合作伙伴、新闻出处等查询有定向检索增强。
- 不依赖外部 Embedding API，启动和检索速度快。

## 技术栈🚀

- Streamlit：页面与聊天交互
- OpenAI SDK：兼容 DeepSeek `deepseek-chat` 和 OpenAI `gpt-4o-mini`
- scikit-learn：TF-IDF 字符级特征与余弦相似度检索
- requests + BeautifulSoup：官网新闻抓取

## 本地运行⚙️

```bash
cd ./Intelligent Service Assistant
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

打开终端提供的本地网址，在侧边栏选择 LLM 提供商并输入 API Key。知识库会自动初始化，官网抓取在后台执行，不阻塞首屏提问。

## 知识库来源📕

- `01 氢璞2025产品单页（1023）校正稿.md`
- `02（已压缩）氢璞2025宣传册（0905）-解决方案全.md`
- `03 氢璞创能氢能产业商机.md`
- 氢璞创能官网首页、新闻中心及有效新闻文章页

官网不可访问时会静默回退到本地 Markdown 知识库，不影响基础问答。

## 测试🧪

```bash
python test_kb.py
python test_kb.py --crawl
python test_kb.py --crawl --with-llm
```

- 默认运行 10 条本地检索断言。
- `--crawl` 额外测试官网新闻检索。
- `--with-llm` 在检测到有效 `DEEPSEEK_API_KEY` 后运行真实 LLM 回答测试。

## 项目结构⬜

```text
Intelligent Service Assistant/
├── app.py                 # Streamlit 主程序
├── kb_engine.py           # Markdown 解析、TF-IDF 检索、官网抓取
├── test_kb.py             # 自测脚本
├── requirements.txt       # Python 依赖
├── README.md
├── .gitignore
├── .streamlit/config.toml
├── 素材/
├── 01 氢璞2025产品单页（1023）校正稿.md
├── 02（已压缩）氢璞2025宣传册（0905）-解决方案全.md
└── 03 氢璞创能氢能产业商机.md
```

## 测试网站
该项目已在Streamlit上线：https://intelligent-service-assistant-ajekavvfmxgzqdqzybnf9n.streamlit.app/