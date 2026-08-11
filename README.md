# T01-01 氢璞创能 · 企业知识与智能服务助手

基于 RAG（检索增强生成）的企业知识库智能问答系统。

## 技术栈

- **前端/全栈**: Streamlit
- **大模型**: DeepSeek API (deepseek-chat + text-embedding-3-small)
- **向量数据库**: ChromaDB
- **PDF 解析**: pdfplumber

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

在侧边栏输入 DeepSeek API Key，点击"构建知识库"即可开始使用。

## 部署到 Streamlit Cloud

### 1. 推送代码到 GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main
```

### 2. 在 Streamlit Cloud 部署

1. 访问 [share.streamlit.io](https://share.streamlit.io)
2. 点击 "New app"
3. 选择你的 GitHub 仓库
4. Main file path: `app.py`
5. 点击 "Advanced settings" -> "Secrets"
6. 添加:
   ```
   DEEPSEEK_API_KEY = "sk-your-key-here"
   ```
7. 点击 "Deploy"

### 3. 访问

部署完成后会获得类似 `https://your-app.streamlit.app` 的公开链接。

## 项目结构

```
t01-hydrogen-assistant/
├── app.py                 # 主程序
├── requirements.txt       # Python 依赖
├── .streamlit/
│   └── secrets.toml       # 本地 API Key（不提交到 Git）
├── .gitignore
├── *.pdf                  # 企业资料 PDF
└── chroma_db/             # 向量数据库（自动生成）
```

## 注意事项

- 首次运行需要构建知识库，会将 PDF 文本分块并向量化存入 ChromaDB
- DeepSeek API 费用极低：Embedding ~¥0.1/百万token，Chat ~¥1/百万token
- Streamlit Cloud 免费版无冷启动延迟，评审期间可正常访问
- 务必在 Streamlit Cloud Secrets 中配置 DEEPSEEK_API_KEY
