# 🤖 AI 知识库问答助手

基于 RAG（检索增强生成）架构的平台无关知识库机器人。支持混合检索、反问澄清、引用溯源和可扩展的多平台适配器架构。

---

## 核心流程

```
用户提问
    ↓
① 反问澄清决策      → 问题跨多领域时主动反问，避免猜错意图
    ↓
② 混合检索          → BM25 关键词 + Dense 语义向量，双路并行召回
    ↓
③ RRF 融合          → 双路命中结果获得加权，取交集增强
    ↓
④ 精细重排序        → Cross-encoder 逐条评分，挑出最相关的 5 条
    ↓
⑤ LLM 生成回答      → 基于检索到的文档片段生成答案，绝不编造
    ↓
⑥ 回答验证 + 溯源    → 检测是否脱离知识库，附上来源文档
    ↓
平台适配器输出        → Streamlit（开箱即用）/ 飞书 / OpenClaw（需自行配置 webhook）
```

---

## 项目结构

```
AI_BOT_DEMO/
├── app/
│   └── streamlit_app.py              # Streamlit 聊天界面
├── config/
│   └── settings.py                   # 全局配置（所有参数集中管理）
├── scripts/                          # 一键运行脚本
│   ├── 01_generate_knowledge.py      # 知识文档脚手架 + 校验
│   ├── 02_ingest_data.py             # 文档分块 → 构建索引
│   ├── 03_start_api.py               # 启动 FastAPI 服务
│   ├── 04_start_ui.py                # 启动 Streamlit 界面
│   ├── 05_run_eval.py                # 运行 RAG 评估
│   └── 06_generate_test_set.py       # 生成评估测试集
├── src/
│   ├── core/
│   │   └── rag_engine.py             # 核心引擎：编排检索→重排→生成
│   ├── adapters/                     # 多平台适配器（拓展新平台只改这里）
│   │   ├── base_adapter.py           # 抽象接口 parse → query → format
│   │   ├── feishu_adapter.py         # 飞书适配器
│   │   ├── openclaw_adapter.py       # OpenClaw 适配器
│   │   └── streamlit_adapter.py      # Streamlit 本地适配器
│   ├── retrieval/                    # 检索层
│   │   ├── bm25_retriever.py         # BM25 关键词检索（精确匹配）
│   │   ├── dense_retriever.py        # Dense 语义检索（模糊匹配）
│   │   ├── rrf_fusion.py             # RRF 融合算法（双路互增强）
│   │   ├── hybrid_searcher.py        # 混合检索引擎
│   │   └── reranker.py               # BGE-reranker 交叉编码器精排
│   ├── generation/                   # 生成层
│   │   ├── llm_client.py             # LLM 客户端（Ollama / Groq / DeepSeek）
│   │   └── prompt_builder.py         # Prompt 模板（六层架构）
│   ├── clarification/                # 反问澄清
│   │   ├── classifier.py             # 领域歧义检测
│   │   ├── templates.py              # 反问话术模板
│   │   └── clarification_service.py  # 澄清决策引擎
│   ├── ingestion/                    # 数据管道
│   │   ├── loader.py                 # 加载 .md 文档 + 解析 YAML 元数据
│   │   ├── chunker.py                # 中文感知的文本分块
│   │   └── pipeline.py               # 编排管道：加载→分块→嵌入→存储
│   ├── embeddings/
│   │   └── embedder.py               # SentenceTransformer 封装
│   ├── evaluation/                   # 评估体系
│   │   ├── evaluator.py              # 4 维度指标计算
│   │   ├── reporter.py               # 评估报告生成（Markdown + JSON + CSV）
│   │   └── test_set_generator.py     # 42 条测试样例（单域/跨域/边界/异常）
│   ├── knowledge_generator/          # 知识管理
│   │   ├── domains.py                # 领域定义（新增领域只改这一个文件）
│   │   └── generator.py              # 脚手架：创建模板 + 完整性校验
│   └── api/                          # HTTP 接口
│       ├── main.py                   # FastAPI 入口
│       ├── routes.py                 # /chat 端点（平台无关，无平台逻辑混入）
│       └── schemas.py                # Pydantic 请求/响应模型
├── data/
│   ├── raw/                          # 知识文档（独立 .md，可直接编辑）
│   │   ├── travel_tips/              # 出差注意事项（4 篇）
│   │   ├── malaysia_visa/            # 马来西亚商务签证（4 篇）
│   │   └── project_applications/     # 项目申报材料（4 篇）
│   └── eval/
│       └── test_set.json             # 评估测试集
├── tests/                            # 单元测试
│   ├── test_retrieval.py             # 检索链路测试
│   ├── test_clarification.py         # 反问澄清测试
│   └── test_error_handling.py        # 错误处理和降级测试
├── .gitignore                        # Git 排除规则
├── .env.example                      # 配置文件模板（无真实 Key）
├── requirements.txt                  # Python 依赖
└── README.md                         # 你正在看的这个文件
```

---

## 快速开始

### 前置条件

- Python 3.12+
- [Ollama](https://ollama.com) 安装并拉取模型：
  ```bash
  ollama pull qwen2.5:3b
  ```
- 或者使用云端 API（Groq 免费额度 / DeepSeek 付费）

### 安装

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 配置

将 '.env.example'的内容复制到用户目录下新建的 `.ai_bot_env` 文件，填入真实配置：

```
Linux/Mac:  ~/.ai_bot_env
Windows:    C:\Users\<你的用户名>\.ai_bot_env
```

示例配置：

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
```

> **为什么不在项目目录里写 `.env`？** 防止 API Key 被意外提交到 Git。配置放在用户目录下，与项目文件夹物理隔离，无论怎么上传 GitHub 都不会泄露。

### 运行

```bash
# 第一步：校验知识文档完整性
python scripts/01_generate_knowledge.py

# 第二步：构建 BM25 和向量索引
python scripts/02_ingest_data.py

# 第三步：启动 Streamlit 聊天界面
streamlit run app/streamlit_app.py --server.port 8501
```

浏览器打开 **http://localhost:8501** 即可开始对话。

---

## 接入新平台

所有平台适配器共享同一套抽象接口。Streamlit 适配器已开箱即用，飞书和 OpenClaw 适配器提供了完整的 parse/format 逻辑，webhook 端点需根据实际部署环境在 `src/api/routes.py` 中自行添加。接入新平台只需实现两个方法：

```python
from src.adapters.base_adapter import BaseAdapter, UnifiedMessage, UnifiedResponse

class WeComAdapter(BaseAdapter):        # 以企业微信为例
    def parse(self, raw_data: dict) -> UnifiedMessage:
        """解析企业微信的消息格式"""
        return UnifiedMessage(
            platform="wecom",
            user_id=raw_data["FromUserName"],
            text=raw_data["Content"],
        )

    def format(self, response: UnifiedResponse) -> dict:
        """转为企业微信要求的 Markdown 格式"""
        return {
            "msgtype": "markdown",
            "markdown": {"content": response.text}
        }
```

然后在 `src/api/routes.py` 中添加 webhook 端点并配置平台回调 URL 即可。**核心 `RAGEngine.query()` 完全不用改。**

---

## 新增知识领域

四步完成，只在 `domains.py` 改一行 Python 代码，其余全是 Markdown 编辑：

1. 在 `src/knowledge_generator/domains.py` 中添加领域定义和子主题树
2. 运行 `python scripts/01_generate_knowledge.py` 创建模板 `.md` 文件
3. 编辑 `data/raw/<新领域>/` 下的 `.md` 文件，填入实际内容
4. 运行 `python scripts/02_ingest_data.py` 重建索引

---

## 技术栈

| 层次 | 方案 |
|------|------|
| 嵌入模型 | paraphrase-multilingual-MiniLM-L12-v2（384 维，中英双语） |
| 关键词检索 | rank_bm25（BM25Okapi，中文 bigram 分词） |
| 语义检索 | Numpy 余弦相似度（纯 numpy，无外部依赖） |
| 融合算法 | Reciprocal Rank Fusion（k=60） |
| 重排序 | BAAI/bge-reranker-base（Cross-encoder） |
| LLM | Qwen2.5:3B（Ollama）/ Llama-3.1-8B（Groq）/ DeepSeek |
| API 框架 | FastAPI |
| 聊天界面 | Streamlit |
| 评估 | 4 维度指标（忠实度 / 相关性 / 检索精度 / 检索召回率），42 条测试样例（可通过 test_set_generator.py 扩展） |

---

## 已排除的内容（安全上传）

以下内容不会出现在 Git 仓库中，clone 后需自行生成：

| 排除内容 | 说明 |
|---------|------|
| `.env` / API Key | 配置文件放在用户目录 `~/.ai_bot_env`，与项目隔离 |
| `data/processed/` | BM25 索引和向量索引，运行 `02_ingest_data.py` 生成 |
| `.chroma/` | 向量存储目录，同上 |
| `data/eval/eval_results/` | 评估报告，运行 `05_run_eval.py` 生成 |
| `venv/` | 虚拟环境，`pip install` 创建 |
| `*.pkl` | 序列化索引文件 |

---

## 常见问题

**Q: 启动后提示 "BM25 index not found"？**
运行 `python scripts/02_ingest_data.py` 构建索引。

**Q: 回答总是 "暂无相关信息"？**
知识库覆盖范围有限，问的内容可能不在已有领域内。可以在 `domains.py` 中添加新领域。

**Q: Ollama 连接失败？**
确保 Ollama 在后台运行，且已拉取模型：`ollama pull qwen2.5:3b`

**Q: 想用 DeepSeek 或 Groq？**
在 `~/.ai_bot_env` 中修改 `LLM_PROVIDER` 并填入对应的 `API_KEY`。

---

## 安全说明

本项目面向作品集展示和教育用途。

- 真实 API Key 应存储在 `~/.ai_bot_env`，不要放在项目目录中。
- `.env`、本地索引、向量存储、日志和生成数据均已被 `.gitignore` 排除。
- API 默认监听 `127.0.0.1`，仅本机可访问。
- 如需公网部署，需配置 API Key 鉴权、限制 CORS、添加速率限制、验证 webhook 签名。
- Pickle 索引文件仅应由项目本地生成和加载，**不要从不可信来源加载 `.pkl` 文件**。
- 飞书 / OpenClaw 适配器的 parse/format 逻辑已实现，但 webhook 端点需根据实际部署环境自行配置。
- 首次启动时嵌入模型会自动从 HuggingFace 下载（约 500MB），请确保网络通畅。

## 已知限制

- 内置知识库为 Demo 数据，12 篇文档覆盖 3 个领域，不足以覆盖真实业务场景。
- 评估脚本使用轻量启发式指标，不是完整的 RAGAS / DeepEval 管线。
- 飞书 / OpenClaw 适配器默认**未启用**为生产就绪的 webhook 端点。
- 签证、政策和项目申报知识可能有时效性，正式使用前需人工审核。
- Streamlit 在运行期间缓存 RAG 引擎；修改知识库后需重启应用。
- 反问澄清使用纯规则判断（领域占比 + 分数差距），未接入 LLM 判断。
- 多轮对话记忆尚未实现（`session_id` 字段预留但未消费）。

## 许可证

MIT — 详见 [LICENSE](LICENSE)
