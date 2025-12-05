# Text2SQL Schema KG (Graph RAG)

本项目是一个基于知识图谱（Knowledge Graph）和检索增强生成（RAG）技术的 Text-to-SQL 辅助系统。它旨在通过构建数据库 Schema 的图谱结构，利用 LLM 和图算法（如 PPR）来提高 Schema Linking（模式链接）的准确性，特别针对 BIRD Benchmark 数据集。

## 📂 项目结构

```text
.
├── bird_data/              # BIRD 数据集目录 (包含数据库、描述文件、Golden Link 等)
├── cache/                  # 缓存目录 (Embedding, 分析结果等)
├── logs/                   # 运行日志
├── scripts/                # 执行脚本
│   ├── build_graph.py      # 构建 Neo4j 知识图谱
│   ├── run_retrieval.py    # 运行检索流程
│   └── evaluate/           # 评估脚本
├── src/                    # 核心源代码
│   ├── graph/              # 图谱构建与检索逻辑
│   ├── llm/                # LLM 交互与 Prompt 管理
│   └── utils/              # 工具函数
├── .env                    # 环境变量配置
└── pyproject.toml          # 项目依赖配置
```

## 🛠️ 环境准备

### 1. 基础环境
*   Python 3.8+
*   [Neo4j Database](https://neo4j.com/) (推荐使用 Docker 部署)

### 2. 安装依赖
在项目根目录下运行：

```bash
pip install -e .
```

### 3. 配置环境变量
复制 `.env` 文件并根据你的环境进行修改。你需要配置 OpenAI 兼容的 API Key 以及 Neo4j 的连接信息。

```ini
# .env 示例
OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_MODEL=qwen3-max
OPENAI_API_KEY=your_api_key_here
RERANK_MODEL=qwen3-rerank
EMBEDDING_MODEL=text-embedding-v4

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# 图算法配置
PPR_SEED_K_EACH=10
PPR_HOPS=2
```

## 🚀 使用指南

### 步骤 1: 构建知识图谱
首先，需要读取 `bird_data` 中的数据库 Schema 信息，并将其构建到 Neo4j 图数据库中。

```bash
python scripts/build_graph.py
```

### 步骤 2: 运行检索 (Schema Retrieval)
针对 BIRD 数据集中的问题，运行 Graph RAG 检索器，提取相关的表和列。

```bash
python scripts/run_retrieval.py
```

### 步骤 3: 评估效果
使用评估脚本计算检索结果的 Precision, Recall 和 F1 Score。

```bash
# 默认评估 bird_data/golden_link/golden_schema_link_test.json
python scripts/evaluate/evaluate_retrieval.py

# 或者指定测试文件
python scripts/evaluate/evaluate_retrieval.py path/to/your/test_file.json
```

## 📊 评估指标

评估脚本 [`scripts/evaluate/evaluate_retrieval.py`](scripts/evaluate/evaluate_retrieval.py) 会输出以下指标：

*   **Table Level**: 表级别的精确率、召回率和 F1 值。
*   **Column Level**: 列级别的精确率、召回率和 F1 值。

## 📝 核心逻辑

*   **图谱构建**: 将数据库的 Table, Column, Primary Key, Foreign Key 映射为图节点和边。
*   **Schema Linking**:
    1.  **实体提取**: 利用 LLM 从自然语言问题中提取关键词。
    2.  **初始检索**: 使用 Embedding 相似度或关键词匹配找到种子节点。
    3.  **图传播 (PPR)**: 使用 Personalized PageRank 算法在图谱上进行相关性传播。
    4.  **重排序 (Rerank)**: 对检索到的 Schema # Text2SQL Schema KG (Graph RAG)

## 📄 License

本项目遵循 MIT License (或参考  中的定义)。
