import sqlite3
import json
import logging
import os
from typing import Dict, Any, List, Optional, Tuple

from src.graph.schema_graph_retriever import GraphRAGRetriever
from src.llm.client import get_competition_json, get_competition

logger = logging.getLogger(__name__)

class Text2SQLAgent:
    def __init__(self, db_name: str, db_path: str, neo4j_config: Tuple[str, str, str], max_retries: int = 3):
        """
        初始化 Agent
        :param db_name: 数据库名称 (用于检索)
        :param db_path: SQLite 文件路径 (用于执行)
        :param neo4j_config: (uri, user, password) 用于初始化检索器
        :param max_retries: SQL 执行失败后的最大重试次数
        """
        self.db_name = db_name
        self.db_path = db_path
        self.max_retries = max_retries
        
        # 初始化检索器
        # 注意：这里假设 schema_json_path 存在于标准位置，用于检索器的初始化
        schema_path = f"bird_data/converted_schemas/{db_name}.json"
        self.retriever = GraphRAGRetriever(*neo4j_config, schema_json_path=schema_path)

    def retrieve_schema(self, question: str, evidence: str) -> Dict[str, List[str]]:
        """
        步骤 1: 调用 GraphRAG 获取 Schema Link 结果
        """
        logger.info(f"Retrieving schema for: {question}")
        query = {
            "question": question,
            "evidence": evidence
        }
        # 调用现有的检索逻辑
        schema_map = self.retriever.retrieve_schema_subgraph(query)
        return schema_map

    def _format_schema_for_prompt(self, schema_map: Dict[str, List[str]]) -> str:
        """
        将检索到的 schema map 转换为 Prompt 友好的字符串
        """
        lines = []
        for table, cols in schema_map.items():
            col_str = ", ".join(cols)
            lines.append(f"Table: {table}\nColumns: {col_str}")
        return "\n\n".join(lines)

    def generate_sql(self, question: str, evidence: str, schema_context: str, error_msg: str = None, previous_sql: str = None) -> str:
        """
        步骤 2 & 4: 生成 SQL (包含错误修正逻辑)
        """
        
        system_prompt = (
            "You are an expert SQL Data Analyst. Your goal is to write a correct SQLite-compatible SQL query to answer the user's question.\n"
            "Strictly follow these rules:\n"
            "1. Only use the tables and columns provided in the 'Retrieved Schema'.\n"
            "2. The output must be a valid JSON object with a single key 'sql'.\n"
            "3. Do not wrap the JSON in markdown code blocks.\n"
            "4. Ensure the SQL is compatible with SQLite (e.g., use `strftime` for dates, avoid MySQL specific functions like `YEAR()`)."
        )

        user_content = f"""
### User Question
{question}

### Evidence / Hint
{evidence}

### Retrieved Schema (Relevant Tables and Columns)
{schema_context}
"""

        if error_msg and previous_sql:
            user_content += f"""
### Previous Failed Attempt
SQL: {previous_sql}
Error: {error_msg}

### Instruction
The previous SQL failed to execute. Please analyze the error and correct the SQL.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        try:
            # 使用 get_competition_json 确保返回 JSON 格式
            response = get_competition_json(messages)
            result = json.loads(response)
            return result.get("sql", "")
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            # Fallback: 尝试直接获取文本并清洗
            resp_text = get_competition(messages)
            # 简单的清洗逻辑，移除 markdown
            clean_sql = resp_text.replace("```sql", "").replace("```json", "").replace("```", "").strip()
            # 如果是 JSON 字符串尝试解析
            try:
                j = json.loads(clean_sql)
                return j.get("sql", clean_sql)
            except:
                return clean_sql

    def execute_sql(self, sql: str) -> Tuple[List[Any], Optional[str]]:
        """
        步骤 3: 在 SQLite 中执行 SQL
        返回: (结果行列表, 错误信息)
        """
        if not os.path.exists(self.db_path):
            return [], f"Database file not found: {self.db_path}"

        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            conn.close()
            return results, None
        except sqlite3.Error as e:
            return [], str(e)

    def solve(self, question: str, evidence: str) -> Dict[str, Any]:
        """
        Agent 主流程
        """
        # 1. Retrieve
        schema_map = self.retrieve_schema(question, evidence)
        schema_context = self._format_schema_for_prompt(schema_map)
        logger.info(f"Retrieved Schema Context:\n{schema_context}")

        current_sql = ""
        error_msg = None
        
        # 2. Generate & Execute Loop (Agentic)
        for attempt in range(self.max_retries + 1):
            logger.info(f"Generating SQL (Attempt {attempt + 1})...")
            
            current_sql = self.generate_sql(question, evidence, schema_context, error_msg, current_sql)
            logger.info(f"Generated SQL: {current_sql}")

            if not current_sql:
                return {"error": "Failed to generate SQL", "steps": attempt}

            results, error_msg = self.execute_sql(current_sql)

            if error_msg is None:
                # Success
                logger.info("Execution Successful.")
                return {
                    "question": question,
                    "sql": current_sql,
                    "result": results,
                    "retrieved_schema": schema_map,
                    "status": "success"
                }
            else:
                logger.warning(f"Execution Failed: {error_msg}")
                # Continue to next iteration to fix

        return {
            "question": question,
            "sql": current_sql,
            "error": error_msg,
            "status": "failed"
        }

    def close(self):
        if self.retriever:
            self.retriever.close()