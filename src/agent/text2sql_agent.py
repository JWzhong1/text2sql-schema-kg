import sqlite3
import json
import logging
import os
from typing import Dict, Any, List, Optional, Tuple

from src.graph.schema_graph_retriever import GraphRAGRetriever
from src.llm.client import get_competition_json, get_competition
from src.llm.prompts import get_sql_generation_prompt

# 添加日志配置，设置级别为 INFO
# 修改：添加 force=True 强制覆盖其他模块可能已设置的日志配置
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # 显式设置当前 logger 级别，确保不被父级过滤

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

    def retrieve_schema(self, question: str, evidence: str) -> Dict[str, Any]:
        """
        步骤 1: 调用 GraphRAG 获取 Schema Link 结果及推理信息
        """
        logger.info(f"Retrieving schema for: {question}")
        query = {
            "question": question,
            "evidence": evidence
        }
        # 调用检索逻辑,获取完整信息
        result = self.retriever.retrieve_schema_subgraph(query, return_reasoning=True)
        return result

    def generate_sql(self, question: str, evidence: str, retrieval_result: Dict[str, Any], error_msg: str = None, previous_sql: str = None) -> str:
        """
        步骤 2 & 4: 生成 SQL (包含推理上下文)
        """
        schema_map = retrieval_result.get("schema_map", {})
        reasoning_ctx = retrieval_result.get("reasoning_context", {})
        
        # 格式化 Schema
        schema_context = self._format_schema_for_prompt(schema_map)
        
        # 格式化推理上下文
        reasoning_context_str = self._format_reasoning_context(reasoning_ctx)
        
        system_prompt, user_content = get_sql_generation_prompt(
            question, 
            evidence, 
            schema_context, 
            reasoning_context_str, 
            error_msg, 
            previous_sql
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        try:
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
    
    def _format_schema_for_prompt(self, schema_map: Dict[str, Any]) -> str:
        """
        格式化 Schema Map 为 Prompt 友好的字符串
        """
        lines = []
        if not schema_map:
            return "No schema information available."
            
        for table_name, columns in schema_map.items():
            # columns 通常是列名列表
            if isinstance(columns, list):
                col_str = ", ".join([str(c) for c in columns])
            else:
                col_str = str(columns)
            lines.append(f"Table: {table_name}\nColumns: {col_str}")
        
        return "\n\n".join(lines)

    def _format_reasoning_context(self, reasoning_ctx: Dict[str, Any]) -> str:
        """
        格式化推理上下文为 Prompt 友好的字符串
        """
        lines = []
        
        # 1. Query Rewrite
        rewrite = reasoning_ctx.get("rewrite_result", {})
        if rewrite:
            lines.append("#### Query Understanding & Rewriting")
            lines.append(f"**Original Question**: {reasoning_ctx.get('original_question', '')}")
            lines.append(f"**Rewritten Question**: {rewrite.get('rewritten_question', '')}")
            lines.append(f"**Keywords Extracted**: {', '.join(rewrite.get('keywords', []))}")
            
            reasoning_trace = rewrite.get("reasoning_trace", [])
            if reasoning_trace:
                lines.append("**Reasoning Trace**:")
                for i, step in enumerate(reasoning_trace, 1):
                    lines.append(f"  {i}. {step}")
        
        # 2. Candidate Schema (已移除)
        
        # 3. Pruning Decision
        pruning = reasoning_ctx.get("pruning_decision", {})
        if pruning:
            lines.append("\n#### Schema Selection Decision")
            llm_dec = pruning.get("llm_decision", {})
            
            norm_rules = llm_dec.get("applied_normalization_rules", [])
            if norm_rules:
                lines.append("**Applied Normalization Rules**:")
                for rule in norm_rules:
                    lines.append(f"  - {rule}")
            
            if not llm_dec.get("is_sufficient", True):
                lines.append(f"**Schema Recovery Applied**: {llm_dec.get('missing_info', '')}")
        
        return "\n".join(lines)
    
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
        # 1. Retrieve (获取完整推理信息)
        retrieval_result = self.retrieve_schema(question, evidence)
        schema_map = retrieval_result.get("schema_map", {})
        reasoning_ctx = retrieval_result.get("reasoning_context", {})
        

        reasoning_context = self._format_reasoning_context(reasoning_ctx)
        
        logger.info(f"Reasoning Context:\n{reasoning_context}")

        current_sql = ""
        error_msg = None
        
        # 2. Generate & Execute Loop
        for attempt in range(self.max_retries + 1):
            logger.info(f"Generating SQL (Attempt {attempt + 1})...")
            
            current_sql = self.generate_sql(question, evidence, retrieval_result, error_msg, current_sql)
            logger.info(f"Generated SQL: {current_sql}")

            if not current_sql:
                return {"error": "Failed to generate SQL", "steps": attempt}

            results, error_msg = self.execute_sql(current_sql)

            if error_msg is None:
                logger.info("Execution Successful.")
                return {
                    "question": question,
                    "sql": current_sql,
                    "result": results,
                    "retrieved_schema": schema_map,
                    "reasoning_context": reasoning_ctx,
                    "status": "success"
                }
            else:
                logger.warning(f"Execution Failed: {error_msg}")

        return {
            "question": question,
            "sql": current_sql,
            "error": error_msg,
            "status": "failed"
        }

    def close(self):
        if self.retriever:
            self.retriever.close()