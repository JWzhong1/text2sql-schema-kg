import sqlite3
import json
import logging
import os
from typing import Dict, Any, List, Optional, Tuple

from src.graph.schema_graph_retriever import GraphRAGRetriever
from src.llm.client import get_competition_json, get_competition, get_competition_from_coder
from src.llm.prompts import get_sql_generation_prompt

# 添加日志配置，设置级别为 INFO
# 修改：添加 force=True 强制覆盖其他模块可能已设置的日志配置
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=False
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
        result = self.retriever.retrieve_schema_subgraph(query)
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
        
        logger.info(f"Reasoning Context:\n{reasoning_context_str}")
        logger.info(f"Retrieved Schema Map: {schema_map}")

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

        response = get_competition_from_coder(messages)
        result = json.loads(response)
        return result.get("sql", "")

    
    def _format_schema_for_prompt(self, schema_map: Any) -> str:
        """
        格式化 Schema Map 为 Prompt 友好的字符串
        支持 Dict[str, List[str]] 或 List[Dict] (BIRD format)
        """
        blocks = []
        if not schema_map:
            return "No schema information available."
            
        # Case 1: List of Table Objects (Rich Schema format)
        if isinstance(schema_map, list):
            for table in schema_map:
                table_lines = []
                t_name = table.get("table_name", "Unknown")
                original_t_name = table.get("original_table_name")
                
                # Table Header
                header = f"Table: {t_name}"
                if original_t_name and original_t_name != t_name:
                    header += f" (Original: {original_t_name})"
                table_lines.append(header)

                # Columns
                columns = table.get("columns", [])
                if columns:
                    table_lines.append("Columns:")
                    for col in columns:
                        if isinstance(col, dict):
                            c_name = col.get("col")
                            c_original_name = col.get("original_column_name")
                            c_type = col.get("type")
                            samples = col.get("sample_values", [])
                            
                            col_info = f"  - {c_name} ({c_type})"

                            if c_original_name:
                                col_info += f" (Original: {c_original_name})"
                            
                            if samples:
                                # Limit samples to avoid context overflow
                                samples_str = ", ".join([str(s) for s in samples[:3]])
                                col_info += f", sample values: [{samples_str}]"
                            
                            c_desc = col.get("column_description")
                            
                            if c_desc:
                                col_info += f", Description: {c_desc}"
                                
                            table_lines.append(col_info)
                        else:
                            table_lines.append(f"  - {str(col)}")
                
                # Primary Keys
                pks = table.get("primary_keys", [])
                if pks:
                    pk_str = ", ".join([str(pk) for pk in pks])
                    table_lines.append(f"Primary Keys: {pk_str}")
                
                # Foreign Keys
                fks = table.get("foreign_keys", [])
                if fks:
                    for fk in fks:
                        fk_table = fk.get("table", "Unknown")
                        fk_columns = fk.get("columns", [])
                        fk_col_str = ", ".join([str(c) for c in fk_columns])
                        table_lines.append(f"Foreign Key -> Table: {fk_table}, Columns: {fk_col_str}")
                
                # Table-level Comments
                t_desc = table.get("table_description")
                if t_desc:
                    table_lines.append(f"Description: {t_desc}")
                
                blocks.append("\n".join(table_lines))

        # Case 2: Simple Dict {table: [cols]}
        elif isinstance(schema_map, dict):
            for table_name, columns in schema_map.items():
                # columns 通常是列名列表
                if isinstance(columns, list):
                    col_str = ", ".join([str(c) for c in columns])
                else:
                    col_str = str(columns)
                blocks.append(f"Table: {table_name}\nColumns: {col_str}")
        
        return "\n\n".join(blocks)

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
        
        # 2. Pruning Decision
        pruning = reasoning_ctx.get("pruning_decision", {})
        if pruning:
            lines.append("\n#### Schema Selection Decision")
            llm_dec = pruning.get("llm_decision", {})
            
            # Handle new format
            if "selected_schema" in llm_dec:
                selected_schema = llm_dec.get("selected_schema", {})
                lines.append("**Selected Schema**:\n (Attention: Not the original names that can be used directly in DB)")
                for t, cols in selected_schema.items():
                    lines.append(f"  - {t}: {cols}")
            else:
                # Fallback to old format
                lines.append(f"**Selected Tables**: {', '.join(llm_dec.get('selected_tables', []))}")
                lines.append(f"**Selected Columns**: {', '.join(llm_dec.get('selected_columns', []))}")

            norm_rules = llm_dec.get("reasoning", [])
            if norm_rules:
                lines.append("**Schema Link Reasoning**:")
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