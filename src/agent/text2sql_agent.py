import sqlite3
import json
import logging
import os
import hashlib
from typing import Dict, Any, List, Optional, Tuple

from src.graph.schema_graph_retriever import GraphRAGRetriever
from src.llm.client import get_competition_json, get_competition, get_competition_from_coder
from src.llm.prompts import get_sql_generation_prompt, get_value_exploration_prompt

# 添加日志配置，设置级别为 INFO
# 修改：添加 force=True 强制覆盖其他模块可能已设置的日志配置
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=False
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Text2SQLAgent:
    def __init__(self, db_name: str, db_path: str, neo4j_config: Tuple[str, str, str], 
                 max_retries: int = 3, enable_value_exploration: bool = True,
                 cache_dir: Optional[str] = None, use_cache: bool = True):
        """
        初始化 Agent
        :param db_name: 数据库名称 (用于检索)
        :param db_path: SQLite 文件路径 (用于执行)
        :param neo4j_config: (uri, user, password) 用于初始化检索器
        :param max_retries: SQL 执行失败后的最大重试次数
        :param enable_value_exploration: 是否启用值探索模块
        :param cache_dir: 缓存目录路径，默认为 cache/{db_name}/retrieval
        :param use_cache: 是否启用缓存，默认 True
        """
        self.db_name = db_name
        self.db_path = db_path
        self.max_retries = max_retries
        self.enable_value_exploration = enable_value_exploration
        self.use_cache = use_cache
        
        # 初始化缓存目录
        self.cache_dir = cache_dir or f"src/agent/schema_retrieval_cache"
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._retrieval_cache = self._load_cache()
        else:
            self._retrieval_cache = {}
        
        # 初始化检索器
        # 注意：这里假设 schema_json_path 存在于标准位置，用于检索器的初始化
        schema_path = f"bird_data/converted_schemas/{db_name}.json"
        self.retriever = GraphRAGRetriever(*neo4j_config, schema_json_path=schema_path)

    def _get_cache_path(self) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, "retrieval_cache.json")
    
    def _load_cache(self) -> Dict[str, Any]:
        """从磁盘加载缓存"""
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    logger.info(f"Loaded retrieval cache from {cache_path}")
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}
    
    def _save_cache(self):
        """保存缓存到磁盘"""
        if not self.use_cache:
            return
        cache_path = self._get_cache_path()
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(self._retrieval_cache, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved retrieval cache to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _generate_cache_key(self, question: str, evidence: str) -> str:
        """生成缓存键（基于问题和证据的哈希）"""
        content = f"{question}|||{evidence}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def retrieve_schema(self, question: str, evidence: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        步骤 1: 调用 GraphRAG 获取 Schema Link 结果及推理信息
        :param question: 自然语言问题
        :param evidence: 背景知识/提示
        :param force_refresh: 是否强制刷新缓存
        """
        cache_key = self._generate_cache_key(question, evidence)
        
        # 检查缓存
        if self.use_cache and not force_refresh and cache_key in self._retrieval_cache:
            logger.info(f"Cache hit for question: {question[:50]}...")
            return self._retrieval_cache[cache_key]
        
        logger.info(f"Retrieving schema for: {question}")
        query = {
            "question": question,
            "evidence": evidence
        }
        # 调用检索逻辑,获取完整信息
        result = self.retriever.retrieve_schema_subgraph(query)
        
        # 保存到缓存
        if self.use_cache:
            self._retrieval_cache[cache_key] = result
            self._save_cache()
            logger.info(f"Cached retrieval result for question: {question[:50]}...")
        
        return result

    def explore_values(self, question: str, evidence: str, retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        值探索模块：生成并执行探索性 SQL 查询
        :return: 包含探索结果的字典
        """
        if not self.enable_value_exploration:
            return {"enabled": False, "exploratory_sql": []}
        
        logger.info("Starting value exploration...")
        
        # 获取关键词和 schema
        reasoning_ctx = retrieval_result.get("reasoning_context", {})
        rewrite_result = reasoning_ctx.get("rewrite_result", {})
        keywords = rewrite_result.get("keywords", [])
        
        schema_map = retrieval_result.get("schema_map", {})
        schema_context = self._format_schema_for_prompt(schema_map)
        
        # 生成探索性 SQL
        exploration_result = self._generate_exploratory_sql(question, evidence, keywords, schema_context)
        
        # 执行探索性 SQL 并收集结果
        executed_explorations = self._execute_exploratory_sql(exploration_result)
        
        logger.info(f"Value exploration completed. Executed {len(executed_explorations)} queries.")
        
        return {
            "enabled": True,
            "exploratory_sql": executed_explorations
        }
    
    def _generate_exploratory_sql(self, question: str, evidence: str, keywords: List[str], schema_context: str) -> List[Dict[str, str]]:
        """
        调用 LLM 生成探索性 SQL
        """
        try:
            system_prompt, user_prompt = get_value_exploration_prompt(
                question, evidence, keywords, schema_context
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = get_competition_json(messages)
            result = json.loads(response)
            
            return result.get("exploratory_sql", [])
        
        except Exception as e:
            logger.warning(f"Failed to generate exploratory SQL: {e}")
            return []
    
    def _execute_exploratory_sql(self, explorations: List[Dict[str, str]], max_results: int = 10) -> List[Dict[str, Any]]:
        """
        执行探索性 SQL 并返回带结果的列表
        :param explorations: 探索性 SQL 列表
        :param max_results: 每个查询返回的最大结果数
        """
        executed = []
        
        if not os.path.exists(self.db_path):
            logger.warning(f"Database file not found: {self.db_path}")
            return executed
        
        for exploration in explorations:
            sql = exploration.get("sql", "")
            purpose = exploration.get("purpose", "")
            
            if not sql:
                continue
            
            result_entry = {
                "sql": sql,
                "purpose": purpose,
                "status": "pending",
                "result": None,
                "error": None,
                "row_count": 0
            }
            
            try:
                conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
                cursor = conn.cursor()
                
                # 添加安全限制
                safe_sql = self._ensure_sql_limit(sql, max_results)
                
                cursor.execute(safe_sql)
                results = cursor.fetchall()
                
                # 获取列名
                column_names = [desc[0] for desc in cursor.description] if cursor.description else []
                
                conn.close()
                
                result_entry["status"] = "success"
                result_entry["result"] = {
                    "columns": column_names,
                    "rows": results
                }
                result_entry["row_count"] = len(results)
                
                logger.debug(f"Exploration query succeeded: {sql[:50]}... -> {len(results)} rows")
                
            except sqlite3.Error as e:
                result_entry["status"] = "error"
                result_entry["error"] = str(e)
                logger.debug(f"Exploration query failed: {sql[:50]}... -> {e}")
            
            except Exception as e:
                result_entry["status"] = "error"
                result_entry["error"] = f"Unexpected error: {str(e)}"
                logger.debug(f"Exploration query unexpected error: {sql[:50]}... -> {e}")
            
            executed.append(result_entry)
        
        return executed
    
    def _ensure_sql_limit(self, sql: str, max_results: int) -> str:
        """
        确保 SQL 有 LIMIT 子句以防止返回过多数据
        """
        sql_upper = sql.upper().strip()
        
        # 如果已经有 LIMIT，不修改
        if "LIMIT" in sql_upper:
            return sql
        
        # 移除末尾分号
        sql = sql.rstrip(";").strip()
        
        return f"{sql} LIMIT {max_results}"
    
    def _format_exploration_context(self, exploration_result: Dict[str, Any]) -> str:
        """
        格式化值探索结果为 Prompt 友好的字符串
        """
        if not exploration_result.get("enabled", False):
            return ""
        
        explorations = exploration_result.get("exploratory_sql", [])
        if not explorations:
            return ""
        
        lines = ["#### Value Exploration Results"]
        lines.append("The following exploratory queries were executed to understand the data:")
        lines.append("")
        
        for i, exp in enumerate(explorations, 1):
            sql = exp.get("sql", "")
            purpose = exp.get("purpose", "")
            status = exp.get("status", "unknown")
            
            lines.append(f"**Query {i}**: `{sql}`")
            lines.append(f"- Purpose: {purpose}")
            lines.append(f"- Status: {status}")
            
            if status == "success":
                result = exp.get("result", {})
                columns = result.get("columns", [])
                rows = result.get("rows", [])
                row_count = exp.get("row_count", 0)
                
                if rows:
                    # 格式化结果
                    lines.append(f"- Found {row_count} row(s)")
                    
                    # 显示列名和前几行数据
                    if columns:
                        col_str = " | ".join(columns)
                        lines.append(f"- Columns: {col_str}")
                    
                    # 限制显示的行数
                    display_rows = rows[:5]
                    for row in display_rows:
                        row_str = " | ".join([str(v) if v is not None else "NULL" for v in row])
                        lines.append(f"  - {row_str}")
                    
                    if len(rows) > 5:
                        lines.append(f"  - ... and {len(rows) - 5} more rows")
                else:
                    lines.append("- No results found")
            
            elif status == "error":
                error = exp.get("error", "Unknown error")
                lines.append(f"- Error: {error}")
            
            lines.append("")
        
        return "\n".join(lines)

    def generate_sql(self, question: str, evidence: str, schema_context: str, 
                      exploration_context_str: str, error_msg: str = None, 
                     previous_sql: str = None) -> str:
        """
        步骤 2 & 4: 生成 SQL (包含推理上下文和值探索结果)
        """
        # schema_map = retrieval_result.get("schema_map", {})
        # reasoning_ctx = retrieval_result.get("reasoning_context", {})
        
        # # 格式化 Schema
        # schema_context = self._format_schema_for_prompt(schema_map)
        
        # # 格式化推理上下文
        # reasoning_context_str = self._format_reasoning_context(reasoning_ctx)
        
        # # 格式化值探索结果
        # exploration_context_str = ""
        # if exploration_result and exploration_result.get("enabled", False):
        #     exploration_context_str = self._format_exploration_context(exploration_result)
        
        # # 合并上下文
        # full_reasoning_context = reasoning_context_str
        # if exploration_context_str:
        #     full_reasoning_context += "\n\n" + exploration_context_str
        
        system_prompt, user_content = get_sql_generation_prompt(
            question, 
            evidence, 
            schema_context, 
            exploration_context_str, 
            error_msg, 
            previous_sql
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = get_competition_json(messages)
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
                header = f"Original_table_name: {original_t_name} \n (数据库中实际保存的名称)"
                if original_t_name and original_t_name != t_name:
                    header += f"(full_name: {t_name}) \n (注意：此名称不可直接用于 SQL 查询,仅展示完整语义)"
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
                            
                            col_info = f"  - {c_original_name} ({c_type})"

                            if c_original_name:
                                col_info += f" (full_name: {c_name})"
                            
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
        
        # 1. Pruning Decision
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

            # norm_rules = llm_dec.get("reasoning", [])
            # if norm_rules:
            #     lines.append("**Schema Link Reasoning**:")
            #     for rule in norm_rules:
            #         lines.append(f"  - {rule}")
            
            # if not llm_dec.get("is_sufficient", True):
            #     lines.append(f"**Schema Recovery Applied**: {llm_dec.get('missing_info', '')}")
        
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

    def solve(self, question: str, evidence: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Agent 主流程
        :param question: 自然语言问题
        :param evidence: 背景知识
        :param force_refresh: 是否强制刷新检索缓存
        """
        # 1. Retrieve (获取完整推理信息)
        retrieval_result = self.retrieve_schema(question, evidence, force_refresh=force_refresh)
        schema_map = retrieval_result.get("schema_map", {})
        reasoning_ctx = retrieval_result.get("reasoning_context", {})
        
        # 2. Value Exploration (可选)
        exploration_result = self.explore_values(question, evidence, retrieval_result)
        
        current_sql = ""
        error_msg = None
        
        # 3. Generate & Execute Loop

        schema_map = retrieval_result.get("schema_map", {})
        reasoning_ctx = retrieval_result.get("reasoning_context", {})
        
        # 格式化 Schema
        schema_context = self._format_schema_for_prompt(schema_map)
        
        # 格式化值探索结果
        exploration_context_str = ""
        if exploration_result and exploration_result.get("enabled", False):
            exploration_context_str = self._format_exploration_context(exploration_result)
        
        
        logger.info(f"Retrieved Schema Map: {schema_map}")
        logger.info(f"exploration_context_str: {exploration_context_str}")

        for attempt in range(self.max_retries + 1):
            logger.info(f"Generating SQL (Attempt {attempt + 1})...")
            
            current_sql = self.generate_sql(
                question, evidence, schema_context, 
                exploration_context_str, error_msg, current_sql
            )
            logger.info(f"Generated SQL: {current_sql}")

            if not current_sql:
                return {"error": "Failed to generate SQL", "steps": attempt}

            results, error_msg = self.execute_sql(current_sql)

            if error_msg is None:
                # 检查结果是否为空
                if results is None or (isinstance(results, list) and len(results) == 0):
                    logger.warning(f"Execution returned empty results (Attempt {attempt + 1})")
                    error_msg = "Query executed successfully but returned no results. The SQL may be incorrect or the filter conditions may be too restrictive."
                    # 继续重试循环
                    continue
                
                logger.info("Execution Successful.")
                return {
                    "question": question,
                    "sql": current_sql,
                    "result": results,
                    "retrieved_schema": schema_map,
                    "reasoning_context": reasoning_ctx,
                    "value_exploration": exploration_result,
                    "status": "success"
                }
            else:
                logger.warning(f"Execution Failed: {error_msg}")

        return {
            "question": question,
            "sql": current_sql,
            "error": error_msg,
            "value_exploration": exploration_result,
            "status": "failed"
        }
    
    def clear_cache(self):
        """清空检索缓存"""
        self._retrieval_cache = {}
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            os.remove(cache_path)
            logger.info(f"Cleared retrieval cache at {cache_path}")

    def close(self):
        # 关闭前保存缓存
        if self.use_cache:
            self._save_cache()
        if self.retriever:
            self.retriever.close()