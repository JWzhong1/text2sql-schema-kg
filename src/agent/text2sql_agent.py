import sqlite3
import json
import logging
import os
import hashlib
from typing import Dict, Any, List, Optional, Tuple

from src.graph.schema_graph_retriever import GraphRAGRetriever
from src.llm.client import get_competition_json, get_competition, get_competition_from_coder
from src.llm.prompts import get_sql_generation_prompt, get_value_exploration_prompt, get_self_correction_prompt

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
                 enable_self_correction: bool = True,
                 cache_dir: Optional[str] = None, use_cache: bool = True):
        """
        初始化 Agent
        :param db_name: 数据库名称 (用于检索)
        :param db_path: SQLite 文件路径 (用于执行)
        :param neo4j_config: (uri, user, password) 用于初始化检索器
        :param max_retries: SQL 执行失败后的最大重试次数
        :param enable_value_exploration: 是否启用值探索模块
        :param enable_self_correction: 是否启用自纠正模块，默认 True
        :param cache_dir: 缓存目录路径，默认为 cache/{db_name}/retrieval
        :param use_cache: 是否启用缓存，默认 True
        """
        self.db_name = db_name
        self.db_path = db_path
        self.max_retries = max_retries
        self.enable_value_exploration = enable_value_exploration
        self.enable_self_correction = enable_self_correction
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

        return "\n".join(lines)
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
                     previous_sql: str = None, attempt: int = 1) -> str:
        """
        步骤 2 & 4: 生成 SQL (包含推理上下文和值探索结果)
        :param attempt: 当前尝试次数，用于调整提示策略
        """
        system_prompt, user_content = get_sql_generation_prompt(
            question, 
            evidence, 
            schema_context, 
            exploration_context_str, 
            error_msg, 
            previous_sql,
            attempt  # 传递尝试次数
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = get_competition_json(messages)
        result = json.loads(response)
        return result.get("sql", "")

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

    def self_correct_sql(self, question: str, evidence: str, schema_context: str, 
                         generated_sql: str, exploration_context_str: str = "") -> Dict[str, Any]:
        """
        Self-Correction 模块：对生成的 SQL 进行逻辑层面的验证和纠错
        
        :param question: 原始自然语言问题
        :param evidence: 背景知识/证据
        :param schema_context: Schema 上下文信息
        :param generated_sql: 生成的 SQL 查询
        :param exploration_context_str: 值探索结果上下文（可选）
        :return: 包含验证结果和可能的修正 SQL 的字典
        """
        logger.info("Starting Self-Correction validation...")
        
        try:
            system_prompt, user_prompt = get_self_correction_prompt(
                question, 
                evidence, 
                schema_context, 
                generated_sql,
                exploration_context_str if exploration_context_str else None
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = get_competition_json(messages)
            correction_result = json.loads(response)
            
            # 验证返回格式
            required_keys = ["is_correct", "confidence", "error_analysis"]
            if not all(key in correction_result for key in required_keys):
                logger.warning(f"Self-correction response missing required keys: {correction_result.keys()}")
                return {
                    "validation_performed": False,
                    "error": "Invalid response format from LLM"
                }
            
            # 记录验证结果
            if correction_result.get("is_correct", False):
                logger.info(f"✓ SQL validated as correct (confidence: {correction_result.get('confidence', 0):.2f})")
            else:
                logger.warning(f"✗ SQL validation failed. Error patterns detected: "
                              f"{len(correction_result.get('error_analysis', {}).get('error_patterns', []))}")
                
                # 详细记录错误模式
                error_patterns = correction_result.get('error_analysis', {}).get('error_patterns', [])
                for i, pattern in enumerate(error_patterns, 1):
                    logger.warning(f"  Error {i}: [{pattern.get('severity', 'UNKNOWN')}] {pattern.get('type', 'UNKNOWN')} - "
                                  f"{pattern.get('description', 'No description')}")
            
            return {
                "validation_performed": True,
                "is_correct": correction_result.get("is_correct", False),
                "confidence": correction_result.get("confidence", 0.0),
                "error_analysis": correction_result.get("error_analysis", {}),
                "corrected_sql": correction_result.get("corrected_sql", ""),
                "correction_reasoning": correction_result.get("correction_reasoning", ""),
                "original_sql": generated_sql
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse self-correction response: {e}")
            return {
                "validation_performed": False,
                "error": f"JSON parse error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Self-correction failed with exception: {e}")
            return {
                "validation_performed": False,
                "error": str(e)
            }

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
        
        # 格式化 Schema
        schema_context = self._format_schema_for_prompt(schema_map)
        
        # 格式化值探索结果
        exploration_context_str = ""
        if exploration_result and exploration_result.get("enabled", False):
            exploration_context_str = self._format_exploration_context(exploration_result)
        
        logger.info(f"Retrieved Schema Map: {schema_map}")
        logger.info(f"exploration_context_str: {exploration_context_str}")

        # 3. Generate & Execute Loop with simplified empty result handling
        execution_history = []  # 记录执行历史
        empty_result_count = 0  # 跟踪空结果次数
        
        for attempt in range(self.max_retries + 1):
            logger.info(f"Generating SQL (Attempt {attempt + 1}/{self.max_retries + 1})...")
            
            # 分析错误模式,决定是否需要特殊处理
            error_analysis = self._analyze_error_pattern(execution_history) if execution_history else None
            
            current_sql = self.generate_sql(
                question, evidence, schema_context, 
                exploration_context_str, 
                error_analysis.get("error_msg") if error_analysis else None,
                error_analysis.get("reference_sql") if error_analysis else None,
                attempt + 1
            )
            
            logger.info(f"Generated SQL: {current_sql}")

            if not current_sql:
                execution_history.append({
                    "attempt": attempt + 1,
                    "sql": None,
                    "error": "Failed to generate SQL",
                    "result": None,
                    "self_correction": None
                })
                continue

            # Self-Correction: 对生成的 SQL 进行逻辑验证（可选功能）
            self_correction_result = {}
            sql_to_execute = current_sql
            
            # 修改：仅在第一次尝试时执行 Self-Correction，重试时跳过
            if self.enable_self_correction and attempt == 0:
                self_correction_result = self.self_correct_sql(
                    question, 
                    evidence, 
                    schema_context, 
                    current_sql,
                    exploration_context_str
                )
                
                # 如果 Self-Correction 发现错误并提供了修正 SQL,使用修正后的 SQL
                if (self_correction_result.get("validation_performed", False) and 
                    not self_correction_result.get("is_correct", True) and 
                    self_correction_result.get("corrected_sql", "")):
                    
                    corrected_sql = self_correction_result.get("corrected_sql", "")
                    logger.info(f"Self-Correction provided corrected SQL: {corrected_sql}")
                    logger.info(f"Correction reasoning: {self_correction_result.get('correction_reasoning', 'N/A')}")
                    
                    # 使用修正后的 SQL
                    sql_to_execute = corrected_sql
            elif self.enable_self_correction and attempt > 0:
                logger.info(f"Skipping Self-Correction for retry attempt {attempt + 1}")
            else:
                logger.debug("Self-Correction is disabled, using generated SQL directly")

            results, error_msg = self.execute_sql(sql_to_execute)
            
            # 检查是否为空结果
            is_empty = error_msg is None and (results is None or len(results) == 0)
            
            # 记录执行历史
            execution_history.append({
                "attempt": attempt + 1,
                "sql": current_sql,  # 记录原始生成的 SQL
                "executed_sql": sql_to_execute,  # 记录实际执行的 SQL
                "self_correction": self_correction_result,  # 记录 Self-Correction 结果
                "error": error_msg,
                "result": results if error_msg is None else None,
                "is_empty": is_empty
            })

            if error_msg is None:
                # 简化空结果处理逻辑
                if is_empty:
                    empty_result_count += 1
                    logger.warning(f"Execution returned empty results (Empty count: {empty_result_count}/2)")
                    
                    # 如果已经连续两次空结果,直接结束
                    if empty_result_count >= 2:
                        logger.warning("Two consecutive empty results, stopping retry.")
                        return {
                            "question": question,
                            "sql": execution_history[-1]["executed_sql"],
                            "original_sql": execution_history[-1]["sql"],
                            "result": [],
                            "retrieved_schema": schema_map,
                            "reasoning_context": reasoning_ctx,
                            "value_exploration": exploration_result,
                            "self_correction": execution_history[-1].get("self_correction", {}),
                            "execution_history": execution_history,
                            "status": "empty_result",
                            "message": "Query executed successfully but returned no results after 2 attempts"
                        }
                    
                    continue  # 第一次空结果,继续重试
                
                # 有结果,执行成功
                logger.info("Execution Successful with results.")
                last_correction = execution_history[-1].get("self_correction", {})
                return {
                    "question": question,
                    "sql": execution_history[-1]["executed_sql"],
                    "original_sql": execution_history[-1]["sql"],
                    "result": results,
                    "retrieved_schema": schema_map,
                    "reasoning_context": reasoning_ctx,
                    "value_exploration": exploration_result,
                    "self_correction": last_correction,
                    "execution_history": execution_history,
                    "status": "success"
                }
            else:
                # 有执行错误,重置空结果计数器
                empty_result_count = 0
                logger.warning(f"Execution Failed: {error_msg}")

        # 所有尝试失败(非空结果导致的失败)
        last_history = execution_history[-1] if execution_history else {}
        return {
            "question": question,
            "sql": last_history.get("executed_sql") or last_history.get("sql"),
            "original_sql": last_history.get("sql"),
            "error": last_history.get("error", "Unknown error"),
            "retrieved_schema": schema_map,
            "reasoning_context": reasoning_ctx,
            "value_exploration": exploration_result,
            "self_correction": last_history.get("self_correction", {}),
            "execution_history": execution_history,
            "status": "failed"
        }
    
    def _analyze_error_pattern(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析历史错误模式，生成智能化的错误提示
        :param history: 执行历史记录
        :return: 包含 error_msg 和 reference_sql 的字典
        """
        if not history:
            return {"error_msg": None, "reference_sql": None}
        
        # 统计错误类型
        syntax_errors = []
        empty_results = []
        execution_errors = []
        
        for record in history:
            if record["error"]:
                if "syntax" in record["error"].lower() or "near" in record["error"].lower():
                    syntax_errors.append(record)
                else:
                    execution_errors.append(record)
            elif record.get("is_empty", False):
                empty_results.append(record)
        
        # 根据错误模式生成不同的提示策略
        
        # 1. 连续多次语法错误 - 重新审视 schema
        if len(syntax_errors) >= 2:
            last_error = syntax_errors[-1]
            return {
                "error_msg": (
                    f"### Pattern Detected: Repeated Syntax Errors\n"
                    f"Last SQL: {last_error['sql']}\n"
                    f"Last Error: {last_error['error']}\n\n"
                    f"**Critical Issue**: Multiple syntax errors suggest schema mismatch.\n"
                    f"**Action Required**:\n"
                    f"1. Carefully verify ALL table and column names against the Retrieved Schema\n"
                    f"2. Check for typos in table/column references\n"
                    f"3. Ensure you're using the original_table_name and original_column_name\n"
                    f"4. Verify JOIN conditions reference existing foreign keys\n"
                    f"5. Double-check SQLite syntax (e.g., use || for concatenation, not +)"
                ),
                "reference_sql": None  # 不参考之前的 SQL
            }
        
        # 2. 连续多次空结果 - 重新思考查询逻辑
        if len(empty_results) >= 2:
            different_sqls = len(set(r["sql"] for r in empty_results)) > 1
            
            if different_sqls:
                # 尝试了不同 SQL 都返回空结果
                return {
                    "error_msg": (
                        f"### Pattern Detected: Multiple Empty Results with Different Queries\n"
                        f"Attempted {len(empty_results)} different SQL queries, all returned empty.\n\n"
                        f"**Root Cause Analysis**:\n"
                        f"This suggests a fundamental misunderstanding of the question or data.\n\n"
                        f"**Recommended Approach**:\n"
                        f"1. **Re-examine the question**: What is REALLY being asked?\n"
                        f"2. **Reconsider table selection**: Are we querying the right tables?\n"
                        f"3. **Review Value Exploration results**: Do the explored values match expectations?\n"
                        f"4. **Simplify the query**: Start with a basic SELECT to verify data exists\n"
                        f"5. **Check filter logic**: Are we applying contradictory conditions?\n\n"
                        f"Start fresh with a completely different approach."
                    ),
                    "reference_sql": None  # 完全重新开始
                }
            else:
                # 同一个 SQL 重复执行返回空结果（不应该发生，防御性编程）
                return {
                    "error_msg": (
                        f"### Warning: Identical SQL Repeated\n"
                        f"The same SQL was generated multiple times: {empty_results[0]['sql']}\n\n"
                        f"Please generate a DIFFERENT query with alternative filter conditions."
                    ),
                    "reference_sql": None
                }
        
        # 3. 单次执行错误 - 标准错误修正
        if history[-1]["error"]:
            last_record = history[-1]
            return {
                "error_msg": (
                    f"### Execution Error\n"
                    f"SQL: {last_record['sql']}\n"
                    f"Error: {last_record['error']}\n\n"
                    f"**Fix Instructions**:\n"
                    f"- Verify column/table names match the schema exactly\n"
                    f"- Check SQLite-specific syntax requirements\n"
                    f"- Ensure data types in comparisons are compatible"
                ),
                "reference_sql": last_record['sql']
            }
        
        # 4. 单次空结果 - 标准空结果处理
        if history[-1].get("is_empty", False):
            last_record = history[-1]
            return {
                "error_msg": (
                    f"### Query Returned Empty Results\n"
                    f"SQL: {last_record['sql']}\n\n"
                    f"**Possible Issues**:\n"
                    f"1. Filter conditions too restrictive (consider using LIKE with wildcards)\n"
                    f"2. Incorrect value matching (check Value Exploration results for actual values)\n"
                    f"3. Case sensitivity issues (use COLLATE NOCASE or LOWER())\n"
                    f"4. Missing/incorrect JOIN conditions\n"
                    f"5. Wrong categorical column selected (prefer type/status columns over names)\n\n"
                    f"**Suggested Actions**:\n"
                    f"- Review the Value Exploration results to validate your filter values\n"
                    f"- Try relaxing filter conditions or using fuzzy matching\n"
                    f"- Verify JOIN keys are correct"
                ),
                "reference_sql": last_record['sql']
            }
        
        # 默认返回
        return {"error_msg": None, "reference_sql": None}

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