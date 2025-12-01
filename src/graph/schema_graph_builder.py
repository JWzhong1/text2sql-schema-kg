import json
import os
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import logging
import time
from src.llm import prompts
from src.llm.client import get_competition_json, get_competition_embedding  # 新增导入
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# 创建日志目录
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"schema_graph_builder_{time.strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class GenericSchemaGraphBuilder:
    """
    通用 Schema Graph 构建器
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, batch_size: int = 50, cache_dir: Optional[str] = None, column_embedding_workers: Optional[int] = None):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        logger.info(f"Connected to Neo4j at {neo4j_uri}")
        self.batch_size = batch_size
        self.table_analysis_cache = {}
        self.columns_batch_cache = {}
        self.relationship_analysis_cache = {}
        self.embedding_cache = {}

        self.column_embedding_workers = max(
            1,
            column_embedding_workers or int(os.getenv("EMBEDDING_WORKERS=8", "4"))
        )

        # 可选：缓存持久化
        self.cache_dir = cache_dir or os.getenv("SCHEMA_KG_CACHE_DIR")
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.table_analysis_cache = self._load_cache_file("table_analysis_cache")
            self.columns_batch_cache = self._load_cache_file("columns_batch_cache")
            self.relationship_analysis_cache = self._load_cache_file("relationship_analysis_cache")
            # 新增：加载 embedding 缓存
            self.embedding_cache = self._load_cache_file("embedding_cache")

    def close(self):
        # 退出时再保险保存一次
        self._save_cache_file("table_analysis_cache", self.table_analysis_cache)
        self._save_cache_file("columns_batch_cache", self.columns_batch_cache)
        self._save_cache_file("relationship_analysis_cache", self.relationship_analysis_cache)
        # 新增：保存 embedding 缓存
        self._save_cache_file("embedding_cache", self.embedding_cache)
        self.driver.close()
        logger.info("Neo4j connection closed.")

    def _log_progress(self, current: int, total: int, start_time: float, task_name: str):
        elapsed = time.time() - start_time
        if current > 0:
            avg_time = elapsed / current
            remaining = total - current
            eta = avg_time * remaining
            print(f"\r  {task_name}: {current}/{total} ({(current/total)*100:.2f}%) | ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}", end='', flush=True)
        if current == total:
            print(f"\r  {task_name}: {current}/{total} (100%) | Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")

    def _call_llm_for_table_analysis(self, table_name: str, table_data: Dict[str, Any]) -> Dict[str, Any]:
        # 使用完整表对象构建提示，并返回严格结构: table_name/original_table_name/description/columns/primary_keys/foreign_keys
        try:
            cache_key = f"ta::{table_name}::{len(json.dumps(table_data, ensure_ascii=False))}"
        except Exception:
            cache_key = f"ta::{table_name}"
        if cache_key in self.table_analysis_cache:
            logger.info(f"Table analysis cache hit: {table_name}")
            return self.table_analysis_cache[cache_key]
        
        logger.info(f"Table analysis cache miss. Calling LLM for table analysis: {table_name}")
        sys_prompt, user_prompt = prompts.get_table_analysis_prompt(table_name, table_data)
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        try:
            response = get_competition_json(messages)
            result = json.loads(response)
            result.setdefault("table_name", table_name)
            result.setdefault("original_table_name", table_data.get("original_table_name", ""))
            result.setdefault("description", table_data.get("table_name", ""))
            result.setdefault("columns", [])
            result.setdefault("primary_keys", table_data.get("primary_keys", []))
            result.setdefault("foreign_keys", [])
            self.table_analysis_cache[cache_key] = result
            self._save_cache_file("table_analysis_cache", self.table_analysis_cache)
            return result
        except Exception as e:
            logger.warning(f"LLM table analysis failed: {table_name} - {e}")
            return {
                "table_name": table_name,
                "original_table_name": "",
                "description": table_data.get("table_description", ""),
                "columns": [],
                "primary_keys": table_data.get("primary_keys", []),
                "foreign_keys": []
            }

    def _call_llm_for_columns_batch(self, table_name: str, table_desc: str, columns_info: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        cache_key = f"{table_name}_columns_batch"
        if cache_key in self.columns_batch_cache:
            logger.info(f"Columns batch analysis cache hit: {table_name}")
            return self.columns_batch_cache[cache_key]
        
        logger.info(f"Columns batch analysis cache miss. Calling LLM for columns batch analysis: {table_name}")
        col_summary = "\n".join([f"- {c['col']} ({c['type']}) original_column_name: {c.get('original_column_name', '')} sample_values: {c.get('sample_values', [])}" for c in columns_info])
        sys_prompt, user_prompt = prompts.get_columns_batch_analysis_prompt(table_name, table_desc, col_summary)
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        details_map: Dict[str, Dict[str, Any]] = {}
        try:
            response = get_competition_json(messages)
            arr = json.loads(response)
            if not isinstance(arr, list):
                raise ValueError("LLM columns batch response must be a list.")
            for item in arr:
                col = item.get("column_name")
                if not col:
                    continue
                original_column_name = item.get("original_column_name", "")
                sample_values = item.get("sample_values") or []
                if not isinstance(sample_values, list):
                    sample_values = []
                constraints = item.get("constraints") or {}
                if not isinstance(constraints, dict):
                    constraints = {}
                normalized = {
                    "column_name": col,
                    "data_type": item.get("data_type") or "",
                    "is_primary_key": bool(item.get("is_primary_key", False)),
                    "not_null": bool(item.get("not_null", False)),
                    "auto_increment": bool(item.get("auto_increment", False)),
                    "description": item.get("description") or "",
                    "original_column_name": original_column_name,
                    "sample_values": sample_values,
                    "constraints": {
                        "primary_key": bool(constraints.get("primary_key", item.get("is_primary_key", False))),
                        "foreign_key": bool(constraints.get("foreign_key", False)),
                        "unique": bool(constraints.get("unique", False)),
                        "check": constraints.get("check", None)
                    }
                }
                details_map[col] = normalized
        except Exception as e:
            logger.warning(f"LLM columns batch failed: {table_name} - {e}")
        # 回补未返回列，使用原始 schema 信息
        for c in columns_info:
            col_name = c["col"]
            if col_name not in details_map:
                details_map[col_name] = {
                    "column_name": col_name,
                    "data_type": c.get("type", "") or "",
                    "is_primary_key": False,
                    "not_null": False,
                    "auto_increment": False,
                    "description": c.get("description", "") or "",
                    "original_column_name": c.get("original_column_name", "") or "",
                    "sample_values": [],
                    "constraints": {
                        "primary_key": False,
                        "foreign_key": False,
                        "unique": False,
                        "check": None
                    }
                }
        self.columns_batch_cache[cache_key] = details_map
        self._save_cache_file("columns_batch_cache", self.columns_batch_cache)
        return details_map

    def _call_llm_for_relationship_analysis(self, table1_info: Dict[str, Any], table2_info: Dict[str, Any]) -> Dict[str, Any]:
        t1_name = table1_info["table_name"]
        t2_name = table2_info["table_name"]
        cache_key = f"{min(t1_name, t2_name)}_{max(t1_name, t2_name)}"
        if cache_key in self.relationship_analysis_cache:
            logger.info(f"Relationship analysis cache hit: {t1_name}-{t2_name}")
            return self.relationship_analysis_cache[cache_key]
        
        logger.info(f"Relationship analysis cache miss. Calling LLM for relationship analysis: {t1_name}-{t2_name}")
        t1_summary = f"{t1_name}: {table1_info.get('table_description', table1_info.get('description', ''))}. \nColumns: {[c['col'] for c in table1_info['columns']]}\n primary_keys: {table1_info.get('primary_keys', [])}\n foreign_keys: {table1_info.get('foreign_keys', [])}"
        t2_summary = f"{t2_name}: {table2_info.get('table_description', table2_info.get('description', ''))}. \nColumns: {[c['col'] for c in table2_info['columns']]}\n primary_keys: {table2_info.get('primary_keys', [])}\n foreign_keys: {table2_info.get('foreign_keys', [])}"
        sys_prompt, user_prompt = prompts.get_relationship_analysis_prompt(t1_summary, t2_summary)
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        try:
            response = get_competition_json(messages)
            result = json.loads(response)
            # 默认值调整为 NONE，避免生成无意义的 RELATED_TO
            result.setdefault("relationship_type", "NONE")
            result.setdefault("strength", 0)
            self.relationship_analysis_cache[cache_key] = result
            self._save_cache_file("relationship_analysis_cache", self.relationship_analysis_cache)
            return result
        except Exception as e:
            logger.warning(f"LLM relationship failed: {t1_name}-{t2_name} - {e}")
            return {"relationship_type": "NONE", "strength": 0}

    def create_table_and_columns(self, session, table_data: Dict[str, Any]):
        table_name = table_data["table_name"]
        table_desc = table_data.get("table_description", "")
        columns = table_data.get("columns", [])
        table_analysis = self._call_llm_for_table_analysis(table_name, table_data)

        original_name = table_analysis.get("original_table_name", "")
        description = table_analysis.get("description", table_desc) or table_desc

        raw_pk = table_analysis.get("primary_keys", table_data.get("primary_keys", []))
        if isinstance(raw_pk, list):
            primary_keys = [str(x) for x in raw_pk]
        elif raw_pk is None:
            primary_keys = []
        else:
            primary_keys = [str(raw_pk)]

        raw_cols = table_analysis.get("columns", [])
        column_names: List[str] = []
        if isinstance(raw_cols, list):
            if raw_cols and isinstance(raw_cols[0], dict):
                for c in raw_cols:
                    name = c.get("col") or c.get("name") or c.get("column_name")
                    if name:
                        column_names.append(str(name))
            else:
                column_names = [str(c) for c in raw_cols]
        # 回退到原始 schema 的列名
        if not column_names:
            column_names = [c.get("col") for c in columns if isinstance(c, dict) and c.get("col")]

        raw_fk = table_analysis.get("foreign_keys", [])
        foreign_keys_norm: List[str] = []
        if isinstance(raw_fk, list):
            for fk in raw_fk:
                if isinstance(fk, dict):
                    col = fk.get("column") or fk.get("from") or fk.get("local_column")
                    ref_table = fk.get("referenced_table") or fk.get("to_table") or fk.get("table")
                    ref_col = fk.get("referenced_column") or fk.get("to_column")
                    if col and ref_table and ref_col:
                        foreign_keys_norm.append(f"{col}->{ref_table}.{ref_col}")
                    else:
                        foreign_keys_norm.append(json.dumps(fk, ensure_ascii=False))
                else:
                    foreign_keys_norm.append(str(fk))
        elif raw_fk:
            # 单个对象或字符串
            if isinstance(raw_fk, dict):
                foreign_keys_norm.append(json.dumps(raw_fk, ensure_ascii=False))
            else:
                foreign_keys_norm.append(str(raw_fk))
        foreign_keys_json = json.dumps(raw_fk, ensure_ascii=False) if isinstance(raw_fk, (list, dict)) else ""

        # 新：获取列完整详情（含约束等）
        column_details_map = self._call_llm_for_columns_batch(table_name, description, columns)

        # 新：生成 Table 的嵌入向量
        table_text = f"{table_name} {description}".strip()
        table_embedding = self._embed_text(table_text) if table_text else []

        # 持久化缓存
        self._save_cache_file("embedding_cache", self.embedding_cache)

        with session.begin_transaction() as tx:
            tx.run("""
                MERGE (t:Table {name: $name})
                SET t.original_name = $original_name,
                    t.description = $description,
                    t.columns = $column_names,
                    t.primary_keys = $primary_keys,
                    t.foreign_keys = $foreign_keys,       // 扁平化后的字符串数组
                    t.foreign_keys_json = $foreign_keys_json,  // 原始外键 JSON 字符串
                    t.embedding = $embedding  // 新增：Table 嵌入向量
            """,
            name=table_name,
            original_name=original_name,
            description=description,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys_norm,
            foreign_keys_json=foreign_keys_json,
            column_names=column_names,
            embedding=table_embedding)

            # 列节点
            column_embeddings = self._generate_column_embeddings(table_name, columns, column_details_map)
            
            # 持久化缓存
            self._save_cache_file("embedding_cache", self.embedding_cache)

            for col_info in columns:
                col_name = col_info['col']
                d = column_details_map.get(col_name, {})
                constraints = d.get("constraints", {}) if isinstance(d, dict) else {}
                constraints_json = json.dumps(constraints, ensure_ascii=False) if constraints else ""
                # 新：生成 Column 的嵌入向量
                col_description = d.get("description", col_info.get('description', ''))
                col_embedding = column_embeddings.get(col_name, [])
                tx.run("""
                    MERGE (c:Column {name: $col_name, table_name: $table_name})
                    SET c.data_type = $data_type,
                        c.description = $description,
                        c.is_primary_key = $is_primary_key,
                        c.not_null = $not_null,
                        c.auto_increment = $auto_increment,
                        c.original_name = $original_name,
                        c.sample_values = $sample_values,
                        c.unique = $unique,
                        c.is_foreign_key = $is_foreign_key,
                        c.constraints_json = $constraints_json,
                        c.embedding = $embedding  // 新增：Column 嵌入向量
                    WITH c
                    MATCH (t:Table {name: $table_name})
                    MERGE (t)-[:CONTAINS_COLUMN]->(c)
                """,
                col_name=col_name,
                table_name=table_name,
                data_type=d.get("data_type", col_info.get('type', '')),
                description=col_description,
                is_primary_key=bool(d.get("is_primary_key", False)),
                not_null=bool(d.get("not_null", False)),
                auto_increment=bool(d.get("auto_increment", False)),
                original_name=d.get("original_column_name", ""),
                sample_values=d.get("sample_values", []),
                unique=bool(constraints.get("unique", False)),
                is_foreign_key=bool(constraints.get("foreign_key", False)),
                constraints_json=constraints_json,
                embedding=col_embedding)

    def create_explicit_relationships(self, session, schema_data: List[Dict[str, Any]]) -> set:
        """
        创建显式外键关系：
        1. (FK Column)-[:FK_TO]->(PK Column)
        2. (Source Table)-[:REFERENCES]->(Target Table)
        3. (FK Column)-[:REFERENCES_TABLE]->(Target Table)
        同时标记 Column.is_foreign_key = true
        返回已连接的表对集合 {(t1, t2), ...}
        """
        connected_pairs = set()
        logger.info("Creating explicit foreign key relationships...")
        
        for table_data in schema_data:
            t1_name = table_data["table_name"]
            # 利用缓存获取清洗后的外键信息
            analysis = self._call_llm_for_table_analysis(t1_name, table_data)
            fks = analysis.get("foreign_keys", [])
            
            for fk in fks:
                if not isinstance(fk, dict):
                    continue
                
                c1_name = fk.get("column")
                t2_name = fk.get("referenced_table")
                c2_name = fk.get("referenced_column")
                
                if c1_name and t2_name and c2_name:
                    pair = tuple(sorted((t1_name, t2_name)))
                    connected_pairs.add(pair)
                    
                    description = f"Foreign Key: {t1_name}.{c1_name} -> {t2_name}.{c2_name}"

                    # 执行 Cypher 创建关系
                    session.run("""
                        MATCH (t1:Table {name: $t1_name})
                        MATCH (c1:Column {name: $c1_name, table_name: $t1_name})
                        MATCH (t2:Table {name: $t2_name})
                        MATCH (c2:Column {name: $c2_name, table_name: $t2_name})
                        
                        MERGE (c1)-[:FK_TO]->(c2)
                        MERGE (t1)-[r:REFERENCES]->(t2)
                        MERGE (c1)-[:REFERENCES_TABLE]->(t2)
                        SET c1.is_foreign_key = true,
                            r.description = $description
                    """, t1_name=t1_name, c1_name=c1_name, t2_name=t2_name, c2_name=c2_name, description=description)
        
        logger.info(f"Created explicit relationships for {len(connected_pairs)} pairs.")
        return connected_pairs

    def create_semantic_relationships(self, session, all_table_data: List[Dict[str, Any]], existing_connections: set):
        """
        仅对未通过外键连接的表对进行 LLM 语义分析。
        创建双向语义关系。
        """
        tables = [td["table_name"] for td in all_table_data]
        table_map = {td["table_name"]: td for td in all_table_data}
        total_pairs = len(tables) * (len(tables) - 1) // 2
        processed = 0
        start = time.time()
        
        # 过滤掉已经有外键连接的对
        candidate_pairs = []
        for i in range(len(tables)):
            for j in range(i + 1, len(tables)):
                t1 = tables[i]
                t2 = tables[j]
                pair = tuple(sorted((t1, t2)))
                if pair not in existing_connections:
                    candidate_pairs.append((t1, t2))
                else:
                    processed += 1 # 视为已处理
        
        logger.info(f"Scanning {len(candidate_pairs)} pairs for semantic relationships (skipped {len(existing_connections)} explicit pairs)...")

        for t1, t2 in candidate_pairs:
            rel = self._call_llm_for_relationship_analysis(table_map[t1], table_map[t2])
            rel_type = rel.get("relationship_type", "NONE")
            
            if rel_type and rel_type != "NONE" and rel_type != "RELATED_TO":
                strength = rel.get("strength", 1)
                description = rel.get("relationship_details", {}).get("description", "")
                
                # 双向创建语义关系
                session.run(f"""
                    MATCH (a:Table {{name: $t1}}) MATCH (b:Table {{name: $t2}})
                    MERGE (a)-[r1:{rel_type}]->(b)
                    MERGE (b)-[r2:{rel_type}]->(a)
                    SET r1.strength = $strength, r1.description = $description,
                        r2.strength = $strength, r2.description = $description
                """, t1=t1, t2=t2, strength=strength, description=description)
            
            processed += 1
            if processed % 50 == 0:
                self._log_progress(processed, total_pairs, start, "Analyzing Relationships")

    def build_graph(self, schema_data: List[Dict[str, Any]]):
        logger.info(f"Starting to build graph for {len(schema_data)} tables...")
        overall_start = time.time()
        with self.driver.session() as session:
            logger.info("Step 1/3: Creating Table and Column Nodes...")
            for i, table_data in enumerate(schema_data):
                self.create_table_and_columns(session, table_data)
                self._log_progress(i + 1, len(schema_data), overall_start, "Creating Tables/Columns")
            
            logger.info("\nStep 2/3: Creating Explicit Foreign Key Relationships...")
            connected_pairs = self.create_explicit_relationships(session, schema_data)
            
            logger.info("\nStep 3/3: Creating Semantic Relationships (LLM)...")
            self.create_semantic_relationships(session, schema_data, connected_pairs)
            
        logger.info(f"\nGraph build completed in {time.time() - overall_start:.2f} seconds.")

    def create_indexes(self):
        logger.info("Creating indexes...")
        with self.driver.session() as session:
            session.run("CREATE INDEX IF NOT EXISTS FOR (t:Table) ON (t.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (c:Column) ON (c.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (c:Column) ON (c.table_name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (t:Table) ON (t.concepts)")
            
            # 修改：指定向量索引名称，并设置维度和相似度函数
            # 注意：请将 1024 替换为你实际使用的 Embedding 模型维度 (如 OpenAI text-embedding-3-small 为 1536, BGE-m3 为 1024 等)
            session.run("""
                CREATE VECTOR INDEX table_embedding IF NOT EXISTS 
                FOR (t:Table) ON (t.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 1024,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
            
            session.run("""
                CREATE VECTOR INDEX column_embedding IF NOT EXISTS 
                FOR (c:Column) ON (c.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 1024,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
        logger.info("Indexes created.")

    # 缓存工具
    def _cache_enabled(self) -> bool:
        return bool(self.cache_dir)

    def _cache_path(self, name: str) -> str:
        return os.path.join(self.cache_dir, f"{name}.json")

    def _load_cache_file(self, name: str) -> Dict[str, Any]:
        if not self._cache_enabled():
            return {}
        try:
            path = self._cache_path(name)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Load cache '{name}' failed: {e}")
        return {}

    def _save_cache_file(self, name: str, data: Dict[str, Any]):
        if not self._cache_enabled():
            return
        try:
            with open(self._cache_path(name), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Save cache '{name}' failed: {e}")

    def _embed_text(self, text: str) -> List[float]:
        if not text:
            return []
        
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        try:
            emb = get_competition_embedding(text)
            self.embedding_cache[text] = emb
            return emb
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return []

    def _generate_column_embeddings(self, table_name: str, columns: List[Dict[str, Any]], column_details_map: Dict[str, Dict[str, Any]]) -> Dict[str, List[float]]:
        texts: Dict[str, str] = {}
        for col in columns:
            col_name = col["col"]
            detail = column_details_map.get(col_name, {})
            col_description = detail.get("description", col.get("description", ""))
            texts[col_name] = f"{col_name} {col_description}".strip()
        non_empty = {k: v for k, v in texts.items() if v}
        embeddings = {k: [] for k in texts.keys()}
        if not non_empty:
            return embeddings
        workers = min(len(non_empty), self.column_embedding_workers)
        if workers == 1:
            for col_name, text in non_empty.items():
                embeddings[col_name] = self._embed_text(text)
            return embeddings
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._embed_text, text): col for col, text in non_empty.items()}
            for future in as_completed(futures):
                col_name = futures[future]
                embeddings[col_name] = future.result() or []
        return embeddings

def load_schema_from_file(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Schema file must contain a JSON array.")
    return data

def main():
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "your_password")
    schema_file_path = "data/converted_schema.json"
    logger.info(f"Loading schema from {schema_file_path}")
    schema_data = load_schema_from_file(schema_file_path)
    builder = GenericSchemaGraphBuilder(neo4j_uri, neo4j_user, neo4j_password)
    try:
        # 修改：先创建索引，确保即使后续数据导入失败，索引也存在
        builder.create_indexes() 
        builder.build_graph(schema_data)
        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
    finally:
        builder.close()

if __name__ == "__main__":
    main()