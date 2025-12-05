import os
import json
import logging
import networkx as nx
import dotenv
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from neo4j import GraphDatabase
from src.llm.client import get_competition_embedding, get_competition_json
from src.llm import prompts

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- 数据结构优化 ---

@dataclass
class NodeItem:
    """统一的节点数据结构，避免反复查询数据库获取属性"""
    id: str
    name: str
    labels: List[str]
    table_name: Optional[str] = None
    original_name: Optional[str] = None
    table_original_name: Optional[str] = None
    score: float = 0.0  # 向量检索分数或PPR分数
    description: str = ""

    @property
    def is_table(self):
        return "Table" in self.labels

    @property
    def is_column(self):
        return "Column" in self.labels

class GraphRAGRetriever:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, schema_json_path: str = None):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        try:
            with open(schema_json_path, 'r') as f:
                self.schema = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load schema JSON: {e}")
            self.schema = {}
            
        # 线程配置
        self.emb_threads = max(1, int(os.getenv("PPR_EMB_THREADS", "8")))
        self.tog_threads = int(os.getenv("TOG_MAX_THREADS", "4"))

    def close(self):
        self.driver.close()

    def retrieve_schema_subgraph(self, query: Dict, return_candidate_map: bool = False) -> Dict[str, Any]:
        """
        入口函数
        """
        # 1. 神经符号检索 (Embedding + PPR + ToG)
        final_nodes, candidate_map = self._neuro_symbolic_retrieve(query)
        
        # 2. 格式化输出 (将 NodeItem 对象转换为题目要求的 Map 格式)
        final_map = self._build_map_from_nodes(final_nodes)

        if return_candidate_map:
            return {"candidate": candidate_map, "final": final_map}
        return final_map

    # -------------------------
    # Core Logic
    # -------------------------

    def _neuro_symbolic_retrieve(self, query: Dict) -> Tuple[Dict[str, NodeItem], Dict[str, List[str]]]:
        """
        执行检索全流程。
        返回: (最终保留的节点字典 {id: NodeItem}, 候选阶段的Map)
        """

        # 1. Phase 0: Query Rewrite
        rewrite_query = self._rewrite_query_for_schema_linking({"question": query}, self.schema)
        rewritten_question = rewrite_query.get("rewritten_question", query.get("question", ""))
        keywords = list(set([k.strip().lower() for k in rewrite_query.get("keywords", [])]))
        logger.info(f"Rewritten Query: {rewritten_question}, Keywords: {keywords}")
        
        # Phase 1: Embedding & PPR (获取种子并扩展)
        # 这里返回的是已经包含元数据的 NodeItem 字典，Key 是 elementId
        
        candidate_nodes, candidate_map = self._embeddings_retrieve(rewrite_query)
        logger.info(f"Phase 1: Retrieved {len(candidate_nodes)} candidate nodes.")
        logger.info(f"Candidate Map: {json.dumps(candidate_map, ensure_ascii=False)}")

        if not candidate_nodes:
            return {}, candidate_map

        # Phase 2: ToG (Graph Traversal)
        if os.getenv("TOG_ENABLED", "1") == "1":
            # 为了代码清晰，我们在 _tog_traversal 中不仅返回 ID，还返回新发现节点的 NodeItem
            final_nodes_map = self._tog_traversal(rewrite_query, candidate_nodes)
            return final_nodes_map, candidate_map
        else:
            logger.info("ToG Traversal disabled.")
            return candidate_nodes, candidate_map

    # -------------------------
    # Phase 1: Seeds Retrieval & PPR
    # -------------------------
    def _embeddings_retrieve(self, query: str) -> Tuple[Dict[str, NodeItem], Dict[str, List[str]]]:
        """
        向量检索 + PPR。
        优化：直接返回 NodeItem 对象，包含 elementId，避免后续名字查ID的开销。
        """
        # # 1. Query Rewrite
        # rewrite_result = self._rewrite_query_for_schema_linking({"question": query}, self.schema)
        # rewritten_query = rewrite_result.get("rewritten_question", query)
        # keywords = list(set([k.strip().lower() for k in rewrite_result.get("keywords", [])]))
        # logger.info(f"Rewritten Query: {rewritten_query}, Keywords: {keywords}")
        keywords = query.get("keywords", [])
        rewritten_question = query.get("rewritten_question")

        # 2. Parallel Embedding
        target_items = keywords + [rewritten_question]
        embeddings_map = self._parallel_get_embeddings(target_items)
        
        if not embeddings_map:
            return {}, {}

        # 3. Vector Search in Neo4j (Fetch Nodes with Metadata immediately)
        # 结果字典: {elementId: NodeItem}
        seed_nodes_map: Dict[str, NodeItem] = {}
        
        # 阈值
        threshold_kw = float(os.getenv("PPR_SCORE_THRESHOLD_KW", "0"))
        threshold_query = float(os.getenv("PPR_SCORE_THRESHOLD_QUERY", "0"))
        seed_k = int(os.getenv("PPR_SEED_K_EACH", "5"))

        with self.driver.session() as session:
            for text, emb in embeddings_map.items():
                threshold = threshold_query if text == rewritten_question else threshold_kw
                # 同时查询 Table 和 Column，减少 session 交互次数 (可以使用 UNION 或两次 run)
                # 这里为了清晰還是分開寫，但關鍵是直接提取所有屬性
                
                # Column Search
                res = session.run("""
                    CALL db.index.vector.queryNodes('column_embedding', $k, $emb)
                    YIELD node AS n, score
                    WHERE score >= $thresh
                    MATCH (t:Table {name: n.table_name})
                    RETURN elementId(n) as id, n.name as name, n.original_name as original_name, labels(n) as labels, n.table_name as table_name, t.original_name as table_original_name, n.description as description, score
                """, k=seed_k, emb=emb, thresh=threshold)
                self._parse_neo4j_result_to_nodes(res, seed_nodes_map)

                # Table Search
                res = session.run("""
                    CALL db.index.vector.queryNodes('table_embedding', $k, $emb)
                    YIELD node AS n, score
                    WHERE score >= $thresh
                    RETURN elementId(n) as id, n.name as name, n.original_name as original_name, labels(n) as labels, null as table_name, null as table_original_name, n.description as description, score
                """, k=seed_k, emb=emb, thresh=threshold)
                self._parse_neo4j_result_to_nodes(res, seed_nodes_map)

        if not seed_nodes_map:
            return {}, {}

        # 4. PPR (PageRank)
        # 构建内存图用于计算 PPR
        G = nx.Graph()
        G.add_nodes_from(seed_nodes_map.keys())
        
        seed_ids = list(seed_nodes_map.keys())
        # 批量获取边
        with self.driver.session() as session:
            res = session.run("""
                MATCH (n)-[r]-(m)
                WHERE elementId(n) IN $ids AND elementId(m) IN $ids AND elementId(n) < elementId(m)
                RETURN elementId(n) as src, elementId(m) as dst
            """, ids=seed_ids)
            edges = [(r["src"], r["dst"]) for r in res]
            G.add_edges_from(edges)

        # PPR Calculation
        max_score = max((n.score for n in seed_nodes_map.values()), default=1.0)
        personalization = {
            nid: (node.score / max_score) for nid, node in seed_nodes_map.items()
        }
        ppr_scores = nx.pagerank(G, alpha=float(os.getenv("PPR_ALPHA", "0.15")), personalization=personalization)

        # 5. Filter Top-K
        topk_tables = int(os.getenv("PPR_TOPK_TABLES", "10"))
        topk_columns = int(os.getenv("PPR_TOPK_COLUMNS", "50"))

        final_candidates: Dict[str, NodeItem] = {}
        t_count, c_count = 0, 0
        
        # 排序
        sorted_nodes = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)
        
        for nid, score in sorted_nodes:
            if nid not in seed_nodes_map: continue
            node = seed_nodes_map[nid]
            
            # 更新分数为 PPR 分数 (可选)
            # node.score = score 
            
            if node.is_table:
                if t_count < topk_tables:
                    final_candidates[nid] = node
                    t_count += 1
            elif node.is_column:
                if c_count < topk_columns:
                    final_candidates[nid] = node
                    c_count += 1
                    # 同时也需要把该列对应的表（如果有）加进来，哪怕表本身分低
                    # 这是一个常见的 Schema Linking 策略：选中列必选中表
                    if node.table_name:
                         # 这里需要查找该表名对应的 Table 节点 ID
                         # 为了避免再次查询，我们在 vector search 时可能没拿到。
                         # 这是一个 trade-off。通常我们会稍后补全 table。
                         pass 

        # 构建 candidate_map 用于返回
        return final_candidates, self._build_map_from_nodes(final_candidates)

    def _parse_neo4j_result_to_nodes(self, result, node_map: Dict[str, NodeItem]):
        """Helper: 解析 Neo4j 结果到 NodeItem，并去重保留最高分"""
        for r in result:
            nid = r["id"]
            score = float(r["score"])
            if nid in node_map:
                if score > node_map[nid].score:
                    node_map[nid].score = score
            else:
                node_map[nid] = NodeItem(
                    id=nid,
                    name=r["name"],
                    labels=r["labels"],
                    table_name=r.get("table_name"),
                    original_name=r.get("original_name"),
                    table_original_name=r.get("table_original_name"),
                    score=score,
                    description=r.get("description", "")
                )

    # -------------------------
    # Phase 2: ToG Traversal (Refactored)
    # -------------------------

    def _tog_traversal(self, query: Dict, initial_nodes: Dict[str, NodeItem]) -> Dict[str, NodeItem]:
        """
        重构后的 ToG 流程：
        Step 1: Structural Expansion - 在图谱上寻找路径连通种子节点，扩展上下文。
        Step 2: Pruning - 利用 LLM 对扩展后的子图进行剪枝，确定最终节点。
        """
        if not initial_nodes:
            return {}

        logger.info(f"ToG Step 1: Structural Expansion on {len(initial_nodes)} seed nodes...")
        expanded_nodes = self._structural_expansion(initial_nodes)
        logger.info(f"ToG Step 1 Done. Expanded to {len(expanded_nodes)} nodes.")
        
        # 打印扩展后的 Map
        expanded_map = self._build_map_from_nodes(expanded_nodes)
        logger.info(f"Expanded Map: {json.dumps(expanded_map, ensure_ascii=False)}")

        logger.info("ToG Step 2: LLM Pruning...")
        final_nodes = self._llm_pruning(query, expanded_nodes)
        logger.info(f"ToG Step 2 Done. Final nodes: {len(final_nodes)}")

        return final_nodes

    def _structural_expansion(self, initial_nodes: Dict[str, NodeItem]) -> Dict[str, NodeItem]:
        """
        Step 1: 拓扑扩展。
        在 Neo4j 中寻找种子节点之间的最短路径，将路径上的节点加入候选集。
        为了性能，主要针对 Table 节点和高分 Column 节点进行路径查找。
        """
        # 1. 确定用于寻路的锚点 (Anchors)
        # 策略：包含所有 Table 节点，以及分数最高的 Top-K Column 节点
        anchors = [n for n in initial_nodes.values() if n.is_table]
              
        anchor_ids = [n.id for n in anchors]

        # 2. Neo4j 路径查询
        # 查找表锚点之间的全部路径 (限制跳数，例如 max 4 hops)
        expanded_pool = initial_nodes.copy()
        
        with self.driver.session() as session:
            # 查询路径上的所有节点
            # 这里的逻辑是：在锚点集合内部寻找两两连通路径
            # 修改：查找表节点之间的全部路径（限制跳数），而不仅仅是最短路径
            res = session.run("""
                MATCH (n), (m)
                WHERE elementId(n) IN $ids AND elementId(m) IN $ids AND elementId(n) < elementId(m)
                MATCH p = (n)-[*..4]-(m)
                UNWIND nodes(p) as node
                WITH DISTINCT node
                OPTIONAL MATCH (t:Table {name: node.table_name})
                RETURN elementId(node) as id, node.name as name, node.original_name as original_name, labels(node) as labels, node.table_name as table_name, t.original_name as table_original_name, node.description as description
            """, ids=anchor_ids)
            
            for r in res:
                nid = r["id"]
                if nid not in expanded_pool:
                    expanded_pool[nid] = NodeItem(
                        id=nid,
                        name=r["name"],
                        labels=r["labels"],
                        table_name=r.get("table_name"),
                        original_name=r.get("original_name"),
                        table_original_name=r.get("table_original_name"),
                        score=0.0, # 路径扩展出来的节点暂时没有分数
                        description=r.get("description", "")
                    )
        
        return expanded_pool

    def _llm_pruning(self, query: Dict, candidate_nodes: Dict[str, NodeItem]) -> Dict[str, NodeItem]:
        """
        Step 2: 节点剪枝。
        构建子图文本描述,让 LLM 选择最终相关的 Table 和 Column。
        """
        candidate_ids = list(candidate_nodes.keys())
        if not candidate_ids:
            return {}

        # 1. 获取子图边信息用于 Prompt
        edges_info = []
        if len(candidate_ids) > 1:
            with self.driver.session() as session:
                res = session.run("""
                    MATCH (n)-[r]-(m)
                    WHERE elementId(n) IN $ids AND elementId(m) IN $ids AND elementId(n) < elementId(m)
                    RETURN elementId(n) as src, elementId(m) as dst, type(r) as type, r.description as description
                """, ids=candidate_ids)
                edges_info = [(r["src"], r["dst"], r["type"], r["description"]) for r in res]

        # 2. 可视化（可选，用于调试）
        G = nx.Graph()
        G.add_nodes_from(candidate_ids)
        G.add_edges_from([(e[0], e[1]) for e in edges_info])
        self._visualize_graph(G, candidate_nodes, "Pruning_Input_Graph", "tog_pruning_input.png")

        # 3. 构建 Prompt
        subgraph_txt = self._format_subgraph(candidate_nodes, edges_info)
        
        # 使用 prompts 模块中的 finalizer prompt (假设其功能是根据子图筛选节点)
        sys_prompt, user_prompt = prompts.get_subgraph_pruning_prompt(
            query,
            subgraph_txt
        )

        # 4. 调用 LLM
        try:
            resp = get_competition_json([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ])
            decision = json.loads(resp)
            
            selected_schema_map = {}
            is_sufficient = True
            missing_info = ""
            
            # 解析 LLM 返回
            if isinstance(decision, dict):
                if "selected_schema" in decision:
                    selected_schema_map = decision["selected_schema"]
                    is_sufficient = decision.get("is_sufficient", True)
                    missing_info = decision.get("missing_info", "")
                else:
                    # 兼容旧格式，尝试将其转换为 map
                    if "table_ids" in decision or "column_ids" in decision or "selected_nodes" in decision:
                        pass # Fall through to old ID parsing logic
                    else:
                        # 假设是 {"t1": ["c1"], ...}
                        selected_schema_map = decision

            # 5. 恢复逻辑 (Recovery)
            if not is_sufficient:
                logger.warning(f"Schema deemed insufficient: {missing_info}. Attempting recovery with full schema...")
                selected_schema_map = self._recover_schema_with_full_context(query, selected_schema_map, missing_info)
                
            # 6. 将 Map 转换为 NodeItem 字典 (Map -> IDs)
            if selected_schema_map:
                final_result = {}
                missing_nodes_map = {} # table -> cols that are not in candidate_nodes

                # 建立查找表
                table_lookup = {n.name: n for n in candidate_nodes.values() if n.is_table}
                col_lookup = {(n.table_name, n.name): n for n in candidate_nodes.values() if n.is_column and n.table_name}

                for t_name, cols in selected_schema_map.items():
                    # Resolve Table
                    if t_name in table_lookup:
                        t_node = table_lookup[t_name]
                        final_result[t_node.id] = t_node
                    else:
                        if t_name not in missing_nodes_map: missing_nodes_map[t_name] = []
                    
                    # Resolve Columns
                    if isinstance(cols, list):
                        for c_name in cols:
                            if (t_name, c_name) in col_lookup:
                                c_node = col_lookup[(t_name, c_name)]
                                final_result[c_node.id] = c_node
                            else:
                                if t_name not in missing_nodes_map: missing_nodes_map[t_name] = []
                                if c_name not in missing_nodes_map[t_name]:
                                    missing_nodes_map[t_name].append(c_name)
                
                # Fetch missing nodes from DB
                if missing_nodes_map:
                    logger.info(f"Fetching missing nodes from DB: {missing_nodes_map.keys()}")
                    fetched_nodes = self._fetch_nodes_by_names(missing_nodes_map)
                    final_result.update(fetched_nodes)
                
                # *** 新增：补充外键列 ***
                final_result = self._ensure_foreign_key_columns(final_result)
                
                # 可视化最终结果
                self._visualize_final_result(final_result, edges_info)
                return final_result

            # --- 旧的 ID 解析逻辑 (Fallback) ---
            # 解析 LLM 返回的 ID
            # 假设返回格式为 {"table_ids": [], "column_ids": []} 或直接是 ID 列表
            selected_ids = set()
            if isinstance(decision, dict):
                # 建立查找表以便将名称映射回 ID
                table_map = {} # name -> id
                col_map = {}   # (table_name, col_name) -> id
                
                for nid, node in candidate_nodes.items():
                    if node.is_table:
                        table_map[node.name] = nid
                    elif node.is_column and node.table_name:
                        col_map[(node.table_name, node.name)] = nid

                # 遍历 LLM 返回的字典
                for t_name, cols in decision.items():
                    # 尝试匹配表
                    if t_name in table_map:
                        selected_ids.add(table_map[t_name])
                    
                    # 尝试匹配列
                    if isinstance(cols, list):
                        for c_name in cols:
                            key = (t_name, c_name)
                            if key in col_map:
                                selected_ids.add(col_map[key])
                
                # 兼容旧格式: {"table_ids": [], "column_ids": []}
                # 如果上述解析没有找到任何东西，但字典里有特定的 keys，则尝试旧逻辑
                if not selected_ids:
                    if "table_ids" in decision or "column_ids" in decision or "selected_nodes" in decision:
                        selected_ids.update(decision.get("table_ids", []))
                        selected_ids.update(decision.get("column_ids", []))
                        selected_ids.update(decision.get("selected_nodes", []))

            elif isinstance(decision, list):
                selected_ids.update(decision)
            
            # 过滤有效 ID
            final_result = {nid: candidate_nodes[nid] for nid in selected_ids if nid in candidate_nodes}
            
            # Fallback: 如果 LLM 返回空或解析失败，返回所有 Table 节点 (保守策略)
            if not final_result and candidate_nodes:
                logger.warning("LLM pruning returned empty set, falling back to all tables in candidates.")
                final_result = {nid: node for nid, node in candidate_nodes.items() if node.is_table}
            
            # 可视化剪枝后的结果
            self._visualize_final_result(final_result, edges_info)

            return final_result

        except Exception as e:
            logger.error(f"LLM Pruning failed: {e}")
            # 出错时返回原候选集
            return candidate_nodes

    def _recover_schema_with_full_context(self, query: Dict, current_selection: Dict, missing_info: str) -> Dict[str, List[str]]:
        """
        当剪枝结果不充分时，使用全量 Schema 进行补全。
        """
        
        schema_str = json.dumps(self.schema, ensure_ascii=False)
        sys_prompt, user_prompt = prompts.get_recover_schema_with_full_context_prompt(query, current_selection, missing_info, schema_str)
    
        try:
            resp = get_competition_json([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ])
            return json.loads(resp)
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return current_selection

    # -------------------------
    # Helpers
    # -------------------------
    
    def _parallel_get_embeddings(self, texts: List[str]) -> Dict[str, List[float]]:
        """并发获取 Embedding"""
        results = {}
        with ThreadPoolExecutor(max_workers=self.emb_threads) as ex:
            future_to_text = {ex.submit(get_competition_embedding, t): t for t in texts}
            for fut in as_completed(future_to_text):
                text = future_to_text[fut]
                try:
                    res = fut.result()
                    if res: results[text] = res
                except Exception as e:
                    logger.warning(f"Embedding failed for {text}: {e}")
        return results

    def _rewrite_query_for_schema_linking(self, query: Dict, schema: dict) -> Dict:
        """Query Rewrite Wrapper"""
        try:
            # p_sys, p_user = prompts.get_query_rewrite_prompt(query.get("question"), query.get("evidence", ""), schema)
            p_sys, p_user = prompts.get_cot_query_rewrite_prompt(query.get("question"), query.get("evidence", ""))
            resp = get_competition_json([{"role": "system", "content": p_sys}, {"role": "user", "content": p_user}])
            return json.loads(resp)
        except Exception as e:
            logger.warning(f"Rewrite failed: {e}")
            return {"rewritten_question": query.get("question"), "keywords": []}

    def _format_subgraph(self, nodes: Dict[str, NodeItem], edges: List[Tuple]) -> str:
        """
        格式化子图信息供 LLM 使用。
        结构：
        1. Nodes: 按表分组，列出表名及其包含的列名。
        2. Relationships: 仅列出表与表之间的关系。
        """
        lines = ["--- Subgraph Nodes ---"]
        
        # 1. Group by Table
        table_dict = {} # table_name -> {'node': NodeItem, 'columns': [NodeItem]}
        
        # Initialize with all tables found in nodes
        for nid, node in nodes.items():
            if node.is_table:
                table_dict[node.name] = {'node': node, 'columns': []}
        
        # Add columns to their tables
        for nid, node in nodes.items():
            if node.is_column:
                t_name = node.table_name
                if t_name:
                    if t_name not in table_dict:
                        # Case: Column selected but Table node not in candidate list
                        # We create a placeholder entry
                        table_dict[t_name] = {'node': None, 'columns': []}
                    table_dict[t_name]['columns'].append(node)

        # Generate Text for Nodes (Sorted by table name)
        for t_name in sorted(table_dict.keys()):
            data = table_dict[t_name]
            t_node = data['node']
            cols = data['columns']
            
            # Table info
            t_info = f"Table name: {t_name} Table description: {t_node.description if t_node else 'N/A'} \n Primary Key: {getattr(t_node, 'primary_key', 'N/A') if t_node else 'N/A'}"
            # if t_node: t_info += f" (ID: {t_node.id})" 
            lines.append(t_info)
            
            # Columns info
            if cols:
                # Sort columns by name
                col_texts = [f"(name: {c.name}, description: {c.description if c.description else 'N/A'})" for c in sorted(cols, key=lambda x: x.name)]
                lines.append(f"Columns: {', '.join(col_texts)}")
            else:
                lines.append(f"Columns: (None)")
        
        lines.append("\n--- Subgraph Relationships (Table-Table) ---")
        
        # 2. Filter and Format Edges
        # Only show edges where both source and target are Tables
        unique_edges = set()
        for src, dst, rtype, rdesc in edges:
            if rtype == 'CONTAINS_COLUMN':
                continue

            src_node = nodes.get(src)
            dst_node = nodes.get(dst)
            
            if src_node and dst_node:
                if src_node.is_table and dst_node.is_table:
                    # Create a string representation
                    desc_part = f"description: {rdesc}" if rdesc else "N/A"
                    edge_str = f"TABLE {src_node.name} - [{rtype}] -> TABLE {dst_node.name}  {desc_part}"
                    unique_edges.add(edge_str)

        for edge_str in sorted(list(unique_edges)):
            lines.append(edge_str)
        
        return "\n".join(lines)

    def _format_neighbors(self, neighbors: List[Dict]) -> str:
        lines = ["--- Neighbors ---"]
        for n in neighbors:
            lines.append(f"ID: {n['neighbor_id']} (Name: {n['name']}, Type: {n['labels'][0]}, Table: {n['table_name']}) via {n['rel_type']}")
        return "\n".join(lines)

    def _build_map_from_nodes(self, nodes: Dict[str, NodeItem]) -> Dict[str, List[str]]:
        """将 NodeItem 字典转换为题目要求的 Table: [Columns] 格式"""
        result = {}
        # 先收集所有涉及的表名
        tables = set()
        columns_map = {} # table_name -> list of col names

        for node in nodes.values():
            if node.is_table:
                t_name = node.original_name if node.original_name else node.name
                tables.add(t_name)
            elif node.is_column:
                t_name = node.table_original_name if node.table_original_name else node.table_name
                if t_name:
                    if t_name not in columns_map: columns_map[t_name] = []
                    c_name = node.original_name if node.original_name else node.name
                    columns_map[t_name].append(c_name)
                    tables.add(t_name) # 确保列所在的表也被包含

        for t in tables:
            result[t] = columns_map.get(t, [])
        return result

    def _visualize_final_result(self, final_result, edges_info):
        """Helper to visualize final graph"""
        final_ids = list(final_result.keys())
        if final_ids:
            G_final = nx.Graph()
            G_final.add_nodes_from(final_ids)
            # 复用 edges_info，只保留两端都在 final_result 中的边
            final_edges = [(e[0], e[1]) for e in edges_info if e[0] in final_result and e[1] in final_result]
            G_final.add_edges_from(final_edges)
            self._visualize_graph(G_final, final_result, "Pruning_Output_Graph", "tog_pruning_output.png")

    def _fetch_nodes_by_names(self, schema_map: Dict[str, List[str]]) -> Dict[str, NodeItem]:
        """根据表名和列名从 Neo4j 获取节点信息"""
        fetched_nodes = {}
        table_names = list(schema_map.keys())
        if not table_names:
            return {}

        with self.driver.session() as session:
            # Fetch Tables
            res = session.run("""
                MATCH (t:Table)
                WHERE t.name IN $names
                RETURN elementId(t) as id, t.name as name, labels(t) as labels, t.original_name as original_name, t.description as description
            """, names=table_names)
            
            for r in res:
                fetched_nodes[r["id"]] = NodeItem(
                    id=r["id"], name=r["name"], labels=r["labels"], 
                    table_name=r["name"], # Table node's table_name is itself
                    original_name=r["original_name"], description=r.get("description", "")
                )

            # Fetch Columns
            col_params = []
            for t_name, cols in schema_map.items():
                for c_name in cols:
                    col_params.append({"t": t_name, "c": c_name})
            
            if col_params:
                # 批量查询列
                res = session.run("""
                    UNWIND $params as p
                    MATCH (c:Column {name: p.c, table_name: p.t})
                    OPTIONAL MATCH (t:Table {name: p.t})
                    RETURN elementId(c) as id, c.name as name, labels(c) as labels, c.table_name as table_name, t.original_name as table_original_name, c.original_name as original_name, c.description as description
                """, params=col_params)
                
                for r in res:
                    fetched_nodes[r["id"]] = NodeItem(
                        id=r["id"], name=r["name"], labels=r["labels"],
                        table_name=r["table_name"], table_original_name=r["table_original_name"],
                        original_name=r["original_name"], description=r.get("description", "")
                    )
                    
        return fetched_nodes

    def _visualize_graph(self, G: nx.Graph, node_map: Dict[str, NodeItem], title: str, filename: str):
        """
        可视化：直接使用 node_map 中的数据，不查库。
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        if G.number_of_nodes() == 0: return

        # 从缓存构建标签和颜色
        labels = {}
        colors = []
        for nid in G.nodes():
            node = node_map.get(nid)
            if not node:
                colors.append('#CCCCCC')
                labels[nid] = str(nid)
                continue
            
            labels[nid] = node.name if len(node.name) < 15 else node.name[:12]+"..."
            if node.is_table:
                colors.append('#FF9999')
            elif node.is_column:
                colors.append('#99FF99')
            else:
                colors.append('#CCCCCC')

        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, k=0.5, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=800, alpha=0.9)
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        plt.title(f"{title}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def _ensure_foreign_key_columns(self, nodes: Dict[str, NodeItem]) -> Dict[str, NodeItem]:
        """
        确保所有选中的表都包含必要的外键列,以支持表间连接。
        
        策略:
        1. 找出所有已选中的表
        2. 查询这些表之间的外键关系
        3. 将缺失的外键列添加到结果中
        """
        # 1. 收集已选中的表名
        selected_tables = set()
        for node in nodes.values():
            if node.is_table:
                selected_tables.add(node.name)
            elif node.is_column and node.table_name:
                selected_tables.add(node.table_name)
        
        if len(selected_tables) < 2:
            return nodes  # 单表或无表,无需处理
        
        logger.info(f"Ensuring foreign key columns for tables: {selected_tables}")
        
        # 2. 查询这些表之间的外键关系
        with self.driver.session() as session:
            res = session.run("""
                MATCH (t1:Table)-[:REFERENCES]->(t2:Table)
                WHERE (t1.name IN $tables OR t1.original_name IN $tables)
                  AND (t2.name IN $tables OR t2.original_name IN $tables)
                MATCH (c1:Column)-[:FK_TO]->(c2:Column)
                WHERE (c1.table_name = t1.name OR c1.table_name = t1.original_name)
                  AND (c2.table_name = t2.name OR c2.table_name = t2.original_name)
                OPTIONAL MATCH (t_parent:Table {name: c1.table_name})
                OPTIONAL MATCH (t_ref:Table {name: c2.table_name})
                RETURN DISTINCT
                    elementId(c1) as fk_col_id,
                    c1.name as fk_col_name,
                    c1.original_name as fk_col_original,
                    c1.table_name as fk_table_name,
                    t_parent.original_name as fk_table_original,
                    c1.description as fk_col_desc,
                    labels(c1) as fk_col_labels,
                    elementId(c2) as ref_col_id,
                    c2.name as ref_col_name,
                    c2.original_name as ref_col_original,
                    c2.table_name as ref_table_name,
                    t_ref.original_name as ref_table_original,
                    c2.description as ref_col_desc,
                    labels(c2) as ref_col_labels
            """, tables=list(selected_tables))
            
            fk_info = list(res)
        
        if not fk_info:
            logger.info("No foreign key relationships found between selected tables")
            return nodes
        
        # 3. 将缺失的外键列添加到结果中
        added_count = 0
        for fk in fk_info:
            # 添加外键列
            fk_col_id = fk["fk_col_id"]
            if fk_col_id not in nodes:
                nodes[fk_col_id] = NodeItem(
                    id=fk_col_id,
                    name=fk["fk_col_name"],
                    labels=fk["fk_col_labels"],
                    table_name=fk["fk_table_name"],
                    original_name=fk["fk_col_original"],
                    table_original_name=fk["fk_table_original"],
                    description=fk.get("fk_col_desc", "")
                )
                added_count += 1
                logger.debug(f"Added FK column: {fk['fk_table_name']}.{fk['fk_col_name']}")
            
            # 添加被引用列
            ref_col_id = fk["ref_col_id"]
            if ref_col_id not in nodes:
                nodes[ref_col_id] = NodeItem(
                    id=ref_col_id,
                    name=fk["ref_col_name"],
                    labels=fk["ref_col_labels"],
                    table_name=fk["ref_table_name"],
                    original_name=fk["ref_col_original"],
                    table_original_name=fk["ref_table_original"],
                    description=fk.get("ref_col_desc", "")
                )
                added_count += 1
                logger.debug(f"Added referenced column: {fk['ref_table_name']}.{fk['ref_col_name']}")
        
        logger.info(f"Added {added_count} foreign key columns to ensure connectivity")
        return nodes