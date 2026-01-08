import json
import networkx as nx
from typing import List, Dict, Set, Tuple, Optional, Union
from openai import OpenAI
import re
import sys
import os
import sqlite3
import pandas as pd

# 将项目 src 目录添加到 sys.path 以解决 ModuleNotFoundError
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from llm.client import get_competition

class SchemaGraphSQL:
    def __init__(self,config_mode: str = "force-union"):
        """
        初始化 SchemaGraphSQL
        
        """
        self.config_mode = config_mode
        
    def load_schema_from_json(self, schema_json_path: str) -> List[Dict]:
        """
        从 JSON 文件加载数据库 schema
        
        参数:
        schema_json_path: schema JSON 文件路径
        
        返回:
        包含表定义的列表
        """
        with open(schema_json_path, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)
        return schema_data
    
    def build_schema_graph(self, schema: List[Dict]) -> nx.Graph:
        """
        基于外键关系构建 schema 图
        
        参数:
        schema: 数据库 schema 列表
        
        返回:
        表示数据库 schema 的图
        """
        G = nx.Graph()
        
        # 添加表作为节点
        for table in schema:
            G.add_node(table['original_table_name'])
        
        # 添加外键关系作为边
        for table in schema:
            src_table = table['original_table_name']
            for fk in table.get('foreign_keys', []):
                dst_table = fk['referenced_table']
                if src_table in G and dst_table in G:
                    G.add_edge(src_table, dst_table)
        
        # 对于稀疏 schema，添加共享 'id' 列名的表之间的边
        if len(G.edges()) < 2:  # 如果边太少
            tables_with_id = {}
            for table in schema:
                table_name = table['original_table_name']
                for column in table['columns']:
                    if 'id' in column['original_column_name'].lower():
                        col_name = column['original_column_name']
                        if col_name not in tables_with_id:
                            tables_with_id[col_name] = []
                        tables_with_id[col_name].append(table_name)
            
            # 为共享相同 'id' 列名的表添加边
            for col_name, tables in tables_with_id.items():
                if len(tables) > 1:
                    for i in range(len(tables)):
                        for j in range(i+1, len(tables)):
                            if tables[i] in G and tables[j] in G:
                                G.add_edge(tables[i], tables[j])
        
        return G
    
    def extract_source_destination_tables(self, question: str, schema: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        
        参数:
        question: 自然语言查询问题
        schema: 数据库 schema
        
        返回:
        (源表列表, 目标表列表)
        """
        # 构建系统提示
        prompt = self._build_extraction_prompt(question, schema)
        
        # 调用 Gemini API
        response = get_competition(
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        

        src_tables, dst_tables = self._parse_extraction_response(response)
        return src_tables, dst_tables
    
    def _build_extraction_prompt(self, question: str, schema: List[Dict]) -> str:
        """构建用于表提取的提示"""
        # 构建 schema 描述
        schema_description = "Database Schema:\n"
        for table in schema:
            schema_description += f"Table: {table['original_table_name']} (Description: {table['table_name']})\n"
            schema_description += "Columns:\n"
            for col in table['columns']:
                schema_description += f"  - {col['original_column_name']} ({col['type']}): {col.get('col', '')}"
                schema_description += "\n"
            schema_description += "\n"
        
        # 构建完整的提示
        prompt = f"""
ROLE & OBJECTIVE
You are a senior data engineer who analyses SQL schemas and maps user questions precisely to source tables (filtering) and destination tables (final result columns).

TASK
Identify:
• Source table(s) (src): contain columns used in filters/conditions.
• Destination table(s) (dst): contain columns returned in the answer.

INSTRUCTIONS
1. Internally inspect every table to determine
   • which tables participate in filtering, and
   • which tables supply the requested output columns.
   Briefly justify your choice internally but do not include that justification in the final answer.
2. Output exactly one line in the following format:
   src=TableA,TableB,dst=TableC,TableD

SCHEMA:
{schema_description}

QUESTION:
{question}
"""
        return prompt
    
    def _parse_extraction_response(self, response: str) -> Tuple[List[str], List[str]]:
        """解析表提取的响应"""
        # 使用正则表达式提取 src 和 dst
        src_match = re.search(r'src\s*=\s*([^,]+(?:,[^,]+)*)', response, re.IGNORECASE)
        dst_match = re.search(r'dst\s*=\s*([^,]+(?:,[^,]+)*)', response, re.IGNORECASE)
        
        src_tables = []
        dst_tables = []
        
        if src_match:
            src_str = src_match.group(1).strip()
            src_tables = [table.strip() for table in src_str.split(',') if table.strip()]
        
        if dst_match:
            dst_str = dst_match.group(1).strip()
            dst_tables = [table.strip() for table in dst_str.split(',') if table.strip()]
        
        return src_tables, dst_tables
    
    def find_candidate_paths(self, G: nx.Graph, src_tables: List[str], dst_tables: List[str]) -> List[List[str]]:
        """
        在 schema 图中找到连接源表和目标表的候选路径
        
        参数:
        G: schema 图
        src_tables: 源表列表
        dst_tables: 目标表列表
        
        返回:
        候选路径列表，每条路径是表名列表
        """
        candidate_paths = []
        
        # 为每对源表-目标表找到最短路径
        for src in src_tables:
            for dst in dst_tables:
                if src in G and dst in G and nx.has_path(G, src, dst):
                    # 获取所有最短路径
                    shortest_paths = list(nx.all_shortest_paths(G, source=src, target=dst))
                    candidate_paths.extend(shortest_paths)
        
        # 去重
        unique_paths = []
        seen = set()
        for path in candidate_paths:
            path_tuple = tuple(path)
            if path_tuple not in seen:
                seen.add(path_tuple)
                unique_paths.append(path)
        
        return unique_paths
    
    def select_relevant_tables(self, candidate_paths: List[List[str]], 
                             src_tables: List[str], dst_tables: List[str]) -> Set[str]:
        """
        根据配置模式选择相关表
        
        参数:
        candidate_paths: 候选路径列表
        src_tables: 源表列表
        dst_tables: 目标表列表
        
        返回:
        选出的相关表集合
        """
        if not candidate_paths:
            # 如果没有找到路径，返回源表和目标表
            return set(src_tables + dst_tables)
        
        if self.config_mode == "force-union":
            # 取所有路径的并集
            relevant_tables = set()
            for path in candidate_paths:
                relevant_tables.update(path)
            return relevant_tables
        
        elif self.config_mode == "n-n":
            # 保留所有最短路径
            relevant_tables = set()
            for path in candidate_paths:
                relevant_tables.update(path)
            return relevant_tables
            
        elif self.config_mode == "1-1" and candidate_paths:
            # 选择第一条路径
            return set(candidate_paths[0])
            
        elif self.config_mode == "force-longest" and candidate_paths:
            # 选择最长的路径
            longest_path = max(candidate_paths, key=len)
            return set(longest_path)
            
        elif self.config_mode == "no-union" and candidate_paths:
            # 选择最短的路径
            shortest_path = min(candidate_paths, key=len)
            return set(shortest_path)
            
        else:
            # 默认：取所有路径的并集
            relevant_tables = set()
            for path in candidate_paths:
                relevant_tables.update(path)
            return relevant_tables
    
    def filter_schema(self, schema: List[Dict], relevant_tables: Set[str]) -> List[Dict]:
        """
        过滤 schema，只保留相关表
        
        参数:
        schema: 完整的数据库 schema
        relevant_tables: 相关表集合
        
        返回:
        过滤后的 schema
        """
        filtered_schema = []
        
        # 过滤表
        for table in schema:
            if table['original_table_name'] in relevant_tables:
                filtered_schema.append(table)
        
        return filtered_schema
    
    def generate_sql(self, question: str, filtered_schema: List[Dict]) -> str:
        """
        使用过滤后的 schema 生成 SQL 查询
        
        参数:
        question: 自然语言查询问题
        filtered_schema: 过滤后的 schema
        
        返回:
        生成的 SQL 查询
        """
        # 构建 SQL 生成提示
        prompt = self._build_sql_generation_prompt(question, filtered_schema)
        
        response = get_competition(
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # 提取 SQL 查询
        # 注意: get_competition 直接返回内容字符串，无需访问 .text
        sql_query = self._extract_sql_from_response(response)
        return sql_query
    
    def _build_sql_generation_prompt(self, question: str, filtered_schema: List[Dict]) -> str:
        """构建 SQL 生成提示"""
        # 构建 schema 描述
        schema_description = "Filtered Database Schema:\n"
        for table in filtered_schema:
            schema_description += f"Table: {table['original_table_name']}\n"
            schema_description += "Columns:\n"
            for col in table['columns']:
                schema_description += f"  - {col['original_column_name']} ({col['type']})"
                if col.get('col') and col['col'] != col['original_column_name']:
                    schema_description += f": {col['col']}"
                schema_description += "\n"
            schema_description += "\n"
        
        # 添加外键信息
        fks_lines = []
        possible_tables = set(t['original_table_name'] for t in filtered_schema)
        
        for table in filtered_schema:
            src_table = table['original_table_name']
            for fk in table.get('foreign_keys', []):
                 ref_table = fk['referenced_table']
                 if ref_table in possible_tables:
                     fks_lines.append(f"  - {src_table}.{fk['column']} -> {ref_table}.{fk['referenced_column']}")

        if fks_lines:
            schema_description += "Foreign Keys:\n" + "\n".join(fks_lines) + "\n"
        
        # 构建完整的提示
        prompt = f"""
ROLE & OBJECTIVE
You are an expert in SQL query generation. Your task is to generate a valid query to answer a user question based on the given schema.

INPUTS
• Schema: 
{schema_description}
• Question: {question}

INSTRUCTIONS
1. Use the provided schema to construct a valid SQL query.
2. Ensure the query correctly answers the user's question.
3. Format the query clearly and confirm it adheres to SQL syntax.
4. Only use tables and columns from the provided schema.
5. Use appropriate JOINs based on the foreign key relationships.

Output only the SQL query, without any additional explanation or formatting.
"""
        return prompt
    
    def _extract_sql_from_response(self, response: str) -> str:
        """从响应中提取 SQL 查询"""
        # 移除代码块标记
        sql = re.sub(r'^```sql\n?', '', response, flags=re.MULTILINE)
        sql = re.sub(r'\n?```$', '', sql, flags=re.MULTILINE)
        
        # 移除任何前导/尾随空格
        sql = sql.strip()
        
        return sql
    
    def execute_sql(self, db_path: str, sql_query: str) -> pd.DataFrame:
        """
        使用 SQLite 执行 SQL 查询并返回结果
        
        参数:
        db_path: SQLite 数据库文件路径
        sql_query: 要执行的 SQL 查询
        
        返回:
        查询结果的 DataFrame
        """
        try:
            # 连接到 SQLite 数据库
            conn = sqlite3.connect(db_path)
            
            # 执行 SQL 查询并返回结果
            df = pd.read_sql_query(sql_query, conn)
            
            # 关闭连接
            conn.close()
            
            return df
        except Exception as e:
            print(f"SQL 执行错误: {str(e)}")
            print(f"执行的 SQL: {sql_query}")
            return pd.DataFrame()  # 返回空的 DataFrame
    
    def run(self, schema_json_path: str, question: str, db_path: Optional[str] = None) -> Dict:
        """
        运行完整的 SchemaGraphSQL 流程
        
        参数:
        schema_json_path: schema JSON 文件路径
        question: 自然语言查询问题
        db_path: SQLite 数据库文件路径 (可选)
        
        返回:
        包含结果的字典，包括过滤后的 schema、生成的 SQL 和执行结果
        """
        # 1. 加载 schema
        schema = self.load_schema_from_json(schema_json_path)
        
        # 2. 构建 schema 图
        schema_graph = self.build_schema_graph(schema)
        
        # 3. 提取源表和目标表
        src_tables, dst_tables = self.extract_source_destination_tables(question, schema)
        
        # 4. 找到候选路径
        candidate_paths = self.find_candidate_paths(schema_graph, src_tables, dst_tables)
        
        # 5. 选择相关表
        relevant_tables = self.select_relevant_tables(candidate_paths, src_tables, dst_tables)
        
        # 6. 过滤 schema
        filtered_schema = self.filter_schema(schema, relevant_tables)
        
        # 7. 生成 SQL
        sql_query = self.generate_sql(question, filtered_schema)
        
        # 8. 执行 SQL 并获取结果 (如果提供了数据库路径)
        execution_result = None
        if db_path:
            execution_result = self.execute_sql(db_path, sql_query)
        
        # 9. 返回结果
        result = {
            "original_schema_path": schema_json_path,
            "question": question,
            "source_tables": src_tables,
            "destination_tables": dst_tables,
            "candidate_paths": candidate_paths,
            "relevant_tables": list(relevant_tables),
            "filtered_schema": filtered_schema,
            "generated_sql": sql_query
        }
        
        if execution_result is not None:
            result["execution_result"] = execution_result
            result["execution_success"] = not execution_result.empty
        
        return result

# 使用示例
if __name__ == "__main__":
    # 初始化 SchemaGraphSQL
    schemagraphsql = SchemaGraphSQL(
        config_mode="force-union"  # 使用论文中效果最好的配置
    )
    question = "For the school with the highest free meal rate in Alameda County, what are its characteristics including whether it's a charter school, what grades it serves, its SAT performance level, and how much its free meal rate deviates from the county average? evidence: Free meal rate = Free Meal Count (K-12) / Enrollment (K-12). SAT performance levels are categorized as: Below Average (total score < 1200), Average (1200-1500), Above Average (> 1500), or No SAT Data if unavailable."
        
    # 运行示例 (需要提供对应的 SQLite 数据库文件路径)
    result = schemagraphsql.run(
        schema_json_path="bird_data\\converted_schemas\\california_schools.json",
        question=question,
        db_path="bird_data\\bird\\llm\\data\\dev_databases\\california_schools\\california_schools.sqlite"  # 替换为实际的数据库路径
    )
    
    print("Source Tables:", result["source_tables"])
    print("Destination Tables:", result["destination_tables"])
    print("Relevant Tables:", result["relevant_tables"])
    print("\nGenerated SQL:")
    print(result["generated_sql"])
    
    # 打印执行结果
    if "execution_result" in result:
        print("\nExecution Result:")
        if result["execution_success"]:
            print(result["execution_result"])
        else:
            print("SQL 执行失败或没有返回结果")
    else:
        print("\nNo database path provided, skipping execution.")