import json
from typing import List, Dict, Set, Tuple, Optional, Any

def get_schema_conversion_prompt(file_type_description: str, target_format_example) -> tuple[str, str]:
    sys_prompt = (
        "You are an expert data schema parser and converter. "
        "Parse the provided schema definition (any text-based format) and convert it into a STRICT JSON array. "
        "Each item represents a table with keys: "
        "table_name (str), display_name (str), description (str), aliases (array of str), business_domain (str), "
        "original_metadata (obj, optional), community (str), "
        "columns (array of { "
        "  column_name (str), data_type (str), description (str, optional), aliases (array of str), "
        "  is_primary_key (bool), not_null (bool), auto_increment (bool), "
        "  sample_values (array of values, optional), "
        "  constraints (obj with primary_key, foreign_key, unique, check info), "
        "  original_attributes (obj, optional) "
        "}), "
        "foreign_keys (array of { "
        "  column (str), referenced_table (str), referenced_column (str), "
        "  constraint_name (str, optional), on_delete (str, optional), on_update (str, optional) "
        "}), "
        "indexes (array of { index_name (str), columns (array of str), is_unique (bool), is_primary (bool) }). "
        "Preserve ALL original info in original_metadata/original_attributes. "
        "If a required value is unknown, use empty string \"\" or empty array []. "
        "Output ONLY the JSON array, with no extra text."
    )
    user_prompt = (
        "Source schema snippet:\n"
        f"{file_type_description}\n\n"
        "Target JSON example (structure only):\n"
        f"{json.dumps(target_format_example, ensure_ascii=False, indent=2)}\n\n"
        "Instructions:\n"
        "- Identify all tables, columns, primary keys, foreign keys, and indexes from the source snippet.\n"
        "- Extract column constraints (PK, FK, NOT NULL, etc.) and data types.\n"
        "- Infer concise table_description and column descriptions when possible.\n"
        "- Include any sample values if available.\n"
        "- Preserve unmapped info in original_metadata/original_attributes.\n"
        "- Output only the JSON array."
    )
    return sys_prompt, user_prompt

def get_table_analysis_prompt(table_name: str, table_data) -> tuple[str, str]:
    sys_prompt = """
    你是一个数据建模专家。请仅返回一个严格的 JSON 对象，包含以下准确的键：: 
    {
      "table_name": "",
      "original_table_name": "",
      "description": ",
      "columns": [],
      "primary_keys": [],
      "foreign_keys": [
        {
          "column": "",
          "referenced_table": "",
          "referenced_column": "",
          "constraint_name": ""
        }
      ]
    }
    说明：
    - table_name：从输入 table_name 原样填充。
    - original_table_name：从输入 original_table_name 原样填充，未知则 ""。
    - description：结合输入中的 original_table_name与各列字段名称简要概述该表的用途。
    - primary_keys：从输入 primary_keys 原样填充，未知则 []。
    - foreign_keys：从输入 foreign_keys 转换键名（ref_table->referenced_table, ref_column->referenced_column），未知则 []。
    - 仅返回上述格式的json文件
    """
    pruned_data = {
        "table_name": table_data.get("table_name"),
        "original_table_name": table_data.get("original_table_name"),
        # 仅提取列名用于语义理解，忽略类型和样本值
        "columns": [
            col.get("col", col.get("name", col.get("column_name"))) 
            if isinstance(col, dict) else col 
            for col in table_data.get("columns", [])
        ],
        "primary_keys": table_data.get("primary_keys", []),
        "foreign_keys": table_data.get("foreign_keys", [])
    }

    user_prompt = (
        f"Input full table object for analysis (use it as the ONLY source of truth):\n"
        f"{json.dumps(pruned_data, ensure_ascii=False, indent=2)}\n\n"
        "Task:\n"
        "- Set table_name,original_table_name from input.\n"
        "- description: prefer input's table_description; otherwise a concise summary.\n"
        "- primary_keys: copy from input primary_keys (or []).\n"
        "- foreign_keys: convert input foreign_keys to keys "
        "  {column, referenced_table, referenced_column, constraint_name?}; "
        "  map ref_table->referenced_table and ref_column->referenced_column.\n"
        "Output strictly the JSON object with EXACT keys."
    )
    return sys_prompt, user_prompt

def get_columns_batch_analysis_prompt(table_name: str, table_desc: str, col_summary: str) -> tuple[str, str]:
    
    
    sys_prompt = """You are a data analyst. For each input column, output EXACTLY one object with the FULL schema details. Return ONLY a strict JSON array (no markdown fences, no extra text). Each item MUST have EXACT keys: "
        期望输出严格 JSON 数组:
    [
      {
        "column_name": "id",
        "data_type": "INT",
        "is_primary_key": true,
        "not_null": true,
        "auto_increment": true,
        "description": "用户唯一标识符",
        "original_column_name": "",
        "sample_values": [1, 2, 3, 4, 5],
        "constraints": {
          "primary_key": true,
          "foreign_key": false,
          "unique": true,
          "check": null
        }
      }
    ]
    Preserve the input column order; do not invent, drop, or rename columns. "
    Use valid JSON (double quotes, no trailing commas). "
    If unknown, use empty string \"\", false, [], or null for check."
    """
    user_prompt = (
        f"Table: {table_name}\n"
        f"Description: {table_desc}\n"
        f"Columns:\n{col_summary}\n\n"
        "Rules:\n"
        "- Keep column_name exactly as listed; preserve order.\n"
        "- Infer data_type from input if present; otherwise \"\".\n"
        "- is_primary_key: true for PK or *_id that is the table identifier; else false.\n"
        "- not_null: true if the column cannot be NULL or is a PK; else false.\n"
        "- auto_increment: true for identity/serial/auto-increment IDs; else false.\n"
        "- description: short human-friendly purpose; empty if unknown.\n"
        "- original_column_name: original column name, empty if unknown.\n"
        "- sample_values: include a few representative values if known; otherwise [].\n"
        "- constraints.primary_key must match is_primary_key.\n"
        "- constraints.foreign_key: true if FK; else false.\n"
        "- constraints.unique: true if unique; else false.\n"
        "- constraints.check: expression string if known; otherwise null.\n"
        "Output ONLY the JSON array."
    )
    return sys_prompt, user_prompt

def get_relationship_analysis_prompt(t1_summary: str, t2_summary: str) -> tuple[str, str]:
    sys_prompt = (
        "You are a data architect. Determine if there is a hidden semantic relationship between two tables "
        "that is NOT an explicit foreign key. "
        "Return ONLY strict JSON with keys 'relationship_type', 'strength' (1-5), "
        "and 'relationship_details' (object with description)."
    )
    user_prompt = (
        f"Table A Summary: {t1_summary}\n"
        f"Table B Summary: {t2_summary}\n\n"
        "Task: Identify semantic connections (e.g., logical grouping, shared business concepts) that are not obvious FKs.\n"
        "Allowed relationship_types: [SEMANTIC_SIMILAR, LOGICAL_GROUP, SHARED_CONTEXT].\n"
        "If no strong semantic relationship exists, return relationship_type: 'NONE'.\n"
        "strength reflects coupling (1-5 scale). Ignore weak links.\n"
        "Output strictly JSON only."
    )
    return sys_prompt, user_prompt

def get_query_rewrite_prompt(nl_query: str, evidence: str, schema: dict) -> tuple[str, str]:
    sys_prompt = (
        "你是一个 Text2SQL 查询重写助手。目标：在不改变语义的前提下，结合evidence和数据库的schema文件，对原始查询进行扩展和重写，并给出重写后查询中包含的关键词列表。"
        "将用户问题标准化、显式化业务指标、实体、时间/条件约束，便于后续 schema linking，严格保留原始语义。\n"
        "在提取关键词时，从高层次对关键词进行抽象，如为具体的值补充抽象后可能对应的字段名，同时保留低层次的具体值，不要重复\n"
        "输出严格 JSON，键：original_question, rewritten_question, keywords (数组)。"
    )
    user_prompt = """
    Natural Language Query: {nl_query}    
    Evidence Context: {evidence}
    Database Schema: {schema} 
    Instructions:
    - Use evidence to clarify ambiguous terms in the query.
    - Ensure the query aligns with the provided schema.
    - Maintain original intent while improving clarity.

    Output Format:
    {{
      "original_question": "...",
      "rewritten_question": "...",
      "keywords": ["...", "..."]
    }}
    """.format(
        nl_query=nl_query,
        evidence=evidence,
        schema=json.dumps(schema, ensure_ascii=False, indent=2)
    )

    return sys_prompt, user_prompt

def get_cot_query_rewrite_prompt(nl_query: str, evidence: str) -> tuple[str, str]:
    sys_prompt = """你是一个 Text2SQL 任务中的**高级逻辑解析与推理专家**。你的核心任务是将用户的自然语言查询（NL）转化为结构化、无歧义的逻辑表达，以便后续模型生成准确的 SQL。
### 核心原则
1. **语义守恒（Semantic Preservation）**：绝对禁止添加用户未提及的过滤条件、时间范围或业务逻辑。你的推理必须严格基于用户输入和提供的 Evidence。
2. **基于证据消歧（Evidence-Based Disambiguation）**：利用提供的 `Evidence`（包含Schema定义、外键关系、数据字典）来解析模糊术语。
   - 例如：用户说“高价值客户”，Evidence 定义为“订单总额 > 1w”，则你需要显式转换逻辑；若无定义，则保留原词。
3. **显性化隐式逻辑（Explicate Implicit Logic）**：挖掘查询中的多跳关系（Multi-hop）和聚合操作（Aggregation）。

### 任务步骤
你需要进行“思维链（Chain of Thought）”推理，并按以下步骤输出：

1. **Analysis & Decomposition (分析与拆解)**：
   - 识别查询涉及的实体（Entities）和属性（Attributes）。
   - 识别隐含的逻辑操作（如：排序、分组、比较、逻辑与/或/非）。
   - 识别多跳路径：如果查询涉及 A 和 C，且 A 与 C 不直接关联，需指出通过 B 进行连接的路径。

2. **Schema Linking & Clarification (模式链接与澄清)**：
   - 根据 Evidence，将自然语言术语映射到可能的数据库概念（表名/列名逻辑，无需精确匹配列名，但需明确意图）。
   - 明确消除歧义的依据。

3. **Structured Rewriting (结构化重写)**：
   - 将查询重写为“中间语言（Intermediate Representation）”。
   - 格式应接近 SQL 的逻辑结构，但保持自然语言的可读性。
   - **关键**：将“最...”转化为“按...排序取第一”；将“...的总和”转化为“Sum(...)”。

### Few-Shot Examples (少样本示例)

**Example 1: 隐式聚合与排序**
**User Input:**
Original Query: "列出上个月销售额最高的产品名称"
Evidence: "订单表包含 order_date 和 amount；产品表包含 product_name；'销售额'指 amount 的总和。"

**Model Output:**
```json
{
  "oringinal_question": "列出上个月销售额最高的产品名称",
  "reasoning_trace": [
    "1. 时间约束识别：'上个月'需要基于当前时间计算日期范围。",
    "2. 语义映射：根据 Evidence，'销售额' = SUM(amount)。",
    "3. 逻辑操作：'最高' imply ORDER BY SUM(amount) DESC LIMIT 1。",
    "4. 关联路径：需按 '产品' 分组计算总销售额。"
  ],
  "rewritten_question": "Find the product_name where the SUM of amount is the maximum, filtered by order_date in (Last Month)",
  "keywords": ["product_name", "SUM(amount)", "MAX", "order_date", "Last Month"]
}

## Example 2: 多跳推理 (Multi-hop Reasoning) 
**User Input:**
 Original Query: "哪个经理管理的员工参与了'阿波罗'项目？" 
 Evidence: "表结构：Employees (id, name, manager_id), Projects (id, name), Project_Assignments (emp_id, proj_id)。"
 **Model Output:**
 ```json
 {
  "reasoning_trace": [
    "1. 目标识别：查询目标是 '经理' 的信息。",
    "2. 路径分析 (多跳)：'阿波罗'项目 -> Project_Assignments (找到员工ID) -> Employees (找到员工及 manager_id) -> Employees (自连接/查找经理信息)。",
    "3. 约束条件：Project.name = '阿波罗'。"
  ],
  "rewritten_question": "Find the name of the manager for employees who are assigned to the project where project_name is '阿波罗'",
  "keywords": ["manager_name", "project_name = '阿波罗'", "JOIN: Projects -> Assignments -> Employees -> Managers"]
}

### 输出格式要求
你必须仅返回一个合法的 JSON 对象，不要包含 markdown 标记（如 json ... ）,使用英文返回内容
格式如下： 
```json
{ 
    "original_question": "用户的原始查询 string",
    "reasoning_trace": ["步骤1...", "步骤2..."], 
    "rewritten_question": "结构化重写后的查询 string", 
    "keywords": ["关键实体", "操作符", "值"] 
} 
"""
    
    user_prompt = f"""请根据以下信息分析并重写查询：
原始查询 (Original Query): {nl_query}
背景知识 (Evidence): {evidence}
注意：
1. 严格遵循 System Prompt 中的 JSON 格式。
2. 不要改变原始查询的意图（Intent）。
3. 如果 Evidence 不足以完全澄清，请在 implicit_assumptions 中注明。 

"""

    return sys_prompt, user_prompt

def get_graph_traversal_prompt(question: str, evidence: str, subgraph_context: str, neighbors_context: str) -> tuple[str, str]:
    sys_prompt = (
        "You are a Schema Graph Traversal Agent. Your task is to navigate a database schema graph to find the minimal set of tables and columns needed to answer a user's question. "
        "An initial subgraph has been identified. You will receive the current subgraph, the original question, and a list of potential neighbors to explore. "
        "Decide which neighbors to add to the subgraph and which existing nodes are relevant or irrelevant. "
        "Provide your decision in a specific JSON format."
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Background Knowledge: {evidence}\n\n"
        f"Current Subgraph:\n{subgraph_context}\n\n"
        f"Potential Neighbors to Explore:\n{neighbors_context}\n\n"
        "Decide which neighbors to add to the subgraph and which nodes in the current subgraph are relevant.\n"
        "Respond with a JSON object containing:\n"
        "- 'add_neighbors': An array of neighbor node IDs to add to the subgraph.\n"
        "- 'relevant_nodes': An array of node IDs from the *current subgraph* that are relevant to the query.\n"
        "- 'stop': A boolean indicating if the traversal should stop."
    )
    return sys_prompt, user_prompt

def get_subgraph_pruning_prompt(query: dict, subgraph_context: str) -> tuple[str, str]:
    sys_prompt = (
        "你是一个数据分析助手，你的任务是根据用户的问题和背景知识，从当前schema子图中选择最相关的表和列，确保查询的可执行性和连接完整性。\n"
        "- 1.保持可连接性，当查询涉及多个表时，保留必要的主键/外键列\n"
        "- 2.确保语义完整性，当查询存在多条路径时，保留全部查询路径可能涉及的字段\n"
        "- 3.严格遵循原始查询和背景知识中的条件、约束和计算公式等，优先级高于schema中字段的说明。\n"
        "- 4.以json格式返回输出结果"
    )

    user_prompt = f"""## 原始查询: {query.get('original_question', '')}
## 背景知识: {query.get('evidence', '')}
## subgraph_nodes: {subgraph_context}

### EXECUTION STEPS:

1. **DECOMPOSE THE QUERY**

2. **PRUNE THE SUBGRAPH**

3. **SUFFICIENCY CHECK**

### OUTPUT FORMAT (STRICT JSON):

{{
  "selected_schema": {{
    "table1": ["col1", "col2"],
    "table2": ["col1"]
  }},
  "is_sufficient": boolean,
  "missing_info": ""
}}
"""

    return sys_prompt, user_prompt



# def get_subgraph_pruning_prompt(query:dict, subgraph_context: str) -> tuple[str, str]:  
#     sys_prompt = (
#         "You are a Schema Graph Finalizer. Your goal is to identify the minimal set of tables and columns "
#         "needed to correctly answer the user's question, while ensuring query executability and join integrity.\n"
#         "Follow these principles:\n"
#         "1. Ensure semantic completeness: keep all columns required for filtering, grouping, calculation, or returning results.\n"
#         "2. Preserve joinability: if multiple tables are involved, keep the necessary primary/foreign key columns.\n"
#         "3. Apply normalization when beneficial: if a descriptive column can reliably be obtained from a related dimension table "
#         "via a stable join, you may prefer using the normalized join path.\n"
#         "4. Never remove information required to correctly interpret or compute the answer.\n"
#     )

#     user_prompt = f"""## original_question: {query.get('original_question', '')}
# ## evidence: {query.get('evidence', '')}
# ## rewritten_question: {query.get('rewritten_question', '')}
# ## reasoning_trace: {query.get('reasoning_trace', [])}
# ## subgraph_nodes: {subgraph_context}

# ### EXECUTION STEPS:

# 1. **DECOMPOSE THE QUERY**
#     - Identify required tables/columns for:
#         a) Filtering or grouping
#         b) Calculations or metrics
#         c) Output fields
#         d) Necessary join keys

# 2. **PRUNE THE SUBGRAPH**
#     - From `subgraph_nodes`, keep all columns identified above.
#     - Remove unrelated columns that do not contribute to answering the query.

# 3. **APPLY OPTIONAL NORMALIZATION**
#     - If a descriptive/denormalized column has an equivalent normalized source and the join is reliable,
#       you may replace it with the normalized table + join key.
#     - If the descriptive column is essential to semantics, keep it.

# 4. **SUFFICIENCY CHECK**
#     - is_sufficient = true only if:
#         (a) All required semantic fields are present
#         (b) All needed join paths are preserved

# ### OUTPUT FORMAT (STRICT JSON):

# {{
#   "selected_schema": {{
#     "table1": ["col1", "col2"],
#     "table2": ["col1"]
#   }},
#   "is_sufficient": boolean,
#   "reasoning": [
#     "Step-by-step explanation of how the query was decomposed, why each table/column was selected, and how normalization was applied or not."
#   ],
#   "missing_info": ""
# }}
# """

#     return sys_prompt, user_prompt

def get_recover_schema_with_full_context_prompt(query: Dict, current_selection: Dict, missing_info: str, schema_str: str) -> tuple[str, str]:
    sys_prompt = (
            "You are a Schema Recovery Expert. The previous schema retrieval was insufficient. "
            "You have access to the FULL database schema. "
            "Your task is to identify the missing tables and columns based on the missing info description and merge them with the currently selected schema."
            "CRITICAL: When adding new tables, you MUST also include the foreign key columns and intermediate tables required to join the new tables with the existing selected schema."
        )
    user_prompt = f"""
        Original Question: {query.get('original_question', '')}
        reasoning_trace: {query.get('reasoning_trace', [])}
        rewritten_question: {query.get('rewritten_question', '')}
        
        Currently Selected Schema (Insufficient):
        {json.dumps(current_selection, ensure_ascii=False, indent=2)}
        
        Missing Information Analysis:
        {missing_info}
        
        FULL Database Schema:
        {schema_str}
        
        Task:
        1. Locate the missing tables/columns in the FULL Schema that address the missing info, attention to the join paths between tables.
        2. Identify any foreign key columns or intermediate tables needed to join these new tables with the Currently Selected Schema.
        3. Merge all found tables and columns into the Selected Schema.
        4. Return the FINAL complete schema map.

        
        Output Format (JSON only):
        {{
            "table_name1": ["col1", "col2"],
            "table_name2": ["col1", "col2"]
        }}
        """
    return sys_prompt, user_prompt

def get_sql_generation_prompt(
    question: str,
    evidence: str,
    schema_context: str,
    reasoning_context: str,
    error_msg: str = None,
    previous_sql: str = None
) -> tuple[str, str]:
    system_prompt = (
        "You are an expert SQL Data Analyst. Your goal is to write a correct SQLite-compatible SQL query.\n"
        "Use only tables and columns from the Retrieved Schema.\n"
        "Map any descriptive names in the reasoning context back to the original schema names.\n"
        "Return a valid JSON object with a single key 'sql'. Do not use markdown."
    )

    user_content = f"""### User Question
{question}

### Evidence / Hint
{evidence}

### Schema Retrieval 
{reasoning_context}

### Retrieved Schema
{schema_context}
"""

    if error_msg and previous_sql:
        # 判断是否是空结果错误
        is_empty_result_error = "returned no results" in error_msg.lower() or "empty" in error_msg.lower()
        
        if is_empty_result_error:
            user_content += f"""
### Previous Attempt Returned Empty Results
SQL: {previous_sql}
Issue: {error_msg}

### Instruction
The previous SQL executed successfully but returned no results. This likely indicates:
1. Incorrect filter conditions (e.g., wrong value matching, case sensitivity issues)
2. Missing or incorrect JOIN conditions
3. Overly restrictive WHERE clauses
4. Column value mismatches (check if you need LIKE instead of = for fuzzy matching)

Please analyze the query and try a different approach. Consider:
- Relaxing filter conditions
- Using LIKE with wildcards for string matching
- Checking for case sensitivity issues (use LOWER() if needed)
- Verifying JOIN conditions are correct
- Removing unnecessary filters to broaden the search
"""
        else:
            user_content += f"""
### Previous Failed Attempt
SQL: {previous_sql}
Error: {error_msg}

### Instruction
The previous SQL failed. Analyze the error and the reasoning context to correct the SQL.
"""
    return system_prompt, user_content

def get_value_exploration_prompt(
    question: str,
    evidence: str,
    keywords: List[str],
    schema_context: str
) -> tuple[str, str]:
    """
    生成值探索 SQL 的 Prompt
    """
    system_prompt = (
        "You are a database exploration expert. Your task is to generate simple exploratory SQL queries "
        "to help understand the data distribution and validate whether keywords from the user's question "
        "correspond to actual values in the database.\n\n"
        "Guidelines:\n"
        "1. Generate 3-5 simple, low-cost SQL queries (e.g., SELECT DISTINCT, COUNT, LIMIT clauses).\n"
        "2. Focus on validating keywords that might be column values (not column/table names).\n"
        "3. Use LIKE patterns for fuzzy matching when appropriate.\n"
        "4. Prioritize queries that help disambiguate the user's intent.\n"
        "5. Each query should have a clear purpose explaining what information it seeks.\n"
        "6. Use the original table/column names from the schema for SQL execution.\n"
        "7. Keep queries simple - avoid complex JOINs or aggregations.\n\n"
        "Output ONLY a valid JSON object with the following structure:\n"
        "{\n"
        '  "exploratory_sql": [\n'
        '    {\n'
        '      "sql": "SELECT DISTINCT column FROM table LIMIT 10;",\n'
        '      "purpose": "Explanation of what this query validates"\n'
        '    }\n'
        "  ]\n"
        "}"
    )
    
    keywords_str = ", ".join(keywords) if keywords else "No specific keywords extracted"
    
    user_prompt = f"""
### User Question
{question}

### Evidence / Hint
{evidence}

### Extracted Keywords
{keywords_str}

### Available Schema (use original_name for SQL)
{schema_context}

### Task
Generate exploratory SQL queries to:
1. Check if any keywords might be actual data values in the database
2. Understand the data distribution in relevant columns
3. Validate entity references mentioned in the question

Return only the JSON object with exploratory_sql array.
"""
    return system_prompt, user_prompt