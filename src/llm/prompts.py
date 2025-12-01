import json

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

    user_prompt = (
        f"Input full table object for analysis (use it as the ONLY source of truth):\n"
        f"{json.dumps(table_data, ensure_ascii=False, indent=2)}\n\n"
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

def get_subgraph_pruning_prompt(query:dict, subgraph_context: str) -> tuple[str, str]:
    sys_prompt = (
        "You are a Schema Graph Finalizer. Given the final traversed subgraph and the original question, identify the final set of relevant table and column nodes.ALWAYS prioritize preserving join paths between tables over keyword matching.ALWAYS prioritize preserving join paths between tables over keyword matching.ALWAYS prioritize preserving join paths between tables over keyword matching.ALWAYS prioritize preserving join paths between tables over keyword matching."
        "ALWAYS prioritize preserving join paths between tables over keyword matching."
    )
    user_prompt = f"""
    Question: {query.get('question', '')}\n Evidence: {query.get('evidence', '')}\n\n
    Evidence Context: {query.get('evidence_context', '')}\n\n
    Subgraph:\n{subgraph_context}\n\n
    Task:
    - Review the subgraph in the context of the question and evidence.
    - Identify and list the relevant table and column nodes needed to answer the question, .
    - If multiple tables are needed to answer the question, MUST retain at least one complete join path between them
    - respond with a JSON object containing:

    Output Format:
    {{
      "tablel_name1": [column_name1, column_name2, ...],
      "tablel_name2": [column_name1, column_name2, ...],
      ...
    }}
"""
    return sys_prompt, user_prompt