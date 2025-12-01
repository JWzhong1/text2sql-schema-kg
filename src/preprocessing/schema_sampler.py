import json
import sqlite3
from typing import List, Dict, Any
import os

def load_and_enhance_schema(schema_json_path: str, sqlite_file_path: str, sample_size: int = 5) -> List[Dict[str, Any]]:
    """
    从 schema.json 与本地 .sqlite 中采样，使用 original_table_name 作为实际查询表名
    
    Args:
        schema_json_path: 原始 schema.json 文件路径
        sqlite_file_path: SQLite .sqlite 文件路径（文件而非服务）
        sample_size: 每个列采样的值数量，默认为5
    """
    if not os.path.isfile(sqlite_file_path):
        raise FileNotFoundError(f"SQLite 文件不存在: {sqlite_file_path}")
    if not sqlite_file_path.endswith(".sqlite"):
        print(f"Warning: 文件扩展名不是 .sqlite -> {sqlite_file_path}")

    with open(schema_json_path, 'r', encoding='utf-8') as f:
        original_schema = json.load(f)

    # 使用只读 URI 方式，避免意外写入
    conn = sqlite3.connect(f'file:{sqlite_file_path}?mode=ro', uri=True)

    enhanced_schema = []

    for table_info in original_schema:
        logical_table_name = table_info['table_name']
        physical_table_name = table_info.get('original_metadata', {}).get('original_table_name', logical_table_name)
        print(f"Processing table: logical={logical_table_name} physical={physical_table_name}")

        enhanced_table = table_info.copy()
        columns_info = []
        for col_info in table_info['columns']:
            column_name = col_info['col']
            enhanced_col = col_info.copy()
            try:
                quoted_table_name = f'"{physical_table_name}"' if ' ' in physical_table_name else physical_table_name
                quoted_column_name = f'"{column_name}"' if ' ' in column_name else column_name
                # 使用随机采样减少偏序（如果表很大）
                query = f"""
                SELECT DISTINCT {quoted_column_name}
                FROM {quoted_table_name}
                WHERE {quoted_column_name} IS NOT NULL
                ORDER BY RANDOM()
                LIMIT {sample_size}
                """
                cursor = conn.execute(query)
                sample_values = [row[0] for row in cursor.fetchall()]
                enhanced_col['sample_values'] = sample_values
            except Exception as e:
                print(f"Warning: 列采样失败 {physical_table_name}.{column_name}: {e}")
                enhanced_col['sample_values'] = []
            columns_info.append(enhanced_col)

        enhanced_table['columns'] = columns_info

        if 'foreign_keys' in enhanced_table:
            new_foreign_keys = []
            for fk in enhanced_table['foreign_keys']:
                new_fk = {
                    'column': fk['column'],
                    'referenced_table': fk['ref_table'],
                    'referenced_column': fk['ref_column']
                }
                new_foreign_keys.append(new_fk)
            enhanced_table['foreign_keys'] = new_foreign_keys

        enhanced_schema.append(enhanced_table)

    conn.close()
    return enhanced_schema

def convert_to_hippo_format(schema_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hippo_format = []
    for table_info in schema_data:
        hippo_table = {
            "table_name": table_info['table_name'],
            "table_description": table_info.get('table_description', ''),
            "columns": [],
            "foreign_keys": []
        }
        for col_info in table_info['columns']:
            hippo_col = {
                "col": col_info['col'],
                "description": col_info.get('description', ''),
                "type": col_info.get('type', ''),
                "sample_values": col_info.get('sample_values', [])
            }
            hippo_table['columns'].append(hippo_col)
        for fk in table_info.get('foreign_keys', []):
            hippo_fk = {
                "column": fk['column'],
                "referenced_table": fk['referenced_table'],
                "referenced_column": fk['referenced_column']
            }
            hippo_table['foreign_keys'].append(hippo_fk)
        hippo_format.append(hippo_table)
    return hippo_format

def main():
    schema_json_path = "data/converted_schema.json"
    sqlite_file_path = "data/bird/llm/data/dev_databases/california_schools/california_schools.sqlite"  # 使用 .sqlite 文件
    output_path = "data/enhanced_schema.json"

    print("从 .sqlite 文件采样生成增强 schema ...")

    enhanced_schema = load_and_enhance_schema(
        schema_json_path=schema_json_path,
        sqlite_file_path=sqlite_file_path,
        sample_size=5
    )

    hippo_format_schema = convert_to_hippo_format(enhanced_schema)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hippo_format_schema, f, ensure_ascii=False, indent=2)

    print(f"已生成: {output_path}")
    print("\nSchema 统计:")
    print(f"表数量: {len(hippo_format_schema)}")
    for table in hippo_format_schema:
        print(f"表: {table['table_name']}")
        print(f"  列数: {len(table['columns'])}")
        sample_cols_with_values = sum(1 for col in table['columns'] if col.get('sample_values'))
        print(f"  有采样值列: {sample_cols_with_values}")
        print(f"  外键数量: {len(table['foreign_keys'])}")
        print()

if __name__ == "__main__":
    main()