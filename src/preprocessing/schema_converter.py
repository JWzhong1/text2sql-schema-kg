import json
import os
import sys
import logging
from typing import Dict, Any, List, Optional

# Add parent directory to path to allow importing 'llm'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import prompts
from src.llm.client import get_competition_json
import sqlite3

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SchemaConverter:
    """
    Schema 转换器：将任意格式的 schema 文件转换为标准化的 JSON 格式，
    并从 SQLite 采样列值进行增强。
    """

    def __init__(self):
        self.conversion_cache = {}  # 简单内存缓存
        # 定义目标格式的示例，用于 LLM 提示
        self.target_format_example = [
            {
                "table_name": "free and reduced-price meals",
                "db_id": "california_schools",
                "original_table_name": "frpm",
                "columns": [
                    {
                        "col": "CDSCode",
                        "type": "text",
                        "original_column_name": ""
                    },
                    {
                        "col": "Academic Year",
                        "type": "text",
                        "original_column_name": ""
                    },
                    {
                        "col": "Enrollment (K-12)",
                        "type": "real",
                        "original_column_name": ""
                    }
                ],
                "primary_keys": ["CDSCode"],
                "foreign_keys": [
                    {
                        "column": "CDSCode",
                        "ref_table": "schools",
                        "ref_column": "CDSCode"
                    }
                ]
            }
        ]

    def _file_to_string(self, file_path: str) -> str:
        """读取文件内容为字符串"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def _convert_by_rule(self, file_content: str, file_path: str) -> List[Dict[str, Any]]:
        """
        使用规则将 BIRD 格式的 schema JSON 转换为目标格式，替代 LLM。
        """
        try:
            data = json.loads(file_content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file {file_path}: {e}")
            raise

        # 简单的 BIRD 格式校验
        if "table_names_original" not in data or "column_names_original" not in data:
             logger.warning(f"File {file_path} might not be in BIRD format. Missing keys.")
        
        db_id = data.get("db_id", "")
        original_tables = data.get("table_names_original", [])
        verbose_tables = data.get("table_names", [])
        original_cols = data.get("column_names_original", [])
        verbose_cols = data.get("column_names", [])
        col_types = data.get("column_types", [])
        pks = data.get("primary_keys", [])
        fks = data.get("foreign_keys", [])

        converted_schema = []
        tables_map = {} # index -> table dict

        # 1. 创建表对象
        for idx, orig_name in enumerate(original_tables):
            # 尝试获取 verbose name，如果没有则用 original
            verbose_name = verbose_tables[idx] if idx < len(verbose_tables) else orig_name
            
            table_def = {
                "table_name": verbose_name,
                "db_id": db_id,
                "original_table_name": orig_name,
                "columns": [],
                "primary_keys": [],
                "foreign_keys": []
            }
            converted_schema.append(table_def)
            tables_map[idx] = table_def

        # 2. 填充列信息
        for col_idx, col_info in enumerate(original_cols):
            # col_info: [table_idx, col_name]
            if not col_info or len(col_info) < 2:
                continue
                
            table_idx, col_name = col_info
            
            # 跳过 -1 (通常是 *)
            if table_idx == -1:
                continue
                
            if table_idx in tables_map:
                # 获取 verbose column name
                col_verbose = col_name
                if col_idx < len(verbose_cols):
                    # verbose_cols[col_idx] 也是 [table_idx, verbose_name]
                    col_verbose = verbose_cols[col_idx][1]
                
                # 获取类型
                c_type = col_types[col_idx] if col_idx < len(col_types) else "text"

                col_def = {
                    "col": col_verbose,
                    "type": c_type,
                    "original_column_name": col_name
                }
                tables_map[table_idx]["columns"].append(col_def)

        # 3. 填充主键
        for pk_col_idx in pks:
            # pk_col_idx 是 original_cols 的索引
            if isinstance(pk_col_idx, int) and pk_col_idx < len(original_cols):
                table_idx, col_name = original_cols[pk_col_idx]
                if table_idx in tables_map:
                    tables_map[table_idx]["primary_keys"].append(col_name)

        # 4. 填充外键
        for fk in fks:
            # fk: [src_col_idx, ref_col_idx]
            if isinstance(fk, list) and len(fk) == 2:
                src_col_idx, ref_col_idx = fk
                
                if src_col_idx < len(original_cols) and ref_col_idx < len(original_cols):
                    src_table_idx, src_col_name = original_cols[src_col_idx]
                    ref_table_idx, ref_col_name = original_cols[ref_col_idx]
                    
                    if src_table_idx in tables_map and ref_table_idx in tables_map:
                        ref_table_orig_name = original_tables[ref_table_idx]
                        
                        fk_def = {
                            "column": src_col_name,
                            "ref_table": ref_table_orig_name,
                            "ref_column": ref_col_name
                        }
                        tables_map[src_table_idx]["foreign_keys"].append(fk_def)

        logger.info(f"Successfully converted schema for {len(converted_schema)} tables from {file_path} using rules.")
        return converted_schema

    @staticmethod
    def _quote_ident(name: str) -> str:
        """安全地为 SQLite 标识符加引号"""
        escaped = name.replace('"', '""')
        return f'"{escaped}"'

    def enhance_with_sqlite_samples(
        self,
        schema_data: List[Dict[str, Any]],
        sqlite_db_path: str,
        sample_size: int = 5
    ) -> List[Dict[str, Any]]:
        """
        基于 SQLite 数据库，为每个列采样非空去重值，写入 sample_values。
        """
        if not os.path.exists(sqlite_db_path):
            raise FileNotFoundError(f"SQLite DB not found: {sqlite_db_path}")

        conn = sqlite3.connect(sqlite_db_path)
        enhanced_schema: List[Dict[str, Any]] = []

        try:
            for table_info in schema_data:
                table_name = table_info.get('original_table_name') or table_info.get('table_name')
                logger.info(f"Sampling values from SQLite for table: {table_name}")

                enhanced_table = dict(table_info)
                columns_info: List[Dict[str, Any]] = []

                qt = self._quote_ident(table_name)

                for col_info in table_info.get('columns', []):
                    column_name = col_info.get('original_column_name') or col_info.get('col')
                    qc = self._quote_ident(column_name)

                    enhanced_col = dict(col_info)
                    try:
                        query = f"""
                            SELECT DISTINCT {qc}
                            FROM {qt}
                            WHERE {qc} IS NOT NULL
                            LIMIT ?
                        """
                        cursor = conn.execute(query, (sample_size,))
                        sample_values = [row[0] for row in cursor.fetchall()]
                        enhanced_col['sample_values'] = sample_values
                    except Exception as e:
                        logger.warning(f"Could not sample values for column {table_name}.{column_name}: {e}")
                        enhanced_col['sample_values'] = []

                    columns_info.append(enhanced_col)

                enhanced_table['columns'] = columns_info

                # 规范外键字段命名
                fks = []
                for fk in enhanced_table.get('foreign_keys', []) or []:
                    if 'referenced_table' in fk and 'referenced_column' in fk:
                        fks.append(fk)
                    else:
                        fks.append({
                            'column': fk.get('column'),
                            'referenced_table': fk.get('ref_table') or fk.get('referenced_table'),
                            'referenced_column': fk.get('ref_column') or fk.get('referenced_column')
                        })
                if fks:
                    enhanced_table['foreign_keys'] = fks

                enhanced_schema.append(enhanced_table)
        finally:
            conn.close()

        return enhanced_schema

    def generate_enhanced_schema(
        self,
        input_file_path: str,
        sqlite_db_path: str,
        output_file_path: str,
        sample_size: int = 5
    ) -> List[Dict[str, Any]]:
        """
        执行转换并增强，只输出最终的增强版 JSON。
        """
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file does not exist: {input_file_path}")

        # 1. LLM 转换
        cache_key = input_file_path
        if cache_key in self.conversion_cache:
            logger.info(f"Cache hit for {input_file_path}.")
            converted_data = self.conversion_cache[cache_key]
        else:
            logger.info(f"Converting schema from file: {input_file_path}")
            file_content = self._file_to_string(input_file_path)
            converted_data = self._convert_by_rule(file_content, input_file_path)
            self.conversion_cache[cache_key] = converted_data

        # 2. SQLite 增强
        enhanced_data = self.enhance_with_sqlite_samples(converted_data, sqlite_db_path, sample_size)

        # 3. 保存结果
        dir_part = os.path.dirname(output_file_path)
        if dir_part:
            os.makedirs(dir_part, exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Enhanced schema saved to: {output_file_path}")

        return enhanced_data


if __name__ == "__main__":
    converter = SchemaConverter()

    input_file = "bird_data/origin_schemas/financial.json"
    sqlite_db_path = "bird_data/bird/llm/data/dev_databases/financial/financial.sqlite"
    output_file = "bird_data/converted_schemas/financial.json"

    try:
        result = converter.generate_enhanced_schema(
            input_file_path=input_file,
            sqlite_db_path=sqlite_db_path,
            output_file_path=output_file,
            sample_size=5
        )
        print(f"Schema conversion and enhancement completed. Output saved to {output_file}.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")