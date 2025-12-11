import json
import sqlite3
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SQLExecutor:
    """SQL 执行器，用于在 SQLite 数据库上执行 SQL 查询"""
    
    def __init__(self, db_root_path: str):
        """
        初始化 SQL 执行器
        
        Args:
            db_root_path: 数据库根目录路径
        """
        self.db_root_path = Path(db_root_path)
        if not self.db_root_path.exists():
            raise FileNotFoundError(f"Database root path not found: {db_root_path}")
    
    def get_db_path(self, db_id: str) -> Path:
        """
        获取数据库文件路径
        
        Args:
            db_id: 数据库 ID
            
        Returns:
            数据库文件的完整路径
        """
        db_path = self.db_root_path / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")
        return db_path
    
    def execute_sql(self, db_id: str, sql: str) -> Tuple[Optional[List[Tuple]], Optional[str]]:
        """
        在指定数据库上执行 SQL 查询
        
        Args:
            db_id: 数据库 ID
            sql: 要执行的 SQL 语句
            
        Returns:
            (查询结果列表, 错误信息)
            - 如果成功，返回 (结果列表, None)
            - 如果失败，返回 (None, 错误信息)
        """
        try:
            db_path = self.get_db_path(db_id)
            
            # 使用只读模式打开数据库
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            
            # 执行 SQL
            cursor.execute(sql)
            results = cursor.fetchall()
            
            # 关闭连接
            conn.close()
            
            return results, None
            
        except sqlite3.Error as e:
            error_msg = f"SQLite Error: {str(e)}"
            logger.error(f"Failed to execute SQL for {db_id}: {error_msg}")
            return None, error_msg
        except FileNotFoundError as e:
            error_msg = str(e)
            logger.error(error_msg)
            return None, error_msg
        except Exception as e:
            error_msg = f"Unexpected Error: {str(e)}"
            logger.error(f"Unexpected error for {db_id}: {error_msg}")
            return None, error_msg


def process_bird_sql_file(
    input_file: str,
    output_file: str,
    db_root_path: str,
    start_from: int = 0,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    处理 BIRD SQL 文件，执行所有 SQL 查询并保存结果
    
    Args:
        input_file: 输入 JSON 文件路径
        output_file: 输出 JSON 文件路径
        db_root_path: 数据库根目录路径
        start_from: 从第几个问题开始处理（用于断点续传）
        limit: 最多处理多少个问题（None 表示处理全部）
        
    Returns:
        统计信息字典
    """
    # 加载输入文件
    logger.info(f"Loading input file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    logger.info(f"Loaded {len(data)} questions")
    
    # 初始化执行器
    executor = SQLExecutor(db_root_path)
    
    # 统计信息
    stats = {
        'total': len(data),
        'processed': 0,
        'success': 0,
        'failed': 0,
        'skipped': 0
    }
    
    # 确定处理范围
    end_index = min(start_from + limit, len(data)) if limit else len(data)
    items_to_process = data[start_from:end_index]
    
    logger.info(f"Processing questions from {start_from} to {end_index}")
    
    # 处理每个问题
    for i, item in enumerate(tqdm(items_to_process, desc="Executing SQL queries")):
        question_id = item.get('question_id')
        db_id = item.get('db_id')
        sql = item.get('SQL')
        
        # 如果已有执行结果，跳过
        if 'execution_result' in item or 'execution_error' in item:
            stats['skipped'] += 1
            logger.debug(f"Skipping question {question_id} (already processed)")
            continue
        
        # 执行 SQL
        logger.info(f"Processing question {question_id} (db: {db_id})")
        results, error = executor.execute_sql(db_id, sql)
        
        # 保存结果
        if error is None:
            # 将结果转换为可序列化的格式
            item['execution_result'] = [list(row) for row in results]
            item['execution_status'] = 'success'
            stats['success'] += 1
            logger.debug(f"Question {question_id} executed successfully, {len(results)} rows returned")
        else:
            item['execution_error'] = error
            item['execution_status'] = 'failed'
            stats['failed'] += 1
            logger.warning(f"Question {question_id} failed: {error}")
        
        stats['processed'] += 1
        
        # 每处理 10 个问题保存一次
        if (i + 1) % 10 == 0:
            save_results(data, output_file)
            logger.info(f"Intermediate save: {stats['processed']} questions processed")
    
    # 最终保存
    save_results(data, output_file)
    logger.info(f"Final save completed: {output_file}")
    
    # 打印统计信息
    print("\n" + "=" * 50)
    print("EXECUTION SUMMARY")
    print("=" * 50)
    print(f"Total questions: {stats['total']}")
    print(f"Processed: {stats['processed']}")
    print(f"Success: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"Skipped: {stats['skipped']}")
    print("=" * 50)
    
    return stats


def save_results(data: List[Dict], output_file: str):
    """
    保存结果到文件
    
    Args:
        data: 数据列表
        output_file: 输出文件路径
    """
    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为 JSONL 格式（每行一个 JSON 对象）
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Execute SQL queries from BIRD dataset and save results"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='bird_data/bird/llm/data/bird_sql_dev_20251106/bird_sql_dev_dev_20251106.jsonl',
        help='Input JSONL file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='scripts/evaluate/exec_evaluate/bird_sql_dev_with_results.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--db_root',
        type=str,
        default='bird_data/bird/llm/data/dev_databases',
        help='Database root directory path'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Start from question index (for resuming)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of questions to process (None = all)'
    )
    
    args = parser.parse_args()
    
    try:
        stats = process_bird_sql_file(
            input_file=args.input,
            output_file=args.output,
            db_root_path=args.db_root,
            start_from=args.start,
            limit=args.limit
        )
        
        # 如果有失败的查询，退出码为 1
        sys.exit(0 if stats['failed'] == 0 else 1)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()