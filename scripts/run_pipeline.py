import os
import json
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

# 添加项目根目录到 sys.path 以便导入 src 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.schema_converter import SchemaConverter
from src.graph.schema_graph_builder import GenericSchemaGraphBuilder

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

def clear_neo4j_database(uri, user, password):
    """清空 Neo4j 数据库中的所有节点和关系"""
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            logger.info("正在清空 Neo4j 数据库...")
            session.run("MATCH (n) DETACH DELETE n")
    except Exception as e:
        logger.error(f"清空数据库失败: {e}")
        raise
    finally:
        driver.close()

def main():
    # 路径定义
    project_root = Path(__file__).parent.parent
    dev_databases_dir = project_root / "bird_data/bird/llm/data/dev_databases"
    dev_tables_path = project_root / "bird_data/bird/llm/data/dev_tables.json"
    origin_schemas_dir = project_root / "bird_data/origin_schemas"
    converted_schemas_dir = project_root / "bird_data/converted_schemas"
    
    # 确保输出目录存在
    origin_schemas_dir.mkdir(parents=True, exist_ok=True)
    converted_schemas_dir.mkdir(parents=True, exist_ok=True)

    # Neo4j 配置
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "your_password")

    # 1. 获取所有 db_name
    if not dev_databases_dir.exists():
        logger.error(f"目录不存在: {dev_databases_dir}")
        return

    # 过滤出目录作为 db_name
    db_names = [d.name for d in dev_databases_dir.iterdir() if d.is_dir()]
    logger.info(f"发现 {len(db_names)} 个数据库: {db_names}")

    # 加载 dev_tables.json
    logger.info(f"加载 {dev_tables_path}...")
    with open(dev_tables_path, "r", encoding="utf-8") as f:
        dev_tables = json.load(f)
    
    # 构建 db_id 到 schema 的映射，方便快速查找
    db_schema_map = {item["db_id"]: item for item in dev_tables}

    # 初始化转换器
    converter = SchemaConverter()

    for db_name in db_names:
        logger.info(f"\n{'='*20} 处理数据库: {db_name} {'='*20}")
        
        # 2. 提取并保存 Origin Schema
        if db_name not in db_schema_map:
            logger.warning(f"在 dev_tables.json 中未找到 {db_name} 的定义，跳过。")
            continue
            
        origin_schema_path = origin_schemas_dir / f"{db_name}.json"
        with open(origin_schema_path, "w", encoding="utf-8") as f:
            # 保存单个 db 的 schema 定义
            json.dump(db_schema_map[db_name], f, indent=4, ensure_ascii=False)
        logger.info(f"已保存 Origin Schema: {origin_schema_path}")

        # 3. 转换 Schema (调用 src/preprocessing/schema_converter.py 的逻辑)
        sqlite_path = dev_databases_dir / db_name / f"{db_name}.sqlite"
        converted_schema_path = converted_schemas_dir / f"{db_name}.json"
        
        if not sqlite_path.exists():
             logger.warning(f"SQLite 文件不存在: {sqlite_path}，跳过转换。")
             continue

        try:
            logger.info(f"正在转换并增强 Schema...")
            converter.generate_enhanced_schema(
                input_file_path=str(origin_schema_path),
                sqlite_db_path=str(sqlite_path),
                output_file_path=str(converted_schema_path)
            )
        except Exception as e:
            logger.error(f"Schema 转换失败: {e}")
            continue

        # 4. 构建 Graph (调用 scripts/build_graph.py 的核心逻辑)
        # 设置缓存目录环境变量
        cache_dir = project_root / "cache" / db_name
        os.environ["SCHEMA_KG_CACHE_DIR"] = str(cache_dir)
        logger.info(f"设置环境变量 SCHEMA_KG_CACHE_DIR={cache_dir}")
        
        # 清空 Neo4j
        try:
            clear_neo4j_database(neo4j_uri, neo4j_user, neo4j_password)
        except Exception:
            logger.error("清空数据库遇到错误，跳过当前 DB 构建")
            continue

        # 构建图谱
        builder = None
        try:
            logger.info(f"开始构建图谱: {db_name}")
            with open(converted_schema_path, "r", encoding="utf-8") as f:
                schema_data = json.load(f)
            
            # 实例化 Builder (会自动读取环境变量中的 cache_dir，也可以显式传入)
            builder = GenericSchemaGraphBuilder(
                neo4j_uri, 
                neo4j_user, 
                neo4j_password, 
                cache_dir=str(cache_dir)
            )
            
            builder.create_indexes()
            builder.build_graph(schema_data)
            logger.info(f"数据库 {db_name} 图谱构建完成。")
            
        except Exception as e:
            logger.error(f"图谱构建失败: {e}")
        finally:
            if builder:
                builder.close()
if __name__ == "__main__":
    main()