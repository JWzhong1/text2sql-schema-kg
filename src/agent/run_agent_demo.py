import os
import sys
import logging
import dotenv
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.text2sql_agent import Text2SQLAgent

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # 加载环境变量
    dotenv.load_dotenv()
    
    # 配置 Neo4j 连接
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "your_password")
    neo4j_config = (neo4j_uri, neo4j_user, neo4j_password)

    # 设定测试参数
    # 这里以 california_schools 为例，你可以修改为你 workspace 中存在的其他 DB
    db_name = "codebase_community" 
    
    # 构造 SQLite 路径 (根据你的 workspace 结构)
    project_root = Path(__file__).parent.parent
    db_path = f"bird_data/bird/llm/data/dev_databases/{db_name}/{db_name}.sqlite"
    
    if not Path(db_path).exists():
        logging.error(f"Database file not found at {db_path}")
        return

    # 初始化 Agent
    agent = Text2SQLAgent(db_name, str(db_path), neo4j_config)

    try:
        # 测试问题
        question = "How many users from New York have a teacher and supporter badge?"
        evidence = "\"Supporter\" and \"Teachers\" are both Name of badge; 'New York' is the Location; user refers to UserId"
        
        print(f"\n{'='*50}")
        print(f"Processing Question: {question}")
        print(f"{'='*50}\n")

        result = agent.solve(question, evidence)

        print(f"\n{'='*50}")
        print("FINAL RESULT")
        print(f"{'='*50}")
        
        if result["status"] == "success":
            print(f"Generated SQL: \n{result['sql']}\n")
            print(f"Execution Result:")
            for row in result["result"]:
                print(row)
        else:
            print(f"Failed to generate correct SQL.")
            print(f"Last SQL: {result.get('sql')}")
            print(f"Error: {result.get('error')}")

    finally:
        agent.close()

if __name__ == "__main__":
    main()