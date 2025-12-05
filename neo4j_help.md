# 查看表节点之间的关系
MATCH (t1:Table)-[r]->(t2:Table)
RETURN t1.name AS from_table,
       type(r) AS rel_type,
       t2.name AS to_table,
       r.join_conditions AS join_conditions,
       r.description AS description,
       r.strength AS strength,
       r.sql_generation_hints AS sql_hints
ORDER BY from_table, to_table;

# 查看所有表节点和边
MATCH (n)-[r]->(m)
RETURN n, r, m;


# 删除全部表节点和边
MATCH (n)-[r]->(m)
DELETE n, r, m;


# neo4j docekr 容器启动指令
docker run -d \
  --name neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -v /home/zjw/neo4j/data:/data \
  -v /home/zjw/neo4j/logs:/logs \
  -v /home/zjw/neo4j/conf:/conf \
  -v /home/zjw/project/Text2SQL/schema_kg/schema_graph_dump:/export \
  -e NEO4J_AUTH=neo4j/Zhong_123456 \
  neo4j:5


## 导入dump文件
docker exec -it neo4j bash

neo4j-admin database load --from-path=/export --overwrite-destination=true financial

eixt

docker restart neo4j

## 导出dump文件
docker stop neo4j

docker run --rm \
  -v /home/zjw/neo4j/data:/data \
  -v /home/zjw/project/Text2SQL/schema_kg/schema_graph_dump:/export \
  neo4j:5 \
  neo4j-admin database dump <db_name> --to-path=/export

docker start neo4j  