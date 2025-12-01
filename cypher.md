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