import sqlite3
import json
import re

def load_sqls_from_json(json_path):
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    gold_sqls = []
    predict_sqls = []
    for item in data:
        gold_sqls.append(item.get('golden_sql', ''))
        predict_sqls.append(item.get('generated_sql', ''))
    return gold_sqls, predict_sqls

def exec_sql(conn, sql):
    try:
        cursor = conn.execute(sql)
        res = cursor.fetchall()
        return res
    except Exception as e:
        return f"ERROR: {e}"

def main():
    db_path = 'bird_data\\bird\\llm\\data\\dev_databases\\california_schools\\california_schools.sqlite'  # 修改为你的sqlite文件路径
    json_path = 'f:\\论文代码复现解析\\text2sql-schema-kg\\evaluate\\exec_evaluate\\output\\test_details_20251216_224453.json'
    output_path = 'f:\\论文代码复现解析\\text2sql-schema-kg\\evaluate\\exec_evaluate\\output\\compare_result.json'

    gold_sqls, predict_sqls = load_sqls_from_json(json_path)

    conn = sqlite3.connect(db_path)
    results = []
    for idx, (gold, pred) in enumerate(zip(gold_sqls, predict_sqls)):
        gold_res = exec_sql(conn, gold)
        pred_res = exec_sql(conn, pred)
        results.append({
            "id": idx,
            "gold_sql": gold,
            "predict_sql": pred,
            "gold_result": gold_res,
            "predict_result": pred_res,
            "is_equal": gold_res == pred_res
        })
    conn.close()

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()