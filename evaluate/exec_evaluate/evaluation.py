import sys
import json
import argparse
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut

# --- 全局变量 ---
exec_result = []  # 存储所有执行结果
incorrect_executions = None  # 将在主函数中初始化为共享列表
# -----------------

def load_json(dir):
    contents = []
    with open(dir, 'r', encoding='utf-8') as j:
        contents = json.load(j)
    return contents

def load_json(dir):
    contents = []
    with open(dir, 'r', encoding='utf-8') as j:
        for line in j:
            line = line.strip()
            if line:
                contents.append(json.loads(line))
    return contents

def result_callback(result):
    exec_result.append(result)

def execute_sql(predicted_sql, ground_truth, db_path):
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    res = 0
    # print(predicted_res)
    # print("-----")
    # print(ground_truth_res)
    # print("=====")
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    # else:
    #     print(f"Predicted Result: {predicted_res}")
    #     print(f"Ground Truth Result: {ground_truth_res}")
    #     print("=====")
    conn.close() # 建议关闭连接
    # 返回结果和查询结果，以便 execute_model 能够记录错误信息
    return res, predicted_res, ground_truth_res

def execute_model(predicted_sql, ground_truth, db_place, idx, meta_time_out, incorrect_list):
    try:
        res, predicted_res, ground_truth_res = func_timeout(meta_time_out, execute_sql,
                                  args=(predicted_sql, ground_truth, db_place))
        # 如果结果不匹配，记录错误信息
        if res == 0:
            incorrect_list.append({
                'sql_idx': idx,
                'predicted_sql': predicted_sql,
                'ground_truth_sql': ground_truth,
                'predicted_result': str(predicted_res),
                'ground_truth_result': str(ground_truth_res),
                'database_path': db_place,
                'error_type': 'result_mismatch'
            })
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        res = 0
        # --- 处理超时错误 ---
        incorrect_list.append({
            'sql_idx': idx,
            'predicted_sql': predicted_sql,
            'ground_truth_sql': ground_truth,
            'predicted_result': 'TIMEOUT',
            'ground_truth_result': 'N/A (due to timeout)',
            'database_path': db_place,
            'error_type': 'timeout'
        })
        # -----------------------
    except Exception as e:
        res = 0
        # --- 处理其他异常 ---
        incorrect_list.append({
            'sql_idx': idx,
            'predicted_sql': predicted_sql,
            'ground_truth_sql': ground_truth,
            'predicted_result': f'ERROR: {str(e)}',
            'ground_truth_result': 'N/A (due to error)',
            'database_path': db_place,
            'error_type': 'execution_error'
        })
        # -----------------------
    # print(result)
    # result = str(set([ret[0] for ret in result]))
    result = {'sql_idx': idx, 'res': res}
    # print(result)
    return result


def package_sqls(sql_path, db_root_path, mode='gpt', data_mode='dev'):
    clean_sqls = []
    db_path_list = []
    if mode == 'gpt':
        sql_data = json.load(open(sql_path + 'predict_' + data_mode + '.json', 'r'))
        for idx, sql_str in sql_data.items():
            if type(sql_str) == str:
                sql, db_name = sql_str.split('\t----- bird -----\t')
            else:
                sql, db_name = " ", "financial"
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    elif mode == 'gt':
        sqls = open(sql_path + data_mode + '_gold.sql')
        sql_txt = sqls.readlines()
        # sql_txt = [sql.split('\t')[0] for sql in sql_txt]
        for idx, sql_str in enumerate(sql_txt):
            sql, db_name = sql_str.strip().split('\t')
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    return clean_sqls, db_path_list

def run_sqls_parallel(sqls, db_places, incorrect_list, num_cpus=1, meta_time_out=30.0):
    pool = mp.Pool(processes=num_cpus)
    for i, sql_pair in enumerate(sqls):
        predicted_sql, ground_truth = sql_pair
        pool.apply_async(execute_model, args=(predicted_sql, ground_truth, db_places[i], i, meta_time_out, incorrect_list), callback=result_callback)
    pool.close()
    pool.join()

def sort_results(list_of_dicts):
  return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

def compute_acc_by_diff(exec_results, diff_json_path):
    num_queries = len(exec_results)
    results = [res['res'] for res in exec_results]
    contents = load_json(diff_json_path)
    simple_results, moderate_results, challenging_results = [], [], []

    for i in range(num_queries):
        content = contents[i]
        if content['difficulty'] == 'simple':
            simple_results.append(exec_results[i])
        if content['difficulty'] == 'moderate':
            moderate_results.append(exec_results[i])
        if content['difficulty'] == 'challenging':
            challenging_results.append(exec_results[i])

    simple_acc = sum([res['res'] for res in simple_results]) / len(simple_results) if simple_results else 0
    moderate_acc = sum([res['res'] for res in moderate_results]) / len(moderate_results) if moderate_results else 0
    challenging_acc = sum([res['res'] for res in challenging_results]) / len(challenging_results) if challenging_results else 0
    all_acc = sum(results) / num_queries if num_queries else 0
    count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
    return simple_acc * 100, moderate_acc * 100, challenging_acc * 100, all_acc * 100, count_lists



def print_data(score_lists,count_lists):
    levels = ['simple', 'moderate', 'challenging', 'total']
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))

    print('======================================    ACCURACY    =====================================')
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('accuracy', *score_lists))

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--predicted_sql_path', type=str, required=True, default='')
    args_parser.add_argument('--ground_truth_path', type=str, required=True, default='')
    args_parser.add_argument('--data_mode', type=str, required=True, default='dev')
    args_parser.add_argument('--db_root_path', type=str, required=True, default='')
    args_parser.add_argument('--num_cpus', type=int, default=1)
    args_parser.add_argument('--meta_time_out', type=float, default=30.0)
    args_parser.add_argument('--mode_gt', type=str, default='gt')
    args_parser.add_argument('--mode_predict', type=str, default='gpt')
    args_parser.add_argument('--difficulty',type=str,default='simple')
    args_parser.add_argument('--diff_json_path',type=str,default='')
    args_parser.add_argument('--output_incorrect_file', type=str, default='incorrect_executions.json', help='Path to save incorrect execution details.')
    args = args_parser.parse_args()

    # Note: exec_result is already a global variable
    # exec_result = [] # No need to re-initialize
    
    # 创建多进程共享的列表
    manager = mp.Manager()
    incorrect_executions = manager.list()

    pred_queries, db_paths = package_sqls(args.predicted_sql_path, args.db_root_path, mode=args.mode_predict,
                                          data_mode=args.data_mode)
    # generate gt sqls:
    gt_queries, db_paths_gt = package_sqls(args.ground_truth_path, args.db_root_path, mode='gt',
                                           data_mode=args.data_mode)

    query_pairs = list(zip(pred_queries, gt_queries))
    run_sqls_parallel(query_pairs, db_places=db_paths, incorrect_list=incorrect_executions, num_cpus=args.num_cpus, meta_time_out=args.meta_time_out)
    exec_result = sort_results(exec_result)
    
    print('start calculate')
    simple_acc, moderate_acc, challenging_acc, acc, count_lists = \
        compute_acc_by_diff(exec_result, args.diff_json_path)
    score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
    print_data(score_lists, count_lists)
    print('===========================================================================================')
    print("Finished evaluation")

    # --- 新增：保存错误执行结果 ---
    # 将共享列表转换为普通列表
    incorrect_executions_list = list(incorrect_executions)
    
    # 加载原始数据以获取 question 信息
    original_data = load_json(args.diff_json_path)
    
    # 为每个错误执行结果添加 question 信息
    enriched_incorrect_executions = []
    for item in incorrect_executions_list:
        sql_idx = item.get('sql_idx', -1)
        if sql_idx >= 0 and sql_idx < len(original_data):
            original_item = original_data[sql_idx]
            enriched_item = {
                'sql_idx': sql_idx,
                'question_id': original_item.get('question_id', sql_idx),
                'db_id': original_item.get('db_id', ''),
                'question': original_item.get('question', ''),
                'evidence': original_item.get('evidence', ''),
                'difficulty': original_item.get('difficulty', ''),
                'golden_sql': item.get('ground_truth_sql', ''),
                'predicted_sql': item.get('predicted_sql', ''),
                'golden_result': item.get('ground_truth_result', ''),
                'predicted_result': item.get('predicted_result', ''),
                'database_path': item.get('database_path', ''),
                'error_type': item.get('error_type', 'result_mismatch')
            }
            enriched_incorrect_executions.append(enriched_item)
        else:
            # 如果找不到对应的原始数据，仍然保存基本信息
            item['error_type'] = item.get('error_type', 'result_mismatch')
            enriched_incorrect_executions.append(item)
    
    print(f"\nSaving {len(enriched_incorrect_executions)} incorrect execution details to '{args.output_incorrect_file}'...")
    with open(args.output_incorrect_file, 'w', encoding='utf-8') as f:
        json.dump(enriched_incorrect_executions, f, indent=2, ensure_ascii=False)
    print(f"Saved.")

    # --- 可选：打印一些统计信息 ---
    timeout_count = sum(1 for item in incorrect_executions_list if item.get('error_type') == 'timeout')
    error_count = sum(1 for item in incorrect_executions_list if item.get('error_type') == 'execution_error')
    mismatch_count = len(incorrect_executions_list) - timeout_count - error_count
    print(f"\nIncorrect Execution Breakdown:")
    print(f"  - Result Mismatches: {mismatch_count}")
    print(f"  - Timeouts: {timeout_count}")
    print(f"  - Execution Errors: {error_count}")
    # ---------------------------------