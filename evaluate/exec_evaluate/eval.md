## evaluate 参数设置
python evaluate\exec_evaluate\evaluation.py --predicted_sql_path evaluate\exec_evaluate\output_dev_2025_qwen_max\ --ground_truth_path evaluate\exec_evaluate\output_dev_2025_qwen_max\ --data_mode "dev" --db_root_path bird_data\bird\llm\data\dev_databases\ --num_cpus 4 --meta_time_out 30.0 --diff_json_path bird_data\bird_sql_dev_dev_20251106.json

## 2025_12_17 qwen3-max结果
![alt text](image.png)

## 2025_12_17 gpt-5结果
![alt text](image-1.png)

## 2025_12_17 qwen3-max 结果（dev_20251106.json）
![alt text](image-2.png)