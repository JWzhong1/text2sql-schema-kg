from datasets import load_dataset

ds = load_dataset("birdsql/bird_sql_dev_20251106")

# 保存所有分割
for split_name, split_data in ds.items():
    split_data.to_json(f"./bird_data/bird_sql_dev_{split_name}.json")