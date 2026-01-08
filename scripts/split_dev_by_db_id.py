import json
from pathlib import Path

def split_dev_by_db_id(input_file, output_dir):
    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load the dev.json file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Group data by db_id
    grouped_data = {}
    for item in data:
        db_id = item.get('db_id')
        if db_id not in grouped_data:
            grouped_data[db_id] = []
        grouped_data[db_id].append(item)

    # Save each group to a separate file
    for db_id, items in grouped_data.items():
        output_file = Path(output_dir) / f"{db_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_file = "f:/论文代码复现解析/text2sql-schema-kg/bird_data/bird/llm/data/dev.json"
    output_dir = "f:/论文代码复现解析/text2sql-schema-kg/bird_data/dev"
    split_dev_by_db_id(input_file, output_dir)