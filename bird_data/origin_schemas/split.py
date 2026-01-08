import json
import os

def split_json_schemas():
    # 定义输入文件路径 (基于您提供的当前文件路径)
    input_file_path = 'bird_data/bird/llm/data/dev_tables.json'
    
    # 定义输出目录
    output_dir = 'bird_data/origin_schemas'

    # 检查输入文件是否存在
    if not os.path.exists(input_file_path):
        print(f"错误: 找不到输入文件: {input_file_path}")
        return

    # 如果输出目录不存在，则创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    try:
        # 读取原始 JSON 文件
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"成功加载 {len(data)} 个数据库模式条目。")

        # 遍历每个条目并保存
        for entry in data:
            db_id = entry.get('db_id')
            
            if db_id:
                # 构建输出文件名
                output_filename = f"{db_id}.json"
                output_path = os.path.join(output_dir, output_filename)
                
                # 写入单个 JSON 文件
                with open(output_path, 'w', encoding='utf-8') as out_f:
                    # indent=4 用于美化输出，ensure_ascii=False 用于正确显示中文等字符
                    json.dump(entry, out_f, indent=4, ensure_ascii=False)
                
                print(f"已保存: {output_path}")
            else:
                print("警告: 发现没有 'db_id' 的条目，已跳过。")

        print("\n所有文件处理完成。")

    except json.JSONDecodeError:
        print(f"错误: 无法解析 JSON 文件 {input_file_path}")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    split_json_schemas()