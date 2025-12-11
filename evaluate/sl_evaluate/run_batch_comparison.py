import os
import subprocess
from pathlib import Path

def main():
    # 定义路径
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    
    dinsql_result_dir = project_root / "scripts/evaluate/dinsql_result"
    ours_result_dir = project_root / "scripts/evaluate/result_2025_12_10"
    visualize_script = current_dir / "visualize_comparison.py"

    # 检查目录是否存在并获取子目录名（即 db_name）
    if not dinsql_result_dir.exists():
        print(f"Warning: Directory not found: {dinsql_result_dir}")
        dinsql_dbs = set()
    else:
        dinsql_dbs = {d.name for d in dinsql_result_dir.iterdir() if d.is_dir()}

    if not ours_result_dir.exists():
        print(f"Warning: Directory not found: {ours_result_dir}")
        ours_dbs = set()
    else:
        ours_dbs = {d.name for d in ours_result_dir.iterdir() if d.is_dir()}

    # 取并集
    all_dbs = sorted(list(dinsql_dbs | ours_dbs))
    
    if not all_dbs:
        print("No databases found in either directory.")
        return

    print(f"Found {len(all_dbs)} databases: {all_dbs}")
    print("Starting batch processing...\n")

    # 批量执行 visualize_comparison.py
    for db_name in all_dbs[:len(all_dbs)-1]:  
        print(f"[{db_name}] Generating comparison report...")
        try:
            cmd = ["python", str(visualize_script), "--db_name", db_name]
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {db_name}: {e}")
        except Exception as e:
            print(f"Unexpected error for {db_name}: {e}")
        print("-" * 40)

if __name__ == "__main__":
    main()