#!/bin/bash

# 定义数据目录路径
DATA_DIR="bird_data/bird/llm/data/dev_databases"
CACHE_DIR="cache"

# 检查目录是否存在
if [ ! -d "$DATA_DIR" ]; then
  echo "Error: Directory $DATA_DIR does not exist."
  exit 1
fi

# 确保 cache 目录存在
if [ ! -d "$CACHE_DIR" ]; then
    mkdir -p "$CACHE_DIR"
fi

echo "=================================================="
echo "Starting batch pipeline for all databases in $DATA_DIR"
echo "=================================================="

# 遍历目录下的所有子目录（即数据库名称）
for db_path in "$DATA_DIR"/*; do
    if [ -d "$db_path" ]; then
        # 获取目录名作为 DB_NAME
        DB_NAME=$(basename "$db_path")
        
        # 如果 cache 中已有该 DB 的文件则跳过
        if ls "$CACHE_DIR"/"$DB_NAME"* 1> /dev/null 2>&1; then
            echo "Skipping $DB_NAME: Found in cache."
            continue
        fi
        
        echo "--------------------------------------------------"
        echo "Processing database: $DB_NAME"
        echo "--------------------------------------------------"

        # 1. Schema Conversion
        echo "[Step 1/3] Running Schema Converter..."
        python src/preprocessing/schema_converter.py --db_name "$DB_NAME"
        if [ $? -ne 0 ]; then
            echo "Error: Schema Converter failed for $DB_NAME. Skipping to next..."
            continue
        fi

        # 2. Golden Link Extraction
        echo -e "\n[Step 2/3] Extracting Golden Schema Links..."
        python src/preprocessing/extracted_golden_schema_link.py --db_name "$DB_NAME"
        if [ $? -ne 0 ]; then
            echo "Error: Golden Link Extraction failed for $DB_NAME. Skipping to next..."
            continue
        fi

        # 3. Graph Building
        echo -e "\n[Step 3/3] Building Schema Graph in Neo4j..."
        python src/graph/schema_graph_builder.py --db_name "$DB_NAME"
        if [ $? -ne 0 ]; then
            echo "Error: Graph Builder failed for $DB_NAME. Skipping to next..."
            continue
        fi
        
        echo "Finished processing $DB_NAME"
    fi
done

echo -e "\nAll databases have been processed."