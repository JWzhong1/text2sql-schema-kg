#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="${1:-cache}"

if [ ! -d "$CACHE_DIR" ]; then
  echo "Cache directory '$CACHE_DIR' not found." >&2
  exit 1
fi

for entry in "$CACHE_DIR"/*; do
  [ -d "$entry" ] || continue
  DB_NAME="$(basename "$entry")"
  echo "========== Processing ${DB_NAME} =========="

  python src/graph/schema_graph_builder.py --db_name "$DB_NAME"

  TEST_FILE="bird_data/golden_link/golden_schema_link_${DB_NAME}.json"
  if [ ! -f "$TEST_FILE" ]; then
    echo "Skipping ${DB_NAME}: missing test file ${TEST_FILE}" >&2
    continue
  fi

  CACHE_FILE="scripts/evaluate/cache/retrieval_results_${DB_NAME}.json"
  REPORT_DIR="scripts/evaluate/result/${DB_NAME}"
  mkdir -p "$(dirname "$CACHE_FILE")" "$REPORT_DIR"
  REPORT_FILE="${REPORT_DIR}/eval_report_${DB_NAME}_$(date +%Y%m%d_%H%M%S).json"

  python scripts/evaluate/evaluate_retrieval.py \
    --db_name "$DB_NAME" \
    --test_file "$TEST_FILE" \
    --cache_file "$CACHE_FILE" \
    --report_file "$REPORT_FILE"
done
