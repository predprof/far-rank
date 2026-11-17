#!/bin/sh
set -eu

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: sh run_all.sh FAR_TRANS_DIR COURSE_STARTER_DIR QWEN_SNAPSHOT_DIR [OUTPUT_DIR]" >&2
    exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
data_dir=$1
starter_dir=$2
model_dir=$3
output_dir=${4:-"$script_dir/outputs/reproduction"}
python=${PYTHON:-python3}
llm_python=${LLM_PYTHON:-"$python"}
device=${DEVICE:-mps}
bpr_source="$script_dir/../session2-ranker/bpr.py"

score() {
    "$python" -B "$starter_dir/starter.py" run --data-dir "$data_dir" \
        --cutoff 2021-06-01 --horizon-days 30 --cohort warm --max-users 100 \
        --candidate-count 50 "$@"
}

"$python" -B "$script_dir/prepare_splits.py" --starter "$starter_dir/starter.py" \
    --data-dir "$data_dir" --output-dir "$output_dir"
score --output-dir "$output_dir/baseline"

"$python" -B "$bpr_source" --starter-dir "$starter_dir" --data-dir "$data_dir" \
    --contexts "$output_dir/baseline/contexts.jsonl" --protocol "$output_dir/baseline/protocol.json" \
    --out "$output_dir/bpr_responses.jsonl" --cutoff 2021-06-01
score --output-dir "$output_dir/bpr" --stage ranker --responses "$output_dir/bpr_responses.jsonl"

"$python" -B "$script_dir/prepare_rerank_inputs.py" --session-dir "$output_dir" \
    --raw-data-dir "$data_dir" --bpr-source "$bpr_source"
rerank_dir="$output_dir/zero_shot_greedy"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$llm_python" -B "$script_dir/run_zero_shot.py" \
    --inputs "$output_dir/rerank_inputs.jsonl" --model-path "$model_dir" \
    --output-dir "$rerank_dir" --device "$device"

for method in llm repeat_first; do
    score --output-dir "$rerank_dir/$method" --stage ranker \
        --responses "$rerank_dir/${method}_responses.jsonl"
done
"$python" -B "$script_dir/verify_results.py" --session-dir "$output_dir"
