#!/bin/sh
# A08 end-to-end pipeline. Reproduces every number in report.md.
#
#   ./run_all.sh /path/to/FAR-Trans
#
# Step 2 (the LLM pass) is the only slow part: ~29 s/user on CPU, ~49 min for
# 100 users. It calls the model once; both re-ranking policies are then derived
# from that single set of generations, so the comparison varies only the policy.
set -e
cd "$(dirname "$0")"

DATA="${1:-data/FAR-Trans}"
STARTER="../RecSys-LLMs/project/far-trans"
LLM="../RecSys-LLMs/project/far-trans-llm"
PY="${PY:-.venv/bin/python}"
MODEL="Qwen/Qwen3-1.7B"
REV="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

echo "== 0. unit tests =="
python3 -m unittest test_repair -v

echo "== 1. four provided baselines =="
python3 "$STARTER/starter.py" run --data-dir "$DATA" --output-dir outputs/baseline

echo "== 2. one LLM pass (slow) =="
[ -f outputs/ranker100/final_responses.jsonl ] || \
  "$PY" "$LLM/ranker.py" --data-dir "$DATA" --output-dir outputs/ranker100 \
    --max-users 100 --candidate-count 50 --model "$MODEL" --revision "$REV" \
    --device cpu --on-invalid fallback

echo "== 3. derive both policies from those generations =="
python3 repair.py \
  --responses outputs/ranker100/final_responses.jsonl \
  --contexts  outputs/baseline/contexts.jsonl \
  --protocol  outputs/baseline/protocol.json \
  --out          outputs/repaired.jsonl \
  --out-provided outputs/provided.jsonl \
  --ranker-dir "$LLM" \
  --stats outputs/repair_stats.json

echo "== 4. score each policy through the same starter evaluation =="
python3 "$STARTER/starter.py" run --data-dir "$DATA" \
  --output-dir outputs/provided --stage ranker --responses outputs/provided.jsonl
python3 "$STARTER/starter.py" run --data-dir "$DATA" \
  --output-dir outputs/repaired --stage ranker --responses outputs/repaired.jsonl

echo "== 5. re-ranking ceiling =="
python3 ceiling.py --starter-dir "$STARTER" --data-dir "$DATA" \
  --out outputs/ceiling.json > /dev/null

echo "== 6. assemble the results table =="
python3 compare.py \
  --baseline outputs/baseline/metrics.json \
  --run provided_policy=outputs/provided/metrics.json \
  --run repair_complete=outputs/repaired/metrics.json \
  --ceiling outputs/ceiling.json \
  --repair-stats outputs/repair_stats.json \
  --out outputs/RESULTS.md

echo "== 7. independent verification =="
python3 verify.py --run outputs/repaired --data-dir "$DATA" --method custom \
  --prompts outputs/ranker100/generations.jsonl

echo "done; see outputs/RESULTS.md"
