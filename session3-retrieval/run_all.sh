#!/bin/sh
# A09 end-to-end pipeline. Reproduces every number the report cites.
#
#   ./run_all.sh [/path/to/FAR-Trans]
#
# Step 3 is the only slow part: three LLM passes of 300 users each. Run in
# parallel they take about 4.8 h wall clock on CPU (~58 s/user/pool with three
# processes resident); run sequentially, about 6.9 h (~27.5 s/user/pool).
# Generation is cached per prompt, so re-running after an interruption resumes
# instead of regenerating.
#
# In A08 the equivalent script silently stopped covering the report, because the
# BPR-MF baseline was added after the script was written and never wired in.
# Every method the A09 report names is produced below; `evaluate.py` is the
# single place that scores them, so a method cannot reach the report without
# passing through this pipeline.
set -e
cd "$(dirname "$0")"

DATA="${1:-data/FAR-Trans}"
STARTER="../RecSys-LLMs/project/far-trans"
PY="${PY:-python3}"
LLM_PY="${LLM_PY:-../Homework 8/.venv/bin/python}"   # needs torch + transformers
MODEL="Qwen/Qwen3-1.7B"
REV="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DEVICE="${DEVICE:-cpu}"
POOLS="cbf own_freq_then_pop pop"

echo "== 1. unit tests =="
"$PY" -m unittest test_pipeline -v 2>&1 | tail -3

echo "== 2. reachability and ceiling, D1000 and D300 =="
for N in 100 300 1000; do
  "$PY" analysis/retrieval.py --starter-dir "$STARTER" --data-dir "$DATA" \
    --max-users "$N" --candidate-count 50 --out "outputs/retrieval_$N.json" >/dev/null
  echo "  wrote outputs/retrieval_$N.json"
done

echo "== 2b. control prompts, mine and the provided ranker's, for check 5 =="
"$PY" generate.py --pool cbf --max-users 300 --data-dir "$DATA" \
  --output-dir outputs/d300_prompts/cbf --prepare-only
"$PY" "$STARTER/../far-trans-llm/ranker.py" --data-dir "$DATA" \
  --output-dir outputs/d300_prompts/provided_prompts --max-users 300 \
  --candidate-count 50 --model "$MODEL" --revision "$REV" --prepare-only

echo "== 3. three LLM passes over three candidate pools (slow) =="
for P in $POOLS; do
  if [ -f "outputs/d300/$P/generation_meta.json" ]; then
    echo "  pool $P already complete; skipping"
  else
    "$LLM_PY" generate.py --pool "$P" --max-users 300 --data-dir "$DATA" \
      --output-dir "outputs/d300/$P" --device "$DEVICE"
  fi
done

echo "== 4. score the LLM cohort: baselines, controls, both policies, BPR-MF, ceiling =="
"$PY" evaluate.py --data-dir "$DATA" --runs outputs/d300 --max-users 300 \
  --pools $POOLS --out outputs/d300/results.json

echo "== 5. score the large development sample (no LLM) =="
"$PY" evaluate.py --data-dir "$DATA" --runs outputs/none --max-users 1000 \
  --pools cbf pop own_then_cbf own_then_pop own_freq_then_pop \
  --out outputs/d1000/results.json

echo "== 6. assemble the report tables =="
"$PY" tables.py --d300 outputs/d300/results.json \
  --d1000 outputs/d1000/results.json \
  --retrieval outputs/retrieval_1000.json \
  --retrieval-300 outputs/retrieval_300.json > outputs/TABLES.md
echo "  wrote outputs/TABLES.md"

echo "== 7. independent verification =="
"$PY" verify.py --results outputs/d300/results.json --data-dir "$DATA" \
  --runs outputs/d300 --max-users 300 \
  --prompt-dir outputs/d300_prompts \
  --method own_freq_then_pop/llm_repaired
"$PY" verify.py --results outputs/d1000/results.json --data-dir "$DATA" \
  --max-users 1000 --method own_freq_then_pop/initial_order_first10

echo "done; see outputs/TABLES.md"
