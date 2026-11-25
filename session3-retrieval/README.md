# Session 3 — retrieval is the binding constraint (Arman Malkhasyan)

Session 2 measured that an oracle reordering of the provided 50-candidate CBF
pool scored only 0.0492, because just 6 of 18 relevant assets were inside the
pool and 5 of 11 labelled users had none at all. Ranking cannot promote an asset
retrieval never supplied, so Session 3 moves the intervention one stage earlier:
rebuild the candidate pool **at the same 50-candidate budget** and measure what
becomes reachable.

The LLM is used as a **Ranker** only. No encoder change is made here; that seam
is Alina's half of the team split.

## What varies, and what is held fixed

| | |
|---|---|
| Varies | the candidate pool, and nothing else |
| Fixed | users, cutoff, horizon, eligible assets, labels, metric |
| Fixed | within a pool, the candidates and their initial order, across all rankers |
| Fixed | one BPR-MF model, trained once, asked to order each pool |
| Imported unchanged | the provided prompt builder, parser, runtime and `ndcg10` |

`P_cbf` is the control. Two checks pin it down: my `P_cbf` candidate list equals
the provided `contexts_for` output for every user, and my `P_cbf` prompts are
**byte-identical** to the ones the provided `ranker.py` writes for all 300 users.
The control condition is therefore the provided experiment, not a
re-implementation of it.

## Pools

| Pool | Composition (capped at 50) |
|---|---|
| `cbf` | provided content-similarity order |
| `pop` | popularity, by distinct pre-cutoff buyers |
| `own_then_cbf` | own pre-cutoff purchases, most recent first, then content order |
| `own_then_pop` | own purchases, most recent first, then popularity |
| `own_freq_then_pop` | own purchases, most frequent first, then popularity |

Repeat purchases are allowed by the protocol, so an asset the customer already
holds is a legitimate candidate rather than a leak. Pools read only pre-cutoff
rows; truth enters solely in scoring and in the after-the-fact oracle ceiling.

**Credit.** The intuition that previously bought assets matter under repeat
purchases is Alina's, from her Session 2 `repeat_first` control, which promotes
them *inside* the provided pool and gained 0.0013 over BPR-MF on the identical
protocol hash. This work asks whether the same intuition belongs at the
retrieval stage instead, where — unlike any reordering — it can change which
assets are reachable at all.

## Development cohorts

Nested prefixes of the starter's deterministic SHA-256 seed-42 warm-user order,
so the smaller is a strict prefix of the larger.

- **D1000** — 1000 users, 119 with future purchases, 285 relevant pairs. Carries
  the retrieval comparison and the ablation.
- **D300** — 300 users, 38 with future purchases, 98 relevant pairs. Carries the
  LLM comparison, which costs about a CPU-minute per user per pool.

Cutoff 2021-06-01 exclusive, 30-day future-Buy window, 50 candidates, 767
eligible assets, repeat purchases allowed. Selection never reads future
purchases or observed scores.

**These are development results, not official benchmark scores.** The official
protocol and common evaluation set are not yet released.

## Ablation

Removing the own-history seed turns `own_then_pop` into `pop`, with budget,
ranker and everything else unchanged. Because popularity alone already supplies
most of the pool's membership, this separates *which assets are reachable* from
*the order they arrive in* — the two distinct ways retrieval can help.

## Files

| File | Role |
|---|---|
| `pools.py` | the five pool definitions; one definition, imported everywhere |
| `analysis/retrieval.py` | reachability and oracle ceiling per pool |
| `generate.py` | swaps the pool, then calls the provided prompt builder and runtime |
| `mf.py` | BPR-MF, trained once, reusable across pools |
| `evaluate.py` | the single place every method is scored |
| `tables.py` | report tables, emitted from the results files rather than retyped |
| `verify.py` | five independent checks (see below) |
| `test_pipeline.py` | 20 unit tests for pool construction and the repair policy |
| `publish.py` | exports aggregate-only results; refuses to write customer IDs |
| `run_all.sh` | end-to-end; reproduces every number the report cites |

## Verification

1. nDCG@10 for one named user **under one named method**, recomputed from truth
   read straight from `transactions.csv`. In Session 2 I reported a hand-check
   without naming its method and the number turned out to belong to
   `popularity`; naming it in every line of output is the fix.
2. No generated prompt carries a date strictly after the cutoff.
3. Pool integrity: inside the eligible universe, inside the size budget, and
   `cbf` identical to the provided `contexts_for`.
4. Each matched control is exactly its own pool prefix.
5. Control prompts byte-identical to the provided `ranker.py`'s.

## What is not in this repository

No dataset, model, cache, log, prompt, raw customer ID or user-level result.
`*_per_user.csv`, `*_predictions.csv`, `pools.json`, `responses.jsonl`,
`prompts.jsonl`, `generations.jsonl` and `generation_cache/` are all keyed on an
anonymised customer ID and stay local; `.gitignore` excludes them and
`publish.py` asserts that no customer-ID-shaped token reaches `results/`.

## Reproduce

```sh
# needs the course starter checked out alongside, and torch + transformers
# for step 3 only
PY=python3 LLM_PY=/path/to/venv/bin/python DEVICE=cpu \
  ./run_all.sh /path/to/FAR-Trans
```

Qwen3-1.7B, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, greedy decoding
with thinking disabled, 8192-token context, 512 max new tokens, CPU float32.
Course starter commit `799e54c`.
