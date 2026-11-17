# Session 2 — Ranker seam (Arman Malkhasyan)

One change at the re-ranking step: a **repair-and-complete** invalid-response
policy, compared against the provided strict-fallback policy on one identical
set of Qwen3-1.7B generations.

| Method | nDCG@10 (all 100) | nDCG@10 (nonempty 11) |
|---|---:|---:|
| popularity | 0.0286 | 0.2599 |
| cbf | 0.0201 | 0.1827 |
| latest_buy | 0.0277 | 0.2520 |
| latest_buy_or_sell | 0.0271 | 0.2464 |
| BPR-MF (team's named baseline) | 0.0268 | 0.2435 |
| llmrank, provided policy | 0.0201 | 0.1827 |
| llmrank, repair-and-complete | 0.0205 | 0.1861 |

Re-ranking ceiling (oracle over the same 50 candidates): **0.0492 / 0.4469**.
Only 6 of 18 relevant assets are in the candidate pool; 5 of 11 labelled users
have none. Retrieval, not ranking, is the binding constraint on this protocol.

BPR-MF — the baseline the team's registered question names — reaches 54.5% of
that ceiling; the LLM re-ranker reaches 41.6%. A matrix factorisation trained in
under a minute beats the 1.7B language model at ordering the same fifty assets.

## Reproduce

```sh
./run_all.sh /path/to/FAR-Trans      # ~49 min, dominated by the single LLM pass
python3 -m unittest test_repair -v   # 11 tests, no model needed
```

Protocol hash for every run in the table:
`50efd4aeb415ba6f0c66dd0793a7c2033cdc32a42b86b58eebbca3e81eaf3417`

`results/` holds aggregates only. User-level outputs (predictions, contexts,
generations) are deliberately not committed: they carry customer identifiers.
