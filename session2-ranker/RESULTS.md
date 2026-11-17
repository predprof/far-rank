# Session 2 results - Alina Raimova

Zero-shot Qwen3 reranking increased mean nDCG@10 from **0.0268 to 0.0313** over BPR-MF on the same candidates, an absolute gain of **0.0045**. The repeat-first control scored **0.0281** using the same BPR scores and historical purchase flags supplied to Qwen.

## Evaluation

The FAR-Trans training data ends before June 1, 2021. Evaluation covers June 1 through June 30, 2021. The deterministic sample contains 100 users with purchase history, 11 of whom bought an eligible asset in June. The all-user average counts empty truth as zero. Repeat purchases are eligible. BPR-MF, repeat-first, and Qwen rerank the same 50 CBF candidates per user.

| Method | nDCG@10, all 100 | nDCG@10, 11 with purchases |
|---|---:|---:|
| Popularity | 0.0286 | 0.2599 |
| CBF | 0.0201 | 0.1827 |
| Latest buy | 0.0277 | 0.2520 |
| Latest buy or sell | 0.0271 | 0.2464 |
| BPR-MF on CBF candidates | 0.0268 | 0.2435 |
| Repeat-first, then BPR-MF score | 0.0281 | 0.2550 |
| **Zero-shot Qwen3-1.7B with BPR-MF fallback** | 0.0313 | 0.2845 |

Qwen produced 84 valid rankings. The other 16 kept BPR-MF order due to an incorrect number of aliases in 15 responses and a duplicate in one. Generation took **349.73 s** on Apple Metal, excluding model loading and scoring. The four course baselines shared a **4.02 s** starter run. BPR-MF training and ranking took **5.60 s**, excluding scoring.

The candidate pools contained 6 of 18 relevant user–asset pairs. Of 54 users with 1–3 historical Buy events, one had an eligible June purchase. Qwen moved that asset from fourth to second.

## Reproduce

The recorded run used course starter commit `f6850861ea20a51e1fb018ee4e5f840c2ac759e0`, Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, NumPy 2.4.6, PyTorch 2.7.1, and Transformers 4.57.6.

From `session2-reranked`, run:

```bash
sh run_all.sh /path/to/FAR-Trans /path/to/RecSys-LLMs/project/far-trans /path/to/pinned-Qwen-snapshot
```

The optional fourth argument sets the output directory. The default is `outputs/reproduction`. Set `PYTHON` and `LLM_PYTHON` to select interpreters and `DEVICE` to select the inference device, which defaults to `mps`.

- [Course baselines](results/metrics_baseline.json), [BPR-MF](results/metrics_bpr_mf.json), [repeat-first](results/metrics_repeat_first.json), and [Qwen](results/metrics_llm.json)
- [Cohort comparisons and fallback counts](results/comparison.json)
