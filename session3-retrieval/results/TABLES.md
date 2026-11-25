## D1000 (large development sample)


**Four required baselines, D1000** (cohort 1000, 119 with future purchases)

| Baseline | nDCG@10 all users | nDCG@10 nonempty |
|---|---:|---:|
| Popularity | 0.0321 | 0.2699 |
| CBF | 0.0344 | 0.2894 |
| Latest buy | 0.0524 | 0.4403 |
| Latest buy or sell | 0.0521 | 0.4374 |

**Pool x ranker, nDCG@10 all users, D1000**

| Candidate pool | initial order (matched control) | BPR-MF | oracle ceiling |
|---|---:|---:|---:|
| P_cbf (provided retrieval) | 0.0344 | 0.0396 | 0.0570 |
| P_pop (popularity only) | 0.0321 | 0.0380 | 0.0972 |
| P_own+cbf | 0.0527 | 0.0461 | 0.0825 |
| P_own+pop (recency-seeded) | 0.0533 | 0.0396 | 0.1057 |
| P_own+pop (frequency-seeded) | 0.0501 | 0.0396 | 0.1057 |

**Pool x ranker, nDCG@10 nonempty truth, D1000**

| Candidate pool | initial order (matched control) | BPR-MF | oracle ceiling |
|---|---:|---:|---:|
| P_cbf (provided retrieval) | 0.2894 | 0.3328 | 0.4794 |
| P_pop (popularity only) | 0.2699 | 0.3190 | 0.8172 |
| P_own+cbf | 0.4429 | 0.3874 | 0.6933 |
| P_own+pop (recency-seeded) | 0.4481 | 0.3326 | 0.8885 |
| P_own+pop (frequency-seeded) | 0.4211 | 0.3326 | 0.8885 |

protocol_hash `526393e6526a`, eligible assets 767, cutoff 2021-06-01, window ends 2021-07-01

**Reachability and ceiling** (cohort 1000, 119 with future purchases, 285 relevant assets, pool size 50)

| Candidate pool | relevant in pool | share | users with none reachable | ceiling all | ceiling nonempty |
|---|---:|---:|---:|---:|---:|
| P_cbf (provided retrieval) | 89 of 285 | 31.2% | 51 of 119 | 0.0570 | 0.4794 |
| P_pop (popularity only) | 212 of 285 | 74.4% | 16 of 119 | 0.0972 | 0.8172 |
| P_own+cbf | 184 of 285 | 64.6% | 29 of 119 | 0.0825 | 0.6933 |
| P_own+pop (recency-seeded) | 238 of 285 | 83.5% | 9 of 119 | 0.1057 | 0.8885 |
| P_own+pop (frequency-seeded) | 239 of 285 | 83.9% | 9 of 119 | 0.1057 | 0.8885 |

## D300 (LLM cohort)


**Four required baselines, D300** (cohort 300, 38 with future purchases)

| Baseline | nDCG@10 all users | nDCG@10 nonempty |
|---|---:|---:|
| Popularity | 0.0276 | 0.2179 |
| CBF | 0.0351 | 0.2775 |
| Latest buy | 0.0498 | 0.3929 |
| Latest buy or sell | 0.0520 | 0.4105 |

**Pool x ranker, nDCG@10 all users, D300**

| Candidate pool | initial order (matched control) | LLM, provided strict policy | LLM, repair-and-complete | BPR-MF | oracle ceiling |
|---|---:|---:|---:|---:|---:|
| P_cbf (provided retrieval) | 0.0351 | 0.0344 | 0.0362 | 0.0369 | 0.0563 |
| P_pop (popularity only) | 0.0276 | 0.0358 | 0.0383 | 0.0323 | 0.0998 |
| P_own+pop (frequency-seeded) | 0.0509 | 0.0513 | 0.0507 | 0.0356 | 0.1126 |

**Pool x ranker, nDCG@10 nonempty truth, D300**

| Candidate pool | initial order (matched control) | LLM, provided strict policy | LLM, repair-and-complete | BPR-MF | oracle ceiling |
|---|---:|---:|---:|---:|---:|
| P_cbf (provided retrieval) | 0.2775 | 0.2716 | 0.2858 | 0.2911 | 0.4443 |
| P_pop (popularity only) | 0.2179 | 0.2829 | 0.3025 | 0.2547 | 0.7877 |
| P_own+pop (frequency-seeded) | 0.4016 | 0.4053 | 0.4000 | 0.2810 | 0.8891 |

**Invalid outputs, repairs and fallbacks**

| Candidate pool | generated | invalid under provided policy | parse repaired | salvageable | nothing salvageable | policies differ | LLM equals control |
|---|---:|---:|---:|---:|---:|---:|---:|
| P_cbf (provided retrieval) | 300 | 61 | 0 | 296 | 4 | 56 | 7 |
| P_pop (popularity only) | 300 | 54 | 0 | 297 | 3 | 49 | 5 |
| P_own+pop (frequency-seeded) | 300 | 75 | 0 | 297 | 3 | 63 | 15 |

protocol_hash `2a04648d0380`, eligible assets 767, cutoff 2021-06-01, window ends 2021-07-01

BPR-MF: 10457 users, 767 items, 38041 pre-cutoff pairs, 32 factors, 10 epochs, lr 0.05, L2 0.01, seed 42; loss 0.6597 to 0.0921; training 3.4 s
