# A08 results

Protocol hash (all runs): `50efd4aeb415ba6f0c66dd0793a7c2033cdc32a42b86b58eebbca3e81eaf3417`

| Method | Users | Users with future purchases | nDCG@10: all users | nDCG@10: nonempty truth |
|---|---:|---:|---:|---:|
| popularity | 100 | 11 | 0.0286 | 0.2599 |
| cbf | 100 | 11 | 0.0201 | 0.1827 |
| latest_buy | 100 | 11 | 0.0277 | 0.2520 |
| latest_buy_or_sell | 100 | 11 | 0.0271 | 0.2464 |
| provided_policy | 100 | 11 | 0.0201 | 0.1827 |
| repair_complete | 100 | 11 | 0.0205 | 0.1861 |

## Re-ranker headroom (oracle over the same candidate pool)

- ceiling nDCG@10, all users: **0.0492**
- ceiling nDCG@10, nonempty truth: **0.4469**
- relevant items present in the candidate pool: **6 of 18**
- labelled users with zero reachable truth: **5 of 11**

## Policy divergence on identical generations

- users: **100**
- invalid under the provided strict policy: **23**
- users with at least one salvageable candidate ISIN: **99**
- users with a complete ten-ISIN model ranking: **77**
- users where nothing was salvageable (policies identical): **1**
- users whose two rankings differ: **22**

Salvaged-count distribution: `{"10": 77, "9": 8, "6": 1, "3": 4, "7": 3, "2": 1, "5": 2, "8": 2, "1": 1, "0": 1}`
Provided-policy failure reasons: `{"ValueError": 23}`
