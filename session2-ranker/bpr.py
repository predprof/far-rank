#!/usr/bin/env python3
"""BPR-MF baseline for FAR-Trans (A08).

The team's registered research question names matrix factorisation trained with
Bayesian Personalised Ranking as the comparison baseline, so this adds it
alongside the four baselines the homework mandates. It is an *additional*
baseline: nothing provided is replaced.

Method: implicit-feedback BPR [Rendle et al., UAI 2009]. Training pairs are
(user, bought item) from pre-cutoff `Buy` rows only, with reconstructed
initial-position rows (negative `transactionID`) excluded exactly as the starter
excludes them. For each sampled positive the negative is drawn uniformly from
the eligible universe minus that user's observed items. Scores are the dot
product of learned user and item factors; the model then orders each evaluated
user's own 50-candidate pool, so it is scored on identical candidates to the
re-rankers.

No future row is read: the cutoff filter is applied before any training pair is
formed, and the evaluation window is never opened here.

Usage:
    python3 bpr.py --starter-dir <far-trans> --data-dir <FAR-Trans> \
                   --contexts <starter_out>/contexts.jsonl \
                   --protocol <starter_out>/protocol.json \
                   --out outputs/bpr.jsonl
"""
from __future__ import annotations

import argparse
import csv
import datetime
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

RANK_LENGTH = 10


def load_starter(starter_dir):
    path = Path(starter_dir) / "starter.py"
    spec = importlib.util.spec_from_file_location("provided_starter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_pre_cutoff_buys(data_dir, cutoff):
    """(user, item) pairs from Buy rows strictly before the cutoff."""
    pairs = defaultdict(set)
    with (Path(data_dir) / "transactions.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["transactionType"] != "Buy":
                continue
            if row["transactionID"].strip().startswith("-"):
                continue                      # reconstructed initial position
            if datetime.date.fromisoformat(row["timestamp"]) >= cutoff:
                continue                      # strictly pre-cutoff only
            pairs[row["customerID"]].add(row["ISIN"])
    return pairs


def train_bpr(pairs, items, factors, epochs, lr, reg, seed):
    rng = np.random.default_rng(seed)
    users = sorted(pairs)
    uidx = {u: n for n, u in enumerate(users)}
    iidx = {i: n for n, i in enumerate(items)}
    observed = {uidx[u]: np.array(sorted(iidx[i] for i in its if i in iidx))
                for u, its in pairs.items()}
    observed = {u: a for u, a in observed.items() if len(a)}

    P = rng.normal(0, 0.1, (len(users), factors))
    Q = rng.normal(0, 0.1, (len(items), factors))

    trainable = [u for u, a in observed.items() if len(a) < len(items)]
    if not trainable:
        raise SystemExit("no trainable users")
    n_pairs = sum(len(observed[u]) for u in trainable)

    for epoch in range(epochs):
        total = 0.0
        for _ in range(n_pairs):
            u = trainable[rng.integers(len(trainable))]
            pos_set = observed[u]
            i = pos_set[rng.integers(len(pos_set))]
            j = rng.integers(len(items))
            while j in pos_set:               # uniform negative, unobserved
                j = rng.integers(len(items))
            pu, qi, qj = P[u], Q[i], Q[j]
            x = float(pu @ (qi - qj))
            sig = 1.0 / (1.0 + np.exp(x))     # d/dx of -log sigmoid(x)
            P[u] += lr * (sig * (qi - qj) - reg * pu)
            Q[i] += lr * (sig * pu - reg * qi)
            Q[j] += lr * (-sig * pu - reg * qj)
            total += np.log1p(np.exp(-x))
        print(f"  epoch {epoch + 1}/{epochs}  mean BPR loss {total / n_pairs:.5f}")
    return P, Q, uidx, iidx


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starter-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cutoff", default="2021-06-01")
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--reg", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    random.seed(args.seed)
    cutoff = datetime.date.fromisoformat(args.cutoff)
    protocol_hash = json.loads(Path(args.protocol).read_text())["protocol_hash"]

    contexts = [json.loads(l) for l in
                Path(args.contexts).read_text(encoding="utf-8").splitlines() if l.strip()]
    candidates = {c["user_id"]: [a["id"] for a in c["candidate_assets"]] for c in contexts}
    eligible = set()
    for c in contexts:
        eligible.update(c["eligible_asset_ids"])

    pairs = read_pre_cutoff_buys(args.data_dir, cutoff)
    pairs = {u: {i for i in its if i in eligible} for u, its in pairs.items()}
    pairs = {u: its for u, its in pairs.items() if its}
    items = sorted(eligible)
    print(f"training BPR-MF: {len(pairs)} users, {len(items)} items, "
          f"{sum(len(v) for v in pairs.values())} observed pairs, "
          f"{args.factors} factors, {args.epochs} epochs, seed {args.seed}")

    P, Q, uidx, iidx = train_bpr(pairs, items, args.factors, args.epochs,
                                 args.lr, args.reg, args.seed)

    rows, cold = [], 0
    for c in contexts:
        uid = c["user_id"]
        pool = candidates[uid]
        if uid in uidx:
            scores = Q[[iidx[i] for i in pool]] @ P[uidx[uid]]
            # Deterministic tie-breaking by ISIN, matching the starter's rule.
            ranked = [i for _, _, i in
                      sorted(((-float(s), i, i) for s, i in zip(scores, pool)))]
        else:
            cold += 1
            ranked = list(pool)               # unseen user: keep candidate order
        rows.append({"user_id": uid, "stage": "ranker",
                     "protocol_hash": protocol_hash,
                     "ranked_ids": ranked[:RANK_LENGTH]})

    with Path(args.out).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rankings to {args.out} "
          f"({cold} users unseen in training, left in candidate order)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
