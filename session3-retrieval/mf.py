#!/usr/bin/env python3
"""BPR-MF baseline for FAR-Trans (A09), reusable across candidate pools.

Carried forward from A08, where the team's registered research question named
matrix factorisation trained with Bayesian Personalised Ranking as the
comparison baseline. A09 adds nothing to the model: it is trained **once** and
then asked to order each of the candidate pools under test, so the trained
ranker is held fixed while retrieval varies. Any difference between pools is
therefore retrieval, not a differently fitted model.

Method: implicit-feedback BPR (Rendle et al., UAI 2009). Training pairs are
(user, bought asset) from `Buy` rows strictly before the cutoff, with
reconstructed initial-position rows (negative `transactionID`) excluded exactly
as the starter excludes them. Negatives are drawn uniformly from the eligible
universe minus that user's observed assets. No future row is read.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RANK_LENGTH = 10


def read_pre_cutoff_buys(data_dir, cutoff):
    """(user, item) pairs from Buy rows strictly before the cutoff."""
    pairs = defaultdict(set)
    with (Path(data_dir) / 'transactions.csv').open(encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            if row['transactionType'] != 'Buy':
                continue
            if row['transactionID'].strip().startswith('-'):
                continue                      # reconstructed initial position
            if datetime.date.fromisoformat(row['timestamp']) >= cutoff:
                continue                      # strictly pre-cutoff only
            pairs[row['customerID']].add(row['ISIN'])
    return pairs


def train(pairs, items, factors=32, epochs=10, lr=0.05, reg=0.01, seed=42,
          log=print):
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
        raise SystemExit('no trainable users')
    n_pairs = sum(len(observed[u]) for u in trainable)
    losses = []

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
        losses.append(total / n_pairs)
        log(f'  epoch {epoch + 1}/{epochs}  mean BPR loss {losses[-1]:.5f}')
    return {'P': P, 'Q': Q, 'uidx': uidx, 'iidx': iidx,
            'n_users': len(users), 'n_items': len(items),
            'n_pairs': sum(len(v) for v in pairs.values()),
            'losses': losses, 'factors': factors, 'epochs': epochs,
            'lr': lr, 'reg': reg, 'seed': seed}


def rank_pool(model, uid, pool):
    """Order one candidate pool; unseen users keep the pool's own order.

    Ties break by ISIN, matching the starter's deterministic rule.
    """
    if uid not in model['uidx']:
        return list(pool)[:RANK_LENGTH], True
    scores = model['Q'][[model['iidx'][i] for i in pool]] @ model['P'][model['uidx'][uid]]
    ranked = [i for _, i in sorted(zip((-float(s) for s in scores), pool))]
    return ranked[:RANK_LENGTH], False


def fit(data_dir, cutoff, eligible, **kwargs):
    """Convenience wrapper: read pre-cutoff buys and train on the eligible set."""
    cut = datetime.date.fromisoformat(cutoff)
    pairs = read_pre_cutoff_buys(data_dir, cut)
    pairs = {u: {i for i in its if i in eligible} for u, its in pairs.items()}
    pairs = {u: its for u, its in pairs.items() if its}
    items = sorted(eligible)
    print(f'training BPR-MF: {len(pairs)} users, {len(items)} items, '
          f'{sum(len(v) for v in pairs.values())} observed pairs, '
          f"{kwargs.get('factors', 32)} factors, {kwargs.get('epochs', 10)} epochs, "
          f"seed {kwargs.get('seed', 42)}")
    return train(pairs, items, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--eligible', required=True,
                        help='JSON file with an eligible_assets list')
    parser.add_argument('--cutoff', default='2021-06-01')
    parser.add_argument('--out', required=True, help='npz factor file')
    parser.add_argument('--factors', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    eligible = set(json.loads(Path(args.eligible).read_text())['eligible_assets'])
    model = fit(args.data_dir, args.cutoff, eligible, factors=args.factors,
                epochs=args.epochs, seed=args.seed)
    np.savez(args.out, P=model['P'], Q=model['Q'],
             users=np.array(sorted(model['uidx'])),
             items=np.array(sorted(model['iidx'])))
    print(f"saved factors to {args.out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
