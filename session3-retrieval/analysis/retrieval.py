#!/usr/bin/env python3
"""Candidate-pool reachability and re-ranking ceiling on FAR-Trans (A09).

HW2 (A08) measured that only 6 of 18 relevant assets were inside the 50-candidate
CBF pool a re-ranker may reorder, so the attainable nDCG@10 was capped far below 1.
This script asks the prior question: does a *different* candidate pool, built from
the same pre-cutoff information and held to the same size budget, put more relevant
assets within reach?

Every pool is constructed from as-of information only (pre-cutoff Buy history,
pre-cutoff popularity, pre-cutoff content similarity). Truth is read afterwards to
score reachability and the oracle ceiling; it never enters pool construction.

Usage:
    python3 analysis/retrieval.py --starter-dir <far-trans> --data-dir <FAR-Trans> \
        --max-users 100 --candidate-count 50 --out outputs/retrieval_100.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pools as poolmod


def load_starter(starter_dir):
    path = Path(starter_dir) / 'starter.py'
    spec = importlib.util.spec_from_file_location('provided_far_trans_starter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ceiling(starter, pool, relevant):
    """Oracle nDCG@10: every reachable relevant asset placed first."""
    reachable = [i for i in pool if i in relevant]
    rest = [i for i in pool if i not in relevant]
    return starter.ndcg10((reachable + rest)[:10], relevant)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--starter-dir', required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--cutoff', default='2021-06-01')
    parser.add_argument('--horizon-days', type=int, default=30)
    parser.add_argument('--max-users', type=int, default=100)
    parser.add_argument('--candidate-count', type=int, default=50)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)

    starter = load_starter(args.starter_dir)
    assets, profiles, history, truth, end, excluded = starter.load_data(
        args.data_dir, args.cutoff, args.horizon_days)
    users = starter.select_users(profiles, history, 'warm', args.max_users, None)
    features = {isin: starter.tokens(row) for isin, row in assets.items()}
    pop = starter.popularity_scores(history, assets)
    events = starter.scan_recent_events(args.data_dir, args.cutoff, assets)

    names = list(poolmod.NAMES)
    agg = {n: {'ceiling_sum': 0.0, 'ceiling_nonempty_sum': 0.0,
               'reachable_items': 0, 'users_zero_reachable': 0} for n in names}
    truth_items = 0
    per_user = []
    nonempty = 0

    for uid in users:
        counts = history.get(uid, Counter())
        eligible = set(assets)
        cbf_order, pop_order, _cold = poolmod.orders_for(
            starter, counts, features, pop, eligible)
        recent_buys = [(t, i) for t, i, kind in events.get(uid, ()) if kind == 'Buy']

        built = poolmod.build(counts, eligible, cbf_order, pop_order,
                              recent_buys, args.candidate_count)
        relevant = truth.get(uid, set()) & eligible
        row = {'user_id': uid, 'truth_size': len(relevant)}
        if relevant:
            nonempty += 1
            truth_items += len(relevant)
        for name in names:
            pool = built[name]
            assert len(pool) <= args.candidate_count, (name, len(pool))
            c = ceiling(starter, pool, relevant)
            agg[name]['ceiling_sum'] += c
            row[name] = {'ceiling': c, 'reachable': len(set(pool) & relevant),
                         'pool_size': len(pool)}
            if relevant:
                agg[name]['ceiling_nonempty_sum'] += c
                agg[name]['reachable_items'] += len(set(pool) & relevant)
                if not set(pool) & relevant:
                    agg[name]['users_zero_reachable'] += 1
        per_user.append(row)

    summary = {
        'cutoff': args.cutoff, 'end_exclusive': end,
        'horizon_days': args.horizon_days,
        'users': len(users), 'nonempty_truth_users': nonempty,
        'candidate_count': args.candidate_count,
        'total_truth_items_nonempty': truth_items,
        'excluded_rows': excluded,
        'pools': {n: {
            'ceiling_ndcg10_all_users': agg[n]['ceiling_sum'] / len(users),
            'ceiling_ndcg10_nonempty': (agg[n]['ceiling_nonempty_sum'] / nonempty
                                        if nonempty else None),
            'reachable_truth_items': agg[n]['reachable_items'],
            'reachable_share': (agg[n]['reachable_items'] / truth_items
                                if truth_items else None),
            'nonempty_users_with_zero_reachable': agg[n]['users_zero_reachable'],
        } for n in names},
    }

    print(json.dumps(summary['pools'], indent=2))
    print(f"users {len(users)}  nonempty {nonempty}  truth items {truth_items}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({**summary, 'per_user': per_user},
                                       indent=2) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
