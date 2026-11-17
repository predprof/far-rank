#!/usr/bin/env python3
"""Headroom analysis for a fixed-candidate re-ranker on FAR-Trans (A08).

A re-ranker may only reorder the candidate pool it is given. It can never add
an asset the retrieval step missed. So the highest nDCG@10 any re-ranker can
reach is the score of the *oracle* ordering, which puts every relevant
candidate first. This script measures that bound with the starter's own
`ndcg10`, and also reports what fraction of each user's truth is reachable at
all.

The truth labels are used here only for after-the-fact analysis of the
measurement ceiling. They are never shown to a model and never enter a prompt.

Usage:
    python3 ceiling.py --starter-dir <far-trans> --data-dir <FAR-Trans>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path


def load_starter(starter_dir):
    path = Path(starter_dir) / "starter.py"
    spec = importlib.util.spec_from_file_location("provided_starter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starter-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cutoff", default="2021-06-01")
    parser.add_argument("--horizon-days", type=int, default=30)
    parser.add_argument("--max-users", type=int, default=100)
    parser.add_argument("--candidate-count", type=int, default=50)
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    starter = load_starter(args.starter_dir)
    assets, profiles, history, truth, _end, _excluded = starter.load_data(
        Path(args.data_dir), args.cutoff, args.horizon_days)
    users = starter.select_users(profiles, history, "warm", args.max_users, None)
    features = {isin: starter.tokens(a) for isin, a in assets.items()}
    pop = starter.popularity_scores(history, assets)
    contexts, _baseline = starter.contexts_for(
        users, assets, profiles, history, features, pop,
        args.cutoff, args.candidate_count, False)

    rows = []
    for uid in users:
        pool = [a["id"] for a in contexts[uid]["candidate_assets"]]
        want = set(truth.get(uid, ()) or ())
        reachable = [i for i in pool if i in want]
        # Oracle ordering: every reachable relevant candidate first.
        oracle = reachable + [i for i in pool if i not in set(reachable)]
        rows.append({
            "user_id": uid,
            "truth_size": len(want),
            "reachable": len(reachable),
            "ceiling_ndcg10": starter.ndcg10(oracle, want),
        })

    nonempty = [r for r in rows if r["truth_size"] > 0]
    zero_reach = [r for r in nonempty if r["reachable"] == 0]
    summary = {
        "users": len(rows),
        "nonempty_truth_users": len(nonempty),
        "ceiling_ndcg10_all_users": statistics.mean(r["ceiling_ndcg10"] for r in rows),
        "ceiling_ndcg10_nonempty": (
            statistics.mean(r["ceiling_ndcg10"] for r in nonempty) if nonempty else 0.0),
        "nonempty_users_with_zero_reachable_truth": len(zero_reach),
        "total_truth_items_nonempty": sum(r["truth_size"] for r in nonempty),
        "reachable_truth_items_nonempty": sum(r["reachable"] for r in nonempty),
        "per_user": sorted(nonempty, key=lambda r: -r["ceiling_ndcg10"]),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
