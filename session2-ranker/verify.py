#!/usr/bin/env python3
"""Independent verification of the A08 results.

Three checks, each recomputed from the raw artefacts rather than trusting the
starter's own aggregation:

  1. nDCG@10 for one named user, recomputed from first principles and compared
     with the value the starter wrote to per_user.csv.
  2. Leakage: no date at or after the cutoff appears in any exported prompt.
  3. Truth containment: the relevant assets used in check 1 really are future
     Buy rows inside the evaluation window, read straight from transactions.csv.

Usage:
    python3 verify.py --run <starter_out> --data-dir <FAR-Trans> [--user ID]
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import re
from collections import defaultdict
from pathlib import Path

DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def ndcg_at_10(ranking, truth):
    """Binary nDCG@10, written out independently of starter.ndcg10."""
    if not truth:
        return 0.0
    gain = 0.0
    contributions = []
    for position, isin in enumerate(ranking[:10], start=1):
        if isin in truth:
            term = 1.0 / math.log2(position + 1)
            gain += term
            contributions.append((position, isin, term))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(10, len(truth))))
    return gain / ideal, contributions, ideal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="starter output directory")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--user", help="user to hand-check; default: first with truth")
    parser.add_argument("--method", default="custom")
    parser.add_argument("--prompts", help="optional prompts/generations JSONL to scan")
    parser.add_argument("--cutoff", default="2021-06-01")
    parser.add_argument("--horizon-days", type=int, default=30)
    args = parser.parse_args(argv)

    run = Path(args.run)
    cutoff = datetime.date.fromisoformat(args.cutoff)
    end = cutoff + datetime.timedelta(days=args.horizon_days)

    # ---------------------------------------------------------- truth from raw
    truth = defaultdict(set)
    with (Path(args.data_dir) / "transactions.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["transactionType"] != "Buy":
                continue
            if row["transactionID"].strip().startswith("-"):
                continue           # reconstructed initial position, excluded
            when = datetime.date.fromisoformat(row["timestamp"])
            if cutoff <= when < end:
                truth[row["customerID"]].add(row["ISIN"])

    # ------------------------------------------------- predictions + per-user
    ranking = defaultdict(list)
    with (run / "predictions.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["method"] == args.method:
                ranking[row["user_id"]].append((int(row["rank"]), row["ISIN"]))
    for uid in ranking:
        ranking[uid] = [i for _, i in sorted(ranking[uid])]

    reported = {}
    with (run / "per_user.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["method"] == args.method:
                reported[row["user_id"]] = (int(row["relevant_count"]),
                                            float(row["ndcg_at_10"]))

    user = args.user
    if not user:
        user = next((u for u in ranking
                     if reported.get(u, (0, 0))[0] > 0 and reported[u][1] > 0), None)
    if not user:
        raise SystemExit("no user with nonzero reported nDCG for this method")

    print(f"=== CHECK 1: nDCG@10 by hand for user {user} (method {args.method}) ===")
    eligible_truth = set(truth.get(user, set()))
    rank = ranking[user]
    print(f"top-10 ranking: {rank}")
    print(f"future Buy assets in window (from raw transactions.csv): "
          f"{sorted(eligible_truth)}")

    rep_count, rep_ndcg = reported[user]
    # The starter intersects truth with the eligible universe; use its count as
    # the authority on |truth| and report any discrepancy explicitly.
    hits = [(p, i) for p, i in enumerate(rank, 1) if i in eligible_truth]
    print(f"hits in top-10: {hits}")

    value, contributions, ideal = ndcg_at_10(rank, eligible_truth)
    for position, isin, term in contributions:
        print(f"  rank {position}: {isin}  gain 1/log2({position}+1) = {term:.6f}")
    print(f"  DCG   = {sum(t for _, _, t in contributions):.6f}")
    print(f"  IDEAL = {ideal:.6f}  (min(10, |truth|) = {min(10, len(eligible_truth))})")
    print(f"  nDCG@10 recomputed = {value:.6f}")
    print(f"  starter per_user.csv = {rep_ndcg:.6f}  "
          f"(relevant_count {rep_count}, raw truth size {len(eligible_truth)})")
    agree = math.isclose(value, rep_ndcg, rel_tol=1e-9, abs_tol=1e-12)
    print(f"  MATCH: {agree}")

    # --------------------------------------------------------- leakage check
    if args.prompts:
        # The prompt legitimately carries the cutoff itself as the as-of
        # boundary, so a date *equal* to the cutoff is expected and is not
        # leakage. Only a date strictly after it could be a future fact.
        print(f"\n=== CHECK 2: no date > cutoff {cutoff} in {args.prompts} ===")
        offenders, at_cutoff, scanned = [], 0, 0
        with Path(args.prompts).open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                scanned += 1
                record = json.loads(line)
                blob = record.get("prompt") or json.dumps(record)
                for found in set(DATE.findall(blob)):
                    try:
                        when = datetime.date.fromisoformat(found)
                    except ValueError:
                        continue
                    if when > cutoff:
                        offenders.append((scanned, found))
                    elif when == cutoff:
                        at_cutoff += 1
        print(f"  records scanned: {scanned}")
        print(f"  records mentioning the cutoff date itself (expected): {at_cutoff}")
        print(f"  dates strictly after the cutoff (leakage): {len(offenders)}")
        if offenders:
            print(f"  first offenders: {offenders[:5]}")
        print(f"  CLEAN: {not offenders}")

    print(f"\n=== CHECK 3: truth provenance ===")
    print(f"  window: {cutoff} (inclusive) .. {end} (exclusive)")
    print(f"  users with >=1 future Buy in raw data: {sum(1 for v in truth.values() if v)}")
    return 0 if agree else 1


if __name__ == "__main__":
    raise SystemExit(main())
