#!/usr/bin/env python3
"""Assemble the A08 results table from completed starter runs.

Every number in report.md is produced here, so the table cannot drift from the
measurements. Each input is a starter `metrics.json`; the `custom` row of a run
is relabelled with the name given on the command line.

Usage:
    python3 compare.py --baseline outputs/baseline/metrics.json \
                       --run provided_policy=outputs/provided/metrics.json \
                       --run repair_complete=outputs/repaired/metrics.json \
                       --out outputs/RESULTS.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASELINE_ORDER = ["popularity", "cbf", "latest_buy", "latest_buy_or_sell"]


def load(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run", action="append", default=[],
                        metavar="LABEL=metrics.json")
    parser.add_argument("--ceiling", help="ceiling.json from ceiling.py")
    parser.add_argument("--repair-stats", help="stats JSON from repair.py")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    base = load(args.baseline)
    hashes = {"baseline": base["protocol_hash"]}
    rows = []
    for name in BASELINE_ORDER:
        m = base["methods"][name]
        rows.append((name, m))

    for spec in args.run:
        label, _, path = spec.partition("=")
        data = load(path)
        hashes[label] = data["protocol_hash"]
        if "custom" not in data["methods"]:
            raise SystemExit(f"{path} has no custom row")
        rows.append((label, data["methods"]["custom"]))

    distinct = set(hashes.values())
    lines = ["# A08 results", ""]
    lines.append(f"Protocol hash (all runs): `{sorted(distinct)[0]}`" if len(distinct) == 1
                 else "**PROTOCOL HASH MISMATCH** — runs are not comparable: "
                      + json.dumps(hashes, indent=2))
    lines += ["", "| Method | Users | Users with future purchases | "
              "nDCG@10: all users | nDCG@10: nonempty truth |",
              "|---|---:|---:|---:|---:|"]
    for name, m in rows:
        lines.append(f"| {name} | {m['users']} | {m['nonempty_truth_users']} | "
                     f"{m['ndcg_at_10']:.4f} | {m['ndcg_at_10_nonempty']:.4f} |")

    if args.ceiling:
        c = load(args.ceiling)
        lines += ["", "## Re-ranker headroom (oracle over the same candidate pool)", "",
                  f"- ceiling nDCG@10, all users: **{c['ceiling_ndcg10_all_users']:.4f}**",
                  f"- ceiling nDCG@10, nonempty truth: "
                  f"**{c['ceiling_ndcg10_nonempty']:.4f}**",
                  f"- relevant items present in the candidate pool: "
                  f"**{c['reachable_truth_items_nonempty']} of "
                  f"{c['total_truth_items_nonempty']}**",
                  f"- labelled users with zero reachable truth: "
                  f"**{c['nonempty_users_with_zero_reachable_truth']} of "
                  f"{c['nonempty_truth_users']}**"]

    if args.repair_stats:
        s = load(args.repair_stats)
        lines += ["", "## Policy divergence on identical generations", "",
                  f"- users: **{s['users']}**",
                  f"- invalid under the provided strict policy: "
                  f"**{s['provided_invalid']}**",
                  f"- users with at least one salvageable candidate ISIN: "
                  f"**{s['salvaged_any']}**",
                  f"- users with a complete ten-ISIN model ranking: "
                  f"**{s['fully_salvaged']}**",
                  f"- users where nothing was salvageable (policies identical): "
                  f"**{s['salvaged_none']}**",
                  f"- users whose two rankings differ: **{s['policies_differ']}**",
                  "", f"Salvaged-count distribution: `{json.dumps(s['salvaged_counts'])}`"]
        if s.get("provided_errors"):
            lines.append(f"Provided-policy failure reasons: "
                         f"`{json.dumps(s['provided_errors'])}`")

    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    if len(distinct) != 1:
        raise SystemExit("protocol hashes differ; refusing to treat runs as comparable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
