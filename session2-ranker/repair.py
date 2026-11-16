#!/usr/bin/env python3
"""Repair-and-complete re-ranking policy for FAR-Trans (A08, Session 2).

The provided LLMRank adaptation treats an unparseable generation as a total
loss: `--on-invalid fallback` replaces that user's ranking with `ids[:10]`,
the first ten candidates in CBF order. That is exactly the
`candidate_order_first10` control, so every invalid user contributes a row
that *is* the baseline. With 22 of 100 users invalid in the reference run, a
measured tie between `llmrank` and the control is partly mechanical.

This module implements a different policy over the *same* generations:

    salvage  - take the candidate ISINs the model actually emitted, in the
               order it emitted them, de-duplicated, capped at ten;
    complete - if fewer than ten survive, append the remaining candidates in
               their original CBF order until ten are present.

Nothing is generated, retried or fabricated: the model is called once by the
provided `ranker.py`, and both policies read that one set of raw responses.
The only difference between the two reported methods is this function.

Usage:
    python3 repair.py --responses  <ranker_out>/final_responses.jsonl \
                      --contexts   <starter_out>/contexts.jsonl \
                      --protocol   <starter_out>/protocol.json \
                      --out        repaired_responses.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path


def load_provided_parser(ranker_dir):
    """Import the provided `parse_ranking` so the replicated policy is identical.

    Re-implementing it would risk a subtle divergence, which would make the
    comparison measure my parser rather than my policy.
    """
    directory = Path(ranker_dir).resolve()
    path = directory / "ranker.py"
    spec = importlib.util.spec_from_file_location("provided_ranker", path)
    module = importlib.util.module_from_spec(spec)
    # ranker.py imports its sibling data_adapter by bare name.
    sys.path.insert(0, str(directory))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(directory))
    return module.parse_ranking

# An ISIN is exactly twelve characters: two letters then ten alphanumerics.
ISIN_TOKEN = re.compile(r"\b[A-Z]{2}[A-Z0-9]{10}\b")

RANK_LENGTH = 10


def salvage(raw, candidates):
    """Candidate ISINs the model emitted, in emission order, de-duplicated.

    Only tokens that are members of this user's candidate pool survive, so a
    hallucinated or out-of-set identifier is dropped rather than repaired into
    something it did not say.
    """
    if not isinstance(raw, str):
        return []
    allowed = set(candidates)
    kept, seen = [], set()
    for match in ISIN_TOKEN.finditer(raw):
        token = match.group(0)
        if token in allowed and token not in seen:
            seen.add(token)
            kept.append(token)
    return kept


def repair_and_complete(raw, candidates):
    """Ten unique in-pool ISINs: salvaged model order, then CBF order."""
    if len(candidates) < RANK_LENGTH:
        raise ValueError("candidate pool shorter than ten")
    ranking = salvage(raw, candidates)[:RANK_LENGTH]
    used = set(ranking)
    for cid in candidates:
        if len(ranking) == RANK_LENGTH:
            break
        if cid not in used:
            used.add(cid)
            ranking.append(cid)
    return ranking


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", required=True,
                        help="final_responses.jsonl from the provided ranker run")
    parser.add_argument("--contexts", required=True,
                        help="contexts.jsonl from the matching starter run")
    parser.add_argument("--protocol", required=True,
                        help="protocol.json from the matching starter run")
    parser.add_argument("--out", required=True, help="repaired responses JSONL")
    parser.add_argument("--out-provided",
                        help="replicate the provided strict-parse/fallback policy "
                             "from the same generations, for a matched control")
    parser.add_argument("--ranker-dir",
                        help="far-trans-llm directory, to import the provided parser")
    parser.add_argument("--stats", help="optional JSON summary of the repair")
    args = parser.parse_args(argv)

    provided_parse = None
    if args.out_provided:
        if not args.ranker_dir:
            parser.error("--out-provided requires --ranker-dir")
        provided_parse = load_provided_parser(args.ranker_dir)

    protocol_hash = json.loads(Path(args.protocol).read_text())["protocol_hash"]
    candidates = {c["user_id"]: [a["id"] for a in c["candidate_assets"]]
                  for c in load_jsonl(args.contexts)}

    rows, provided_rows, summary = [], [], {
        "users": 0,
        "provided_invalid": 0,       # users the provided policy discarded
        "salvaged_any": 0,           # users where >=1 model ISIN survived
        "fully_salvaged": 0,         # users where all ten came from the model
        "salvaged_none": 0,          # users where the policies agree exactly
        "policies_differ": 0,        # users whose two rankings are not identical
        "salvaged_counts": {},
        "provided_errors": {},
    }
    for record in load_jsonl(args.responses):
        uid = record["user_id"]
        pool = candidates[uid]
        kept = salvage(record.get("raw_response"), pool)[:RANK_LENGTH]
        ranking = repair_and_complete(record.get("raw_response"), pool)

        summary["users"] += 1
        summary["salvaged_any"] += int(len(kept) > 0)
        summary["fully_salvaged"] += int(len(kept) == RANK_LENGTH)
        summary["salvaged_none"] += int(len(kept) == 0)
        summary["salvaged_counts"][str(len(kept))] = \
            summary["salvaged_counts"].get(str(len(kept)), 0) + 1

        rows.append({"user_id": uid, "stage": "ranker",
                     "protocol_hash": protocol_hash, "ranked_ids": ranking})

        if provided_parse is not None:
            try:
                provided_ranking, _ = provided_parse(record.get("raw_response"), set(pool))
            except (ValueError, TypeError):
                # Exactly the provided policy: discard everything, take CBF order.
                provided_ranking = pool[:RANK_LENGTH]
                summary["provided_invalid"] += 1
                reason = (record.get("error") or "parse failure").split(":")[0]
                summary["provided_errors"][reason] = \
                    summary["provided_errors"].get(reason, 0) + 1
            summary["policies_differ"] += int(provided_ranking != ranking)
            provided_rows.append({"user_id": uid, "stage": "ranker",
                                  "protocol_hash": protocol_hash,
                                  "ranked_ids": provided_ranking})

    with Path(args.out).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.out_provided:
        with Path(args.out_provided).open("w", encoding="utf-8") as stream:
            for row in provided_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.stats:
        Path(args.stats).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
