#!/usr/bin/env python3
"""Export aggregate-only result files for the public team repository (A09).

The provided starter is explicit that a distribution must contain
"source/tests/documentation only, not models, data, caches, logs, raw customer
IDs, or user-level results". Several of my local artefacts are user-level by
construction -- `*_per_user.csv`, `*_predictions.csv`, `pools.json`,
`responses.jsonl`, `prompts.jsonl`, `generations.jsonl` and the generation
cache all key on an anonymised customer ID -- so none of them may be committed.

This script writes the publishable subset: aggregate metrics, pool-level
reachability, repair/fallback counts, timings and the protocol hash, with every
per-user structure removed. It then asserts that nothing customer-ID-shaped
survives, and refuses to write a file that still contains one.

Usage:
    python3 publish.py --results outputs/d1000/results.json \
        --retrieval outputs/retrieval_1000.json --out-dir ../far-rank/session3-retrieval/results
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# FAR-Trans anonymised customer IDs are 20 uppercase hex characters.
CUSTOMER_ID = re.compile(r'\b[0-9A-F]{20}\b')


def assert_no_customer_ids(text, label):
    found = sorted(set(CUSTOMER_ID.findall(text)))
    if found:
        raise SystemExit(
            f'refusing to publish {label}: {len(found)} customer-ID-shaped '
            f'tokens remain (first: {found[0]})')


def write(path, payload, label):
    text = json.dumps(payload, indent=2, sort_keys=True) + '\n'
    assert_no_customer_ids(text, label)
    path.write_text(text, encoding='utf-8')
    print(f'  wrote {path} ({len(text)} bytes)')


def strip_retrieval(payload):
    """Pool-level reachability and ceilings; the per-user list is dropped."""
    return {k: v for k, v in payload.items() if k != 'per_user'}


def strip_results(payload):
    """Aggregate metrics only. `metrics` is keyed by method, never by user."""
    out = {k: v for k, v in payload.items() if k != 'per_user'}
    # generation_meta records cache counts and wall time, not user content.
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, action='append', default=[])
    parser.add_argument('--retrieval', type=Path, action='append', default=[])
    parser.add_argument('--tables', type=Path)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.results:
        if not path.exists():
            print(f'  skipping missing {path}')
            continue
        payload = json.loads(path.read_text())
        name = f"metrics_d{payload['cohort_size']}.json"
        write(args.out_dir / name, strip_results(payload), name)
    for path in args.retrieval:
        if not path.exists():
            print(f'  skipping missing {path}')
            continue
        payload = json.loads(path.read_text())
        name = f"retrieval_d{payload['users']}_aggregate.json"
        write(args.out_dir / name, strip_retrieval(payload), name)
    if args.tables and args.tables.exists():
        text = args.tables.read_text(encoding='utf-8')
        assert_no_customer_ids(text, 'TABLES.md')
        (args.out_dir / 'TABLES.md').write_text(text, encoding='utf-8')
        print(f"  wrote {args.out_dir / 'TABLES.md'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
