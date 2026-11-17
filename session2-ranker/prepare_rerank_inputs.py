#!/usr/bin/env python3
"""Freeze scored reranker candidates and verify the original BPR-MF order."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path


SEED = 42
RANK_LENGTH = 10


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_bpr(path):
    spec = importlib.util.spec_from_file_location('team_bpr', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_split_buys(path, cutoff, eligible):
    pairs = defaultdict(set)
    with path.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['timestamp'] >= cutoff or row['ISIN'] not in eligible:
                raise ValueError('Train split contains a future event or an ineligible asset')
            if row['transactionType'] == 'Buy':
                pairs[row['customerID']].add(row['ISIN'])
    return dict(pairs)


def score_context(context, factors, history, reference, protocol):
    user_id = context['user_id']
    pool = [item['id'] for item in context['candidate_assets']]
    if (len(pool) != protocol['candidate_count'] or len(set(pool)) != len(pool)
            or not set(pool) <= set(context['eligible_asset_ids'])):
        raise ValueError('Invalid baseline candidate pool')
    if context['cutoff'] != protocol['cutoff']:
        raise ValueError('Baseline context cutoff differs from protocol')
    if set(history) != {item['id'] for item in context['history']}:
        raise ValueError('Context history differs from the verified train split')
    user_factors, asset_factors, users, items = factors
    values = asset_factors[[items[asset] for asset in pool]] @ user_factors[users[user_id]]
    scores = dict(zip(pool, map(float, values), strict=True))
    if not all(map(math.isfinite, scores.values())):
        raise ValueError('Non-finite BPR score')
    bpr_order = sorted(pool, key=lambda asset: (-scores[asset], asset))
    if bpr_order[:RANK_LENGTH] != reference['ranked_ids']:
        raise ValueError('Recomputed BPR top 10 differs from the verified baseline')

    rounded = {asset: round(score, 8) for asset, score in scores.items()}
    if sorted(pool, key=lambda asset: (-rounded[asset], asset)) == bpr_order:
        scores = rounded
    repeat_first = sorted(pool, key=lambda asset: (asset not in history, -scores[asset], asset))
    presentation = sorted(pool, key=lambda asset: (
        hashlib.sha256(f'{SEED}:{asset}'.encode()).hexdigest(), asset))
    return {
        'user_id': user_id,
        'protocol_hash': protocol['protocol_hash'],
        'candidates': [
            {'asset_id': asset, 'alias': f'A{position:02}', 'score': scores[asset],
             'bought_before': asset in history}
            for position, asset in enumerate(presentation, 1)
        ],
        'bpr_top10': bpr_order[:RANK_LENGTH],
        'repeat_first_top10': repeat_first[:RANK_LENGTH],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-dir', required=True, type=Path)
    parser.add_argument('--raw-data-dir', required=True, type=Path)
    parser.add_argument('--bpr-source', required=True, type=Path)
    args = parser.parse_args()
    root = args.session_dir
    protocol = json.loads((root / 'baseline/protocol.json').read_text(encoding='utf-8'))
    if json.loads((root / 'bpr/protocol.json').read_text(encoding='utf-8')) != protocol:
        raise ValueError('Baseline and BPR protocols differ')
    contexts = read_jsonl(root / 'baseline/contexts.jsonl')
    references = read_jsonl(root / 'bpr_responses.jsonl')
    users = protocol['users']
    if (len(users) != 100 or len(set(users)) != 100 or [row['user_id'] for row in contexts] != users
            or [row['user_id'] for row in references] != users
            or (root / 'evaluation/users.txt').read_text(encoding='utf-8').splitlines() != users):
        raise ValueError('Contexts, responses, and split must use the same 100 users in protocol order')
    if any(row['stage'] != 'ranker' or row['protocol_hash'] != protocol['protocol_hash']
           for row in references):
        raise ValueError('BPR responses do not match the ranker protocol')
    cutoff = protocol['cutoff']
    eligible = set(contexts[0]['eligible_asset_ids'])
    split_assets = root / 'train/eligible_assets.csv'
    with split_assets.open(encoding='utf-8', newline='') as stream:
        if {row['ISIN'] for row in csv.DictReader(stream)} != eligible:
            raise ValueError('Train split catalog differs from the baseline catalog')
    if (eligible != set(protocol['eligible_assets'])
            or any(set(row['eligible_asset_ids']) != eligible for row in contexts)):
        raise ValueError('Baseline asset catalogs differ')

    bpr = load_bpr(args.bpr_source)
    # Upstream samples training users by insertion order, so keep the original
    # CSV order and verify that its filtered pairs equal the reusable split.
    raw = bpr.read_pre_cutoff_buys(args.raw_data_dir, date.fromisoformat(cutoff))
    pairs = {user: assets & eligible for user, assets in raw.items()}
    pairs = {user: assets for user, assets in pairs.items() if assets}
    if pairs != read_split_buys(root / 'train/transactions.csv', cutoff, eligible):
        raise ValueError('Original BPR training pairs differ from the reusable split')
    factors = bpr.train_bpr(pairs, sorted(eligible), factors=32, epochs=10,
                            lr=0.05, reg=0.01, seed=SEED)
    records = [score_context(context, factors, pairs[context['user_id']], reference, protocol)
               for context, reference in zip(contexts, references, strict=True)]
    content = ''.join(json.dumps(record) + '\n' for record in records)
    output = root / 'rerank_inputs.jsonl'
    if output.exists():
        if output.read_text(encoding='utf-8') != content:
            raise ValueError('Existing reranker input artifact differs from the verified reconstruction')
    else:
        temporary = output.with_suffix('.jsonl.tmp')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(output)
    print(f'Verified {len(records)} BPR rankings on their original fixed pools; inputs at {output}')


if __name__ == '__main__':
    main()
