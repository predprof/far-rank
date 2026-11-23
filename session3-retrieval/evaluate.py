#!/usr/bin/env python3
"""Score every A09 method on every candidate pool under one fixed protocol.

The comparison this assembles:

  four mandated baselines      popularity, cbf, latest_buy, latest_buy_or_sell
                               (pool-independent; computed by the provided
                               starter over the full eligible universe)
  per pool, matched control    initial_order_first10 - the pool's own first ten
  per pool, provided LLM       llm_provided  - provided strict parse, fallback
                               to the pool's initial order on any failure
  per pool, repaired LLM       llm_repaired  - A08 repair-and-complete policy
                               over the identical generations
  per pool, classical ranker   bpr_mf        - one BPR-MF model, trained once,
                               asked to order each pool
  per pool, oracle             ceiling       - every reachable relevant asset
                               first; an upper bound no re-ranker can exceed

Users, cutoff, eligible assets, labels and the metric are identical everywhere;
the candidate pool is the only thing that varies between pool blocks, and within
a pool block the candidates and their initial order are fixed across rankers.

Usage:
    python3 evaluate.py --data-dir data/FAR-Trans --runs outputs/d300 \
        --max-users 300 --out outputs/d300/results.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import mf
import pools as poolmod

HERE = Path(__file__).resolve().parent
LLM_DIR = (HERE / '..' / 'RecSys-LLMs' / 'project' / 'far-trans-llm').resolve()
STARTER_DIR = (HERE / '..' / 'RecSys-LLMs' / 'project' / 'far-trans').resolve()
RANK_LENGTH = 10
BASELINES = ('popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell')


def import_provided(name, directory):
    path = Path(directory) / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'provided_{name}', path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(directory))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(directory))
    return module


def load_jsonl(path):
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            yield json.loads(line)


# --- the A08 repair-and-complete policy, carried forward unchanged -----------
import re

ISIN_TOKEN = re.compile(r'\b[A-Z]{2}[A-Z0-9]{10}\b')


def salvage(raw, candidates):
    """Candidate ISINs the model emitted, in emission order, de-duplicated."""
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
    """Ten unique in-pool ISINs: salvaged model order, then the pool's order."""
    if len(candidates) < RANK_LENGTH:
        raise ValueError('candidate pool shorter than ten')
    ranking = salvage(raw, candidates)[:RANK_LENGTH]
    used = set(ranking)
    for cid in candidates:
        if len(ranking) == RANK_LENGTH:
            break
        if cid not in used:
            used.add(cid)
            ranking.append(cid)
    return ranking


def oracle(starter, pool, relevant):
    reachable = [i for i in pool if i in relevant]
    rest = [i for i in pool if i not in relevant]
    return (reachable + rest)[:RANK_LENGTH]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True,
                        help='directory holding one subdirectory per generated pool')
    parser.add_argument('--cutoff', default='2021-06-01')
    parser.add_argument('--horizon-days', type=int, default=30)
    parser.add_argument('--max-users', type=int, default=300)
    parser.add_argument('--candidate-count', type=int, default=50)
    parser.add_argument('--pools', nargs='*', default=None,
                        help='pools to score; default: every generated one')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--factors', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    starter = import_provided('starter', STARTER_DIR)
    ranker = import_provided('ranker', LLM_DIR)

    timings = {}
    _t0 = time.perf_counter()
    assets, profiles, history, truth, end, excluded = starter.load_data(
        args.data_dir, args.cutoff, args.horizon_days)
    users = starter.select_users(profiles, history, 'warm', args.max_users, None)
    features = {isin: starter.tokens(row) for isin, row in assets.items()}
    pop = starter.popularity_scores(history, assets)
    events = starter.scan_recent_events(args.data_dir, args.cutoff, assets)
    eligible = set(assets)
    timings['data_load_seconds'] = time.perf_counter() - _t0

    # --- the four mandated baselines, from the provided starter --------------
    _t0 = time.perf_counter()
    contexts, baseline = starter.contexts_for(
        users, assets, profiles, history, features, pop, args.cutoff,
        args.candidate_count, False)
    recent = starter.recency_rankings(contexts, pop, events, k=RANK_LENGTH)
    for uid in users:
        baseline[uid].update(recent[uid])
    timings['baseline_seconds'] = time.perf_counter() - _t0

    # --- protocol identity, computed exactly as the starter does -------------
    protocol = {'teaching_defaults_not_official_grading_protocol': True,
                'cutoff': args.cutoff, 'end_exclusive': end,
                'users': users, 'cohort_selection': 'warm',
                'eligible_assets': sorted(assets), 'exclude_seen': False,
                'candidate_count': args.candidate_count, 'k': RANK_LENGTH,
                'empty_truth_policy': 'zero in primary mean; nonempty mean also reported',
                'initial_positions': 'excluded',
                'duplicate_policy': 'exact full rows only',
                'profile_use': 'display only in classical baselines',
                'asset_feature_fields': ['assetCategory', 'assetSubCategory', 'sector',
                                         'industry', 'marketID']}
    protocol['source_sha256'] = {
        name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
        for name in ('transactions.csv', 'asset_information.csv',
                     'customer_information.csv', 'close_prices.csv')}
    protocol_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()).hexdigest()

    # --- candidate pools, from as-of information only ------------------------
    _t0 = time.perf_counter()
    built = {}
    for uid in users:
        counts = history.get(uid, Counter())
        cbf_order, pop_order, _cold = poolmod.orders_for(
            starter, counts, features, pop, eligible)
        recent_buys = [(t, i) for t, i, k in events.get(uid, ()) if k == 'Buy']
        built[uid] = poolmod.build(counts, eligible, cbf_order, pop_order,
                                   recent_buys, args.candidate_count)

    timings['pool_build_seconds'] = time.perf_counter() - _t0

    available = sorted(d.name for d in args.runs.iterdir()
                       if d.is_dir() and (d / 'responses.jsonl').exists()) \
        if args.runs.exists() else []
    wanted = args.pools if args.pools is not None else available

    # --- one BPR-MF model, trained once and reused across every pool ---------
    train_started = time.perf_counter()
    model = mf.fit(args.data_dir, args.cutoff, eligible, factors=args.factors,
                   epochs=args.epochs, seed=args.seed)
    train_seconds = time.perf_counter() - train_started

    timings['bpr_train_seconds'] = train_seconds
    _t0 = time.perf_counter()
    records, repair_stats, coverage, generation_meta = [], {}, {}, {}

    predictions = []

    def add(uid, method, ranking):
        starter.validate_ranking(ranking, contexts[uid]['eligible_asset_ids'])
        relevant = truth.get(uid, set()) & eligible
        records.append({'user_id': uid, 'method': method,
                        'relevant_count': len(relevant),
                        'ndcg_at_10': starter.ndcg10(ranking, relevant)})
        predictions.extend({'method': method, 'user_id': uid, 'rank': rank,
                            'ISIN': isin}
                           for rank, isin in enumerate(ranking, 1))

    for uid in users:
        for name in BASELINES:
            add(uid, name, baseline[uid][name])

    for pool_name in wanted:
        responses = {}
        run_dir = args.runs / pool_name
        if (run_dir / 'responses.jsonl').exists():
            responses = {r['user_id']: r for r in load_jsonl(run_dir / 'responses.jsonl')}
        if (run_dir / 'generation_meta.json').exists():
            generation_meta[pool_name] = json.loads(
                (run_dir / 'generation_meta.json').read_text())
        stats = {'users_generated': 0, 'provided_invalid': 0, 'parse_repaired': 0,
                 'salvaged_any': 0, 'fully_salvaged': 0, 'salvaged_none': 0,
                 'policies_differ': 0, 'llm_equals_control': 0,
                 'salvaged_counts': {}, 'provided_errors': {}}
        cold_users = 0

        for uid in users:
            pool = built[uid][pool_name]
            add(uid, f'{pool_name}/initial_order_first10', pool[:RANK_LENGTH])
            ranked, cold = mf.rank_pool(model, uid, pool)
            cold_users += int(cold)
            add(uid, f'{pool_name}/bpr_mf', ranked)
            relevant = truth.get(uid, set()) & eligible
            add(uid, f'{pool_name}/ceiling', oracle(starter, pool, relevant))

            record = responses.get(uid)
            if record is None:
                continue
            stats['users_generated'] += 1
            raw = record.get('raw_response')
            try:
                provided_ranking, repaired = ranker.parse_ranking(raw, set(pool))
                stats['parse_repaired'] += int(repaired)
            except (ValueError, TypeError) as exc:
                provided_ranking = pool[:RANK_LENGTH]
                stats['provided_invalid'] += 1
                reason = str(exc).split(':')[0]
                stats['provided_errors'][reason] = \
                    stats['provided_errors'].get(reason, 0) + 1
            kept = salvage(raw, pool)[:RANK_LENGTH]
            repaired_ranking = repair_and_complete(raw, pool)
            stats['salvaged_any'] += int(len(kept) > 0)
            stats['fully_salvaged'] += int(len(kept) == RANK_LENGTH)
            stats['salvaged_none'] += int(len(kept) == 0)
            key = str(len(kept))
            stats['salvaged_counts'][key] = stats['salvaged_counts'].get(key, 0) + 1
            stats['policies_differ'] += int(provided_ranking != repaired_ranking)
            stats['llm_equals_control'] += int(repaired_ranking == pool[:RANK_LENGTH])
            add(uid, f'{pool_name}/llm_provided', provided_ranking)
            add(uid, f'{pool_name}/llm_repaired', repaired_ranking)

        stats['bpr_cold_users'] = cold_users
        repair_stats[pool_name] = stats
        coverage[pool_name] = {
            'users_in_cohort': len(users),
            'users_generated': stats['users_generated'],
            'generation_complete': stats['users_generated'] == len(users),
        }

    timings['scoring_seconds'] = time.perf_counter() - _t0
    methods = sorted({r['method'] for r in records})
    metrics = {m: starter.aggregate([r for r in records if r['method'] == m])
               for m in methods}

    out = {
        'protocol_hash': protocol_hash,
        'development_results_not_official_benchmark': True,
        'cutoff': args.cutoff, 'end_exclusive': end,
        'horizon_days': args.horizon_days,
        'cohort_size': len(users), 'candidate_count': args.candidate_count,
        'eligible_assets': len(assets),
        'excluded_rows': excluded,
        'metrics': metrics,
        'repair_stats': repair_stats,
        'coverage': coverage,
        'generation_meta': generation_meta,
        'timings': timings,
        'bpr': {'n_users': model['n_users'], 'n_items': model['n_items'],
                'n_pairs': model['n_pairs'], 'factors': model['factors'],
                'epochs': model['epochs'], 'lr': model['lr'], 'reg': model['reg'],
                'seed': model['seed'], 'losses': model['losses'],
                'train_seconds': train_seconds},
        'note': 'Offline purchase relevance, not profitability or asset suitability.',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')

    per_user_path = args.out.with_name(args.out.stem + '_per_user.csv')
    with per_user_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    predictions_path = args.out.with_name(args.out.stem + '_predictions.csv')
    with predictions_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', 'user_id', 'rank', 'ISIN'])
        writer.writeheader()
        writer.writerows(predictions)

    width = max(len(m) for m in methods)
    print(f'\nprotocol_hash {protocol_hash[:12]}  cohort {len(users)}  '
          f'(development results, not official benchmark scores)')
    print(f"{'method'.ljust(width)}  users  nonempty  nDCG@10(all)  nDCG@10(nonempty)")
    for m in methods:
        row = metrics[m]
        ne = row['ndcg_at_10_nonempty']
        print(f"{m.ljust(width)}  {row['users']:5d}  {row['nonempty_truth_users']:8d}  "
              f"{row['ndcg_at_10']:12.4f}  "
              f"{(f'{ne:.4f}' if ne is not None else 'n/a'):>17}")
    print(f'\nper-user rows written to {per_user_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
