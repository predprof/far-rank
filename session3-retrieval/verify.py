#!/usr/bin/env python3
"""Independent verification of the A09 results.

Each check recomputes something from the raw release or the provided code path
rather than trusting `evaluate.py`'s own aggregation.

  1. nDCG@10 for one named user **under one named method**, recomputed from
     first principles against truth read straight from `transactions.csv`.
     The method is printed in every line of output: in A08 I reported a
     hand-check without naming its method, and the number turned out to belong
     to `popularity` rather than to the re-ranker the section was about. Naming
     it is the fix.
  2. Leakage: no date strictly after the cutoff appears in any generated prompt.
     The cutoff itself is expected, since the prompt carries it as the as-of
     boundary.
  3. Pool integrity: every pool stays inside the eligible universe and inside
     the size budget, and the `cbf` pool reproduces the provided
     `contexts_for` candidate list exactly.
  4. Control identity: each pool's `initial_order_first10` equals that pool's
     own first ten, so the matched control is the pool and nothing else.
  5. Control prompt equivalence: for the `cbf` pool my driver's prompt is
     byte-identical to the one the provided `ranker.py` writes, so the control
     condition is the provided experiment and not a re-implementation of it.

Usage:
    python3 verify.py --results outputs/d300/results.json --data-dir data/FAR-Trans \
        --max-users 300 --method own_freq_then_pop/llm_repaired
"""
from __future__ import annotations

import argparse
import csv
import datetime
import importlib.util
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pools as poolmod

HERE = Path(__file__).resolve().parent
STARTER_DIR = (HERE / '..' / 'RecSys-LLMs' / 'project' / 'far-trans').resolve()
DATE = re.compile(r'\b(\d{4}-\d{2}-\d{2})\b')


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


def ndcg_at_10(ranking, truth):
    """Binary nDCG@10, written out independently of the starter's ndcg10."""
    contributions = []
    for position, isin in enumerate(ranking[:10], start=1):
        if isin in truth:
            contributions.append((position, isin, 1.0 / math.log2(position + 1)))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(10, len(truth))))
    gain = sum(term for _, _, term in contributions)
    return (gain / ideal if ideal else 0.0), contributions, ideal, gain


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--prompt-dir', type=Path,
                        help='directory holding cbf/prompts.jsonl and '
                             'provided_prompts/ranker_prompts.jsonl for check 5')
    parser.add_argument('--cutoff', default='2021-06-01')
    parser.add_argument('--horizon-days', type=int, default=30)
    parser.add_argument('--max-users', type=int, default=300)
    parser.add_argument('--candidate-count', type=int, default=50)
    parser.add_argument('--method', required=True,
                        help='method to hand-check, e.g. pool/llm_repaired')
    parser.add_argument('--user', help='user to hand-check; default: first with a hit')
    args = parser.parse_args(argv)

    cutoff = datetime.date.fromisoformat(args.cutoff)
    end = cutoff + datetime.timedelta(days=args.horizon_days)
    starter = import_provided('starter', STARTER_DIR)
    failures = []

    # ---------------------------------------------------------- truth, raw
    truth = defaultdict(set)
    with (args.data_dir / 'transactions.csv').open(encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            if row['transactionType'] != 'Buy':
                continue
            if row['transactionID'].strip().startswith('-'):
                continue                      # reconstructed initial position
            when = datetime.date.fromisoformat(row['timestamp'])
            if cutoff <= when < end:
                truth[row['customerID']].add(row['ISIN'])

    # ------------------------------------------------- reported per-user rows
    per_user_path = args.results.with_name(args.results.stem + '_per_user.csv')
    reported = {}
    with per_user_path.open(encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            if row['method'] == args.method:
                reported[row['user_id']] = (int(row['relevant_count']),
                                            float(row['ndcg_at_10']))
    if not reported:
        raise SystemExit(f'method {args.method!r} not present in {per_user_path}')

    # ------------------------------------------- the ranking the method output
    assets, profiles, history, _t, _end, _exc = starter.load_data(
        args.data_dir, args.cutoff, args.horizon_days)
    users = starter.select_users(profiles, history, 'warm', args.max_users, None)
    features = {isin: starter.tokens(row) for isin, row in assets.items()}
    pop = starter.popularity_scores(history, assets)
    events = starter.scan_recent_events(args.data_dir, args.cutoff, assets)
    eligible = set(assets)

    built = {}
    for uid in users:
        counts = history.get(uid, Counter())
        cbf_order, pop_order, _cold = poolmod.orders_for(
            starter, counts, features, pop, eligible)
        recent_buys = [(t, i) for t, i, k in events.get(uid, ()) if k == 'Buy']
        built[uid] = poolmod.build(counts, eligible, cbf_order, pop_order,
                                   recent_buys, args.candidate_count)

    rankings = defaultdict(list)
    predictions_path = args.results.with_name(args.results.stem + '_predictions.csv')
    if predictions_path.exists():
        with predictions_path.open(encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                if row['method'] == args.method:
                    rankings[row['user_id']].append((int(row['rank']), row['ISIN']))
        for uid in rankings:
            rankings[uid] = [i for _, i in sorted(rankings[uid])]

    user = args.user
    if not user:
        user = next((u for u in users
                     if reported.get(u, (0, 0.0))[1] > 0), None)
    if not user:
        raise SystemExit('no user with a nonzero reported score for this method')

    print(f'=== CHECK 1: nDCG@10 by hand ===')
    print(f'  method : {args.method}')
    print(f'  user   : {user}')
    ranking = rankings.get(user)
    if ranking is None:
        print('  (no predictions file; recomputing the arithmetic from the '
              'reported hits is not possible, so this check is skipped)')
    else:
        relevant = truth.get(user, set()) & eligible
        print(f'  top-10 ({args.method}): {ranking}')
        print(f'  future Buy assets in window, read from transactions.csv: '
              f'{sorted(relevant)}')
        value, contributions, ideal, gain = ndcg_at_10(ranking, relevant)
        for position, isin, term in contributions:
            print(f'    rank {position:2d}: {isin}  1/log2({position}+1) = {term:.6f}')
        print(f'    DCG   = {gain:.6f}')
        print(f'    IDEAL = {ideal:.6f}  (min(10, |truth|) = {min(10, len(relevant))})')
        print(f'    nDCG@10 recomputed = {value:.6f}')
        rep_count, rep_ndcg = reported[user]
        print(f'    evaluate.py reported = {rep_ndcg:.6f} '
              f'(relevant_count {rep_count}, raw truth size {len(relevant)})')
        agree = math.isclose(value, rep_ndcg, rel_tol=1e-9, abs_tol=1e-12)
        print(f'    MATCH: {agree}')
        if not agree:
            failures.append('check 1: hand-computed nDCG@10 disagrees')

    # ------------------------------------------------------------- leakage
    if args.runs:
        print(f'\n=== CHECK 2: no prompt date strictly after the cutoff {cutoff} ===')
        for prompts in sorted(args.runs.glob('*/prompts.jsonl')) + \
                sorted(args.runs.glob('*/generations.jsonl')):
            offenders, at_cutoff, scanned = [], 0, 0
            for line in prompts.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                scanned += 1
                record = json.loads(line)
                blob = record.get('prompt') or json.dumps(record)
                for found in set(DATE.findall(blob)):
                    try:
                        when = datetime.date.fromisoformat(found)
                    except ValueError:
                        continue
                    if when > cutoff:
                        offenders.append((scanned, found))
                    elif when == cutoff:
                        at_cutoff += 1
            print(f'  {prompts.parent.name}/{prompts.name}: scanned {scanned}, '
                  f'mention cutoff itself {at_cutoff}, after cutoff {len(offenders)}')
            if offenders:
                failures.append(f'check 2: leakage in {prompts}')
                print(f'    offenders: {offenders[:5]}')

    # ------------------------------------------------------ pool integrity
    print(f'\n=== CHECK 3: pool integrity over {len(users)} users ===')
    contexts, _baseline = starter.contexts_for(
        users, assets, profiles, history, features, pop, args.cutoff,
        args.candidate_count, False)
    bad_size = bad_subset = bad_cbf = 0
    for uid in users:
        provided = [a['id'] for a in contexts[uid]['candidate_assets']]
        if built[uid]['cbf'] != provided:
            bad_cbf += 1
        for name, pool in built[uid].items():
            if len(pool) > args.candidate_count or len(pool) != len(set(pool)):
                bad_size += 1
            if not set(pool) <= eligible:
                bad_subset += 1
    print(f'  pools exceeding the size budget or repeating a candidate: {bad_size}')
    print(f'  pools leaving the eligible universe: {bad_subset}')
    print(f'  users where my cbf pool differs from the provided contexts_for: {bad_cbf}')
    for label, count in (('size', bad_size), ('subset', bad_subset), ('cbf', bad_cbf)):
        if count:
            failures.append(f'check 3: {label} violation on {count} users')

    # ---------------------------------------------------- control identity
    print(f'\n=== CHECK 4: each matched control is its own pool prefix ===')
    results = json.loads(args.results.read_text())
    rows = defaultdict(dict)
    with per_user_path.open(encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            rows[row['method']][row['user_id']] = float(row['ndcg_at_10'])
    for method in sorted(results['metrics']):
        if not method.endswith('/initial_order_first10'):
            continue
        pool_name = method.split('/')[0]
        mismatch = sum(
            1 for uid in users
            if not math.isclose(
                rows[method][uid],
                starter.ndcg10(built[uid][pool_name][:10],
                               truth.get(uid, set()) & eligible),
                rel_tol=1e-12, abs_tol=1e-15))
        print(f'  {method}: users disagreeing with the pool prefix = {mismatch}')
        if mismatch:
            failures.append(f'check 4: {method} is not its pool prefix')

    # ------------------------------------------- control prompt equivalence
    if args.prompt_dir:
        prepared = args.prompt_dir / 'cbf' / 'prompts.jsonl'
        provided = args.prompt_dir / 'provided_prompts' / 'ranker_prompts.jsonl'
        if prepared.exists() and provided.exists():
            print('\n=== CHECK 5: control prompts equal the provided ranker\'s ===')
            mine = {json.loads(l)['user_id']: json.loads(l)['prompt']
                    for l in prepared.read_text(encoding='utf-8').splitlines() if l.strip()}
            theirs = {json.loads(l)['user_id']: json.loads(l)['prompt']
                      for l in provided.read_text(encoding='utf-8').splitlines() if l.strip()}
            same = sum(1 for u in theirs if mine.get(u) == theirs[u])
            print(f'  byte-identical prompts: {same} of {len(theirs)}')
            if same != len(theirs):
                failures.append('check 5: control prompt differs from the provided ranker')

    print()
    if failures:
        for failure in failures:
            print(f'FAILED  {failure}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
