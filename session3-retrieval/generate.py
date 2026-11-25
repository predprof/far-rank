#!/usr/bin/env python3
"""Run the provided LLMRank adaptation over a chosen candidate pool (A09).

The provided `ranker.py` always re-ranks the CBF pool that `contexts_for` builds.
My A09 intervention is the retrieval stage, so this driver swaps the candidate
pool and leaves the provided prompt builder, parser and runtime untouched:

  * `ranker.build_prompt`  - imported, not reimplemented, so prompt wording,
    field order and chronology notes are the provided ones;
  * `ranker.parse_ranking` - imported, so the strict policy I compare against is
    byte-identical in semantics to the provided one;
  * `runtime.LocalGenerator/LoggedGenerator` - imported, so generation settings,
    the 8192-token budget, the immutable log and the resume cache are unchanged.

Only `context['candidate_assets']` differs between pools, which is exactly the
component under test. Generation is greedy and cached, so re-running after an
interruption reuses completed work rather than regenerating it.

Usage:
    python3 generate.py --pool own_freq_then_pop --max-users 300 \
        --data-dir data/FAR-Trans --output-dir outputs/d300/own_freq_then_pop
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pools as poolmod

HERE = Path(__file__).resolve().parent
LLM_DIR = (HERE / '..' / 'RecSys-LLMs' / 'project' / 'far-trans-llm').resolve()
STARTER_DIR = (HERE / '..' / 'RecSys-LLMs' / 'project' / 'far-trans').resolve()


def import_provided(name, directory):
    """Import a provided module, with its own directory importable.

    `ranker.py` imports its sibling `data_adapter` by bare name, so the
    directory must be on sys.path while the module executes.
    """
    path = Path(directory) / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'provided_{name}', path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(directory))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(directory))
    return module


def asset_record(starter, isin, assets, features, cutoff):
    return {'id': isin, 'name': assets[isin]['assetName'],
            'attributes': features[isin],
            'metadata_timestamp': assets[isin]['timestamp']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', required=True, choices=list(poolmod.NAMES))
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cutoff', default='2021-06-01')
    parser.add_argument('--horizon-days', type=int, default=30)
    parser.add_argument('--max-users', type=int, default=300)
    parser.add_argument('--candidate-count', type=int, default=50)
    parser.add_argument('--history-events', type=int, default=15)
    parser.add_argument('--model', default='Qwen/Qwen3-1.7B')
    parser.add_argument('--revision',
                        default='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--prepare-only', action='store_true',
                        help='write prompts and pool manifest; call no model')
    args = parser.parse_args(argv)

    starter = import_provided('starter', STARTER_DIR)
    ranker = import_provided('ranker', LLM_DIR)

    assets, profiles, history, truth, end, excluded = starter.load_data(
        args.data_dir, args.cutoff, args.horizon_days)
    users = starter.select_users(profiles, history, 'warm', args.max_users, None)
    features = {isin: starter.tokens(row) for isin, row in assets.items()}
    pop = starter.popularity_scores(history, assets)
    events = starter.scan_recent_events(args.data_dir, args.cutoff, assets)
    eligible = set(assets)
    eligible_sorted = sorted(eligible)

    # ---- pools and prompts, from as-of information only -------------------
    prompts, pool_ids = [], {}
    for uid in users:
        counts = history.get(uid, Counter())
        cbf_order, pop_order, _cold = poolmod.orders_for(
            starter, counts, features, pop, eligible)
        recent_buys = [(t, i) for t, i, kind in events.get(uid, ()) if kind == 'Buy']
        pool = poolmod.build(counts, eligible, cbf_order, pop_order,
                             recent_buys, args.candidate_count)[args.pool]
        if len(pool) < 10:
            raise ValueError(f'{uid}: pool {args.pool} shorter than ten')
        pool_ids[uid] = pool
        context = {
            'user_id': uid, 'cutoff': args.cutoff,
            'profile': starter.profile_context(profiles[uid]),
            'history': [dict(asset_record(starter, i, assets, features, args.cutoff),
                             buy_count=n)
                        for i, n in sorted(counts.items(), key=lambda p: (-p[1], p[0]))],
            'candidate_assets': [asset_record(starter, i, assets, features, args.cutoff)
                                 for i in pool],
            'eligible_asset_ids': eligible_sorted,
        }
        prompts.append((uid, ranker.build_prompt(
            context, assets, history.get(uid, {}), events.get(uid, ()),
            args.history_events, None)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'pools.json').write_text(
        json.dumps({'pool': args.pool, 'candidate_count': args.candidate_count,
                    'users': users, 'pool_ids': pool_ids}, indent=2) + '\n',
        encoding='utf-8')

    if args.prepare_only:
        with (args.output_dir / 'prompts.jsonl').open('w', encoding='utf-8') as fh:
            for uid, prompt in prompts:
                fh.write(json.dumps({'user_id': uid, 'pool': args.pool,
                                     'stage': 'prepare_only_no_inference',
                                     'prompt': prompt}, ensure_ascii=False) + '\n')
        print(f'prepared {len(prompts)} prompts for pool {args.pool}; no model called')
        return 0

    # ---- generation, through the provided runtime --------------------------
    runtime = import_provided('runtime', LLM_DIR)
    generator = runtime.LoggedGenerator(
        runtime.LocalGenerator(args.model, args.revision, args.device, False),
        args.output_dir)

    done = {}
    responses_path = args.output_dir / 'responses.jsonl'
    if responses_path.exists():
        for line in responses_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                done[row['user_id']] = row

    started = time.perf_counter()
    with responses_path.open('a', encoding='utf-8') as fh:
        for index, (uid, prompt) in enumerate(prompts, 1):
            if uid in done:
                continue
            try:
                result = generator.generate(prompt, max_new_tokens=512)
                record = {'user_id': uid, 'pool': args.pool,
                          'raw_response': result['text'],
                          'input_tokens': result['input_tokens'],
                          'output_tokens': result['output_tokens'],
                          'seconds': result['seconds'],
                          'generation_failed': False}
            except ValueError as exc:
                # The provided runtime enforces a fixed 8192-token budget and
                # the provided prompt builder does not cap `buy_history_counts`,
                # so a customer with a long enough history cannot be generated
                # for at all. That is a property of the provided adaptation, not
                # of a pool: the same users overflow in every pool, so the
                # cohort stays matched. Record the failure and keep the user;
                # dropping them would change the cohort between methods.
                record = {'user_id': uid, 'pool': args.pool,
                          'raw_response': None, 'generation_failed': True,
                          'error': f'{type(exc).__name__}: {exc}'}
                print(f'  {uid}: generation failed, kept with fallback '
                      f'({exc})', flush=True)
            fh.write(json.dumps(record, ensure_ascii=False) + '\n')
            fh.flush()
            if index % 10 == 0 or index == len(prompts):
                elapsed = time.perf_counter() - started
                print(f'  {index}/{len(prompts)} users  {elapsed/60:.1f} min elapsed',
                      flush=True)

    (args.output_dir / 'generation_meta.json').write_text(
        json.dumps(dict(generator.metadata, pool=args.pool,
                        users=len(prompts), resumed_from=len(done),
                        wall_seconds=time.perf_counter() - started), indent=2) + '\n',
        encoding='utf-8')
    print(f'pool {args.pool}: {len(prompts)} users complete')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
