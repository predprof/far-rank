#!/usr/bin/env python3
"""Generate the report's result tables from the results files.

Report numbers are emitted from the measured artefacts rather than retyped, so a
table cannot drift from what the code produced. Metrics are printed to four
decimal places, as the homework requires.

Usage:
    python3 tables.py --d300 outputs/d300/results.json \
        --d1000 outputs/d1000/results.json \
        --retrieval outputs/retrieval_1000.json > outputs/TABLES.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASELINES = ('popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell')
LABEL = {
    'popularity': 'Popularity',
    'cbf': 'CBF',
    'latest_buy': 'Latest buy',
    'latest_buy_or_sell': 'Latest buy or sell',
    'initial_order_first10': 'initial order (matched control)',
    'llm_provided': 'LLM, provided strict policy',
    'llm_repaired': 'LLM, repair-and-complete',
    'bpr_mf': 'BPR-MF',
    'ceiling': 'oracle ceiling',
}
POOL_LABEL = {
    'cbf': 'P_cbf (provided retrieval)',
    'pop': 'P_pop (popularity only)',
    'own_then_cbf': 'P_own+cbf',
    'own_then_pop': 'P_own+pop (recency-seeded)',
    'own_freq_then_pop': 'P_own+pop (frequency-seeded)',
}


def fmt(value):
    return f'{value:.4f}' if value is not None else 'n/a'


def metric_row(metrics, key):
    row = metrics[key]
    return row['users'], row['nonempty_truth_users'], row['ndcg_at_10'], \
        row['ndcg_at_10_nonempty']


def baseline_table(results, title):
    metrics = results['metrics']
    lines = [f'**{title}** (cohort {results["cohort_size"]}, '
             f'{metrics[BASELINES[0]]["nonempty_truth_users"]} with future purchases)',
             '',
             '| Baseline | nDCG@10 all users | nDCG@10 nonempty |',
             '|---|---:|---:|']
    for name in BASELINES:
        _u, _ne, allv, nev = metric_row(metrics, name)
        lines.append(f'| {LABEL[name]} | {fmt(allv)} | {fmt(nev)} |')
    return '\n'.join(lines)


def pool_table(results, pools, title, rankers):
    metrics = results['metrics']
    lines = [f'**{title}**', '',
             '| Candidate pool | ' + ' | '.join(LABEL[r] for r in rankers) + ' |',
             '|---|' + '---:|' * len(rankers)]
    for pool in pools:
        cells = []
        for ranker in rankers:
            key = f'{pool}/{ranker}'
            cells.append(fmt(metrics[key]['ndcg_at_10']) if key in metrics else '—')
        lines.append(f'| {POOL_LABEL.get(pool, pool)} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def pool_table_nonempty(results, pools, title, rankers):
    metrics = results['metrics']
    lines = [f'**{title}**', '',
             '| Candidate pool | ' + ' | '.join(LABEL[r] for r in rankers) + ' |',
             '|---|' + '---:|' * len(rankers)]
    for pool in pools:
        cells = []
        for ranker in rankers:
            key = f'{pool}/{ranker}'
            cells.append(fmt(metrics[key]['ndcg_at_10_nonempty'])
                         if key in metrics else '—')
        lines.append(f'| {POOL_LABEL.get(pool, pool)} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def reachability_table(retrieval, pools):
    lines = [f'**Reachability and ceiling** (cohort {retrieval["users"]}, '
             f'{retrieval["nonempty_truth_users"]} with future purchases, '
             f'{retrieval["total_truth_items_nonempty"]} relevant assets, '
             f'pool size {retrieval["candidate_count"]})',
             '',
             '| Candidate pool | relevant in pool | share | users with none reachable '
             '| ceiling all | ceiling nonempty |',
             '|---|---:|---:|---:|---:|---:|']
    total = retrieval['total_truth_items_nonempty']
    ne = retrieval['nonempty_truth_users']
    for pool in pools:
        row = retrieval['pools'][pool]
        lines.append(
            f'| {POOL_LABEL.get(pool, pool)} | {row["reachable_truth_items"]} of {total} '
            f'| {row["reachable_share"]*100:.1f}% '
            f'| {row["nonempty_users_with_zero_reachable"]} of {ne} '
            f'| {fmt(row["ceiling_ndcg10_all_users"])} '
            f'| {fmt(row["ceiling_ndcg10_nonempty"])} |')
    return '\n'.join(lines)


def failure_table(results, pools):
    stats = results.get('repair_stats', {})
    lines = ['**Invalid outputs, repairs and fallbacks**', '',
             '| Candidate pool | generated | invalid under provided policy | parse '
             'repaired | salvageable | nothing salvageable | policies differ | '
             'LLM equals control |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for pool in pools:
        s = stats.get(pool)
        if not s:
            continue
        lines.append(
            f'| {POOL_LABEL.get(pool, pool)} | {s["users_generated"]} '
            f'| {s["provided_invalid"]} | {s["parse_repaired"]} '
            f'| {s["salvaged_any"]} | {s["salvaged_none"]} '
            f'| {s["policies_differ"]} | {s["llm_equals_control"]} |')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--d300', type=Path)
    parser.add_argument('--d1000', type=Path)
    parser.add_argument('--retrieval', type=Path)
    parser.add_argument('--retrieval-300', type=Path)
    args = parser.parse_args(argv)

    blocks = []
    if args.d1000 and args.d1000.exists():
        r = json.loads(args.d1000.read_text())
        pools = sorted({m.split('/')[0] for m in r['metrics'] if '/' in m})
        order = [p for p in ('cbf', 'pop', 'own_then_cbf', 'own_then_pop',
                             'own_freq_then_pop') if p in pools]
        blocks.append('## D1000 (large development sample)\n')
        blocks.append(baseline_table(r, 'Four required baselines, D1000'))
        blocks.append(pool_table(r, order, 'Pool x ranker, nDCG@10 all users, D1000',
                                 ['initial_order_first10', 'bpr_mf', 'ceiling']))
        blocks.append(pool_table_nonempty(
            r, order, 'Pool x ranker, nDCG@10 nonempty truth, D1000',
            ['initial_order_first10', 'bpr_mf', 'ceiling']))
        blocks.append(f'protocol_hash `{r["protocol_hash"][:12]}`, '
                      f'eligible assets {r["eligible_assets"]}, '
                      f'cutoff {r["cutoff"]}, window ends {r["end_exclusive"]}')
    if args.retrieval and args.retrieval.exists():
        rt = json.loads(args.retrieval.read_text())
        order = [p for p in ('cbf', 'pop', 'own_then_cbf', 'own_then_pop',
                             'own_freq_then_pop') if p in rt['pools']]
        blocks.append(reachability_table(rt, order))
    if args.d300 and args.d300.exists():
        r = json.loads(args.d300.read_text())
        pools = sorted({m.split('/')[0] for m in r['metrics'] if '/' in m})
        order = [p for p in ('cbf', 'pop', 'own_freq_then_pop') if p in pools]
        blocks.append('## D300 (LLM cohort)\n')
        blocks.append(baseline_table(r, 'Four required baselines, D300'))
        blocks.append(pool_table(
            r, order, 'Pool x ranker, nDCG@10 all users, D300',
            ['initial_order_first10', 'llm_provided', 'llm_repaired', 'bpr_mf',
             'ceiling']))
        blocks.append(pool_table_nonempty(
            r, order, 'Pool x ranker, nDCG@10 nonempty truth, D300',
            ['initial_order_first10', 'llm_provided', 'llm_repaired', 'bpr_mf',
             'ceiling']))
        blocks.append(failure_table(r, order))
        blocks.append(f'protocol_hash `{r["protocol_hash"][:12]}`, '
                      f'eligible assets {r["eligible_assets"]}, '
                      f'cutoff {r["cutoff"]}, window ends {r["end_exclusive"]}')
        bpr = r['bpr']
        blocks.append(
            f'BPR-MF: {bpr["n_users"]} users, {bpr["n_items"]} items, '
            f'{bpr["n_pairs"]} pre-cutoff pairs, {bpr["factors"]} factors, '
            f'{bpr["epochs"]} epochs, lr {bpr["lr"]}, L2 {bpr["reg"]}, '
            f'seed {bpr["seed"]}; loss {bpr["losses"][0]:.4f} to '
            f'{bpr["losses"][-1]:.4f}; training {bpr["train_seconds"]:.1f} s')
    print('\n\n'.join(blocks))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
