#!/usr/bin/env python3
"""Verify saved ranks, fallbacks, and June metrics from a reranker run."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--session-dir', required=True, type=Path)
root = parser.parse_args().session_dir
run = root / 'zero_shot_greedy'
baseline = json.loads((root / 'baseline/metrics.json').read_text())

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))

spec = importlib.util.spec_from_file_location('ranker', Path(__file__).with_name('run_zero_shot.py'))
ranker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ranker)
inputs = read_jsonl(root / 'rerank_inputs.jsonl')
truth = read_jsonl(root / 'evaluation/truth.jsonl')
llm = read_jsonl(run / 'llm_responses.jsonl')
control = read_jsonl(run / 'repeat_first_responses.jsonl')
hash_value = baseline['protocol_hash']
settings = json.loads((run / 'settings.json').read_text())
assert settings['protocol_hash'] == hash_value
assert settings['inputs_sha256'] == hashlib.sha256((root / 'rerank_inputs.jsonl').read_bytes()).hexdigest()
assert settings['runner_sha256'] == hashlib.sha256((Path(__file__).with_name('run_zero_shot.py')).read_bytes()).hexdigest()
assert settings['do_sample'] is settings['use_model_generation_defaults'] is False
official_protocol = (root / 'baseline/protocol.json').read_bytes()
bpr = json.loads((root / 'bpr/metrics.json').read_text())
assert (root / 'bpr/protocol.json').read_bytes() == official_protocol
assert bpr['protocol_hash'] == hash_value and not bpr['custom_failures']
bpr_per_user = {
    row['user_id']: float(row['ndcg_at_10'])
    for row in read_csv(root / 'bpr/per_user.csv') if row['method'] == 'custom'
}
verified = {'bpr': (bpr['methods']['custom'], bpr_per_user)}
for label in ('llm', 'repeat_first'):
    scored = run / label
    assert (scored / 'protocol.json').read_bytes() == official_protocol
    metrics = json.loads((scored / 'metrics.json').read_text())
    assert metrics['protocol_hash'] == hash_value and not metrics['custom_failures']
    assert {k: v for k, v in metrics['methods'].items() if k != 'custom'} == baseline['methods']
    per_user = {row['user_id']: float(row['ndcg_at_10']) for row in read_csv(scored / 'per_user.csv')
                if row['method'] == 'custom'}
    verified[label] = (metrics['methods']['custom'], per_user)

def ndcg(ranking, labels):
    if not labels:
        return 0.0
    dcg = math.fsum(1 / math.log2(i + 2) for i, asset in enumerate(ranking) if asset in labels)
    idcg = math.fsum(1 / math.log2(i + 2) for i in range(min(10, len(labels))))
    return dcg / idcg

details = []
fallback_reasons = Counter()
for index, (features, labels, model, repeat) in enumerate(zip(inputs, truth, llm, control, strict=True), 1):
    uid = features['user_id']
    assert uid == labels['user_id'] == model['user_id'] == repeat['user_id']
    assert features['protocol_hash'] == model['protocol_hash'] == repeat['protocol_hash'] == hash_value
    candidates = features['candidates']
    allowed = {item['asset_id'] for item in candidates}
    assert len(allowed) == len(candidates) == 50
    generation = json.loads((run / 'generations' / f'{index:03d}.json').read_text())
    assert generation['user_id'] == uid and generation['prompt'] == ranker.prompt_for(candidates)
    assert uid not in generation['prompt'] and all(asset not in generation['prompt'] for asset in allowed)
    try:
        validated = ranker.parse_ranking(generation['raw_response'], candidates)
    except ValueError as error:
        validated = features['bpr_top10']
        assert generation['fallback_reason'] == str(error)
        fallback_reasons[str(error)] += 1
    else:
        assert generation['fallback_reason'] is None
    assert validated == generation['ranked_ids'] == model['ranked_ids']
    assert features['repeat_first_top10'] == repeat['ranked_ids']
    labels_set = set(labels['relevant_asset_ids'])
    algorithms = {'bpr': features['bpr_top10'], 'llm': model['ranked_ids'], 'repeat_first': repeat['ranked_ids']}
    for order in algorithms.values():
        assert len(order) == len(set(order)) == 10 and set(order) <= allowed
    computed = {name: ndcg(order, labels_set) for name, order in algorithms.items()}
    assert all(math.isclose(computed[name], scores[uid], abs_tol=1e-15)
               for name, (_, scores) in verified.items())
    details.append({
        'scores': computed, 'labelled': bool(labels_set), 'reachable_pairs': len(labels_set & allowed),
        'pairs': len(labels_set), 'sparse': 1 <= labels['pre_cutoff_buy_count'] <= 3,
        'llm_changed_order': model['ranked_ids'] != features['bpr_top10'],
    })
assert len(details) == 100
summary = json.loads((run / 'generation_summary.json').read_text())
assert summary['fallback_reasons'] == fallback_reasons and summary['users'] == len(details)
assert summary['valid_responses'] == len(details) - sum(fallback_reasons.values())

cohort_metrics = {}
for group, rows in (
    ('all', details), ('nonempty', [r for r in details if r['labelled']]),
    ('sparse', [r for r in details if r['sparse']]),
    ('sparse_nonempty', [r for r in details if r['sparse'] and r['labelled']]),
):
    scores = {method: statistics.mean(row['scores'][method] for row in rows)
              for method in ('bpr', 'llm', 'repeat_first')}
    scores.update(llm_delta_vs_bpr=scores['llm'] - scores['bpr'])
    cohort_metrics[group] = {'users': len(rows), 'users_with_purchases': sum(r['labelled'] for r in rows),
                             'scores': scores}
summary.update({
    'protocol_hash': hash_value, 'cohorts': cohort_metrics,
    'llm_changes_to_bpr_top10': sum(row['llm_changed_order'] for row in details),
    'reachable_relevant_pairs': sum(row['reachable_pairs'] for row in details),
    'relevant_pairs': sum(row['pairs'] for row in details),
})
(run / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
