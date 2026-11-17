#!/usr/bin/env python3
"""Export the Session 2 homework split with the course starter's data rules."""

import argparse
import csv
import hashlib
import importlib.util
import json
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path


CUTOFF = '2021-06-01'
END_EXCLUSIVE = '2021-07-01'
USER_LIMIT = 100
EVENT_FIELDS = ('customerID', 'ISIN', 'transactionType', 'timestamp')
ASSET_FIELDS = ('ISIN', 'assetName', 'assetShortName', 'assetCategory',
                'assetSubCategory', 'marketID', 'sector', 'industry', 'timestamp')
LABEL_FIELDS = ('customerID', 'ISIN')
SOURCE_FILES = ('transactions.csv', 'asset_information.csv',
                'customer_information.csv', 'close_prices.csv')


def load_starter(path):
    spec = importlib.util.spec_from_file_location('course_starter', path)
    starter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(starter)
    return starter


def write_csv(path, header, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_export(root, starter, assets, history, labels, users, expected_truth, events):
    buy_counts = Counter()
    event_types = Counter()
    past_events = ((uid, asset_id, kind, timestamp)
                   for uid in sorted(events) for timestamp, asset_id, kind in events[uid])
    with (root / 'train/transactions.csv').open(encoding='utf-8', newline='') as stream:
        reader = csv.reader(stream)
        if tuple(next(reader)) != EVENT_FIELDS:
            raise ValueError('Unexpected train transaction schema')
        for uid, asset_id, kind, timestamp in reader:
            if (uid, asset_id, kind, timestamp) != next(past_events, None):
                raise ValueError('Train events differ from the starter')
            if date.fromisoformat(timestamp) >= date.fromisoformat(CUTOFF) or asset_id not in assets:
                raise ValueError('Invalid train event date or asset')
            event_types[kind] += 1
            if kind == 'Buy':
                buy_counts[uid, asset_id] += 1
    expected_buys = Counter({(uid, asset_id): count for uid, counts in history.items()
                             for asset_id, count in counts.items()})
    if next(past_events, None) is not None or buy_counts != expected_buys:
        raise ValueError('Incomplete train export or inconsistent Buy counts')

    catalog = list(starter.rows(root / 'train/eligible_assets.csv'))
    if catalog != [assets[asset_id] for asset_id in sorted(assets)]:
        raise ValueError('Asset export differs from the starter catalog')
    if any(row['timestamp'][:10] >= CUTOFF for row in catalog):
        raise ValueError('Asset metadata at or after the train cutoff')
    saved_labels = [(row['customerID'], row['ISIN'])
                    for row in starter.rows(root / 'evaluation/relevant_assets.csv')]
    if saved_labels != labels:
        raise ValueError('Evaluation labels differ from the starter')
    saved_users = (root / 'evaluation/users.txt').read_text(encoding='utf-8').splitlines()
    if saved_users != users:
        raise ValueError('Evaluation user order differs from the starter')
    with (root / 'evaluation/truth.jsonl').open(encoding='utf-8') as stream:
        if [json.loads(line) for line in stream] != expected_truth:
            raise ValueError('Evaluation truth differs from the starter')
    return event_types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--starter', required=True, type=Path)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()

    starter = load_starter(args.starter)
    horizon_days = (date.fromisoformat(END_EXCLUSIVE) - date.fromisoformat(CUTOFF)).days
    assets, profiles, history, truth, end, excluded = starter.load_data(
        args.data_dir, CUTOFF, horizon_days)
    users = starter.select_users(profiles, history, 'warm', USER_LIMIT, None)
    if end != END_EXCLUSIVE or len(users) != USER_LIMIT:
        raise ValueError('Unexpected evaluation window or user count')
    events = starter.scan_recent_events(args.data_dir, CUTOFF, assets)
    labels = [(uid, asset_id) for uid in sorted(truth) for asset_id in sorted(truth[uid])]
    selected_truth = [{'user_id': uid, 'relevant_asset_ids': sorted(truth.get(uid, ())),
                       'pre_cutoff_buy_count': sum(history[uid].values())} for uid in users]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-splits-', dir=args.output_dir) as staging:
        root = Path(staging)
        (root / 'train').mkdir()
        (root / 'evaluation').mkdir()
        write_csv(root / 'train/transactions.csv', EVENT_FIELDS,
                  ((uid, asset_id, kind, timestamp) for uid in sorted(events)
                   for timestamp, asset_id, kind in events[uid]))
        write_csv(root / 'train/eligible_assets.csv', ASSET_FIELDS,
                  ((assets[asset_id][field] for field in ASSET_FIELDS) for asset_id in sorted(assets)))
        write_csv(root / 'evaluation/relevant_assets.csv', LABEL_FIELDS, labels)
        (root / 'evaluation/users.txt').write_text('\n'.join(users) + '\n', encoding='utf-8')
        with (root / 'evaluation/truth.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
            for record in selected_truth:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')

        event_types = verify_export(root, starter, assets, history, labels, users, selected_truth, events)
        nonempty = sum(bool(row['relevant_asset_ids']) for row in selected_truth)
        files = {
            'train/transactions.csv': ('Pre-cutoff eligible events; preserve retained event counts '
                                       'even when their projected columns match.', sum(event_types.values())),
            'train/eligible_assets.csv': ('Latest metadata before cutoff, one row per eligible asset.', len(assets)),
            'evaluation/relevant_assets.csv': ('All unique eligible June Buy pairs.', len(labels)),
            'evaluation/users.txt': ('Official evaluated customer IDs, one per line in sample order.', len(users)),
            'evaluation/truth.jsonl': ('Per-user truth and pre-cutoff Buy counts in sample order.', len(users)),
        }
        manifest = {
            'cutoff': CUTOFF,
            'train_date_condition': f'timestamp < {CUTOFF}',
            'evaluation_date_condition': f'{CUTOFF} <= timestamp < {END_EXCLUSIVE}',
            'end_exclusive': END_EXCLUSIVE,
            'cohort': 'Course starter deterministic sample of 100 warm users',
            'repeat_purchases': 'allowed',
            'starter_sha256': sha256(args.starter),
            'source_sha256': {name: sha256(args.data_dir / name) for name in SOURCE_FILES},
            'excluded_source_rows': excluded,
            'train_events_by_type': dict(event_types),
            'train_users': len(events),
            'evaluation_users_with_purchases': nonempty,
            'evaluation_users_without_purchases': len(users) - nonempty,
            'evaluation_sample_relevant_pairs': sum(len(row['relevant_asset_ids']) for row in selected_truth),
            'files': {path: {'description': description, 'rows': count, 'sha256': sha256(root / path)}
                      for path, (description, count) in files.items()},
        }
        name = 'split_manifest.json'
        (root / name).write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        for path in (*files, name):
            target = args.output_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            (root / path).replace(target)

    print(f'Train: {sum(event_types.values()):,} events, {len(events):,} users, {len(assets):,} assets')
    print(f'Evaluation: {len(labels):,} relevant pairs, {len(users)} evaluated users, {nonempty} with purchases')


if __name__ == '__main__':
    main()
