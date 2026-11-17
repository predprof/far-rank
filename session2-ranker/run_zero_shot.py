#!/usr/bin/env python3
"""Run Qwen3 zero-shot reranking on the frozen Session 2 BPR-MF candidates."""

import argparse
import hashlib
import json
import platform
import re
import time
from collections import Counter
from pathlib import Path

MODEL_ID = 'Qwen/Qwen3-1.7B'
MODEL_REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
MAX_NEW_TOKENS = 128
INSTRUCTION = (
    'Rerank the assets for this customer by likelihood of a future purchase. '
    'Each row is alias,bpr_score,previously_bought. Higher BPR score means higher '
    'baseline relevance. previously_bought is 1 if the customer bought the asset '
    'before the cutoff and 0 otherwise. A previous purchase may recur. Use only '
    'the supplied scores and purchase flags.\n'
)
RESPONSE_FORMAT = ('\nReturn a JSON array of exactly 10 distinct quoted aliases, most likely first. '
                   'Return the aliases alone. Omit scores, flags, objects, and explanations.')


def read_inputs(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    if len(rows) != 100 or len({row['user_id'] for row in rows}) != 100:
        raise ValueError('Expected the same 100 unique evaluation users as the starter')
    if len({row['protocol_hash'] for row in rows}) != 1:
        raise ValueError('Inputs use different evaluation protocols')
    return rows


def prompt_for(candidates):
    return INSTRUCTION + '\n'.join(
        f"{item['alias']},{item['score']:.8f},{int(item['bought_before'])}"
        for item in candidates
    ) + RESPONSE_FORMAT


def parse_ranking(raw, candidates):
    aliases = {item['alias']: item['asset_id'] for item in candidates}
    fence = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', raw.strip(), re.IGNORECASE | re.DOTALL)
    payload = fence.group(1) if fence else raw
    try:
        ranking = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError('invalid JSON') from error
    if not isinstance(ranking, list) or len(ranking) != 10:
        raise ValueError('expected exactly ten aliases')
    normalized = []
    for alias in ranking:
        if not isinstance(alias, str) or not re.fullmatch(r'A[0-9]{1,2}', alias):
            raise ValueError('invalid alias')
        candidate = f'A{int(alias[1:]):02d}'
        if candidate not in aliases:
            raise ValueError('alias outside candidate set')
        normalized.append(candidate)
    if len(set(normalized)) != 10:
        raise ValueError('duplicate alias')
    return [aliases[alias] for alias in normalized]


def write_json(path, record):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_responses(path, inputs, rankings):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        for row, ranking in zip(inputs, rankings, strict=True):
            record = {'user_id': row['user_id'], 'protocol_hash': row['protocol_hash'],
                      'stage': 'ranker', 'ranked_ids': ranking}
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    temporary.replace(path)


class LocalRanker:
    def __init__(self, model_path, device):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        self.device = device
        self.torch = torch
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('Apple Metal is unavailable to this process')
        torch.set_num_threads(4)
        dtype = torch.bfloat16 if device == 'mps' else torch.float32
        options = {'local_files_only': True, 'trust_remote_code': False}
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **options)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, low_cpu_mem_usage=True, **options
        ).to(device).eval()
        self.generation_config = GenerationConfig(
            do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def generate(self, prompt):
        rendered = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer(rendered, add_special_tokens=False, return_tensors='pt')
        input_tokens = inputs.input_ids.shape[1]
        inputs = inputs.to(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, generation_config=self.generation_config, use_model_defaults=False,
            )
        # Moving the generated tokens back to the CPU synchronizes Metal.
        tokens = output[0, input_tokens:].tolist()
        return {
            'raw_response': self.tokenizer.decode(tokens, skip_special_tokens=True),
            'input_tokens': input_tokens,
            'output_tokens': len(tokens),
            'generation_seconds': time.perf_counter() - started,
        }


def main():
    from importlib.metadata import version

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--model-path', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    args = parser.parse_args()
    rows = read_inputs(args.inputs)
    output = args.output_dir
    artifact_dir = output / 'generations'
    artifact_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        'model': MODEL_ID,
        'revision': MODEL_REVISION,
        'architecture': platform.machine(),
        'os': platform.platform(),
        'python': platform.python_version(),
        'torch': version('torch'),
        'transformers': version('transformers'),
        'device': args.device,
        'dtype': 'bfloat16' if args.device == 'mps' else 'float32',
        'max_new_tokens': MAX_NEW_TOKENS,
        'thinking': False,
        'do_sample': False,
        'use_model_generation_defaults': False,
        'instruction': INSTRUCTION,
        'response_format_instruction': RESPONSE_FORMAT,
        'protocol_hash': rows[0]['protocol_hash'],
        'inputs_sha256': hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'validation': ('JSON array of ten unique allowed aliases; strip optional Markdown fences '
                       'and normalize missing leading zeros'),
        'invalid_response_policy': 'fallback to BPR-MF top ten',
        'same_input_control': 'sort by previously bought first, then BPR-MF score',
    }
    settings_path = output / 'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('Output directory contains generations for different run settings')
    write_json(settings_path, settings)
    ranker = None
    completed = []
    for index, row in enumerate(rows, 1):
        path = artifact_dir / f'{index:03d}.json'
        prompt = prompt_for(row['candidates'])
        if path.exists():
            result = json.loads(path.read_text(encoding='utf-8'))
            if result['user_id'] != row['user_id'] or result['prompt'] != prompt:
                raise ValueError(f'Generation artifact {index} does not match frozen input')
        else:
            if ranker is None:
                print('Loading Qwen3-1.7B locally.', flush=True)
                ranker = LocalRanker(args.model_path, args.device)
            result = ranker.generate(prompt)
            result.update(user_id=row['user_id'], prompt=prompt)
            try:
                result['ranked_ids'] = parse_ranking(result['raw_response'], row['candidates'])
                result['fallback_reason'] = None
            except ValueError as error:
                result['ranked_ids'] = row['bpr_top10']
                result['fallback_reason'] = str(error)
            write_json(path, result)
        completed.append(result)
        if index in (1, len(rows)) or index % 10 == 0:
            fallbacks = sum(record['fallback_reason'] is not None for record in completed)
            print(f'{index}/{len(rows)} generated, {fallbacks} BPR-MF fallbacks.', flush=True)

    write_responses(output / 'llm_responses.jsonl', rows,
                    (record['ranked_ids'] for record in completed))
    write_responses(output / 'repeat_first_responses.jsonl', rows,
                    (row['repeat_first_top10'] for row in rows))
    reasons = Counter(record['fallback_reason'] for record in completed if record['fallback_reason'])
    summary = {
        'users': len(completed),
        'valid_responses': len(completed) - sum(reasons.values()),
        'fallback_reasons': reasons,
        'total_generation_seconds': sum(record['generation_seconds'] for record in completed),
        'total_input_tokens': sum(record['input_tokens'] for record in completed),
        'total_output_tokens': sum(record['output_tokens'] for record in completed),
    }
    write_json(output / 'generation_summary.json', summary)


if __name__ == '__main__':
    main()
