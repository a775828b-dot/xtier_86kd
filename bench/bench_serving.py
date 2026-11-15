#!/usr/bin/env python3
"""Measure successful output throughput, TTFT and TPOT at fixed concurrency."""

import argparse
import concurrent.futures
import json
import math
import statistics
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path

PROMPTS = [
    'Write a Python LRU cache with get and put methods. Include tests and explain complexity.',
    '用中文详细解释多级缓存一致性，给出一个具体的并发读写例子。',
    'Explain branch prediction, two-level predictors, and their cost when they miss.',
    'Implement a bounded concurrent queue in Python and discuss race conditions.',
]

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=30000)
parser.add_argument('--host', default='127.0.0.1')
parser.add_argument('--model', default='DeepSeek-V4.1-Flash-1017')
parser.add_argument('--api-key-file')
parser.add_argument('--concurrencies', default='1,20,64')
parser.add_argument('--rounds', type=int, default=2)
parser.add_argument('--output-tokens', type=int, default=512)
parser.add_argument('--prompt-tokens', type=int, default=0,
                    help='Use this many corpus tokens for each independent request')
parser.add_argument('--corpus', default='/data/bench/sglang-corpus.txt')
parser.add_argument('--tokenizer', default='/data/models/DeepSeek-V4.1-Flash-1017')
parser.add_argument('--out', default='/data/bench/serving-64.json')
args = parser.parse_args()
run_id = uuid.uuid4().hex[:12]
key = open(args.api_key_file).read().strip() if args.api_key_file else None
url = f'http://{args.host}:{args.port}/v1/chat/completions'
long_prompt = ''
if args.prompt_tokens:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    ids = tokenizer.encode(Path(args.corpus).read_text(errors='ignore'),
                           add_special_tokens=False)
    if len(ids) < args.prompt_tokens:
        raise SystemExit(f'corpus has {len(ids)} tokens, need {args.prompt_tokens}')
    long_prompt = tokenizer.decode(ids[:args.prompt_tokens])


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))]


def run_one(index, case_prefix):
    content = f'Case {case_prefix}-{index}: ' + (long_prompt or PROMPTS[index % len(PROMPTS)])
    body = {
        'model': args.model,
        'messages': [{'role': 'user', 'content': content}],
        'temperature': 0,
        'max_tokens': args.output_tokens,
        'ignore_eos': True,
        'stream': True,
        'stream_options': {'include_usage': True},
        'chat_template_kwargs': {'enable_thinking': False},
    }
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = f'Bearer {key}'
    request = urllib.request.Request(url, json.dumps(body).encode(), headers)
    started = time.perf_counter()
    first = last = None
    usage = {}
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            for raw in response:
                line = raw.decode('utf-8', errors='replace').strip()
                if not line.startswith('data: ') or line == 'data: [DONE]':
                    continue
                event = json.loads(line[6:])
                usage = event.get('usage') or usage
                for choice in event.get('choices') or []:
                    delta = choice.get('delta') or {}
                    if delta.get('content') or delta.get('reasoning_content'):
                        now = time.perf_counter()
                        first = first or now
                        last = now
        finished = time.perf_counter()
        tokens = int(usage.get('completion_tokens') or 0)
        if not first or not tokens:
            raise ValueError('empty stream or missing completion token count')
        return {
            'ok': True,
            'input_tokens': int(usage.get('prompt_tokens') or 0),
            'output_tokens': tokens,
            'started': started,
            'first': first,
            'last': last,
            'finished': finished,
            'ttft_s': first - started,
            'tpot_ms': 1000 * (last - first) / (tokens - 1) if tokens > 1 else 0,
        }
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
        return {'ok': False, 'started': started, 'finished': time.perf_counter(),
                'error': str(exc)[:300]}


rows = []
for concurrency in [int(value) for value in args.concurrencies.split(',')]:
    for round_index in range(args.rounds):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            case_prefix = f'{run_id}-{concurrency}-{round_index}'
            results = list(pool.map(lambda index: run_one(index, case_prefix), range(concurrency)))
        good = [row for row in results if row['ok']]
        summary = {
            'concurrency': concurrency,
            'round': round_index,
            'requests': len(results),
            'success': len(good),
            'errors': [row['error'] for row in results if not row['ok']],
        }
        if good:
            summary.update({
                'input_tokens': sum(row['input_tokens'] for row in good),
                'output_tokens': sum(row['output_tokens'] for row in good),
                'prefill_window_tps_est': round(sum(row['input_tokens'] for row in good) /
                                                (max(row['first'] for row in good) -
                                                 min(row['started'] for row in results)), 2),
                'successful_output_tps': round(sum(row['output_tokens'] for row in good) /
                                               (max(row['finished'] for row in results) -
                                                min(row['started'] for row in results)), 2),
                'decode_window_tps': round(sum(row['output_tokens'] for row in good) /
                                           (max(row['last'] for row in good) -
                                            min(row['first'] for row in good)), 2),
                'p50_ttft_s': round(statistics.median(row['ttft_s'] for row in good), 3),
                'p95_ttft_s': round(percentile([row['ttft_s'] for row in good], .95), 3),
                'p50_tpot_ms': round(statistics.median(row['tpot_ms'] for row in good), 3),
                'p95_tpot_ms': round(percentile([row['tpot_ms'] for row in good], .95), 3),
            })
        rows.append({'summary': summary, 'requests': results})
        print(json.dumps(summary, ensure_ascii=False), flush=True)
with open(args.out, 'w') as stream:
    json.dump({'options': vars(args), 'rounds': rows}, stream, indent=2, ensure_ascii=False)

