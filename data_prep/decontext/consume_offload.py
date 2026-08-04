import os
import sys
import argparse
import asyncio
import gzip
import json

import httpx
from openai import AsyncOpenAI
from tqdm import tqdm


def postprocess(text):
    try:
        x = text.strip().split('</s>')
        x = [xx.strip().replace('<s>', '').replace('</s>', '') for xx in x]
        return [xx for xx in x if xx not in ['', ' ', '\n']]
    except Exception:
        return None


def load_done_ids(output_file):
    if not os.path.exists(output_file):
        return set()
    print(f"Found existing output file: {output_file}")
    done_ids = set()
    with open(output_file, 'rt') as f:
        for line in f:
            done_ids.add(json.loads(line)['id'])
    return done_ids


def load_shard(input_file, shard_index, num_shards, done_ids):
    prompts = {}
    with gzip.open(input_file, 'rt') as f:
        for idx, line in enumerate(f):
            if idx % num_shards != shard_index:
                continue
            item = json.loads(line.strip())
            if item['id'] in done_ids:
                continue
            prompts[item['id']] = item['messages']
    return prompts


async def process_one(client, model, docid, messages, max_tokens, semaphore):
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
            )
            statements = postprocess(resp.choices[0].message.content)
        except Exception as e:
            print(f"[{docid}] request failed: {e}", file=sys.stderr)
            statements = None
    return docid, statements


async def run(prompts, endpoint, model, max_tokens, max_concurrency, timeout, output_file):
    limits = httpx.Limits(max_connections=max_concurrency, max_keepalive_connections=max_concurrency)
    http_client = httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(timeout))
    client = AsyncOpenAI(base_url=f"http://{endpoint}/v1", api_key="EMPTY", http_client=http_client)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        asyncio.create_task(process_one(client, model, docid, messages, max_tokens, semaphore))
        for docid, messages in prompts.items()
    ]

    with open(output_file, 'at') as f:
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=endpoint):
            docid, statements = await coro
            if statements is not None:
                f.write(json.dumps({"id": docid, "statements": statements}, ensure_ascii=False) + '\n')
                f.flush()

    await http_client.aclose()


def main(args):
    done_ids = load_done_ids(args.output_file)
    prompts = load_shard(args.input_file, args.shard_index, args.num_shards, done_ids)

    print(f"{args.endpoint}: {len(prompts)} documents remaining "
          f"(shard {args.shard_index}/{args.num_shards} of {args.input_file})")
    if not prompts:
        return 0

    asyncio.run(run(prompts, args.endpoint, args.model, args.max_tokens,
                     args.max_concurrency, args.timeout, args.output_file))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
                         help='Offloaded prompt file (nuggetized_corpus.<lang>.jsonl.gz) with "messages" per doc')
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--endpoint', type=str, required=True,
                         help='host:port of a running vLLM OpenAI-compatible server')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.3-70B-Instruct')
    parser.add_argument('--max_tokens', type=int, default=8196)
    parser.add_argument('--max_concurrency', type=int, default=128,
                         help='In-flight requests to this endpoint at once (server does continuous batching)')
    parser.add_argument('--timeout', type=float, default=600.0)
    parser.add_argument('--shard_index', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    args = parser.parse_args()
    sys.exit(main(args))
