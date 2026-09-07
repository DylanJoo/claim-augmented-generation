"""
LLM reranking demo -- reranks an existing first-stage TREC run file using
APRIL's ModularReranker (github.com/DylanJoo/APRIL, checked out at
$HOME/APRIL). Queries and document text come from this project's own
topics/corpus files; APRIL only supplies the reranking algorithm (prompt
building + LLM calls + result parsing), driven here through its documented
Python API rather than its ir_datasets-oriented CLI loaders, which don't know
this project's data layout. See src/retrieval/rerank_llm.py.

Requires an OpenAI-compatible LLM endpoint reachable at --base-url when
--backend is request/openai (the default) -- see scripts/reranknig/*.sh for
how to bring one up on LUMI before calling this script.

Usage:
    python pipeline/run_rerank_llm.py \
        --topics <topics.jsonl> \
        --run-file <path/to/first-stage-run.txt> \
        --corpus <path/to/collection.jsonl.gz> [<more files/globs>...] \
        --output <results.txt> \
        [--method rankgpt] [--model Qwen/Qwen2.5-7B-Instruct] \
        [--backend request] [--base-url http://localhost:8000/v1] \
        [--k 100] [--query-batch-size 32] [--tag rerank-rankgpt]
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from retrieval import rerank_llm
from utils import load_topics, load_corpus, load_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_APRIL_SRC = os.path.join(os.path.expanduser("~"), "APRIL", "src")


def write_trec(reranked_run, output_path, tag):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for qid, hits in reranked_run.items():
            for rank, (docid, score) in enumerate(hits.items(), start=1):
                out.write(f"{qid} Q0 {docid} {rank} {score:.6f} {tag}\n")
            logger.info("topic %s: wrote %d reranked hits", qid, len(hits))
    logger.info("Done. Results saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Rerank a first-stage TREC run with APRIL's ModularReranker")
    parser.add_argument("--topics", required=True,
                        help="JSONL file with topics; each line must have 'qid' and 'query'")
    parser.add_argument("--run-file", required=True,
                        help="First-stage TREC run file to rerank")
    parser.add_argument("--corpus", required=True, nargs="+",
                        help="JSONL or JSONL.gz document corpus file(s); globs accepted")
    parser.add_argument("--output", required=True,
                        help="Output file path (TREC run format)")
    parser.add_argument("--method", default="rankgpt",
                        choices=["rankgpt", "point", "judge", "judge_expr", "umbrela",
                                 "setmaxheaptopk", "pairtopk", "rankfirst", "rankzephyr", "lancer"],
                        help="APRIL prebuilt reranking method / prompt-parser combo (default: rankgpt)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="LLM name or path for the reranker (default: Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--backend", default="request", choices=["request", "openai", "vllm"],
                        help="APRIL LLM backend: 'request'/'openai' hit an OpenAI-compatible endpoint "
                             "given by --base-url, 'vllm' loads the model in-process (default: request)")
    parser.add_argument("--base-url", default="http://localhost:8000/v1",
                        help="OpenAI-compatible endpoint for --backend request/openai "
                             "(default: http://localhost:8000/v1)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=100,
                        help="Pool size taken from the run file and reranked (default: 100)")
    parser.add_argument("--max_doc_length", type=int, default=1024)
    parser.add_argument("--query-batch-size", type=int, default=32,
                        help="Number of queries processed per LLM batch (default: 32)")
    parser.add_argument("--april-src", default=DEFAULT_APRIL_SRC,
                        help=f"Path to APRIL's src/ directory (default: {DEFAULT_APRIL_SRC})")
    parser.add_argument("--tag", default=None,
                        help="Run tag written in the TREC output (default: rerank-<method>)")
    args = parser.parse_args()

    topics = load_topics(args.topics)
    logger.info("Loaded %d topic(s) from %s", len(topics), args.topics)

    initial_run = load_run(args.run_file)
    logger.info("Loaded run for %d quer(ies) from %s", len(initial_run), args.run_file)

    corpus = load_corpus(args.corpus)
    logger.info("Loaded %d document(s) from corpus", len(corpus))

    reranked_run = rerank_llm.run(
        topics=topics,
        initial_run=initial_run,
        corpus=corpus,
        april_src=args.april_src,
        method=args.method,
        model=args.model,
        backend=args.backend,
        base_url=args.base_url,
        temperature=args.temperature,
        k=args.k,
        query_batch_size=args.query_batch_size,
        max_doc_length=args.max_doc_length
    )

    write_trec(reranked_run, args.output, args.tag or f"rerank-{args.method}")


if __name__ == "__main__":
    main()
