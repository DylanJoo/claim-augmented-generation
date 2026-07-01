"""
Iterative RAG generation pipeline.

Reads a pre-computed BM25 run file (TREC format), loads the corpus, and
iteratively generates a report for each topic by reading retrieved documents
in fixed-size batches.

Example:
    python pipeline/run_iterative_rag.py \\
        --topics   ~/scratch/neuclir1/topics/neuclir24-test-request.jsonl \\
        --run-file ~/scratch/neuclir1/runs/bm25-documents.txt \\
        --corpus   ~/scratch/neuclir1/*.processed.jsonl.gz \\
        --output   ~/scratch/neuclir1/runs/iterative-rag.jsonl \\
        --model    llama3.3-70b-instruct \\
        --base-url http://10.162.95.158:4000/v1 \\
        --topk 50 --n-rounds 3 --docs-per-round 5 --max-chars 2000
"""

import argparse
import glob
import gzip
import json
import logging
import os
import sys
from collections import defaultdict

import openai

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from generation.iterative_rag import run as generate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_topics(path: str) -> list[dict]:
    topics = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                topics.append(json.loads(line))
    return topics


def _open(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def _resolve_files(patterns: list[str]) -> list[str]:
    files = []
    for p in patterns:
        expanded = glob.glob(p)
        files.extend(expanded if expanded else [p])
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No corpus files matched: {patterns}")
    return files


def load_corpus(patterns: list[str]) -> dict[str, dict]:
    """Load a JSONL / JSONL.gz corpus into {docid: {text, title}}."""
    files = _resolve_files(patterns)
    logger.info("Loading document corpus from %d file(s)", len(files))

    corpus = {}
    for fpath in files:
        with _open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                corpus[doc["id"]] = {
                    "title": (doc.get("title") or "").strip(),
                    "text": (doc.get("text") or "").strip(),
                }
    logger.info("Loaded %d documents into corpus", len(corpus))
    return corpus


def load_corpus_claims(patterns: list[str]) -> dict[str, dict]:
    """Load a claims JSONL corpus into {parent_id#i: {text, title}}.

    Each source document has a 'statements' list; entries are indexed as
    '{doc_id}#{i}' to match the docids produced by the claims BM25 index.
    """
    files = _resolve_files(patterns)
    logger.info("Loading claims corpus from %d file(s)", len(files))

    corpus = {}
    for fpath in files:
        with _open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                parent_id = doc["id"]
                title = (doc.get("title") or "").strip()
                for i, claim in enumerate(doc.get("statements") or []):
                    corpus[f"{parent_id}#{i}"] = {"title": title, "text": claim.strip()}
    logger.info("Loaded %d claims into corpus", len(corpus))
    return corpus


def load_run(path: str, topk: int) -> dict[str, list[tuple[str, float]]]:
    """Parse a TREC run file → {qid: [(docid, score), ...]} sorted by score desc."""
    run: dict[str, list] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, docid, rank, score = parts[0], parts[2], int(parts[3]), float(parts[4])
            if rank <= topk:
                run[qid].append((docid, score))
    return {
        qid: sorted(hits, key=lambda x: x[1], reverse=True)
        for qid, hits in run.items()
    }


# ---------------------------------------------------------------------------
# LLM wrapper
# ---------------------------------------------------------------------------

def make_llm(model: str, base_url: str, api_key: str, max_tokens: int, temperature: float):
    client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=5)

    def llm(messages: list[dict]) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    return llm


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Iterative RAG generation pipeline")

    # Data
    parser.add_argument("--topics", required=True,
                        help="JSONL file with topics (request_id, problem_statement, background)")
    parser.add_argument("--run-file", required=True,
                        help="TREC run file from retrieval (BM25 or other)")
    parser.add_argument("--corpus", required=True, nargs="+",
                        help="JSONL or JSONL.gz corpus file(s); globs are accepted")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file (one {qid, response} per line)")

    # Retrieval
    parser.add_argument("--topk", type=int, default=50,
                        help="Maximum number of retrieved docs to consider per topic (default: 50)")
    parser.add_argument("--claim-level", action="store_true", default=False,
                        help="Corpus contains claim-level entries (statements); "
                             "docids in the run file are expected as 'parent_id#i'")

    # Generation loop
    parser.add_argument("--n-rounds", type=int, default=3,
                        help="Maximum generation rounds (default: 3)")
    parser.add_argument("--docs-per-round", type=int, default=5,
                        help="Documents fed to the LLM per round (default: 5)")
    parser.add_argument("--max-chars", type=int, default=2000,
                        help="Target character budget for the output report (default: 2000)")

    # LLM
    parser.add_argument("--model", default="llama3.3-70b-instruct",
                        help="Model name served at the endpoint (default: llama3.3-70b-instruct)")
    parser.add_argument("--base-url", default="http://10.162.95.158:4000/v1",
                        help="OpenAI-compatible endpoint base URL")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                        help="API key (reads OPENAI_API_KEY env var by default)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum tokens to generate per LLM call (default: 1024)")
    parser.add_argument("--temperature", type=float, default=0.0)

    args = parser.parse_args()

    # Load data
    topics = load_topics(args.topics)
    logger.info("Loaded %d topic(s)", len(topics))

    run_results = load_run(args.run_file, topk=args.topk)
    logger.info("Loaded run file: %d queries", len(run_results))

    corpus = load_corpus_claims(args.corpus) if args.claim_level else load_corpus(args.corpus)

    # Build LLM
    llm = make_llm(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Generate
    outputs = generate(
        topics=topics,
        run_results=run_results,
        corpus=corpus,
        llm=llm,
        n_rounds=args.n_rounds,
        docs_per_round=args.docs_per_round,
        max_chars=args.max_chars,
    )

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for qid, response in outputs.items():
            f.write(json.dumps({"qid": qid, "response": response}, ensure_ascii=False) + "\n")

    logger.info("Wrote %d responses to %s", len(outputs), args.output)


if __name__ == "__main__":
    main()
