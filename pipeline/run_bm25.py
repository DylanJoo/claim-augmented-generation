"""
Raw BM25 retrieval demo — queries a pre-built bm25s index and writes TREC-format output.

Works with both document-level and claim-level indices built by scripts/index/.
For claim-level indices the docids carry a '#<i>' suffix; downstream tools strip it
when aggregating to document level.

Usage:
    python pipeline/run_bm25.py \
        --topics <topics.jsonl> \
        --index  <path/to/bm25s-index-dir> \
        --output <results.txt> \
        [--k 1000] [--tag bm25-doc] [--stopwords en] [--fusion sum|rrf|max|first]
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from retrieval import search
from utils import Result, load_topics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def write_trec(results, output_path, tag):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for result in results:
            qid = result.topic["qid"]
            for hit in result.evidences:
                out.write(f"{qid} Q0 {hit.docid} {hit.rank} {hit.score:.6f} {tag}\n")
            logger.info("topic %s: wrote %d hits of (document) evidence", qid, len(result.evidences))
    logger.info("Done. Results saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Raw BM25 retrieval over a bm25s index")
    parser.add_argument("--topics", required=True,
                        help="JSONL file with topics; each line must have 'qid' and 'query'")
    parser.add_argument("--index", required=True,
                        help="Directory of the pre-built bm25s index")
    parser.add_argument("--output", required=True,
                        help="Output file path (TREC run format)")
    parser.add_argument("--k", type=int, default=1000,
                        help="Number of results to retrieve per query (default: 1000)")
    parser.add_argument("--stopwords", default="en",
                        help="Stopword list passed to bm25s.tokenize (default: en)")
    parser.add_argument("--stemmer", default=None,
                        help="Stemmer to use, e.g. 'snowball' (default: none)")
    parser.add_argument("--tag", default="bm25",
                        help="Run tag written in the TREC output (default: bm25)")
    parser.add_argument("--fusion", default="sum",
                        choices=["sum", "rrf", "max", "first"],
                        help="Score fusion strategy for claim-level indices (default: sum)")
    args = parser.parse_args()

    topics = load_topics(args.topics)
    logger.info("Loaded %d topic(s) from %s", len(topics), args.topics)

    inputs = [Result(topic=t, subquestions=[]) for t in topics]
    results = search.run(
        inputs,
        index=args.index,
        k=args.k,
        stopwords=args.stopwords,
        stemmer=args.stemmer,
        fusion=args.fusion
    )

    write_trec(results, args.output, args.tag)


if __name__ == "__main__":
   main()
