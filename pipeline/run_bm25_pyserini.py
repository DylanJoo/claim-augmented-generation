"""
Pyserini/Lucene BM25 retrieval — queries a pre-built Lucene index and writes TREC-format output.

Works with both document-level and claim-level indices built by scripts/index/.
For claim-level indices the docids carry a '#<i>' suffix; downstream tools strip it
when aggregating to document level.

Usage:
    python pipeline/run_bm25_pyserini.py \
        --topics <topics.jsonl> \
        --index  <path/to/lucene-index-dir> \
        --output <results.txt> \
        [--k 1000] [--k1 0.9] [--b 0.4] [--tag bm25-doc] [--fusion sum|rrf|max|first]
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from retrieval import search_pyserini
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
    parser = argparse.ArgumentParser(description="Pyserini/Lucene BM25 retrieval over a pre-built index")
    parser.add_argument("--topics", required=True,
                        help="JSONL file with topics; each line must have 'qid' and 'query'")
    parser.add_argument("--index", required=True,
                        help="Directory of the pre-built Lucene index")
    parser.add_argument("--output", required=True,
                        help="Output file path (TREC run format)")
    parser.add_argument("--k", type=int, default=1000,
                        help="Number of results to retrieve per query (default: 1000)")
    parser.add_argument("--k1", type=float, default=0.9,
                        help="BM25 k1 term-frequency saturation parameter (default: 0.9, Anserini default)")
    parser.add_argument("--b", type=float, default=0.4,
                        help="BM25 b length-normalization parameter (default: 0.4, Anserini default)")
    parser.add_argument("--language", default=None,
                        help="Analyzer language passed to LuceneSearcher.set_language (default: none)")
    parser.add_argument("--tag", default="bm25",
                        help="Run tag written in the TREC output (default: bm25)")
    parser.add_argument("--fusion", default="sum",
                        choices=["sum", "rrf", "max", "first"],
                        help="Score fusion strategy for claim-level indices (default: sum)")
    args = parser.parse_args()

    topics = load_topics(args.topics)
    logger.info("Loaded %d topic(s) from %s", len(topics), args.topics)

    inputs = [Result(topic=t, subquestions=[]) for t in topics]
    results = search_pyserini.run(
        inputs,
        index=args.index,
        k=args.k,
        language=args.language,
        fusion=args.fusion,
        k1=args.k1,
        b=args.b,
    )

    write_trec(results, args.output, args.tag)


if __name__ == "__main__":
    main()
