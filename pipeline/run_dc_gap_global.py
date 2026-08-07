"""
dc-gap retrieval demo, global claim-index variant -- base relevance is read
from a pre-computed doc-level TREC run file, followed by a greedy
re-ranking that selects, at each step, the pooled document whose claims
show the largest max-min "gap" against the currently selected set, exactly
as in run_dc_gap.py. The difference: score(D_i, c) is looked up from a
single pre-built claim-level BM25 index covering the whole corpus, instead
of a fresh per-topic index over just the pooled docs' claims. See
src/retrieval/dc_gap_global.py.

Usage:
    python pipeline/run_dc_gap_global.py \
        --topics <topics.jsonl> \
        --run-file <path/to/doc-level-run.txt> \
        --corpus <path/to/collection.jsonl.gz> [<more files/globs>...] \
        --claim-index <path/to/claims.bm25s> \
        --output <results.txt> \
        [--k 1000] \
        [--stopwords en] [--stemmer snowball] [--tag dc-gap-global]
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from retrieval import dc_gap_global
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
    parser = argparse.ArgumentParser(description="Doc-claim gap re-ranking over a doc-level run file, using a global claim index")
    parser.add_argument("--topics", required=True,
                        help="JSONL file with topics; each line must have 'qid' and 'query'")
    parser.add_argument("--run-file", required=True,
                        help="Pre-computed doc-level TREC run file used as the base relevance score")
    parser.add_argument("--corpus", required=True, nargs="+",
                        help="JSONL or JSONL.gz document corpus file(s) with 'text' and 'statements' fields; globs accepted")
    parser.add_argument("--claim-index", required=True,
                        help="Path to a prebuilt claim-level bm25s index (see indexing.py --claim-level)")
    parser.add_argument("--output", required=True,
                        help="Output file path (TREC run format)")
    parser.add_argument("--k", type=int, default=1000,
                        help="Pool size taken from the run file and re-ranked (default: 1000)")
    parser.add_argument("--stopwords", default="en",
                        help="Stopword list passed to bm25s.tokenize (default: en)")
    parser.add_argument("--stemmer", default=None,
                        help="Stemmer to use, e.g. 'snowball' (default: none). Must match how "
                             "--claim-index was built, or query tokens will miss its vocabulary.")
    parser.add_argument("--tag", default="dc-gap-global",
                        help="Run tag written in the TREC output (default: dc-gap-global)")
    args = parser.parse_args()

    topics = load_topics(args.topics)
    logger.info("Loaded %d topic(s) from %s", len(topics), args.topics)

    inputs = [Result(topic=t, subquestions=[]) for t in topics]
    results = dc_gap_global.run(
        inputs,
        run_file=args.run_file,
        corpus=args.corpus,
        claim_index=args.claim_index,
        k=args.k,
        stopwords=args.stopwords,
        stemmer_name=args.stemmer,
    )

    write_trec(results, args.output, args.tag)


if __name__ == "__main__":
    main()
