"""
cc-MMR retrieval demo -- base relevance is read from a pre-computed doc-level
TREC run file, followed by a greedy Maximal Marginal Relevance re-ranking
whose diversity signal is claim-claim MaxSim rather than whole-document
similarity. Claims per doc come from the document corpus's "statements"
field. See src/retrieval/cc_mmr.py.

Usage:
    python pipeline/run_cc_mmr.py \
        --topics <topics.jsonl> \
        --run-file <path/to/doc-level-run.txt> \
        --corpus <path/to/collection.jsonl.gz> [<more files/globs>...] \
        --output <results.txt> \
        [--k 1000] [--lambda-mult 0.9] \
        [--stopwords en] [--stemmer snowball] [--tag cc-mmr]
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from retrieval import cc_mmr
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
    parser = argparse.ArgumentParser(description="Claim-claim MaxSim MMR re-ranking over a doc-level run file")
    parser.add_argument("--topics", required=True,
                        help="JSONL file with topics; each line must have 'qid' and 'query'")
    parser.add_argument("--run-file", required=True,
                        help="Pre-computed doc-level TREC run file used as the base relevance score")
    parser.add_argument("--corpus", required=True, nargs="+",
                        help="JSONL or JSONL.gz document corpus file(s) with a 'statements' field; globs accepted")
    parser.add_argument("--output", required=True,
                        help="Output file path (TREC run format)")
    parser.add_argument("--k", type=int, default=1000,
                        help="Pool size taken from the run file and re-ranked by MMR (default: 1000)")
    parser.add_argument("--lambda-mult", type=float, default=0.9,
                        help="MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity (default: 0.9)")
    parser.add_argument("--stopwords", default="en",
                        help="Stopword list passed to bm25s.tokenize (default: en)")
    parser.add_argument("--stemmer", default=None,
                        help="Stemmer to use, e.g. 'snowball' (default: none)")
    parser.add_argument("--local-k1", type=float, default=1.2,
                        help="k1 for the local claim-claim BM25 similarity index (default: 1.2)")
    parser.add_argument("--local-b", type=float, default=0.5,
                        help="b for the local claim-claim BM25 similarity index (default: 0.5)")
    parser.add_argument("--tag", default="cc-mmr",
                        help="Run tag written in the TREC output (default: cc-mmr)")
    args = parser.parse_args()

    topics = load_topics(args.topics)
    logger.info("Loaded %d topic(s) from %s", len(topics), args.topics)

    inputs = [Result(topic=t, subquestions=[]) for t in topics]
    results = cc_mmr.run(
        inputs,
        run_file=args.run_file,
        corpus=args.corpus,
        k=args.k,
        lambda_mult=args.lambda_mult,
        stopwords=args.stopwords,
        stemmer_name=args.stemmer,
        local_k1=args.local_k1,
        local_b=args.local_b,
    )

    write_trec(results, args.output, args.tag)


if __name__ == "__main__":
    main()
