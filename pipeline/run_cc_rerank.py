"""
cc-rerank retrieval demo -- base relevance is read from a pre-computed
doc-level TREC run file, followed by a greedy re-ranking whose signal is
claim-claim MaxSim against already-selected docs rather than whole-document
similarity. `--mode subtract` is Maximal Marginal Relevance (penalize claim
overlap with what's selected); `--mode add` treats claim overlap as a
relevance-corroborating boost instead. Claims per doc come from the document
corpus's "statements" field. See src/retrieval/cc_rerank.py.

Usage:
    python pipeline/run_cc_rerank.py \
        --topics <topics.jsonl> \
        --run-file <path/to/doc-level-run.txt> \
        --corpus <path/to/collection.jsonl.gz> [<more files/globs>...] \
        --output <results.txt> \
        [--k 1000] [--lambda-mult 0.9] [--mode subtract|add] \
        [--stopwords en] [--stemmer snowball] [--tag cc-rerank]
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from retrieval import cc_rerank
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
    parser = argparse.ArgumentParser(description="Claim-claim MaxSim re-ranking over a doc-level run file")
    parser.add_argument("--topics", required=True,
                        help="JSONL file with topics; each line must have 'qid' and 'query'")
    parser.add_argument("--run-file", required=True,
                        help="Pre-computed doc-level TREC run file used as the base relevance score")
    parser.add_argument("--corpus", required=True, nargs="+",
                        help="JSONL or JSONL.gz document corpus file(s) with a 'statements' field; globs accepted")
    parser.add_argument("--output", required=True,
                        help="Output file path (TREC run format)")
    parser.add_argument("--k", type=int, default=1000,
                        help="Pool size taken from the run file and re-ranked (default: 1000)")
    parser.add_argument("--lambda-mult", type=float, default=0.9,
                        help="Relevance/claim-signal trade-off: 1.0 = pure relevance, 0.0 = pure claim signal (default: 0.9)")
    parser.add_argument("--mode", choices=["subtract", "add"], default="subtract",
                        help="subtract = MMR (penalize claim overlap with selected docs); "
                             "add = claim-echo boost (reward it) (default: subtract)")
    parser.add_argument("--stopwords", default="en",
                        help="Stopword list passed to bm25s.tokenize (default: en)")
    parser.add_argument("--stemmer", default=None,
                        help="Stemmer to use, e.g. 'snowball' (default: none)")
    parser.add_argument("--local-k1", type=float, default=1.2,
                        help="k1 for the local claim-claim BM25 similarity index (default: 1.2)")
    parser.add_argument("--local-b", type=float, default=0.5,
                        help="b for the local claim-claim BM25 similarity index (default: 0.5)")
    parser.add_argument("--tag", default="cc-rerank",
                        help="Run tag written in the TREC output (default: cc-rerank)")
    args = parser.parse_args()

    topics = load_topics(args.topics)
    logger.info("Loaded %d topic(s) from %s", len(topics), args.topics)

    inputs = [Result(topic=t, subquestions=[]) for t in topics]
    results = cc_rerank.run(
        inputs,
        run_file=args.run_file,
        corpus=args.corpus,
        k=args.k,
        lambda_mult=args.lambda_mult,
        mode=args.mode,
        stopwords=args.stopwords,
        stemmer_name=args.stemmer,
        local_k1=args.local_k1,
        local_b=args.local_b,
    )

    write_trec(results, args.output, args.tag)


if __name__ == "__main__":
    main()
