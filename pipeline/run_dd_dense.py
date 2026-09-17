"""
MMR retrieval demo, dense embedding variant -- base relevance is read from a
pre-computed doc-level TREC run file, followed by a greedy re-ranking of the
evidence pool, exactly as in run_dd.py. The difference: sim(d, d') is a dot
product of pre-computed tevatron doc-level embeddings instead of a local
BM25 self-similarity score. `--mode subtract` is MMR (penalize doc overlap
with what's selected); `--mode add` treats doc overlap as a
relevance-corroborating boost instead. See src/retrieval/dd_dense.py.

Usage:
    python pipeline/run_dd_dense.py \
        --topics <topics.jsonl> \
        --run-file <path/to/doc-level-run.txt> \
        --corpus <path/to/collection.jsonl.gz> [<more files/globs>...] \
        --doc-reps <'docs_emb/docs_emb.*.pkl'> \
        --output <results.txt> \
        [--k 100] [--lambda-mult 0.9] [--mode subtract|add] [--tag dd-dense]
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from retrieval import dd_dense
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
    parser = argparse.ArgumentParser(description="MMR re-ranking over a doc-level run file, using dense embeddings")
    parser.add_argument("--topics", required=True,
                        help="JSONL file with topics; each line must have 'qid' and 'query'")
    parser.add_argument("--run-file", required=True,
                        help="Pre-computed doc-level TREC run file used as the base relevance score")
    parser.add_argument("--corpus", required=True, nargs="+",
                        help="JSONL or JSONL.gz document corpus file(s); globs accepted")
    parser.add_argument("--doc-reps", required=True,
                        help="Glob pattern matching tevatron doc-level embedding shard pkl(s) (see "
                             "scripts/dense-index/*/*-encode-docs.sh), e.g. 'docs_emb/docs_emb.*.pkl'")
    parser.add_argument("--output", required=True,
                        help="Output file path (TREC run format)")
    parser.add_argument("--k", type=int, default=100,
                        help="Pool size taken from the run file and re-ranked by MMR (default: 100)")
    parser.add_argument("--lambda-mult", type=float, default=0.9,
                        help="MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity (default: 0.9)")
    parser.add_argument("--mode", choices=["subtract", "add"], default="subtract",
                        help="subtract = MMR (penalize doc overlap with selected docs); "
                             "add = doc-echo boost (reward it) (default: subtract)")
    parser.add_argument("--tag", default="dd-dense",
                        help="Run tag written in the TREC output (default: dd-dense)")
    args = parser.parse_args()

    topics = load_topics(args.topics)
    logger.info("Loaded %d topic(s) from %s", len(topics), args.topics)

    inputs = [Result(topic=t, subquestions=[]) for t in topics]
    results = dd_dense.run(
        inputs,
        run_file=args.run_file,
        corpus=args.corpus,
        doc_reps=args.doc_reps,
        k=args.k,
        lambda_mult=args.lambda_mult,
        mode=args.mode,
    )

    write_trec(results, args.output, args.tag)


if __name__ == "__main__":
    main()
