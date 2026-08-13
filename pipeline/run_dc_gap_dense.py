"""
dc-gap retrieval demo, dense embedding variant -- base relevance is read
from a pre-computed doc-level TREC run file, followed by a greedy
re-ranking that selects, at each step, the pooled document whose claims
show the largest max-min "gap" against the currently selected set, exactly
as in run_dc_gap.py. The difference: score(D_i, c) is a dot product of
pre-computed tevatron doc/claim embeddings instead of a BM25 score. See
src/retrieval/dc_gap_dense.py.

Usage:
    python pipeline/run_dc_gap_dense.py \
        --topics <topics.jsonl> \
        --run-file <path/to/doc-level-run.txt> \
        --corpus <path/to/collection.jsonl.gz> [<more files/globs>...] \
        --doc-reps <'docs_emb/docs_emb.*.pkl'> \
        --claim-reps <'claims_emb/claims_emb.*.pkl'> \
        --output <results.txt> \
        [--k 1000] [--tag dc-gap-dense]
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from retrieval import dc_gap_dense
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
    parser = argparse.ArgumentParser(description="Doc-claim gap re-ranking over a doc-level run file, using dense embeddings")
    parser.add_argument("--topics", required=True,
                        help="JSONL file with topics; each line must have 'qid' and 'query'")
    parser.add_argument("--run-file", required=True,
                        help="Pre-computed doc-level TREC run file used as the base relevance score")
    parser.add_argument("--corpus", required=True, nargs="+",
                        help="JSONL or JSONL.gz document corpus file(s) with 'text' and 'statements' fields; "
                             "globs accepted. Only used for display fields, not scoring.")
    parser.add_argument("--doc-reps", required=True,
                        help="Glob pattern matching tevatron doc-level embedding shard pkl(s) (see "
                             "scripts/dense-index/*/*-encode-docs.sh), e.g. 'docs_emb/docs_emb.*.pkl'")
    parser.add_argument("--claim-reps", required=True,
                        help="Glob pattern matching tevatron claim-level embedding shard pkl(s) (see "
                             "scripts/dense-index/*/*-encode-claims.sh), e.g. 'claims_emb/claims_emb.*.pkl'")
    parser.add_argument("--output", required=True,
                        help="Output file path (TREC run format)")
    parser.add_argument("--k", type=int, default=1000,
                        help="Pool size taken from the run file and re-ranked (default: 1000)")
    parser.add_argument("--tag", default="dc-gap-dense",
                        help="Run tag written in the TREC output (default: dc-gap-dense)")
    args = parser.parse_args()

    topics = load_topics(args.topics)
    logger.info("Loaded %d topic(s) from %s", len(topics), args.topics)

    inputs = [Result(topic=t, subquestions=[]) for t in topics]
    results = dc_gap_dense.run(
        inputs,
        run_file=args.run_file,
        corpus=args.corpus,
        doc_reps=args.doc_reps,
        claim_reps=args.claim_reps,
        k=args.k,
    )

    write_trec(results, args.output, args.tag)


if __name__ == "__main__":
    main()
