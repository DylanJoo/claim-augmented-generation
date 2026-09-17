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
        [--k 100] [--tag dc-gap-dense]
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
    parser.add_argument("--k", type=int, default=100,
                        help="Pool size taken from the run file and re-ranked (default: 100)")
    parser.add_argument("--tag", default="dc-gap-dense",
                        help="Run tag written in the TREC output (default: dc-gap-dense)")
    parser.add_argument("--query-reps",
                        help="Path to a single tevatron query embedding pickle (reps, lookup) with one "
                             "vector per topic qid (see scripts/dense-index/*/*-encode-q.sh). "
                             "Required when --claim-filter is not 'none'.")
    parser.add_argument("--claim-filter", choices=["none", "topn", "threshold"], default="none",
                        help="Filter each pooled doc's claims by query-claim cosine similarity before "
                             "bridge/novelty scoring, so an off-topic claim can't tank a doc's novelty "
                             "score. 'topn': keep each doc's N highest-similarity claims (see "
                             "--claims-per-doc/--scale-topn-by-relevance). 'threshold': keep claims at or "
                             "above --claim-sim-threshold, same cutoff for every doc (falls back to a "
                             "doc's single best claim if none pass). Default 'none' keeps every claim "
                             "found in the shards, matching prior behavior.")
    parser.add_argument("--claims-per-doc", type=int, default=5,
                        help="'topn' claim_filter: number of claims kept per doc (default: 5)")
    parser.add_argument("--scale-topn-by-relevance", action="store_true",
                        help="'topn' claim_filter: scale each doc's claim quota by its own base "
                             "relevance (normalized 0-1 over the pool), so higher-relevance docs keep "
                             "more claims: n = max(1, round(claims_per_doc * relevance))")
    parser.add_argument("--claim-sim-threshold", type=float, default=0.0,
                        help="'threshold' claim_filter: minimum query-claim cosine similarity to keep a "
                             "claim (default: 0.0)")
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
        query_reps=args.query_reps,
        claim_filter=args.claim_filter,
        claims_per_doc=args.claims_per_doc,
        scale_topn_by_relevance=args.scale_topn_by_relevance,
        claim_sim_threshold=args.claim_sim_threshold,
    )

    write_trec(results, args.output, args.tag)


if __name__ == "__main__":
    main()
