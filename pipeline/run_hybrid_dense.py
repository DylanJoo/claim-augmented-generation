"""
Hybrid retrieval demo, dense embedding variant -- claim-level dense
retrieval as the base signal, combined with doc-level dense retrieval, at
the parent-doc level. Both legs reuse retrieval.search_dense.run; see
src/retrieval/hybrid_dense.py for the combination step.

Usage:
    python pipeline/run_hybrid_dense.py \
        --topics <topics.jsonl> \
        --query-reps <queries_emb.pkl> \
        --claim-shard-groups <'claims_emb.lang1-*.pkl'> <'claims_emb.lang2-*.pkl'> ... \
        --doc-passage-reps <'docs_emb.*.pkl'> \
        --output <results.txt> \
        [--corpus <claims_flat/*.claims.jsonl.gz>] \
        [--k-claim 1000] [--k-doc 1000] \
        [--claim-fusion sum|rrf|max|first] \
        [--alpha 0.5] [--no-normalize] [--tag hybrid-dense]
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from retrieval import hybrid_dense
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
    parser = argparse.ArgumentParser(description="Hybrid (claim + doc) retrieval over tevatron dense embedding shards")
    parser.add_argument("--topics", required=True,
                        help="JSONL file with topics; each line must have 'qid' and 'query'")
    parser.add_argument("--query-reps", required=True,
                        help="Tevatron query embeddings pkl (a (reps, lookup) tuple)")
    parser.add_argument("--claim-shard-groups", nargs="+", required=True,
                        help="One glob pattern per claim-level shard group (e.g. one per language); each "
                             "group is loaded, searched, and dropped before the next loads (claim-level scale)")
    parser.add_argument("--doc-passage-reps", required=True,
                        help="Glob pattern matching doc-level embedding shard pkl(s); loaded all at once "
                             "into one index (doc-level scale)")
    parser.add_argument("--corpus", nargs="+", default=None,
                        help="Corpus file(s)/glob(s), for claim/doc text (display only, not scoring)")
    parser.add_argument("--output", required=True,
                        help="Output file path (TREC run format)")
    parser.add_argument("--k-claim", type=int, default=1000,
                        help="Number of claim hits to fuse per query from the claim leg (default: 1000)")
    parser.add_argument("--k-doc", type=int, default=1000,
                        help="Number of doc hits to retrieve per query from the doc leg (default: 1000)")
    parser.add_argument("--claim-fusion", default="sum",
                        choices=["sum", "rrf", "max", "first"],
                        help="Claim-level within-leg fusion strategy (default: sum)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight on the claim-level leg when combining (1-alpha on the doc leg; default: 0.5)")
    parser.add_argument("--no-normalize", action="store_true", default=False,
                        help="Disable min-max score normalization before combining the two legs")
    parser.add_argument("--tag", default="hybrid-dense",
                        help="Run tag written in the TREC output (default: hybrid-dense)")
    args = parser.parse_args()

    topics = load_topics(args.topics)
    logger.info("Loaded %d topic(s) from %s", len(topics), args.topics)

    inputs = [Result(topic=t, subquestions=[]) for t in topics]
    results = hybrid_dense.run(
        inputs,
        query_reps=args.query_reps,
        claim_shard_groups=args.claim_shard_groups,
        doc_passage_reps=args.doc_passage_reps,
        corpus_paths=args.corpus,
        k_claim=args.k_claim,
        k_doc=args.k_doc,
        claim_fusion=args.claim_fusion,
        alpha=args.alpha,
        normalize=not args.no_normalize,
    )

    write_trec(results, args.output, args.tag)


if __name__ == "__main__":
    main()
