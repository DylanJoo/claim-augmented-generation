"""
Hybrid retrieval demo -- claim-level retrieval as the base signal, combined
with a second, coarser signal (doc-level or concat-claims-level retrieval)
at the parent-doc level. Both legs reuse retrieval.search.run; see
src/retrieval/hybrid.py for the combination step.

Usage:
    python pipeline/run_hybrid.py \
        --topics <topics.jsonl> \
        --claim-index <path/to/claims.bm25s> \
        --aux-index   <path/to/concat-claims.bm25s | documents.bm25s> \
        --output <results.txt> \
        [--k-claim 1000] [--k-aux 1000] \
        [--claim-fusion sum|rrf|max|first] [--aux-fusion sum|rrf|max|first] \
        [--alpha 0.5] [--combine sum|rrf|max] [--no-normalize] \
        [--stopwords en] [--stemmer snowball] [--tag hybrid]
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from retrieval import hybrid
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
    parser = argparse.ArgumentParser(description="Hybrid (claim + aux) retrieval over bm25s indices")
    parser.add_argument("--topics", required=True,
                        help="JSONL file with topics; each line must have 'qid' and 'query'")
    parser.add_argument("--claim-index", required=True,
                        help="Directory of the pre-built claim-level bm25s index (base signal)")
    parser.add_argument("--aux-index", required=True,
                        help="Directory of the pre-built doc-level or concat-claims-level bm25s index")
    parser.add_argument("--output", required=True,
                        help="Output file path (TREC run format)")
    parser.add_argument("--k-claim", type=int, default=1000,
                        help="Number of results to retrieve per query from the claim index (default: 1000)")
    parser.add_argument("--k-aux", type=int, default=1000,
                        help="Number of results to retrieve per query from the aux index (default: 1000)")
    parser.add_argument("--claim-fusion", default="sum",
                        choices=["sum", "rrf", "max", "first"],
                        help="Claim-level within-index fusion strategy (default: sum)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight on the claim-level leg when combining (1-alpha on aux; default: 0.5)")
    parser.add_argument("--combine", default="rrf",
                        choices=["sum", "rrf", "max"],
                        help="Cross-leg combination strategy (default: sum)")
    parser.add_argument("--no-normalize", action="store_true", default=False,
                        help="Disable min-max score normalization before combining the two legs")
    parser.add_argument("--stopwords", default="en",
                        help="Stopword list passed to bm25s.tokenize (default: en)")
    parser.add_argument("--stemmer", default=None,
                        help="Stemmer to use, e.g. 'snowball' (default: none)")
    parser.add_argument("--tag", default="hybrid",
                        help="Run tag written in the TREC output (default: hybrid)")
    args = parser.parse_args()

    topics = load_topics(args.topics)
    logger.info("Loaded %d topic(s) from %s", len(topics), args.topics)

    inputs = [Result(topic=t, subquestions=[]) for t in topics]
    results = hybrid.run(
        inputs,
        claim_index=args.claim_index,
        aux_index=args.aux_index,
        k_claim=args.k_claim,
        k_aux=args.k_aux,
        claim_fusion=args.claim_fusion,
        alpha=args.alpha,
        normalize=not args.no_normalize,
        stopwords=args.stopwords,
        stemmer=args.stemmer,
    )

    write_trec(results, args.output, args.tag)


if __name__ == "__main__":
    main()
