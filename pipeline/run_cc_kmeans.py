"""
cc-kmeans retrieval demo -- base relevance is read from a pre-computed
doc-level TREC run file, followed by a greedy re-ranking that selects docs to
maximize a `--lambda-mult` blend of that base relevance and alpha-nDCG's own
per-cluster diminishing-returns gain over per-doc k-means cluster-membership
vectors (same construction as src/evaluator/rac_eval_ub.py's
greedy_alpha_ndcg_oracle, with k-means clusters standing in for ground-truth
subtopics). `--lambda-mult 0.0` (default) is pure coverage, reproducing the
original behavior exactly; `--alpha` is the novelty discount within that
coverage term. See src/retrieval/cc_kmeans.py for the full method
description. For exact-relevance MMR-style reranking instead, use
cc_dense.py's subtract/add modes.

`--kmeans-top-m` controls whether k-means fits on the whole pool (omit the
flag, or set it >= pool depth -- the original single-fit behavior) or only
on the top-M docs by base relevance (the "relevant core"), with the rest of
the pool scored against those fitted centroids via .predict() instead of
being included in the fit itself.

Usage:
    python pipeline/run_cc_kmeans.py \
        --topics <topics.jsonl> \
        --run-file <path/to/doc-level-run.txt> \
        --corpus <path/to/collection.jsonl.gz> [<more files/globs>...] \
        --claim-reps <'claims_emb/claims_emb.*.pkl'> \
        --output <results.txt> \
        [--k 100] [--alpha 0.5] [--lambda-mult 0.0] \
        [--kmeans-n-clusters 20] [--kmeans-top-m <int, default: whole pool>] \
        [--kmeans-label-mode binary|scaled] [--kmeans-n-init 10] \
        [--tag cc-kmeans]
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from retrieval import cc_kmeans
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
    parser = argparse.ArgumentParser(
        description="K-means cluster-based doc-doc diversity re-ranking over a doc-level run file")
    parser.add_argument("--topics", required=True,
                        help="JSONL file with topics; each line must have 'qid' and 'query'")
    parser.add_argument("--run-file", required=True,
                        help="Pre-computed doc-level TREC run file used as the base relevance score")
    parser.add_argument("--corpus", required=True, nargs="+",
                        help="JSONL or JSONL.gz document corpus file(s) with a 'statements' field; "
                             "globs accepted. Only used for display fields, not scoring.")
    parser.add_argument("--claim-reps", required=True,
                        help="Glob pattern matching tevatron claim-level embedding shard pkl(s) (see "
                             "scripts/dense-index/*/*-encode-claims.sh), e.g. 'claims_emb/claims_emb.*.pkl'")
    parser.add_argument("--output", required=True,
                        help="Output file path (TREC run format)")
    parser.add_argument("--k", type=int, default=100,
                        help="Pool size taken from the run file and re-ranked (default: 100)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Novelty discount: a cluster's marginal gain is multiplied by "
                             "(1 - alpha) for each already-selected doc that touches it, same "
                             "semantics as rac_eval_ub.py's --alpha (default: 0.5)")
    parser.add_argument("--lambda-mult", type=float, default=0.0,
                        help="Relevance/coverage trade-off, same convention as cc_dense.py's "
                             "--lambda-mult: 1.0 = pure relevance, 0.0 = pure cluster-coverage gain "
                             "(default: 0.0, reproduces pre-existing pure-coverage selection exactly)")
    parser.add_argument("--discount-floor", type=float, default=0.0,
                        help="Minimum weight a covered cluster keeps in the gain, i.e. the novelty "
                             "discount is max((1 - alpha) ** covered_count, floor). Lets docs that "
                             "overlap the already-selected set still earn credit for it "
                             "(default: 0.0, no floor -- reproduces existing behavior exactly)")
    parser.add_argument("--kmeans-n-clusters", type=int, default=20,
                        help="Number of k-means clusters fit on the core docs' claims (clamped down if "
                             "fewer claims are available) (default: 20)")
    parser.add_argument("--kmeans-top-m", type=int, default=None,
                        help="How many top-ranked (by base relevance) pooled docs count as the "
                             "'relevant core' that k-means is fit on; the rest of the pool is scored "
                             "against those fitted centroids via .predict(), not included in the fit "
                             "itself. Omit (or set >= pool depth) to fit on the whole pool instead "
                             "(default: whole pool)")
    parser.add_argument("--kmeans-label-mode", choices=["binary", "scaled"], default="binary",
                        help="How a doc's claims-per-cluster counts become its cluster vector. "
                             "'binary': multi-hot, 1 if the doc has >=1 claim in that cluster. "
                             "'scaled': within-doc fraction of claims per cluster, summing to 1 "
                             "(default: binary)")
    parser.add_argument("--kmeans-n-init", type=int, default=10,
                        help="Number of k-means initializations (sklearn KMeans n_init) (default: 10)")
    parser.add_argument("--tag", default="cc-kmeans",
                        help="Run tag written in the TREC output (default: cc-kmeans)")
    args = parser.parse_args()

    topics = load_topics(args.topics)
    logger.info("Loaded %d topic(s) from %s", len(topics), args.topics)

    inputs = [Result(topic=t, subquestions=[]) for t in topics]
    results = cc_kmeans.run(
        inputs,
        run_file=args.run_file,
        corpus=args.corpus,
        claim_reps=args.claim_reps,
        k=args.k,
        alpha=args.alpha,
        lambda_mult=args.lambda_mult,
        discount_floor=args.discount_floor,
        n_clusters=args.kmeans_n_clusters,
        top_m=args.kmeans_top_m,
        label_mode=args.kmeans_label_mode,
        kmeans_n_init=args.kmeans_n_init,
    )

    write_trec(results, args.output, args.tag)


if __name__ == "__main__":
    main()
