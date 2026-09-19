"""
cc-dbscan retrieval -- same as run_cc_kmeans.py, but claim clusters come from
DBSCAN, so the number of clusters is discovered per topic. See
src/retrieval/cc_dbscan.py for the method (eps / min_samples / noise_mode).

Usage:
    python pipeline/run_cc_dbscan.py \
        --topics <topics.jsonl> --run-file <doc-level-run.txt> \
        --corpus <collection.jsonl.gz> --claim-reps <'claims_emb/claims_emb.*.pkl'> \
        --output <results.txt> [--k 100] [--alpha 0.5] [--lambda-mult 0.0] \
        [--dbscan-eps 0.3 | --dbscan-eps-quantile 0.9] [--dbscan-min-samples 5] \
        [--dbscan-noise-mode single|drop|singleton] [--dbscan-top-m <int>]
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from retrieval import cc_dbscan
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
        description="DBSCAN cluster-based doc-doc diversity re-ranking over a doc-level run file")
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
    parser.add_argument("--dbscan-eps", type=float, default=0.3,
                        help="DBSCAN neighbourhood radius in cosine distance (default: 0.3)")
    parser.add_argument("--dbscan-eps-quantile", type=float, default=None,
                        help="If set, overrides --dbscan-eps: eps becomes this quantile of each "
                             "claim's distance to its min_samples-th nearest claim, per topic")
    parser.add_argument("--dbscan-min-samples", type=int, default=5,
                        help="Claims within eps (self included) needed to form a core point (default: 5)")
    parser.add_argument("--dbscan-noise-mode", choices=["single", "drop", "singleton"], default="single",
                        help="single: all noise claims share one extra cluster; drop: ignore them; "
                             "singleton: each noise claim is its own cluster (default: single)")
    parser.add_argument("--dbscan-top-m", type=int, default=None,
                        help="Fit DBSCAN on only the top-M docs' claims; the rest are assigned to the "
                             "nearest core sample within eps (default: whole pool)")
    parser.add_argument("--dbscan-label-mode", choices=["binary", "scaled"], default="binary",
                        help="binary: multi-hot per doc; scaled: within-doc fraction (default: binary)")
    parser.add_argument("--tag", default="cc-dbscan",
                        help="Run tag written in the TREC output (default: cc-dbscan)")
    args = parser.parse_args()

    topics = load_topics(args.topics)
    logger.info("Loaded %d topic(s) from %s", len(topics), args.topics)

    inputs = [Result(topic=t, subquestions=[]) for t in topics]
    results = cc_dbscan.run(
        inputs,
        run_file=args.run_file,
        corpus=args.corpus,
        claim_reps=args.claim_reps,
        k=args.k,
        alpha=args.alpha,
        lambda_mult=args.lambda_mult,
        discount_floor=args.discount_floor,
        top_m=args.dbscan_top_m,
        label_mode=args.dbscan_label_mode,
        eps=args.dbscan_eps,
        eps_quantile=args.dbscan_eps_quantile,
        min_samples=args.dbscan_min_samples,
        noise_mode=args.dbscan_noise_mode,
    )

    write_trec(results, args.output, args.tag)


if __name__ == "__main__":
    main()
