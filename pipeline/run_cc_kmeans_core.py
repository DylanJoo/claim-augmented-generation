"""
cc-kmeans-core retrieval demo -- base relevance is read from a pre-computed
doc-level TREC run file, followed by a greedy re-ranking whose diversity
signal is cosine similarity between per-doc k-means cluster-membership
vectors, exactly as in cc_dense.py's agg="kmeans" -- except the k-means fit
itself only sees the top `--top-m` docs by base relevance (the "relevant
core"), not the whole pool. See src/retrieval/cc_kmeans_core.py for why this
needs its own module rather than a flag on run_cc_dense.py. `--mode
subtract` is MMR (penalize cluster overlap with selected docs); `--mode add`
treats cluster overlap as a relevance-corroborating boost instead.

Usage:
    python pipeline/run_cc_kmeans_core.py \
        --topics <topics.jsonl> \
        --run-file <path/to/doc-level-run.txt> \
        --corpus <path/to/collection.jsonl.gz> [<more files/globs>...] \
        --claim-reps <'claims_emb/claims_emb.*.pkl'> \
        --output <results.txt> \
        [--k 1000] [--lambda-mult 0.9] [--mode subtract|add] \
        [--kmeans-n-clusters 20] [--kmeans-top-m 100] \
        [--kmeans-label-mode binary|scaled] [--kmeans-n-init 10] \
        [--kmeans-min-doc-support 1] \
        [--tag cc-kmeans-core]
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from retrieval import cc_kmeans_core
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
        description="K-means cluster-based doc-doc diversity re-ranking, fit only on the top-M "
                     "relevant docs and predicted onto the rest of the pool")
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
    parser.add_argument("--k", type=int, default=1000,
                        help="Pool size taken from the run file and re-ranked (default: 1000)")
    parser.add_argument("--lambda-mult", type=float, default=0.9,
                        help="Relevance/cluster-signal trade-off: 1.0 = pure relevance, 0.0 = pure cluster signal (default: 0.9)")
    parser.add_argument("--mode", choices=["subtract", "add"], default="subtract",
                        help="subtract = MMR (penalize cluster overlap with selected docs); "
                             "add = claim-echo boost (reward it) (default: subtract)")
    parser.add_argument("--kmeans-n-clusters", type=int, default=20,
                        help="Number of k-means clusters fit on the top-M core docs' claims "
                             "(clamped down if fewer claims are available) (default: 20)")
    parser.add_argument("--kmeans-top-m", type=int, default=100,
                        help="How many top-ranked (by base relevance) pooled docs count as the "
                             "'relevant core' that k-means is fit on. The rest of the pool is scored "
                             "against those fitted centroids via .predict(), not included in the fit "
                             "itself (default: 100)")
    parser.add_argument("--kmeans-label-mode", choices=["binary", "scaled"], default="binary",
                        help="How a doc's claims-per-cluster counts become its cluster vector. "
                             "'binary': multi-hot, 1 if the doc has >=1 claim in that cluster. "
                             "'scaled': within-doc fraction of claims per cluster, summing to 1 "
                             "(default: binary)")
    parser.add_argument("--kmeans-n-init", type=int, default=10,
                        help="Number of k-means initializations (sklearn KMeans n_init) (default: 10)")
    parser.add_argument("--kmeans-min-doc-support", type=int, default=1,
                        help="A cluster touched by fewer than this many distinct docs is zeroed out of "
                             "every doc's vector (not redundant -- one doc's unique claim, not "
                             "corroboration). Default 1 is a no-op; raise to e.g. 2 to require actual "
                             "cross-document overlap (default: 1)")
    parser.add_argument("--tag", default="cc-kmeans-core",
                        help="Run tag written in the TREC output (default: cc-kmeans-core)")
    args = parser.parse_args()

    topics = load_topics(args.topics)
    logger.info("Loaded %d topic(s) from %s", len(topics), args.topics)

    inputs = [Result(topic=t, subquestions=[]) for t in topics]
    results = cc_kmeans_core.run(
        inputs,
        run_file=args.run_file,
        corpus=args.corpus,
        claim_reps=args.claim_reps,
        k=args.k,
        lambda_mult=args.lambda_mult,
        mode=args.mode,
        n_clusters=args.kmeans_n_clusters,
        top_m=args.kmeans_top_m,
        label_mode=args.kmeans_label_mode,
        kmeans_n_init=args.kmeans_n_init,
        min_doc_support=args.kmeans_min_doc_support,
    )

    write_trec(results, args.output, args.tag)


if __name__ == "__main__":
    main()
