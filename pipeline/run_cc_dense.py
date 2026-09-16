"""
cc-rerank retrieval demo, dense embedding variant -- base relevance is read
from a pre-computed doc-level TREC run file, followed by a greedy
re-ranking whose signal is claim-claim MaxSim against already-selected
docs, exactly as in run_cc.py. The difference: sim_claim(c, c') is a dot
product of pre-computed tevatron claim embeddings instead of a local BM25
score. `--mode subtract` is MMR (penalize claim overlap with what's
selected); `--mode add` treats claim overlap as a relevance-corroborating
boost instead. See src/retrieval/cc_dense.py.

Usage:
    python pipeline/run_cc_dense.py \
        --topics <topics.jsonl> \
        --run-file <path/to/doc-level-run.txt> \
        --corpus <path/to/collection.jsonl.gz> [<more files/globs>...] \
        --claim-reps <'claims_emb/claims_emb.*.pkl'> \
        --output <results.txt> \
        [--k 1000] [--lambda-mult 0.9] [--mode subtract|add] \
        [--agg maxsim|mean|kmeans] \
        [--kmeans-n-clusters 20] [--kmeans-label-mode binary|scaled] [--kmeans-n-init 10] \
        [--kmeans-min-doc-support 1] \
        [--tag cc-dense]
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from retrieval import cc_dense
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
    parser = argparse.ArgumentParser(description="Claim-claim MaxSim re-ranking over a doc-level run file, using dense embeddings")
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
                        help="Relevance/claim-signal trade-off: 1.0 = pure relevance, 0.0 = pure claim signal (default: 0.9)")
    parser.add_argument("--mode", choices=["subtract", "add"], default="subtract",
                        help="subtract = MMR (penalize claim overlap with selected docs); "
                             "add = claim-echo boost (reward it) (default: subtract)")
    parser.add_argument("--agg", choices=["maxsim", "mean", "kmeans"], default="maxsim",
                        help="How to reduce each doc-doc claim x claim block to one similarity. "
                             "'maxsim': ColBERT-style sum of each claim's best match, row-normalized "
                             "(equivalent to mean-of-max over d's claims -- see src/retrieval/cc_dense.py). "
                             "'mean': plain mean over every claim-claim pair, already bounded in [-1, 1] "
                             "so no normalization is applied. "
                             "'kmeans': cluster the topic's pooled claims into topic buckets first, then "
                             "score doc-doc similarity as cosine over cluster-membership vectors -- see "
                             "the --kmeans-* flags below (default: maxsim)")
    parser.add_argument("--kmeans-n-clusters", type=int, default=20,
                        help="Only used when --agg kmeans. Number of k-means clusters fit per topic over "
                             "that topic's pooled claim embeddings (clamped down if fewer claims are "
                             "available) (default: 20)")
    parser.add_argument("--kmeans-label-mode", choices=["binary", "scaled"], default="binary",
                        help="Only used when --agg kmeans. How a doc's claims-per-cluster counts become "
                             "its cluster vector. 'binary': multi-hot, 1 if the doc has >=1 claim in that "
                             "cluster. 'scaled': within-doc fraction of claims per cluster, summing to 1 "
                             "(default: binary)")
    parser.add_argument("--kmeans-n-init", type=int, default=10,
                        help="Only used when --agg kmeans. Number of k-means initializations (sklearn "
                             "KMeans n_init) (default: 10)")
    parser.add_argument("--kmeans-min-doc-support", type=int, default=1,
                        help="Only used when --agg kmeans. A cluster touched by fewer than this many "
                             "distinct docs is zeroed out of every doc's vector (not redundant -- one "
                             "doc's unique claim, not corroboration). Default 1 is a no-op; raise to "
                             "e.g. 2 to require actual cross-document overlap (default: 1)")
    parser.add_argument("--tag", default="cc-dense",
                        help="Run tag written in the TREC output (default: cc-dense)")
    args = parser.parse_args()

    topics = load_topics(args.topics)
    logger.info("Loaded %d topic(s) from %s", len(topics), args.topics)

    inputs = [Result(topic=t, subquestions=[]) for t in topics]
    results = cc_dense.run(
        inputs,
        run_file=args.run_file,
        corpus=args.corpus,
        claim_reps=args.claim_reps,
        k=args.k,
        lambda_mult=args.lambda_mult,
        mode=args.mode,
        agg=args.agg,
        n_clusters=args.kmeans_n_clusters,
        label_mode=args.kmeans_label_mode,
        kmeans_n_init=args.kmeans_n_init,
        min_doc_support=args.kmeans_min_doc_support,
    )
    logger.info(f"the run file is {args.run_file}")

    write_trec(results, args.output, args.tag)


if __name__ == "__main__":
    main()
