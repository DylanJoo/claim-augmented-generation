"""
Sweep cc_kmeans configs in one process: claim shards and the base run are
loaded ONCE, then every (n_clusters, alpha, lambda, cluster_reweight) combo for
the given top_m is selected and written. Same outputs, same file naming, as
looping pipeline/run_cc_kmeans.py, minus re-reading ~180GB of shards per run.
Existing non-empty outputs are skipped.

Usage:
    python pipeline/sweep_cc_kmeans.py --top-m 20 --n-clusters 50 75 100 \
        --alpha 0.1 0.3 --lambda-mult 0.4 0.5 --reweight none idf ...
"""
import argparse
import itertools
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from retrieval import cc_kmeans
from utils import Hit, load_run, load_topics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", required=True)
    ap.add_argument("--run-file", required=True)
    ap.add_argument("--claim-reps", required=True)
    ap.add_argument("--query-reps", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-prefix", required=True, help="e.g. run.ragtime1.documents.Qwen3-Embedding-0.6B")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--top-m", type=int, required=True)
    ap.add_argument("--n-clusters", type=int, nargs="+", required=True)
    ap.add_argument("--alpha", type=float, nargs="+", required=True)
    ap.add_argument("--lambda-mult", type=float, nargs="+", required=True)
    ap.add_argument("--floor", type=float, nargs="+", default=[0.0])
    ap.add_argument("--reweight", nargs="+", default=["none"], choices=["none", "idf"])
    ap.add_argument("--label-mode", default="binary", choices=["binary", "scaled", "centroid", "centroid_count"])
    args = ap.parse_args()

    def out_path(nc, a, fl, lm, rw):
        name = args.label_mode + ("" if rw == "none" else f"-{rw}")
        return os.path.join(args.out_dir, f"{args.out_prefix}.cckmeans.top{args.top_m}-k{nc}-{name}"
                                          f".alpha-{a}.floor-{fl}.lambda-{lm}.txt")

    combos = [c for c in itertools.product(args.n_clusters, args.alpha, args.floor, args.lambda_mult, args.reweight)
              if not (os.path.exists(out_path(*c)) and os.path.getsize(out_path(*c)) > 0)]
    logger.info("%d config(s) to run", len(combos))
    if not combos:
        return

    topics = load_topics(args.topics)
    base_run = load_run(args.run_file, k=args.k)
    needed = {d for pool in base_run.values() for d, _ in pool}
    claim_reps_by_id = cc_kmeans._load_claim_reps(args.claim_reps, needed)
    rows_by_parent = cc_kmeans._rows_by_parent(claim_reps_by_id)
    query_vecs = cc_kmeans._load_query_reps(args.query_reps) if args.label_mode.startswith("centroid") else {}
    os.makedirs(args.out_dir, exist_ok=True)

    for n, (nc, a, fl, lm, rw) in enumerate(combos, 1):
        path = out_path(nc, a, fl, lm, rw)
        tag = f"doc-cckmeans-top{args.top_m}-k{nc}-a{a}-f{fl}-l{lm}" + ("" if rw == "none" else f"-{rw}")
        logger.info("[%d/%d] %s", n, len(combos), path)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as out:
            for t in topics:
                qid = str(t["qid"])
                hits = [Hit(docid=d, score=s, rank=r, content_dict={"text": None, "title": None})
                        for r, (d, s) in enumerate(base_run.get(qid, []), start=1)]
                sel = cc_kmeans._select(
                    hits, claim_reps_by_id, rows_by_parent, args.k, n_clusters=nc, top_m=args.top_m,
                    label_mode=args.label_mode, alpha=a, lambda_mult=lm, discount_floor=fl, qid=qid,
                    query_vec=query_vecs.get(qid), cluster_reweight=rw)
                for h in sel:
                    out.write(f"{qid} Q0 {h.docid} {h.rank} {h.score:.6f} {tag}\n")
        os.replace(tmp, path)  # only complete runs count as existing


if __name__ == "__main__":
    main()
