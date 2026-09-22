"""
Sweep cc_kmeans_weighted configs in one process (claim shards and base run are
loaded ONCE). Unlike sweep_cc_kmeans.py this takes an explicit list of run
specs instead of a cross-product, so relevance-weighted (A) and staged
re-clustering (C) variants can be mixed freely:

    --run "<stages>|<rel>[|<trigger>]"

  stages : top_m:n_clusters[:rel_tau],... e.g. "100:50" or "20:50,40:100"
  rel    : "none" | "softmax:<tau>"
  trigger: coverage fraction that fires the next stage (multi-stage only,
           default 1.0 = every cluster touched)

alpha / lambda / --label-mode / --reweight are crossed with every --run. Existing non-empty outputs
are skipped. Next to each run file a `.stagelog` keeps the per-query cluster
sizes, weight ranges, re-cluster events and ARI lines from cc_kmeans_weighted.

Usage:
    python pipeline/sweep_cc_kmeans_weighted.py ... --alpha 0.5 --lambda-mult 0.5 \
        --run "100:50|none" --run "100:50|softmax:0.05" --run "20:50,40:100|none|0.9"
"""
import argparse
import contextlib
import io
import itertools
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from retrieval import cc_kmeans_weighted as w
from utils import Hit, load_run, load_topics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_KEEP = ("clustered into", "re-clustered", "ARI(", "rel_tau=", "no claims found")


def parse_run(spec):
    parts = spec.split("|")
    if len(parts) not in (2, 3):
        raise ValueError(f"bad --run {spec!r}: expected stages|rel[|trigger]")
    stages = w.parse_stages(parts[0])
    rel = parts[1]
    if rel == "none":
        tau = None
    elif rel.startswith("softmax:"):
        tau = float(rel.split(":", 1)[1])
    else:
        raise ValueError(f"bad rel {rel!r}: none | softmax:<tau>")
    trigger = float(parts[2]) if len(parts) == 3 else 1.0
    return stages, tau, trigger, parts[0], rel, len(parts) == 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", required=True)
    ap.add_argument("--run-file", required=True)
    ap.add_argument("--claim-reps", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-prefix", required=True, help="e.g. run.ragtime1.documents.Qwen3-Embedding-0.6B")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--run", action="append", required=True, help="stages|rel[|trigger] (repeatable)")
    ap.add_argument("--alpha", type=float, nargs="+", required=True)
    ap.add_argument("--lambda-mult", type=float, nargs="+", required=True)
    # discount floor was removed (it did not help); the flag and the "floor-0.0" filename token are kept only
    # so already-submitted jobs and existing result files stay compatible. Drop both once nothing uses them.
    ap.add_argument("--floor", type=float, nargs="+", default=[0.0], help=argparse.SUPPRESS)
    ap.add_argument("--label-mode", nargs="+", default=["binary"],
                    choices=["binary", "scaled", "centroid", "centroid_count"])
    ap.add_argument("--reweight", nargs="+", default=["none"], choices=["none", "idf"],
                    help="cluster_reweight values, crossed with --label-mode (idf: rare clusters count more)")
    ap.add_argument("--query-reps", default=None, help="tevatron query embedding pkl; needed for centroid label modes")
    args = ap.parse_args()

    if any(f != 0.0 for f in args.floor):
        raise SystemExit("--floor was removed; only 0.0 is accepted")
    runs = [parse_run(s) for s in args.run]
    need_query = any(m.startswith("centroid") for m in args.label_mode)
    if need_query and not args.query_reps:
        raise SystemExit("centroid label modes need --query-reps")

    def out_path(run, a, lm, mode, rw):
        _, _, trig, stage_spec, rel, has_trig = run
        name = stage_spec.replace(":", "-").replace(",", "_") + f"-{mode}" + ("" if rw == "none" else f"-{rw}")
        if rel != "none":
            name += "." + rel.replace(":", "-")
        if has_trig:
            name += f".trig-{trig}"
        return os.path.join(args.out_dir, f"{args.out_prefix}.cckmeansw.{name}"
                                          f".alpha-{a}.floor-0.0.lambda-{lm}.txt")

    combos = [c for c in itertools.product(runs, args.label_mode, args.reweight, args.alpha, args.lambda_mult)
              if not (os.path.exists(out_path(c[0], c[3], c[4], c[1], c[2]))
                      and os.path.getsize(out_path(c[0], c[3], c[4], c[1], c[2])) > 0)]
    logger.info("%d config(s) to run", len(combos))
    if not combos:
        return

    topics = load_topics(args.topics)
    base_run = load_run(args.run_file, k=args.k)
    needed = {d for pool in base_run.values() for d, _ in pool}
    claim_reps_by_id = w._load_claim_reps(args.claim_reps, needed)
    rows_by_parent = w._rows_by_parent(claim_reps_by_id)
    query_vecs = w._load_query_reps(args.query_reps) if need_query else {}
    os.makedirs(args.out_dir, exist_ok=True)

    for n, (run, mode, rw, a, lm) in enumerate(combos, 1):
        stages, tau, trigger, *_ = run
        path = out_path(run, a, lm, mode, rw)
        tag = "doc-cckmeansw-" + os.path.basename(path).split(".cckmeansw.")[1].rsplit(".txt", 1)[0]
        logger.info("[%d/%d] %s", n, len(combos), path)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as out, open(path + ".stagelog", "w", encoding="utf-8") as slog:
            for t in topics:
                qid = str(t["qid"])
                if mode.startswith("centroid") and qid not in query_vecs:
                    raise KeyError(f"qid {qid} not found in query reps {args.query_reps}")
                hits = [Hit(docid=d, score=s, rank=r, content_dict={"text": None, "title": None})
                        for r, (d, s) in enumerate(base_run.get(qid, []), start=1)]
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    sel = w._select(
                        hits, claim_reps_by_id, rows_by_parent, args.k,
                        n_clusters=stages[0][1], top_m=stages[0][0],
                        label_mode=mode, cluster_reweight=rw, query_vec=query_vecs.get(qid),
                        alpha=a, lambda_mult=lm, qid=qid,
                        rel_tau=tau,
                        stages=stages, stage_trigger=trigger)
                slog.writelines(line + "\n" for line in buf.getvalue().splitlines()
                                if any(key in line for key in _KEEP))
                for h in sel:
                    out.write(f"{qid} Q0 {h.docid} {h.rank} {h.score:.6f} {tag}\n")
        os.replace(tmp, path)  # only complete runs count as existing


if __name__ == "__main__":
    main()
