"""
Nugget-per-document coverage report -- for a cc_dense.py retrieval pool,
classifies claims ("nuggets") by how many OTHER documents carry a similar
claim, using src/retrieval/dev/nugget_coverage.py. Answers: how many
documents carry a "hard"/unique nugget (a fact found nowhere else in the
pool), and do those documents otherwise look like ordinary, on-topic pool
members (mostly common/mainstream claims, plus one extra detail) or outliers?

Read-only: loads the same inputs as pipeline/run_cc_dense.py, builds the same
claim-claim matrix _select uses (claim_filter="none"), and reports without
writing a run file or changing any selection behavior. Claim-embedding shard
loading is cached across runs on the same pool (see _claim_cache.py) since it
otherwise takes 15-20 minutes per topic.

Usage:
    python pipeline/dev/run_nugget_coverage.py \
        --topics <topics.jsonl> \
        --run-file <path/to/doc-level-run.txt> \
        --claim-reps <'claims_emb/claims_emb.*.pkl'> \
        [--corpus <path/to/collection.jsonl.gz> [<more files/globs>...]] \
        [--qid <QID>] [--k 100] [--agg maxsim|mean] \
        [--threshold 0.75] [--hard-max 0] [--common-min 3] \
        [--examples 5] [--example-claims 2]

--corpus is optional and only used to print actual claim text so you can
eyeball the hard-vs-common nuggets and their topical relatedness.
"""
import argparse
import glob
import gzip
import json
import logging
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src", "retrieval", "dev"))

from retrieval.cc_dense import _load_claim_reps, _rows_by_parent, _claim_aggsim_matrix
from utils import load_run, load_topics

import nugget_coverage as nug
from _claim_cache import load_claim_reps_cached

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _open_file(fpath):
    return gzip.open(fpath, "rt", encoding="utf-8") if fpath.endswith(".gz") else open(fpath, encoding="utf-8")


def _load_statements_subset(corpus_patterns, needed_docids):
    files = []
    for pat in corpus_patterns:
        files.extend(sorted(glob.glob(pat)) or [pat])
    statements = {}
    remaining = set(needed_docids)
    for fpath in files:
        if not remaining:
            break
        with _open_file(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                docid = doc.get("id")
                if docid in remaining:
                    statements[docid] = doc.get("statements") or []
                    remaining.discard(docid)
    return statements


def _claim_ids_flat(list_docids, rows_by_parent):
    """Mirrors _claim_aggsim_matrix's own claim-gathering order (claim_filter
    ="none" path) so a global row index can be resolved to "{parent}#{i}"."""
    flat = []
    for docid in list_docids:
        ids = rows_by_parent.get(docid) or []
        flat.extend(ids) if ids else flat.append(None)
    return flat


def _claim_text(claim_ids_flat, statements, global_idx, width=200):
    repid = claim_ids_flat[global_idx]
    if not repid or not statements:
        return None
    parent, _, suffix = repid.rpartition("#")
    stmts = statements.get(parent)
    if stmts and suffix.isdigit() and int(suffix) < len(stmts):
        return stmts[int(suffix)][:width]
    return None


def main():
    parser = argparse.ArgumentParser(description="Nugget-per-document coverage report over a cc_dense pool")
    parser.add_argument("--topics", required=True)
    parser.add_argument("--run-file", required=True)
    parser.add_argument("--claim-reps", required=True)
    parser.add_argument("--corpus", nargs="+", default=None,
                        help="Optional -- if given, prints example claim text")
    parser.add_argument("--qid", default=None, help="Restrict to one topic qid (default: all topics in --topics)")
    parser.add_argument("--k", type=int, default=100, help="Pool size taken from the run file (default: 100)")
    parser.add_argument("--agg", choices=["maxsim", "mean"], default="maxsim")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Cosine similarity above which two claims count as the same nugget (default: 0.75)")
    parser.add_argument("--hard-max", type=int, default=1,
                        help="A claim with coverage <= this many other docs counts as a 'hard'/unique nugget "
                             "-- 0 = appears nowhere else, 1 = appears in at most one other doc (default: 1)")
    parser.add_argument("--common-min", type=int, default=3,
                        help="A claim with coverage >= this many other docs counts as a 'common' nugget (default: 3)")
    parser.add_argument("--examples", type=int, default=5, help="How many example hard-nugget docs to print (default: 5)")
    parser.add_argument("--example-claims", type=int, default=2, help="How many hard/common claims to print per example doc (default: 2)")
    args = parser.parse_args()

    topics = load_topics(args.topics)
    qids = [str(t["qid"]) for t in topics] if not args.qid else [args.qid]
    base_run = load_run(args.run_file, k=args.k)

    needed_docids = {docid for qid in qids for docid, _ in base_run.get(qid, [])}
    logger.info("Loading claim embeddings for %d pooled docid(s) (cached across runs)", len(needed_docids))
    claim_reps_by_id = load_claim_reps_cached(_load_claim_reps, args.claim_reps, needed_docids)
    dim = next(iter(claim_reps_by_id.values())).shape[0]
    rows_by_parent = _rows_by_parent(claim_reps_by_id)

    statements = _load_statements_subset(args.corpus, needed_docids) if args.corpus else {}

    for qid in qids:
        pool = base_run.get(qid, [])
        if not pool:
            continue
        list_docids = [docid for docid, _ in pool]
        relevance_rank = {docid: rank for rank, docid in enumerate(list_docids, start=1)}

        sim, claim_sim, doc_of_claim = _claim_aggsim_matrix(
            list_docids=list_docids,
            claim_reps_by_id=claim_reps_by_id,
            rows_by_parent=rows_by_parent,
            dim=dim,
            agg=args.agg,
        )
        claim_ids_flat = _claim_ids_flat(list_docids, rows_by_parent)

        pcts = nug.similarity_percentiles(claim_sim, doc_of_claim)
        coverage = nug.nugget_coverage_counts(claim_sim, doc_of_claim, threshold=args.threshold)
        dist = nug.pool_distribution(coverage)
        reports = nug.per_doc_report(list_docids, doc_of_claim, coverage, hard_max=args.hard_max, common_min=args.common_min)

        n_total_claims = len(coverage)
        docs_with_hard = [r for r in reports if r["n_hard"] > 0]

        print(f"\n=== qid={qid}  pool size {len(list_docids)}  total claims {n_total_claims} ===")
        print(f"  cross-doc claim-claim cosine percentiles: " +
              "  ".join(f"p{p}={v:.3f}" for p, v in pcts.items()) +
              f"   (threshold={args.threshold})")
        print(f"  nugget coverage distribution (n_other_docs -> n_claims): "
              + "  ".join(f"{k}:{v}" for k, v in sorted(dist.items())))
        print(f"  documents with >=1 hard nugget (coverage<={args.hard_max}): "
              f"{len(docs_with_hard)}/{len(reports)}")

        if docs_with_hard:
            avg_common_frac = float(np.mean([r["common_fraction"] for r in docs_with_hard]))
            avg_rank = float(np.mean([relevance_rank[r["docid"]] for r in docs_with_hard]))
            print(f"  among those docs: avg fraction of their OWN claims that are common (>={args.common_min} other docs) "
                  f"= {avg_common_frac:.2f}   avg base-relevance rank = {avg_rank:.1f}/{len(list_docids)}")

        top = sorted(docs_with_hard, key=lambda r: -r["n_hard"])[: args.examples]
        for r in top:
            print(f"\n  --- docid={r['docid']}  base-rank={relevance_rank[r['docid']]}  "
                  f"n_claims={r['n_claims']}  n_hard={r['n_hard']}  n_common={r['n_common']}  "
                  f"common_fraction={r['common_fraction']:.2f} ---")
            hard_idxs = r["claim_global_idx"][r["claim_coverage"] <= args.hard_max]
            common_idxs = r["claim_global_idx"][r["claim_coverage"] >= args.common_min]
            for gi in hard_idxs[: args.example_claims]:
                gi = int(gi)
                cov = int(coverage[gi])
                text = _claim_text(claim_ids_flat, statements, gi)
                label = f"[hard, cov={cov}]"
                print(f"    {label} {text!r}" if text else f"    {label} claim_global_idx={gi}")
                if cov >= 1:
                    match = nug.find_sharing_doc(
                        claim_sim, doc_of_claim, list_docids,
                        claim_global_idx=gi, threshold=args.threshold, exclude_doc_idx=r["doc_idx"],
                    )
                    if match:
                        shared_text = _claim_text(claim_ids_flat, statements, match["claim_global_idx"])
                        print(f"      -> shared with {match['docid']} (sim={match['similarity']:.3f})"
                              + (f": {shared_text!r}" if shared_text else ""))
            for gi in common_idxs[: args.example_claims]:
                gi = int(gi)
                cov = int(coverage[gi])
                text = _claim_text(claim_ids_flat, statements, gi)
                print(f"    [common, cov={cov}] {text!r}" if text else f"    [common, cov={cov}] claim_global_idx={gi}")


if __name__ == "__main__":
    main()
