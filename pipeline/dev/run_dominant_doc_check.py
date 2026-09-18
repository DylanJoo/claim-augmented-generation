"""
Dominant-document report -- checks whether the top-relevance ("first-picked")
document in a cc_dense.py MMR pool is a broad/generic document that creates
false redundancy signals for other, more informative docs, using
src/retrieval/dev/dominant_doc_check.py.

Read-only: loads the same inputs as pipeline/run_cc_dense.py, builds the same
claim-claim aggsim matrix _select uses (claim_filter="none", matching
_select's default), and reports on doc rank 1 (or --target-rank) without
writing a run file or changing any selection behavior.

Usage:
    python pipeline/dev/run_dominant_doc_check.py \
        --topics <topics.jsonl> \
        --run-file <path/to/doc-level-run.txt> \
        --claim-reps <'claims_emb/claims_emb.*.pkl'> \
        [--corpus <path/to/collection.jsonl.gz> [<more files/globs>...]] \
        [--qid <QID>] [--k 100] [--agg maxsim|mean] \
        [--target-rank 1] [--top-claims 3]

--corpus is optional and only used to print the magnet claim's actual text
(via each doc's "statements" field) so you can eyeball whether it reads as
generic ("this report covers various aspects of...") or genuinely specific.
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

import dominant_doc_check as ddc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _open_file(fpath):
    return gzip.open(fpath, "rt", encoding="utf-8") if fpath.endswith(".gz") else open(fpath, encoding="utf-8")


def _load_statements_subset(corpus_patterns, needed_docids):
    """Same scan-and-filter pattern as cc_dense._load_claim_reps -- only
    keeps statements for docids actually in the pool being checked."""
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
    """Mirrors _claim_aggsim_matrix's own claim-gathering order exactly
    (claim_filter="none" path) so a global row index into claim_sim/
    doc_of_claim can be resolved back to the real "{parent}#{i}" claim id.
    A `None` entry marks the synthetic all-zero row _claim_aggsim_matrix
    inserts for a doc with no claims found in the shards."""
    flat = []
    for docid in list_docids:
        ids = rows_by_parent.get(docid) or []
        flat.extend(ids) if ids else flat.append(None)
    return flat


def main():
    parser = argparse.ArgumentParser(description="Check whether the top-relevance doc in a cc_dense pool is a dominant/generic document")
    parser.add_argument("--topics", required=True)
    parser.add_argument("--run-file", required=True)
    parser.add_argument("--claim-reps", required=True)
    parser.add_argument("--corpus", nargs="+", default=None,
                        help="Optional -- if given, prints the magnet claim's actual statement text")
    parser.add_argument("--qid", default=None, help="Restrict to one topic qid (default: all topics in --topics)")
    parser.add_argument("--k", type=int, default=100, help="Pool size taken from the run file (default: 100)")
    parser.add_argument("--agg", choices=["maxsim", "mean"], default="maxsim")
    parser.add_argument("--target-rank", type=int, default=1,
                        help="1-based base-relevance rank of the doc to check (default: 1, the doc _select always picks first)")
    parser.add_argument("--top-claims", type=int, default=3, help="How many top magnet claims to print (default: 3)")
    args = parser.parse_args()

    topics = load_topics(args.topics)
    qids = [str(t["qid"]) for t in topics] if not args.qid else [args.qid]
    base_run = load_run(args.run_file, k=args.k)

    needed_docids = {docid for qid in qids for docid, _ in base_run.get(qid, [])}
    logger.info("Loading claim embeddings for %d pooled docid(s)", len(needed_docids))
    claim_reps_by_id = _load_claim_reps(args.claim_reps, needed_docids)
    dim = next(iter(claim_reps_by_id.values())).shape[0]
    rows_by_parent = _rows_by_parent(claim_reps_by_id)

    statements = _load_statements_subset(args.corpus, needed_docids) if args.corpus else {}

    for qid in qids:
        pool = base_run.get(qid, [])
        if not pool:
            continue
        list_docids = [docid for docid, _ in pool]
        if args.target_rank > len(list_docids):
            logger.warning("qid=%s: pool has only %d docs, skipping (target-rank=%d)", qid, len(list_docids), args.target_rank)
            continue

        sim, claim_sim, doc_of_claim = _claim_aggsim_matrix(
            list_docids=list_docids,
            claim_reps_by_id=claim_reps_by_id,
            rows_by_parent=rows_by_parent,
            dim=dim,
            agg=args.agg,
        )
        target_idx = args.target_rank - 1

        row_report = ddc.row_breadth_report(list_docids, sim, target_idx=target_idx)
        magnet_report = ddc.claim_magnet_report(
            list_docids, claim_sim, doc_of_claim, target_idx=target_idx, top_claims=args.top_claims
        )
        v = ddc.verdict(row_report, magnet_report)
        claim_ids_flat = _claim_ids_flat(list_docids, rows_by_parent)

        print(f"\n=== qid={qid}  target=rank-{args.target_rank} docid={row_report['docid']} (pool size {len(list_docids)}) ===")
        print(f"  row mean sim to rest of pool: {row_report['target_row_mean']:.4f}  "
              f"(rank {row_report['pool_row_mean_rank']}/{len(list_docids)} among all pooled docs, 1=broadest)")
        print(f"  row median sim: {row_report['target_row_median']:.4f}  "
              f"max (excl self): {row_report['target_row_max_excl_self']:.4f}")
        print(f"  pool's own median pairwise sim: {row_report['pool_median_pairwise_sim']:.4f}  "
              f"-- target is >= that against {row_report['n_docs_above_pool_median_sim']}/{row_report['n_other_docs']} other docs")
        print(f"  claims in target doc: {magnet_report['n_claims']}")
        for c in magnet_report["top_magnet_claims"]:
            print(f"    claim local-idx {c['claim_local_idx']}: wins the argmax for "
                  f"{c['win_count']}/{magnet_report['n_other_docs']} other docs ({c['win_fraction']:.0%})")
            repid = claim_ids_flat[c["claim_global_idx"]]
            if repid and statements:
                parent, _, suffix = repid.rpartition("#")
                stmts = statements.get(parent)
                if stmts and suffix.isdigit() and int(suffix) < len(stmts):
                    print(f"      text: {stmts[int(suffix)][:200]!r}")
        print(f"  verdict: broad_across_pool={v['is_broad_across_pool']}  "
              f"single_claim_magnet={v['is_single_claim_magnet']}  "
              f"=> likely_dominant_doc={v['likely_dominant_doc']}")


if __name__ == "__main__":
    main()
