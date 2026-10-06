"""
Whole-document embedding similarity check for rare-vs-mainstream-nugget
documents. Tests the hypothesis: a document that carries a rare/unique qrel
subtopic ("nugget") still looks very similar, at the whole-document dense
embedding level, to documents that only carry mainstream/common subtopics --
i.e. document-level similarity is dominated by the shared mainstream content
and effectively blind to the rare detail. This matters for any doc-doc
(non-claim) MMR/dedup step: if true, it would penalize a rare-nugget document
for looking redundant with the mainstream cluster even though it uniquely
covers something nothing else does.

Uses the qrel's diversity subtopic judgments (via
src/evaluator/nugget_redundancy.py's load_subtopic_map) to define two groups
for one topic:
  rare group:       docs with >=1 subtopic judged relevant for <= --rare-max
                     other docs pool-wide.
  mainstream group: docs whose ENTIRE judged subtopic set is common
                     (each subtopic judged relevant for >= --common-min docs)
                     -- i.e. they never touch a rare subtopic at all.

Then loads each group's single whole-document dense embedding (Qwen3, or
whatever --docs-emb points at) and reports:
  rare  x mainstream cosine similarity (cross-group)
  mainstream x mainstream cosine similarity (in-group baseline)
  rare  x rare cosine similarity (in-group, if >1 rare doc)

Read-only. Embedding shard loading is cached across runs (see
_claim_cache.py's generic cache, reused here for doc reps too).

Usage:
    python pipeline/dev/run_rare_nugget_doc_similarity.py \
        --qrel <path/to/ragtime25-test-request.qrel> \
        --docs-emb '<docs_emb/docs_emb.*.pkl>' \
        --qid 1001 [--rare-max 2] [--common-min 20] [--mainstream-sample 100]
"""
import argparse
import glob
import logging
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src", "evaluator"))

from nugget_redundancy import load_subtopic_map
from _claim_cache import load_claim_reps_cached

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _pickle_load(path):
    with open(path, "rb") as f:
        reps, lookup = pickle.load(f)
    return np.asarray(reps), lookup


def _load_doc_reps(doc_reps_glob, needed_docids):
    files = sorted(glob.glob(doc_reps_glob))
    if not files:
        raise FileNotFoundError(f"No doc rep shards matched: {doc_reps_glob}")
    reps_by_id = {}
    for i, fpath in enumerate(files, 1):
        reps, lookup = _pickle_load(fpath)
        for vec, repid in zip(reps, lookup):
            if repid in needed_docids:
                reps_by_id[repid] = vec.copy()
        del reps, lookup
        logger.info("doc reps: [%d/%d] loaded %s, %d/%d matched so far",
                    i, len(files), fpath, len(reps_by_id), len(needed_docids))
    return reps_by_id


def _cos_matrix(mat_a, mat_b):
    a = mat_a / np.linalg.norm(mat_a, axis=1, keepdims=True)
    b = mat_b / np.linalg.norm(mat_b, axis=1, keepdims=True)
    return a @ b.T


def main():
    parser = argparse.ArgumentParser(description="Rare-vs-mainstream-nugget whole-document similarity check")
    parser.add_argument("--qrel", required=True)
    parser.add_argument("--docs-emb", required=True)
    parser.add_argument("--qid", required=True)
    parser.add_argument("--rare-max", type=int, default=2,
                        help="A subtopic judged relevant for <= this many docs counts as rare (default: 2)")
    parser.add_argument("--common-min", type=int, default=20,
                        help="A subtopic judged relevant for >= this many docs counts as common (default: 20)")
    parser.add_argument("--mainstream-sample", type=int, default=100,
                        help="Cap on how many mainstream-only docs to load/compare against (default: 100)")
    args = parser.parse_args()

    doc_subtopics, _universe = load_subtopic_map(args.qrel)
    ds = doc_subtopics.get(args.qid, {})
    if not ds:
        raise SystemExit(f"No qrel rows found for qid={args.qid}")

    pop = {}
    for docid, subs in ds.items():
        for s in subs:
            pop[s] = pop.get(s, 0) + 1
    rare_subs = {s for s, c in pop.items() if c <= args.rare_max}
    common_subs = {s for s, c in pop.items() if c >= args.common_min}

    rare_docs = [d for d, subs in ds.items() if subs & rare_subs]
    mainstream_docs = [d for d, subs in ds.items() if subs and subs <= common_subs][: args.mainstream_sample]

    logger.info("qid=%s: %d rare-nugget docs, %d mainstream-only docs (capped at %d)",
                args.qid, len(rare_docs), len(mainstream_docs), args.mainstream_sample)
    if not rare_docs:
        raise SystemExit("No rare-nugget documents found under these thresholds -- try raising --rare-max")

    needed = set(rare_docs) | set(mainstream_docs)
    reps_by_id = load_claim_reps_cached(_load_doc_reps, args.docs_emb, needed)
    missing = needed - reps_by_id.keys()
    if missing:
        logger.warning("%d docid(s) had no doc embedding found: %s", len(missing), list(missing)[:5])
    rare_docs = [d for d in rare_docs if d in reps_by_id]
    mainstream_docs = [d for d in mainstream_docs if d in reps_by_id]

    rare_mat = np.stack([reps_by_id[d] for d in rare_docs]).astype(np.float32)
    main_mat = np.stack([reps_by_id[d] for d in mainstream_docs]).astype(np.float32)

    cross = _cos_matrix(rare_mat, main_mat)  # rare x mainstream
    main_main = _cos_matrix(main_mat, main_mat)
    main_main_offdiag = main_main[~np.eye(len(mainstream_docs), dtype=bool)]

    print(f"\n=== qid={args.qid}  rare docs={len(rare_docs)}  mainstream-only docs={len(mainstream_docs)} ===")
    print(f"  mainstream x mainstream cosine (baseline, off-diagonal): "
          f"mean={main_main_offdiag.mean():.4f}  median={np.median(main_main_offdiag):.4f}  "
          f"p90={np.percentile(main_main_offdiag,90):.4f}")
    print(f"  rare x mainstream cosine (cross-group): "
          f"mean={cross.mean():.4f}  median={np.median(cross):.4f}  max={cross.max():.4f}")

    if len(rare_docs) > 1:
        rare_rare = _cos_matrix(rare_mat, rare_mat)
        rare_rare_offdiag = rare_rare[~np.eye(len(rare_docs), dtype=bool)]
        print(f"  rare x rare cosine (in-group, off-diagonal): "
              f"mean={rare_rare_offdiag.mean():.4f}  median={np.median(rare_rare_offdiag):.4f}")

    print("\n  per rare-doc: mean/max cosine to the mainstream-only group, and its best-matching mainstream doc")
    for i, d in enumerate(rare_docs):
        row = cross[i]
        best = int(row.argmax())
        print(f"    {d}: mean={row.mean():.4f}  max={row.max():.4f}  "
              f"best_match={mainstream_docs[best]} (sim={row[best]:.4f})")


if __name__ == "__main__":
    main()
