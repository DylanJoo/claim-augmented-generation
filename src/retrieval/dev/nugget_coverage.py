"""
Nugget-per-document coverage: classifies claims ("nuggets") in a retrieval
pool by how many DISTINCT OTHER documents carry a similar claim -- the same
"does something else already cover this claim" direction cc_dense.py's
_select coverage tracker uses (see its votes/coverage_threshold), computed
once over the whole pool instead of incrementally during greedy selection.

"Hard"/unique nuggets: coverage <= hard_max other docs (default 0 -- this
fact appears nowhere else in the pool). "Common" nuggets: coverage >=
common_min other docs (default 3 -- echoed widely; likely core/background
information everyone touching the topic repeats).

The interesting question this exists to answer: do documents that carry a
hard nugget otherwise look like ordinary, on-topic pool members (i.e. most of
their OTHER claims are common/mainstream, so they're "the same topic, plus
one extra detail nobody else has"), or are they off-topic outliers that just
happen to contain one stray claim? per_doc_report's common_fraction is the
number to look at for that.

Read-only: does not change any selection or scoring behavior.
"""
import numpy as np


def nugget_coverage_counts(claim_sim, doc_of_claim, threshold=0.75):
    """coverage[c] = number of DISTINCT other documents with at least one
    claim >= threshold similar to claim c (a doc's own other claims don't
    count, and multiple matching claims within the same other doc count once)."""
    n_claims = claim_sim.shape[0]
    coverage = np.zeros(n_claims, dtype=np.int32)
    for c in range(n_claims):
        own_doc = doc_of_claim[c]
        hit_docs = set(doc_of_claim[claim_sim[c] >= threshold].tolist())
        hit_docs.discard(own_doc)
        coverage[c] = len(hit_docs)
    return coverage


def pool_distribution(coverage):
    """Histogram of coverage counts across every claim in the pool."""
    values, counts = np.unique(coverage, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def similarity_percentiles(claim_sim, doc_of_claim, percentiles=(50, 75, 90, 95, 99)):
    """Cross-document claim-claim similarity percentiles, to help pick/sanity
    check `threshold` -- only pairs from DIFFERENT docs (same-doc pairs are
    almost all near-duplicates of the doc's own writing style, not signal)."""
    n = len(doc_of_claim)
    same_doc = doc_of_claim[:, None] == doc_of_claim[None, :]
    cross = claim_sim[~same_doc]
    return {p: float(np.percentile(cross, p)) for p in percentiles}


def per_doc_report(list_docids, doc_of_claim, coverage, hard_max=0, common_min=3):
    n = len(list_docids)
    reports = []
    for i in range(n):
        idxs = np.where(doc_of_claim == i)[0]
        if len(idxs) == 0:
            continue
        cov = coverage[idxs]
        n_hard = int((cov <= hard_max).sum())
        n_common = int((cov >= common_min).sum())
        reports.append({
            "doc_idx": i,
            "docid": list_docids[i],
            "n_claims": len(idxs),
            "n_hard": n_hard,
            "n_common": n_common,
            "hard_fraction": n_hard / len(idxs),
            "common_fraction": n_common / len(idxs),
            "claim_global_idx": idxs,
            "claim_coverage": cov,
        })
    return reports


def find_sharing_doc(claim_sim, doc_of_claim, list_docids, claim_global_idx, threshold, exclude_doc_idx):
    """For a claim that IS covered (coverage >= 1), name one other doc that
    shares it and that doc's best-matching claim's global index -- lets a
    caller print the paired example text."""
    own_doc = doc_of_claim[claim_global_idx]
    sims = claim_sim[claim_global_idx].copy()
    sims[doc_of_claim == own_doc] = -1.0
    if exclude_doc_idx is not None:
        sims[doc_of_claim == exclude_doc_idx] = -1.0
    best = int(sims.argmax())
    if sims[best] < threshold:
        return None
    return {
        "docid": list_docids[doc_of_claim[best]],
        "claim_global_idx": best,
        "similarity": float(sims[best]),
    }
