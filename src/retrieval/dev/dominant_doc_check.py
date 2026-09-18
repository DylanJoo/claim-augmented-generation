"""
Diagnose whether the highest-relevance ("first-picked") document in a
cc_dense.py MMR candidate pool is a "dominant"/generic document -- one that
touches a little bit of everything without being genuinely on-topic for most
of it -- which would make _select's max_sim_to_selected look falsely high for
many other, more specific/informative docs once this doc is picked in round 1
(see src/retrieval/cc_dense.py's _select: round 1's score is pure relevance,
so hits[0]/rank-1 is always what gets picked first, and every later round's
max_sim_to_selected only grows from whatever it contributes).

Two independent signals, both computed straight from cc_dense.py's own
_claim_aggsim_matrix output (sim, claim_sim, doc_of_claim) so this sees
exactly what _select sees -- no separate re-scoring path:

1. Row-breadth (row_breadth_report): sim[target, j] compared across the
   whole pool, and against every other doc's own average similarity to the
   rest of the pool. A genuinely on-topic-with-a-few-docs document should be
   highly similar to a handful of docs and weakly similar to the rest; a
   "broad but shallow" document is elevated across most of the pool instead
   of concentrated on the docs that actually overlap with it.

2. Claim-level "magnet" (claim_magnet_report): for each other doc j, which
   one of target's claims drives sim(target, j) -- i.e. wins the row_max in
   the claim x claim block, the same asymmetric direction _select's
   max_sim_to_selected and its coverage tracker already use. If a single
   claim of target wins that argmax for most of the pool, that's the "just a
   tiny bit covers everything" pattern: one broad/generic claim, not the
   document as a whole, is responsible for the inflated redundancy signal.

Read-only: does not change any selection behavior, just reports on a pool
that's already been scored.
"""
import numpy as np


def row_breadth_report(list_docids, sim, target_idx=0):
    """How broadly similar is doc `target_idx` to the rest of the pool,
    compared to how broadly similar every other pooled doc is to the rest of
    the pool? A dominant/generic doc should rank near the top here -- it's
    elevated against nearly everyone, not just its real topical neighbors."""
    n = len(list_docids)
    others = np.array([j for j in range(n) if j != target_idx])
    target_row = sim[target_idx, others]

    row_means = np.array([
        float(sim[i, [j for j in range(n) if j != i]].mean()) for i in range(n)
    ])
    off_diag = sim[~np.eye(n, dtype=bool)]
    pool_median_sim = float(np.median(off_diag)) if off_diag.size else 0.0

    rank = int((row_means > row_means[target_idx]).sum()) + 1  # 1 = broadest in the pool

    return {
        "docid": list_docids[target_idx],
        "n_other_docs": len(others),
        "target_row_mean": float(target_row.mean()),
        "target_row_median": float(np.median(target_row)),
        "target_row_max_excl_self": float(target_row.max()) if len(target_row) else 0.0,
        "pool_row_mean_rank": rank,
        "pool_median_pairwise_sim": pool_median_sim,
        "n_docs_above_pool_median_sim": int((target_row >= pool_median_sim).sum()),
    }


def claim_magnet_report(list_docids, claim_sim, doc_of_claim, target_idx=0, top_claims=3):
    """For each other pooled doc, which single claim of `target_idx` wins the
    row_max (mirrors _select's own coverage-tracker direction, cc_dense.py
    lines ~374-379). Tallies how often each of target's claims is that
    winner across the whole pool -- a single claim winning for most of the
    pool is the "one generic claim drags the whole doc up" signature."""
    n = len(list_docids)
    target_claim_idx = np.where(doc_of_claim == target_idx)[0]
    win_counts = np.zeros(len(target_claim_idx), dtype=np.int64)
    winner_sims = []

    for j in range(n):
        if j == target_idx:
            continue
        j_idx = np.where(doc_of_claim == j)[0]
        if len(j_idx) == 0 or len(target_claim_idx) == 0:
            continue
        block = claim_sim[np.ix_(target_claim_idx, j_idx)]
        row_max = block.max(axis=1)
        winner_local = int(row_max.argmax())
        win_counts[winner_local] += 1
        winner_sims.append(float(row_max[winner_local]))

    order = np.argsort(-win_counts)
    n_other_docs = max(n - 1, 1)
    top = [
        {
            "claim_local_idx": int(li),
            "claim_global_idx": int(target_claim_idx[li]),
            "win_count": int(win_counts[li]),
            "win_fraction": float(win_counts[li] / n_other_docs),
        }
        for li in order[:top_claims]
    ]
    return {
        "docid": list_docids[target_idx],
        "n_claims": len(target_claim_idx),
        "n_other_docs": n_other_docs,
        "top_magnet_claims": top,
        "mean_winner_sim": float(np.mean(winner_sims)) if winner_sims else None,
    }


def verdict(row_report, magnet_report, row_rank_threshold=3, magnet_fraction_threshold=0.5):
    """Heuristic flag, not a hard rule -- both thresholds are eyeball starting
    points (top-3-broadest-in-the-pool, one claim winning half the pool),
    tune per corpus/pool size. Always look at the underlying numbers too."""
    top_magnet_frac = (
        magnet_report["top_magnet_claims"][0]["win_fraction"]
        if magnet_report["top_magnet_claims"] else 0.0
    )
    is_broad = row_report["pool_row_mean_rank"] <= row_rank_threshold
    is_magnet_driven = top_magnet_frac >= magnet_fraction_threshold
    return {
        "is_broad_across_pool": is_broad,
        "is_single_claim_magnet": is_magnet_driven,
        "likely_dominant_doc": is_broad and is_magnet_driven,
    }
