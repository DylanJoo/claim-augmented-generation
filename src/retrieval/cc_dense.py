"""
cc-rerank (Claim-Claim rerank) on top of a doc-level run file -- dense
embedding variant.

Identical selection logic to cc.py (see that module's docstring for the
full MaxSim/mode explanation). The only difference is where sim_claim(c, c')
comes from:

    sim_claim(c, c') = dot(embed(c), embed(c'))

Claim vectors are read from pre-computed tevatron claim-level embedding
shards (see scripts/dense-index/), keyed by docid "{parent_id}#{i}",
instead of being scored by a local BM25 index at rerank time. 

`agg` controls how the claim x claim block between two docs is reduced to a
single doc-doc similarity, sim(d, d'):

  - "maxsim" (default): ColBERT-style MaxSim, sum_{c in d} max_{c' in d'}
    dot(embed(c), embed(c')). Because embeddings are unit-normalized,
    dot(c, c) = 1 is the global max any term in that sum can reach, so
    sim(d, d) = sum_{c in d} 1 = |d| exactly -- meaning sim(d, d) is always
    the largest entry in d's row. The row-normalization step (divide by
    that row's max) is therefore not an arbitrary [0, 1] rescale; it is
    mathematically equivalent to dividing by |d|, i.e. sim(d, d') / |d| ==
    mean_{c in d} max_{c' in d'} dot(c, c') -- "mean of MaxSim" over d's
    claims, exactly. This equivalence is specific to normalized dense
    embeddings (cosine's hard ceiling of 1) and would not hold for BM25.

  - "mean": plain mean over every one of the |d| x |d'| claim-claim pairs,
    sum_{c in d, c' in d'} dot(c, c') / (|d| * |d'|). Unlike "maxsim", this
    divides by both docs' claim counts, not just d's, and folds in the
    near-zero non-matching pairs rather than only the best match per claim
    -- a weaker but broader corroboration signal. Since each dot(c, c')
    term is already bounded in [-1, 1] (cosine), their mean is too, so no
    separate row-normalization is needed or applied here.

For the k-means-clustered variant of this same idea (claims collapsed into
topic buckets instead of scored pairwise), see cc_kmeans.py -- it was
detached from this module since it needs its own fit/predict machinery and
knobs (n_clusters, label_mode, top_m) that don't apply to
maxsim/mean at all.

query_reps/claim_filter (optional): the same query-claim similarity filter
dc_gap_dense.py uses, applied here before the claim-claim aggsim matrix is
built. Every claim in a doc counts equally in both "maxsim" and "mean", so a
claim that is essentially unrelated to the query still gets to (a) be the
arg-max some other claim latches onto in "maxsim", or (b) dilute the "mean"
average with a near-zero pair -- either way it can suppress a doc's true
claim-echo/overlap signal or, in "subtract" mode, its diversity signal, for
reasons that have nothing to do with the query. Dropping (or, for "topn",
reweighting the quota of) off-topic claims by query-claim cosine similarity
before aggregation keeps sim(d, d') focused on claims that are actually
about the query. Default claim_filter="none" keeps every claim, matching
prior behavior exactly.
"""
import copy
import glob
import logging
import pickle
from collections import defaultdict
from typing import List

import numpy as np

from utils import Result, Hit, load_run

logger = logging.getLogger(__name__)


def _pickle_load(path):
    with open(path, "rb") as f:
        reps, lookup = pickle.load(f)
    return np.asarray(reps), lookup

def _load_query_reps(query_reps_path):
    reps, lookup = _pickle_load(query_reps_path)
    return reps, {str(qid): i for i, qid in enumerate(lookup)}

def _load_claim_reps(claim_reps_path, needed_docids):
    files = sorted(glob.glob(claim_reps_path))
    if not files:
        raise FileNotFoundError(f"No claim rep shards matched: {claim_reps_path}")

    reps_by_id = {}
    for i, fpath in enumerate(files, 1):
        reps, lookup = _pickle_load(fpath)
        for vec, repid in zip(reps, lookup):
            parent_id = repid.rsplit("#", 1)[0]
            if parent_id in needed_docids:
                reps_by_id[repid] = vec.copy()
        del reps, lookup
        logger.info("cc-dense: [%d/%d] loaded shard %s, %d claim vector(s) matched so far",
                    i, len(files), fpath, len(reps_by_id))
    return reps_by_id

def _rows_by_parent(claim_reps_by_id):
    rows_by_parent = defaultdict(list)
    for claim_docid in claim_reps_by_id:
        parent_id = claim_docid.rsplit("#", 1)[0]
        rows_by_parent[parent_id].append(claim_docid)
    return rows_by_parent


def _filter_claims_for_doc(cids, claim_reps_by_id, query_vec, doc_relevance, claim_filter,
                            claims_per_doc, scale_topn_by_relevance, claim_sim_threshold):
    """Query-claim similarity filter, identical to dc_gap_dense.py's
    per-doc claim selection (see module docstring)."""
    sims = np.asarray(
        [float(np.dot(query_vec, claim_reps_by_id[cid])) for cid in cids],
        dtype=np.float32,
    )
    orders = np.argsort(sims)[::-1]  # descending by query-claim similarity

    if claim_filter == "topn":
        n = claims_per_doc
        if scale_topn_by_relevance:
            n = max(1, round(claims_per_doc * doc_relevance))
        keep = orders[:n]
    elif claim_filter == "threshold":
        keep = orders[sims[orders] >= claim_sim_threshold]
        if len(keep) == 0:
            keep = orders[:1]  # never zero out a doc entirely -- keep its single best claim
    else:
        raise ValueError(f"Unknown claim_filter: {claim_filter!r}")

    return [cids[i] for i in keep]


def _claim_aggsim_matrix(list_docids, claim_reps_by_id, rows_by_parent, dim, agg="maxsim",
                          list_relevance=None, query_vec=None, claim_filter="none",
                          claims_per_doc=5, scale_topn_by_relevance=False, claim_sim_threshold=0.0):
    """Doc-doc similarity via claim-level aggregation of dense embeddings.

    See module docstring for the "maxsim" vs "mean" trade-off. "maxsim" is
    row-normalized to [0, 1] (mathematically equivalent to averaging each
    max over d's claim count, see module docstring); "mean" is already
    bounded in [-1, 1] by construction and left as-is.

    claim_filter != "none" applies the query-claim similarity filter (see
    module docstring) to each doc's claims before they enter the matrix.
    """
    if agg not in ("maxsim", "mean"):
        raise ValueError(f"agg must be 'maxsim' or 'mean', got {agg!r}")

    if claim_filter != "none":
        values = np.asarray(list_relevance, dtype=np.float32)
        lo, hi = values.min(), values.max()
        norm_relevance = np.ones_like(values) if hi - lo < 1e-9 else (values - lo) / (hi - lo)

    n = len(list_docids)
    doc_of_claim = []
    claim_vecs = []
    for doc_idx, docid in enumerate(list_docids):
        claim_ids = rows_by_parent.get(docid) or []
        if claim_filter != "none" and claim_ids:
            claim_ids = _filter_claims_for_doc(
                claim_ids, claim_reps_by_id, query_vec, float(norm_relevance[doc_idx]),
                claim_filter, claims_per_doc, scale_topn_by_relevance, claim_sim_threshold,
            )
        vecs = [claim_reps_by_id[cid] for cid in claim_ids]
        if not vecs:
            vecs = [np.zeros(dim, dtype=np.float32)]
        for vec in vecs:
            claim_vecs.append(vec)
            doc_of_claim.append(doc_idx)
    doc_of_claim = np.asarray(doc_of_claim)
    claim_matrix = np.stack(claim_vecs).astype(np.float32)

    claim_sim = claim_matrix @ claim_matrix.T

    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        rows = claim_sim[doc_of_claim == i]
        for j in range(n):
            block = rows[:, doc_of_claim == j]
            sim[i, j] = block.max(axis=1).sum() if agg == "maxsim" else block.mean()

    if agg == "maxsim":
        row_max = sim.max(axis=1, keepdims=True)
        row_max[row_max < 1e-9] = 1.0
        return sim / row_max
    return sim


def _print_sim_matrix(list_docids, sim, agg, topn=10):
    """Print the top-N x top-N doc-doc similarity submatrix (hits are already
    rank-ordered by base relevance, so the first `topn` are the top-N)."""
    n = min(topn, len(list_docids))
    ids = [str(d)[:12] for d in list_docids[:n]]
    header = " " * 14 + "".join(f"{c:>8}" for c in ids)
    print(f"[cc_dense] top-{n} x top-{n} {agg} doc-doc matrix:")
    print(header)
    for i in range(n):
        row = "".join(f"{sim[i, j]:>8.3f}" for j in range(n))
        print(f"{ids[i]:<14}{row}")


def _select(
    hits,
    claim_reps_by_id,
    rows_by_parent,
    dim,
    k,
    lambda_mult,
    mode,
    agg,
    query_vec=None,
    claim_filter="none",
    claims_per_doc=5,
    scale_topn_by_relevance=False,
    claim_sim_threshold=0.0,
):
    if len(hits) <= 1:
        return hits
    if mode not in ("subtract", "add"):
        raise ValueError(f"mode must be 'subtract' (MMR) or 'add' (claim-echo boost), got {mode!r}")
    sign = -1.0 if mode == "subtract" else 1.0

    relevance = np.asarray([h.score for h in hits], dtype=np.float32)
    list_docids = [h.docid for h in hits]

    sim = _claim_aggsim_matrix(
        list_docids=list_docids,
        claim_reps_by_id=claim_reps_by_id,
        rows_by_parent=rows_by_parent,
        dim=dim,
        agg=agg,
        list_relevance=relevance,
        query_vec=query_vec,
        claim_filter=claim_filter,
        claims_per_doc=claims_per_doc,
        scale_topn_by_relevance=scale_topn_by_relevance,
        claim_sim_threshold=claim_sim_threshold,
    )
    _print_sim_matrix(list_docids, sim, agg=agg, topn=10)

    n = len(hits)
    n_select = min(k, n)
    selected = []
    selected_scores = []
    max_sim_to_selected = np.zeros(n, dtype=np.float32)

    for _ in range(n_select):
        scores = lambda_mult * relevance + sign * (1 - lambda_mult) * max_sim_to_selected
        scores[selected] = -np.inf
        pick = int(np.argmax(scores))
        selected_scores.append(float(scores[pick]))
        selected.append(pick)
        max_sim_to_selected = np.maximum(max_sim_to_selected, sim[pick])

    # "subtract" mode's selected_scores are already non-increasing by
    # construction (max_sim_to_selected only grows, so each round's best
    # achievable score can only shrink). "add" mode has no such guarantee:
    # round 1's score is pure lambda*relevance (max_sim_to_selected is
    # exactly zero -- nothing is selected yet), while every later round adds
    # (1 - lambda_mult) * max_sim_to_selected on top. Once that boost term
    # is weighted heavily (low lambda_mult, e.g. the boost-doc lambda=0.1/0.2
    # runs), round 1's score is virtually guaranteed to be the *smallest*
    # value in the whole sequence -- so a running minimum (previously used
    # here) never moves past it, and every doc in the query is written out
    # with that one identical score. Downstream TREC tooling sorts by the
    # score column, not by file order, so a tied column silently discards
    # the entire selection order this loop just computed. Re-sorting the
    # achieved values descending keeps every actually-computed magnitude but
    # reassigns them by rank, guaranteeing a strictly decreasing column that
    # matches the greedy order above.
    if mode == "add" and len(selected_scores) > 1:
        selected_scores = sorted(selected_scores, reverse=True)

    return [
        Hit(docid=hits[idx].docid, score=selected_scores[rank - 1], rank=rank, content_dict=hits[idx].content_dict)
        for rank, idx in enumerate(selected, start=1)
    ]


def run(
    inputs: List[Result],
    run_file: str,
    corpus: List[str],
    claim_reps: str,
    k: int = 100,
    lambda_mult: float = 0.9,
    mode: str = "subtract",
    agg: str = "maxsim",
    query_reps: str = None,
    claim_filter: str = "none",
    claims_per_doc: int = 5,
    scale_topn_by_relevance: bool = False,
    claim_sim_threshold: float = 0.0,
) -> List[Result]:
    """query_reps/claim_filter: optional query-claim similarity filter on
    each pooled doc's claims before the claim-claim aggsim matrix is built
    (see module docstring). claim_filter="none" (default) keeps every claim
    found in the shards, matching pre-existing behavior; query_reps is
    required for "topn"/"threshold"."""
    if claim_filter != "none" and not query_reps:
        raise ValueError(f"claim_filter={claim_filter!r} requires query_reps")

    logger.info("cc-dense: mode=%s, agg=%s, base relevance from run file %s, pool k=%d, lambda=%.2f, claim_reps=%s, claim_filter=%s",
                mode, agg, run_file, k, lambda_mult, claim_reps, claim_filter)
    base_run = load_run(run_file, k=k)
    claim_corpus = {}  # corpus text unused downstream; skip loading to save memory

    needed_docids = {docid for pool in base_run.values() for docid, _ in pool}
    logger.info("cc-dense: %d unique pooled docid(s) across %d topic(s) need claim embeddings",
                len(needed_docids), len(base_run))

    claim_reps_by_id = _load_claim_reps(claim_reps, needed_docids)
    dim = next(iter(claim_reps_by_id.values())).shape[0]
    rows_by_parent = _rows_by_parent(claim_reps_by_id)

    q_reps, q_pos = (None, {})
    if query_reps:
        q_reps, q_pos = _load_query_reps(query_reps)
        logger.info("cc-dense: loaded %d query embedding(s) from %s for claim_filter=%r",
                    len(q_pos), query_reps, claim_filter)

    outputs = copy.deepcopy(inputs)
    for i, inp in enumerate(inputs):
        qid = str(inp.topic["qid"])
        pool = base_run.get(qid, [])

        query_vec = None
        if claim_filter != "none":
            query_vec = q_reps[q_pos[qid]]

        hits = [
            Hit(
                docid=docid,
                score=score,
                rank=rank,
                content_dict={
                    "text": claim_corpus.get(docid, {}).get("statements"),
                    "title": claim_corpus.get(docid, {}).get("title"),
                },
            )
            for rank, (docid, score) in enumerate(pool, start=1)
        ]

        n_claims = sum(len(rows_by_parent.get(docid, [])) for docid, _ in pool)
        logger.info("cc-dense: qid=%s pooled %d documents, %d claims found in embedding shards", qid, len(pool), n_claims)

        outputs[i].hits = hits
        outputs[i].evidences = _select(
            hits, claim_reps_by_id, rows_by_parent, dim, k, lambda_mult, mode=mode, agg=agg,
            query_vec=query_vec,
            claim_filter=claim_filter,
            claims_per_doc=claims_per_doc,
            scale_topn_by_relevance=scale_topn_by_relevance,
            claim_sim_threshold=claim_sim_threshold,
        )

    return outputs
