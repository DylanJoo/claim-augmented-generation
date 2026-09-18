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

query_reps/claim_filter (optional), applied before the claim-claim aggsim
matrix is built. Every claim in a doc counts equally in both "maxsim" and
"mean", so a claim that is essentially unrelated to the query still gets to
(a) be the arg-max some other claim latches onto in "maxsim", or (b) dilute
the "mean" average with a near-zero pair -- either way it can suppress a
doc's true claim-echo/overlap signal or, in "subtract" mode, its diversity
signal, for reasons that have nothing to do with the query. Dropping
off-topic claims by query-claim cosine similarity before aggregation keeps
sim(d, d') focused on claims that are actually about the query. Three modes:

  - "abs_threshold": one fixed cosine cutoff (claim_sim_threshold) applied
    identically to every claim in every doc.
  - "topic_percentile": the cutoff is instead resolved once per topic as a
    percentile (topic_percentile) of that topic's own pool-wide claim
    query-cosine distribution, so it adapts to each topic's score scale
    without being tied to any single document's own relevance (see
    _topic_percentile_threshold's docstring for why a per-document margin,
    tried and dropped, over-penalizes exactly the comprehensive documents).
  - "neighbor_rescue": before the claim_sim_threshold cutoff is applied, a
    claim's own query-cosine is raised to the best score among its "neighbor"
    claims elsewhere in the pool (other claims at or above neighbor_threshold
    claim-claim similarity to it, see _neighbor_rescue_scores) -- so a claim
    that's a near-duplicate of something that scores well against the query
    isn't dropped just because its own phrasing happens to score lower.

Default claim_filter="none" keeps every claim, matching prior behavior
exactly.

Claim coverage / novelty bonus (novelty_weight, coverage_threshold,
opt-in, default 0.0): sim(pick, j) (used by _select as
max_sim_to_selected[j]) is a sum over pick's claims of each one's best
match in j -- an asymmetric direction where only pick's claims get to
"vote" for a match in j. If several of pick's claims all happen to land on
the same one dominant claim in j (a claim broadly similar to a lot of
things), sim(pick, j) reads as heavily redundant even though most of j's
*other* claims were never compared against anything and may be entirely
novel information -- the doc-doc scalar just can't see them. novelty_weight
> 0 adds a claim-level coverage tracker that can: every already-selected
claim "votes" for its best match (>= coverage_threshold) within each
remaining candidate's own claim set (mirroring the same forward direction
sim(pick, j) already uses, just kept at claim granularity instead of
summed away), and any claim in a candidate doc that never receives a vote
counts toward that doc's uncovered_fraction -- the share of its claims
still unrepresented by anything selected so far. Each round's score is
lambda_mult * relevance + sign * (1 - lambda_mult) * max_sim_to_selected +
novelty_weight * uncovered_fraction, so with novelty_weight left at 0.0
this term drops out exactly and selection matches pre-existing behavior
bit-for-bit -- built this way specifically so the bonus can be ablated
in/out rather than replacing the existing tradeoff.
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


def _keep_by_score(cids, scores, threshold):
    """Shared keep/drop rule for every claim_filter mode: keep every cid whose
    score clears threshold, but never zero out a doc entirely -- always keep
    at least its single best-scoring claim, regardless of mode."""
    order = sorted(range(len(cids)), key=lambda i: -scores[i])
    keep = [i for i in order if scores[i] >= threshold]
    if not keep:
        keep = order[:1]
    return [cids[i] for i in keep]


def _filter_claims_for_doc(cids, claim_reps_by_id, query_vec, claim_sim_threshold):
    """"abs_threshold" claim_filter: one absolute cosine cutoff applied to
    every claim in this doc, regardless of that doc's own base relevance --
    claim_sim_threshold is used directly; "topic_percentile" (see
    _topic_percentile_threshold) resolves a pool-wide percentile once and
    passes it in as claim_sim_threshold instead, reusing this same logic.

    An earlier per-document margin variant (keep a claim only if it didn't
    fall more than a fixed gap below that document's own base-run relevance)
    was tried and dropped: it set a *higher* absolute bar for already-
    high-relevance docs, disproportionately gutting exactly the comprehensive
    documents whose claims have the widest internal score spread."""
    scores = [float(np.dot(query_vec, claim_reps_by_id[cid])) for cid in cids]
    return _keep_by_score(cids, scores, claim_sim_threshold)


def _neighbor_rescue_scores(list_docids, rows_by_parent, claim_reps_by_id, query_vec, neighbor_threshold):
    """"neighbor_rescue" claim_filter: per-claim effective score = max(this
    claim's own query-cosine, the query-cosine of its best "neighbor" --
    another doc's claim at or above neighbor_threshold claim-claim cosine
    similarity to it). A claim that's a near-duplicate of a claim elsewhere
    in the pool that happens to score better against the query borrows that
    higher score instead of being judged solely on its own (often
    vocabulary-mismatched) phrasing; a genuinely rare/unique claim with no
    such neighbor keeps its own score.

    neighbor_threshold should be loose (~0.5), not the ~0.75 bar that would
    call two claims "the same nugget": at 0.75, a truly rare claim has zero
    neighbors by construction (rarity is exactly why it scored low in the
    first place), so nothing could ever rescue it. At a loose bar, neighbor
    *existence* stops being the discriminating factor at all (nearly every
    claim has some neighbor) -- it's the neighbor's own query-score magnitude,
    compared against claim_sim_threshold downstream, that actually separates
    real nuggets from coincidental noise. Validated empirically to correctly
    rescue real nugget-matching claims sitting well below a flat cutoff while
    still rejecting the clearest noise (claims with no qualifying neighbor at
    any threshold); it does still let a handful of borderline/tangential
    claims through when they happen to share vocabulary with something
    on-topic elsewhere -- an acceptable trade since keeping a little noise is
    far cheaper than discarding a real nugget outright."""
    cids_flat, doc_of = [], []
    for doc_idx, docid in enumerate(list_docids):
        for cid in (rows_by_parent.get(docid) or []):
            if cid in claim_reps_by_id:
                cids_flat.append(cid)
                doc_of.append(doc_idx)
    if not cids_flat:
        return {}
    doc_of = np.asarray(doc_of)
    claim_matrix = np.stack([claim_reps_by_id[c] for c in cids_flat]).astype(np.float32)
    own_query_sim = claim_matrix @ np.asarray(query_vec, dtype=np.float32)
    claim_claim_sim = claim_matrix @ claim_matrix.T

    scores = {}
    for i, cid in enumerate(cids_flat):
        row = claim_claim_sim[i].copy()
        row[doc_of == doc_of[i]] = -1.0  # a claim's own doc never counts as its "neighbor"
        neighbor_mask = row >= neighbor_threshold
        best_neighbor = float(own_query_sim[neighbor_mask].max()) if neighbor_mask.any() else -np.inf
        scores[cid] = max(float(own_query_sim[i]), best_neighbor)
    return scores


def _topic_percentile_threshold(list_docids, rows_by_parent, claim_reps_by_id, query_vec, percentile):
    """The pool-wide counterpart to a per-doc margin: gather every claim's
    query-cosine across every doc in this topic's pool (not just one doc),
    and return the given percentile of that single, shared distribution.
    Applying this one number to every doc as an "abs_threshold" cutoff avoids
    two failure modes seen empirically: (1) a single hand-picked constant
    across topics doesn't adapt to a topic's own score scale (some topics run
    systematically higher/lower on this dense retriever than others), and (2)
    a per-doc margin (doc_relevance - claim_sim_threshold, see "threshold"
    above) sets a *higher* absolute bar for already-high-relevance docs,
    disproportionately gutting exactly the comprehensive documents whose
    claims have the widest internal score spread. A pool-wide percentile is
    fixed per topic, not per document, so it doesn't have that doc-relative
    penalty, while still adapting to each topic's own scale."""
    all_sims = []
    for docid in list_docids:
        for cid in (rows_by_parent.get(docid) or []):
            vec = claim_reps_by_id.get(cid)
            if vec is not None:
                all_sims.append(float(np.dot(query_vec, vec)))
    if not all_sims:
        return 0.0
    return float(np.percentile(np.asarray(all_sims, dtype=np.float32), percentile))


_CLAIM_FILTERS = ("none", "abs_threshold", "topic_percentile", "neighbor_rescue")


def _claim_aggsim_matrix(
    list_docids,
    claim_reps_by_id,
    rows_by_parent,
    dim,
    agg="maxsim",
    query_vec=None,
    claim_filter="none",
    claim_sim_threshold=0.0,
    topic_percentile=25.0,
    neighbor_threshold=0.5,
):
    """Doc-doc similarity via claim-level aggregation of dense embeddings.

    See module docstring for the "maxsim" vs "mean" trade-off. "maxsim" is
    row-normalized to [0, 1] (mathematically equivalent to averaging each
    max over d's claim count, see module docstring); "mean" is already
    bounded in [-1, 1] by construction and left as-is.

    claim_filter != "none" applies the query-claim similarity filter (see
    module docstring) to each doc's claims before they enter the matrix.
    "topic_percentile" and "neighbor_rescue" both resolve to a precomputed,
    pool-wide per-claim quantity (a shared cutoff, or a per-claim rescued
    score -- see _topic_percentile_threshold / _neighbor_rescue_scores)
    before the same per-doc keep/drop loop below runs for every mode.

    Returns (sim, claim_sim, doc_of_claim): the aggregated n x n doc-doc
    matrix, plus the raw claims x claims cosine matrix and the doc index
    each row/column of it belongs to -- both needed by _select's coverage
    tracker (see its docstring) so it doesn't have to redo claim gathering
    and filtering a second time.
    """
    if agg not in ("maxsim", "mean"):
        raise ValueError(f"agg must be 'maxsim' or 'mean', got {agg!r}")
    if claim_filter not in _CLAIM_FILTERS:
        raise ValueError(f"claim_filter must be one of {_CLAIM_FILTERS}, got {claim_filter!r}")

    effective_threshold = claim_sim_threshold
    neighbor_scores = None
    if claim_filter == "topic_percentile":
        effective_threshold = _topic_percentile_threshold(
            list_docids, rows_by_parent, claim_reps_by_id, query_vec, topic_percentile,
        )
        logger.info("cc-dense: topic_percentile=%.1f resolved to abs_threshold=%.4f over %d doc(s)",
                    topic_percentile, effective_threshold, len(list_docids))
    elif claim_filter == "neighbor_rescue":
        neighbor_scores = _neighbor_rescue_scores(
            list_docids, rows_by_parent, claim_reps_by_id, query_vec, neighbor_threshold,
        )

    n = len(list_docids)
    doc_of_claim = []
    claim_vecs = []

    ## Pre-filter: replace claim_ids with filter claim only
    for doc_idx, docid in enumerate(list_docids):
        claim_ids = rows_by_parent.get(docid) or []
        if claim_filter == "neighbor_rescue" and claim_ids:
            scores = [neighbor_scores.get(cid, -np.inf) for cid in claim_ids]
            claim_ids = _keep_by_score(claim_ids, scores, claim_sim_threshold)
        elif claim_filter != "none" and claim_ids:
            claim_ids = _filter_claims_for_doc(claim_ids, claim_reps_by_id, query_vec, effective_threshold)
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
        # mean over maxsim regardless of number of claims
        row_max = sim.max(axis=1, keepdims=True)
        row_max[row_max < 1e-9] = 1.0
        return sim / row_max, claim_sim, doc_of_claim
    return sim, claim_sim, doc_of_claim


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
    claim_sim_threshold=0.0,
    topic_percentile=25.0,
    neighbor_threshold=0.5,
    novelty_weight=0.0,
    coverage_threshold=0.0,
):
    """novelty_weight/coverage_threshold (both opt-in, default 0.0 =
    identical to pre-existing behavior -- see module docstring's "claim
    coverage" section): a per-round bonus for candidates that still have
    claims uncovered by everything selected so far, layered on top of the
    existing relevance/max_sim_to_selected tradeoff rather than replacing
    it, so novelty_weight=0.0 is a true no-op for ablating against."""
    if len(hits) <= 1:
        return hits
    if mode not in ("subtract", "add"):
        raise ValueError(f"mode must be 'subtract' (MMR) or 'add' (claim-echo boost), got {mode!r}")
    sign = -1.0 if mode == "subtract" else 1.0

    relevance = np.asarray([h.score for h in hits], dtype=np.float32)
    list_docids = [h.docid for h in hits]

    sim, claim_sim, doc_of_claim = _claim_aggsim_matrix(
        list_docids=list_docids,
        claim_reps_by_id=claim_reps_by_id,
        rows_by_parent=rows_by_parent,
        dim=dim,
        agg=agg,
        query_vec=query_vec,
        claim_filter=claim_filter,
        claim_sim_threshold=claim_sim_threshold,
        topic_percentile=topic_percentile,
        neighbor_threshold=neighbor_threshold,
    )
    _print_sim_matrix(list_docids, sim, agg=agg, topn=10)

    n = len(hits)
    n_select = min(k, n)
    selected = []
    selected_scores = []
    max_sim_to_selected = np.zeros(n, dtype=np.float32)

    # Claim-level coverage tracker for the novelty bonus (see docstring
    # above): claims_per_doc_idx[j] are doc j's row/col positions in
    # claim_sim; votes[c] counts how many times claim c has "won" as some
    # already-selected claim's best (>= coverage_threshold) match within
    # its own doc's block -- i.e. it's the one _claim_aggsim_matrix's
    # maxsim sum would have picked up, the same asymmetric direction
    # max_sim_to_selected already uses (sim[pick, j] = pick's claims
    # matched forward into j). A claim that never wins stays "uncovered"
    # regardless of how the doc's aggregate maxsim score looks -- a single
    # dominant claim can carry the whole doc-doc score while every other
    # claim in the doc never gets compared to anything and is invisible to
    # that aggregate. This is deliberately lazy (skipped entirely when
    # novelty_weight == 0.0) so the default path pays zero extra cost.
    claims_per_doc_idx = [np.where(doc_of_claim == j)[0] for j in range(n)]
    votes = np.zeros(len(doc_of_claim), dtype=np.int32)

    def _uncovered_fraction():
        frac = np.zeros(n, dtype=np.float32)
        for j, idxs in enumerate(claims_per_doc_idx):
            if len(idxs):
                frac[j] = float((votes[idxs] == 0).mean())
        return frac

    for _ in range(n_select):
        novelty_term = _uncovered_fraction() if novelty_weight != 0.0 else 0.0
        scores = (
            lambda_mult * relevance
            + sign * (1 - lambda_mult) * max_sim_to_selected
            + novelty_weight * novelty_term
        )
        scores[selected] = -np.inf
        pick = int(np.argmax(scores))
        selected_scores.append(float(scores[pick]))
        selected.append(pick)
        max_sim_to_selected = np.maximum(max_sim_to_selected, sim[pick])

        if novelty_weight != 0.0:
            pick_claim_idx = claims_per_doc_idx[pick]
            selected_set = set(selected)
            for j, idxs in enumerate(claims_per_doc_idx):
                if j in selected_set or len(idxs) == 0 or len(pick_claim_idx) == 0:
                    continue
                # for each document, log the maxsim contributor to vote
                block = claim_sim[np.ix_(pick_claim_idx, idxs)]
                row_max = block.max(axis=1)
                row_argmax = block.argmax(axis=1)
                winners = row_argmax[row_max >= coverage_threshold]
                for w in winners:
                    votes[idxs[w]] += 1

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
    claim_sim_threshold: float = 0.0,
    topic_percentile: float = 25.0,
    neighbor_threshold: float = 0.5,
    novelty_weight: float = 0.0,
    coverage_threshold: float = 0.0,
) -> List[Result]:
    """query_reps/claim_filter: optional query-claim similarity filter on
    each pooled doc's claims before the claim-claim aggsim matrix is built
    (see module docstring). claim_filter="none" (default) keeps every claim
    found in the shards, matching pre-existing behavior; query_reps is
    required for "abs_threshold"/"topic_percentile"/"neighbor_rescue".

    "abs_threshold": one fixed cosine cutoff (--claim-sim-threshold) applied
    identically to every claim in every doc, regardless of that doc's own
    base relevance. (An earlier per-document margin variant, relative to
    each doc's own score, was tried and dropped -- it over-penalized
    high-relevance, comprehensive documents; see
    _topic_percentile_threshold's docstring.)

    "topic_percentile": like "abs_threshold", but the cutoff is resolved once
    per topic from --topic-percentile of that topic's own pool-wide claim
    query-cosine distribution, instead of one hand-picked constant reused
    across every topic regardless of its own score scale.

    "neighbor_rescue": a claim's own query-cosine is replaced by
    max(its own score, its best same-pool "neighbor" claim's score at or
    above --neighbor-threshold claim-claim cosine similarity, see
    _neighbor_rescue_scores) before the same --claim-sim-threshold cutoff is
    applied -- a claim that's a near-duplicate of something elsewhere that
    scores better against the query borrows that higher score, instead of
    being judged solely on its own (often vocabulary-mismatched) phrasing.

    novelty_weight/coverage_threshold: optional per-claim coverage bonus
    added on top of the existing relevance/max_sim_to_selected tradeoff
    during greedy selection (see module docstring's "claim coverage"
    section and _select's docstring). Both default to 0.0, which is a
    no-op reproducing pre-existing selection exactly -- set novelty_weight
    > 0 to run the ablation."""
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
            claim_sim_threshold=claim_sim_threshold,
            topic_percentile=topic_percentile,
            neighbor_threshold=neighbor_threshold,
            novelty_weight=novelty_weight,
            coverage_threshold=coverage_threshold,
        )

    return outputs
