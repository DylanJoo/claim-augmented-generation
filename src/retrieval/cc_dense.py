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

  - "kmeans": rather than scoring exact claim-pair similarity, every claim
    among the topic's pooled documents is first clustered with k-means into
    `n_clusters` groups, fit fresh per topic (a cluster id is only
    meaningful within one topic's own pool). Each pooled doc d is then
    re-described as a vector over cluster ids ("multi-hot", since a doc can
    have claims landing in several clusters): label_mode="binary" sets
    v_d[j] = 1 if d has >=1 claim in cluster j; label_mode="scaled" sets
    v_d[j] = (# of d's claims in cluster j) / (# of d's claims total), a
    within-doc distribution over clusters summing to 1. sim(d, d') is then
    cosine similarity between these cluster-membership vectors. A doc with
    zero claims in the shards gets the all-zero vector (it is *not* padded
    into the k-means fit with a dummy zero-vector claim -- doing so would
    inject a fake "no claim" cluster and skew the fit for every doc that
    does have real claims). Rationale: maxsim/mean tie diversity to exact
    claim-pair similarity, which can be noisy claim-by-claim; clustering
    first collapses that into a shared, topic-specific vocabulary of "claim
    topics" every pooled doc is scored against, letting diversity be
    reasoned about in terms of topic coverage rather than pairwise echoes.

    min_doc_support (kmeans only, default 1 = off) guards against the
    other failure mode: a cluster that only one doc happens to touch is
    definitionally not redundant (it's that doc's unique claim, not
    corroboration), so any cluster with fewer than min_doc_support distinct
    docs is zeroed out of every doc's vector before label_mode is applied.
    This also matters because kmeans is fit over the whole 1000-doc pool
    (mostly irrelevant docs, in practice), so without this guard a chance
    collision between one relevant doc and one irrelevant doc's claims can
    register as "overlap" with no real corroboration behind it.
"""
import copy
import glob
import logging
import pickle
from collections import defaultdict
from typing import List

import numpy as np
from sklearn.cluster import KMeans

from utils import Result, Hit, load_run

logger = logging.getLogger(__name__)

_KMEANS_RANDOM_STATE = 42  # fixed for reproducibility; not exposed as a CLI knob


def _pickle_load(path):
    with open(path, "rb") as f:
        reps, lookup = pickle.load(f)
    return np.asarray(reps), lookup

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


def _claim_aggsim_matrix(list_docids, claim_reps_by_id, rows_by_parent, dim, agg="maxsim"):
    """Doc-doc similarity via claim-level aggregation of dense embeddings.

    See module docstring for the "maxsim" vs "mean" trade-off. "maxsim" is
    row-normalized to [0, 1] (mathematically equivalent to averaging each
    max over d's claim count, see module docstring); "mean" is already
    bounded in [-1, 1] by construction and left as-is.
    """
    if agg not in ("maxsim", "mean"):
        raise ValueError(f"agg must be 'maxsim' or 'mean', got {agg!r}")

    n = len(list_docids)
    doc_of_claim = []
    claim_vecs = []
    for doc_idx, docid in enumerate(list_docids):
        claim_ids = rows_by_parent.get(docid) or []
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

def _kmeans_doc_vectors(
    list_docids,
    claim_reps_by_id,
    rows_by_parent,
    n_clusters,
    label_mode="binary",
    kmeans_n_init=10,
    min_doc_support=1,
):
    """Cluster every claim found for docs in `list_docids` with k-means (fit
    fresh per topic), then re-describe each doc as a fixed-length vector
    over cluster ids. See module docstring's "kmeans" bullet for the
    binary/scaled label_mode trade-off.

    min_doc_support: a cluster touched by fewer than this many distinct
    docs is, by definition, not redundant -- it's one doc's unique claim,
    not corroboration -- so it is zeroed out of every doc's vector before
    label_mode is applied. Default 1 is a no-op (every cluster counts,
    matching prior behavior); raise it (e.g. 2) to require actual
    cross-document overlap before two docs are treated as similar.

    Returns (doc_vecs [n_docs, effective_k], labels [n_claims] or None --
    None only when no pooled doc has any claim in the shards at all).
    """
    if label_mode not in ("binary", "scaled"):
        raise ValueError(f"label_mode must be 'binary' or 'scaled', got {label_mode!r}")

    doc_of_claim = []
    claim_vecs = []
    missing_docs = []
    for doc_idx, docid in enumerate(list_docids):
        ids = [cid for cid in (rows_by_parent.get(docid) or []) if cid in claim_reps_by_id]
        if not ids:
            missing_docs.append(docid)
            continue
        for cid in ids:
            claim_vecs.append(claim_reps_by_id[cid])
            doc_of_claim.append(doc_idx)

    n = len(list_docids)
    if missing_docs:
        print(f"[cc_dense] {len(missing_docs)} pooled doc(s) have no claims in embedding shards; "
              f"their cluster vector is all-zero, e.g. {missing_docs[:3]!r}")

    if not claim_vecs:
        return np.zeros((n, 0), dtype=np.float32), None

    doc_of_claim = np.asarray(doc_of_claim, dtype=np.int64)
    claim_matrix = np.stack(claim_vecs).astype(np.float32)
    n_claims = claim_matrix.shape[0]

    effective_k = min(n_clusters, n_claims)
    if effective_k < n_clusters:
        print(f"[cc_dense] only {n_claims} claim(s) available in this pool; "
              f"clamping n_clusters {n_clusters} -> {effective_k}")

    if effective_k < 2:
        # Degenerate pool (0 or 1 distinct claim to cluster): every claim is
        # trivially its own/the only cluster -- skip fitting k-means.
        labels = np.zeros(n_claims, dtype=np.int64)
        effective_k = 1
    else:
        km = KMeans(n_clusters=effective_k, n_init=kmeans_n_init, random_state=_KMEANS_RANDOM_STATE)
        labels = km.fit_predict(claim_matrix)

    counts = np.zeros((n, effective_k), dtype=np.float32)
    for doc_idx, cluster_id in zip(doc_of_claim, labels):
        counts[doc_idx, cluster_id] += 1.0

    if min_doc_support > 1:
        doc_freq = (counts > 0).sum(axis=0)
        weak = doc_freq < min_doc_support
        if weak.any():
            print(f"[cc_dense] {int(weak.sum())}/{effective_k} cluster(s) touched by <{min_doc_support} "
                  f"distinct doc(s); zeroing them out as non-redundant (no corroboration), "
                  f"e.g. cluster ids {np.nonzero(weak)[0][:5].tolist()}")
            counts[:, weak] = 0.0

    if label_mode == "binary":
        doc_vecs = (counts > 0).astype(np.float32)
    else:  # "scaled"
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-9] = 1.0
        doc_vecs = counts / row_sums

    return doc_vecs, labels


def _kmeans_sim_matrix(doc_vecs):
    """Cosine similarity between doc cluster-membership vectors. An
    all-zero row (doc with no claims in the shards) stays zero-similarity
    to everyone, including itself."""
    norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    normalized = doc_vecs / norms
    return normalized @ normalized.T


def _print_cluster_summary(qid, n_clusters, labels):
    if labels is None:
        print(f"[cc_dense] qid={qid}: no claims found in this pool, nothing clustered")
        return
    sizes = np.bincount(labels, minlength=n_clusters)
    print(f"[cc_dense] qid={qid}: {len(labels)} claim(s) clustered into {n_clusters} cluster(s); "
          f"cluster sizes={sizes.tolist()}")


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


def _print_cluster_assignment(list_docids, doc_vecs, agg, topn=10):
    """Print the top-N docs' cluster-membership vectors (rows: docs, cols:
    cluster ids) -- which claim-topic cluster(s) each doc touches, and how
    strongly (label_mode="binary" -> 0/1, "scaled" -> within-doc fraction).
    More directly diagnostic than the doc-doc matrix for kmeans-based aggs,
    since it shows *why* two docs are/aren't similar (shared cluster ids)
    rather than just the collapsed pairwise score."""
    n = min(topn, len(list_docids))
    k = doc_vecs.shape[1]
    ids = [str(d)[:12] for d in list_docids[:n]]
    header = " " * 14 + "".join(f"c{j:<5}" for j in range(k))
    print(f"[cc_dense] top-{n} doc x {k}-cluster ({agg}) assignment:")
    print(header)
    for i in range(n):
        row = "".join(f"{doc_vecs[i, j]:>6.2f}" for j in range(k))
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
    n_clusters=20,
    label_mode="binary",
    kmeans_n_init=10,
    min_doc_support=1,
    qid=None,
):
    if len(hits) <= 1:
        return hits
    if mode not in ("subtract", "add"):
        raise ValueError(f"mode must be 'subtract' (MMR) or 'add' (claim-echo boost), got {mode!r}")
    sign = -1.0 if mode == "subtract" else 1.0

    relevance = np.asarray([h.score for h in hits], dtype=np.float32)
    list_docids = [h.docid for h in hits]

    if agg in ("maxsim", "mean"):
        sim = _claim_aggsim_matrix(
            list_docids=list_docids,
            claim_reps_by_id=claim_reps_by_id,
            rows_by_parent=rows_by_parent,
            dim=dim,
            agg=agg
        )
        _print_sim_matrix(list_docids, sim, agg=agg, topn=10)
    elif agg == "kmeans":
        doc_vecs, labels = _kmeans_doc_vectors(
            list_docids=list_docids,
            claim_reps_by_id=claim_reps_by_id,
            rows_by_parent=rows_by_parent,
            n_clusters=n_clusters,
            label_mode=label_mode,
            kmeans_n_init=kmeans_n_init,
            min_doc_support=min_doc_support,
        )
        _print_cluster_summary(qid, doc_vecs.shape[1], labels)
        _print_cluster_assignment(list_docids, doc_vecs, agg=agg, topn=10)
        sim = _kmeans_sim_matrix(doc_vecs)
    else:
        raise ValueError(f"agg must be 'maxsim', 'mean', or 'kmeans', got {agg!r}")

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
    k: int = 1000,
    lambda_mult: float = 0.9,
    mode: str = "subtract",
    agg: str = "maxsim",
    n_clusters: int = 20,
    label_mode: str = "binary",
    kmeans_n_init: int = 10,
    min_doc_support: int = 1,
) -> List[Result]:
    logger.info("cc-dense: mode=%s, agg=%s, base relevance from run file %s, pool k=%d, lambda=%.2f, claim_reps=%s",
                mode, agg, run_file, k, lambda_mult, claim_reps)
    base_run = load_run(run_file, k=k)
    claim_corpus = {}  # corpus text unused downstream; skip loading to save memory

    needed_docids = {docid for pool in base_run.values() for docid, _ in pool}
    logger.info("cc-dense: %d unique pooled docid(s) across %d topic(s) need claim embeddings",
                len(needed_docids), len(base_run))

    claim_reps_by_id = _load_claim_reps(claim_reps, needed_docids)
    dim = next(iter(claim_reps_by_id.values())).shape[0]
    rows_by_parent = _rows_by_parent(claim_reps_by_id)

    outputs = copy.deepcopy(inputs)
    for i, inp in enumerate(inputs):
        qid = str(inp.topic["qid"])
        pool = base_run.get(qid, [])
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
            n_clusters=n_clusters, label_mode=label_mode, kmeans_n_init=kmeans_n_init,
            min_doc_support=min_doc_support, qid=qid,
        )

    return outputs
