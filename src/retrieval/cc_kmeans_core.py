"""
cc-kmeans-core: like cc_dense.py's agg="kmeans" (see that module's docstring
for the base rationale -- clustering claims into topic buckets to smooth out
claim-pair noise), but the k-means fit itself only sees the "relevant core"
of the pool instead of the whole thing.

Why this needs to be a separate module rather than another cc_dense.py
option: cc_dense.py's agg="kmeans" fits k-means fresh on every pooled doc's
claims -- but a pool is typically k=1000 docs deep while only ~15-20 are
actually relevant (checked directly against the neuclir1/ragtime1 qrels'
subtopic-id column). That means the fit is done on claims that are ~98%
noise, which can dilute the topic buckets the whole method is trying to
recover. Fixing that means restructuring the fit itself (fit on a filtered
subset, predict everyone else onto it), not just adding a knob to the
existing single-fit function -- hence its own module.

Mechanism:
  1. `hits` arrive already sorted by base relevance (descending -- see
     utils.load_run). The first `top_m` docs are treated as the "core":
     the pool members most likely to actually be relevant.
  2. K-means is fit ONLY on the core's claims, into `n_clusters` groups.
  3. Every doc's claims -- core AND the long tail beyond top_m alike -- are
     then assigned cluster ids: core docs reuse the labels already computed
     by fit_predict (no need to re-score them), tail docs are scored with
     km.predict() against those same fitted centroids. This keeps every
     doc's cluster-membership vector living in the same topic-bucket space,
     but that space itself was defined only by the likely-relevant core, not
     diluted by the tail's noise.
  4. Doc cluster-membership vectors, min_doc_support redundancy filtering,
     cosine similarity, and the greedy subtract/add selection loop are all
     identical to cc_dense.py's kmeans path -- see that module's docstring
     for the label_mode and min_doc_support explanations. Both low-level
     helpers (_load_claim_reps, _rows_by_parent, _kmeans_sim_matrix,
     _print_cluster_summary, _print_cluster_assignment) are imported from
     cc_dense rather than duplicated, since they don't depend on the
     fit-vs-predict split.

top_m is a hard cutoff on rank, not a relevance-score threshold -- simple
and requires no extra tuning beyond picking a number comfortably above the
qrels' observed ~15-20 true subtopics-per-topic (e.g. 50-100), to leave
headroom for retrieval error while still cutting the ~1000-doc pool down by
an order of magnitude before fitting.
"""
import copy
import logging
from typing import List

import numpy as np
from sklearn.cluster import KMeans

from utils import Result, Hit, load_run
from retrieval.cc_dense import (
    _KMEANS_RANDOM_STATE,
    _load_claim_reps,
    _rows_by_parent,
    _kmeans_sim_matrix,
    _print_cluster_summary,
    _print_cluster_assignment,
)

logger = logging.getLogger(__name__)


def _core_doc_cluster_vectors(
    list_docids,
    claim_reps_by_id,
    rows_by_parent,
    n_clusters,
    top_m,
    label_mode="binary",
    kmeans_n_init=10,
    min_doc_support=1,
):
    """Fit k-means on the claims of the top `top_m` docs only (by base
    relevance -- `list_docids` must already be sorted that way), then assign
    every pooled doc's claims to those fitted centroids. See module
    docstring for the full mechanism and cc_dense.py's _kmeans_doc_vectors
    for label_mode/min_doc_support semantics (identical here).

    Returns (doc_vecs [n_docs, effective_k], core_labels [n_core_claims] or
    None -- None only when the core itself has no claims in the shards).
    """
    if label_mode not in ("binary", "scaled"):
        raise ValueError(f"label_mode must be 'binary' or 'scaled', got {label_mode!r}")

    n = len(list_docids)
    core_docids = list_docids[:top_m]

    # 1. Gather claims for the core only -- this is what k-means is fit on.
    core_doc_of_claim = []
    core_claim_vecs = []
    for doc_idx, docid in enumerate(core_docids):
        for cid in (rows_by_parent.get(docid) or []):
            if cid not in claim_reps_by_id:
                continue
            core_claim_vecs.append(claim_reps_by_id[cid])
            core_doc_of_claim.append(doc_idx)

    missing_docs = [d for d in list_docids if not (rows_by_parent.get(d) or [])]
    if missing_docs:
        print(f"[cc_kmeans_core] {len(missing_docs)}/{n} pooled doc(s) have no claims in embedding "
              f"shards; their cluster vector is all-zero, e.g. {missing_docs[:3]!r}")

    if not core_claim_vecs:
        print(f"[cc_kmeans_core] no claims found among the top-{top_m} core doc(s); "
              f"returning all-zero vectors for the whole pool")
        return np.zeros((n, 0), dtype=np.float32), None

    core_claim_matrix = np.stack(core_claim_vecs).astype(np.float32)
    n_core_claims = core_claim_matrix.shape[0]

    effective_k = min(n_clusters, n_core_claims)
    if effective_k < n_clusters:
        print(f"[cc_kmeans_core] only {n_core_claims} claim(s) available among the top-{top_m} core "
              f"doc(s); clamping n_clusters {n_clusters} -> {effective_k}")

    km = None
    if effective_k < 2:
        # Degenerate core (0 or 1 distinct claim to cluster): every claim is
        # trivially its own/the only cluster -- skip fitting k-means, and
        # the tail has no meaningful centroid space to be scored against.
        core_labels = np.zeros(n_core_claims, dtype=np.int64)
        effective_k = 1
    else:
        km = KMeans(n_clusters=effective_k, n_init=kmeans_n_init, random_state=_KMEANS_RANDOM_STATE)
        core_labels = km.fit_predict(core_claim_matrix)

    # 2. Score every doc's claims against the core's centroids: core docs
    # reuse the labels fit_predict already computed, tail docs (beyond
    # top_m) are assigned via .predict() into that same fitted space.
    counts = np.zeros((n, effective_k), dtype=np.float32)
    for doc_idx, cluster_id in zip(core_doc_of_claim, core_labels):
        counts[doc_idx, cluster_id] += 1.0

    if km is not None:
        for doc_idx in range(top_m, n):
            docid = list_docids[doc_idx]
            ids = [cid for cid in (rows_by_parent.get(docid) or []) if cid in claim_reps_by_id]
            if not ids:
                continue
            tail_claim_matrix = np.stack([claim_reps_by_id[cid] for cid in ids]).astype(np.float32)
            for cluster_id in km.predict(tail_claim_matrix):
                counts[doc_idx, cluster_id] += 1.0

    if min_doc_support > 1:
        doc_freq = (counts > 0).sum(axis=0)
        weak = doc_freq < min_doc_support
        if weak.any():
            print(f"[cc_kmeans_core] {int(weak.sum())}/{effective_k} cluster(s) touched by "
                  f"<{min_doc_support} distinct doc(s); zeroing them out as non-redundant, "
                  f"e.g. cluster ids {np.nonzero(weak)[0][:5].tolist()}")
            counts[:, weak] = 0.0

    if label_mode == "binary":
        doc_vecs = (counts > 0).astype(np.float32)
    else:  # "scaled"
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-9] = 1.0
        doc_vecs = counts / row_sums

    return doc_vecs, core_labels


def _select(
    hits,
    claim_reps_by_id,
    rows_by_parent,
    k,
    lambda_mult,
    mode,
    n_clusters,
    top_m,
    label_mode,
    kmeans_n_init,
    min_doc_support,
    qid=None,
):
    if len(hits) <= 1:
        return hits
    if mode not in ("subtract", "add"):
        raise ValueError(f"mode must be 'subtract' (MMR) or 'add' (claim-echo boost), got {mode!r}")
    sign = -1.0 if mode == "subtract" else 1.0

    list_docids = [h.docid for h in hits]  # already relevance-sorted, see utils.load_run
    relevance = np.asarray([h.score for h in hits], dtype=np.float32)

    doc_vecs, labels = _core_doc_cluster_vectors(
        list_docids=list_docids,
        claim_reps_by_id=claim_reps_by_id,
        rows_by_parent=rows_by_parent,
        n_clusters=n_clusters,
        top_m=top_m,
        label_mode=label_mode,
        kmeans_n_init=kmeans_n_init,
        min_doc_support=min_doc_support,
    )
    _print_cluster_summary(qid, doc_vecs.shape[1], labels)
    _print_cluster_assignment(list_docids, doc_vecs, agg=f"kmeans-core(top{top_m})", topn=10)
    sim = _kmeans_sim_matrix(doc_vecs)

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

    # See cc_dense.py's _select for why "add" mode needs this re-sort (score
    # sequence is not guaranteed monotonic with rank the way "subtract" is,
    # and downstream TREC tooling sorts by score, not file order).
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
    n_clusters: int = 20,
    top_m: int = 100,
    label_mode: str = "binary",
    kmeans_n_init: int = 10,
    min_doc_support: int = 1,
) -> List[Result]:
    logger.info("cc-kmeans-core: mode=%s, n_clusters=%d, top_m=%d, base relevance from run file %s, "
                "pool k=%d, lambda=%.2f, claim_reps=%s",
                mode, n_clusters, top_m, run_file, k, lambda_mult, claim_reps)
    base_run = load_run(run_file, k=k)
    claim_corpus = {}  # corpus text unused downstream; skip loading to save memory

    needed_docids = {docid for pool in base_run.values() for docid, _ in pool}
    logger.info("cc-kmeans-core: %d unique pooled docid(s) across %d topic(s) need claim embeddings",
                len(needed_docids), len(base_run))

    claim_reps_by_id = _load_claim_reps(claim_reps, needed_docids)
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
        logger.info("cc-kmeans-core: qid=%s pooled %d documents, %d claims found in embedding shards "
                    "(top-%d treated as the relevant core)", qid, len(pool), n_claims, top_m)

        outputs[i].hits = hits
        outputs[i].evidences = _select(
            hits, claim_reps_by_id, rows_by_parent, k, lambda_mult, mode=mode,
            n_clusters=n_clusters, top_m=top_m, label_mode=label_mode,
            kmeans_n_init=kmeans_n_init, min_doc_support=min_doc_support, qid=qid,
        )

    return outputs
