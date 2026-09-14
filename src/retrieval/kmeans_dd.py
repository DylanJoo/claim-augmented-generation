"""
kmeans-dd (K-Means cluster re-labeled Doc-Doc) rerank on top of a doc-level
run file -- dense embedding variant. There is no BM25 counterpart: k-means
only makes sense over a continuous embedding space, so (unlike dd/cc/dc_gap,
which each have a BM25-scored sibling module) this module is dense-only and
carries no "_dense" suffix.

Same greedy selection loop as dd_dense.py/cc_dense.py (see those modules'
docstrings for the MMR/"add"-boost mechanics), but sim(d, d') comes from
neither a whole-doc embedding (dd_dense) nor raw claim-claim MaxSim
(cc_dense). Instead:

  1. Every claim embedding among the topic's pooled documents is clustered
     with k-means into `n_clusters` groups, fit fresh per topic -- a cluster
     id is only meaningful within one topic's own pool, so it is not fit
     once globally.
  2. Each claim is re-labeled with its cluster id, and each pooled doc d is
     re-described as a vector over cluster ids ("multi-hot", since a doc can
     have claims landing in several clusters):
       - label_mode="binary" (default): v_d[j] = 1 if d has >=1 claim in
         cluster j, else 0.
       - label_mode="scaled": v_d[j] = (# of d's claims in cluster j) /
         (# of d's claims total) -- a within-doc distribution over clusters,
         summing to 1.
     A doc with zero claims in the shards gets the all-zero vector (it is
     *not* padded into the k-means fit with a dummy zero-vector claim --
     doing so would inject a fake "no claim" cluster and skew the fit for
     every doc that does have real claims).
  3. sim(d, d') = cosine similarity between these cluster-membership vectors
     (L2-normalized dot product), so sim(d, d) = 1 for any doc with at least
     one claim -- two docs are "similar" to the extent their claims land in
     the same topical clusters, regardless of which specific claims they
     were.
  4. The same greedy loop as cc_dense.py: mode="subtract" is MMR (penalize
     cluster overlap with what's already selected), mode="add" is a
     claim-echo boost (reward it).

Rationale: cc_dense's claim-claim MaxSim ties diversity to exact claim-pair
similarity, which can be noisy claim-by-claim. Clustering first collapses
that into a shared, topic-specific vocabulary of "claim topics" that every
pooled doc is scored against, which is smoother and lets us reason about
diversity in terms of topic coverage rather than pairwise claim echoes.
"""
import copy
import glob
import logging
import pickle
from collections import defaultdict
from typing import List

import numpy as np
from sklearn.cluster import KMeans

from utils import Result, Hit, load_run, load_corpus

logger = logging.getLogger(__name__)


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
        logger.info("kmeans-dd: [%d/%d] loaded shard %s, %d claim vector(s) matched so far",
                    i, len(files), fpath, len(reps_by_id))
    return reps_by_id

def _rows_by_parent(claim_reps_by_id):
    rows_by_parent = defaultdict(list)
    for claim_docid in claim_reps_by_id:
        parent_id = claim_docid.rsplit("#", 1)[0]
        rows_by_parent[parent_id].append(claim_docid)
    return rows_by_parent


def _doc_cluster_vectors(
    list_docids,
    claim_reps_by_id,
    rows_by_parent,
    n_clusters,
    label_mode="binary",
    kmeans_n_init=10,
    random_state=42,
):
    """Cluster every claim found for docs in `list_docids`, then re-describe
    each doc as a fixed-length vector over cluster ids. Returns
    (doc_vecs [n_docs, effective_k], labels [n_claims] or None, doc_of_claim
    or None, claim_ids or None) -- the last three are None only when no
    pooled doc has any claim in the shards at all.
    """
    if label_mode not in ("binary", "scaled"):
        raise ValueError(f"label_mode must be 'binary' or 'scaled', got {label_mode!r}")

    doc_of_claim = []
    claim_vecs = []
    claim_ids = []
    missing_docs = []
    for doc_idx, docid in enumerate(list_docids):
        ids = [cid for cid in (rows_by_parent.get(docid) or []) if cid in claim_reps_by_id]
        if not ids:
            missing_docs.append(docid)
            continue
        for cid in ids:
            claim_vecs.append(claim_reps_by_id[cid])
            doc_of_claim.append(doc_idx)
            claim_ids.append(cid)

    n = len(list_docids)
    if missing_docs:
        print(f"[kmeans_dd] {len(missing_docs)} pooled doc(s) have no claims in embedding shards; "
              f"their cluster vector is all-zero, e.g. {missing_docs[:3]!r}")

    if not claim_vecs:
        return np.zeros((n, 0), dtype=np.float32), None, None, None

    doc_of_claim = np.asarray(doc_of_claim, dtype=np.int64)
    claim_matrix = np.stack(claim_vecs).astype(np.float32)
    n_claims = claim_matrix.shape[0]

    effective_k = min(n_clusters, n_claims)
    if effective_k < n_clusters:
        print(f"[kmeans_dd] only {n_claims} claim(s) available in this pool; "
              f"clamping n_clusters {n_clusters} -> {effective_k}")

    if effective_k < 2:
        # Degenerate pool (0 or 1 distinct claim to cluster): every claim is
        # trivially its own/the only cluster -- skip fitting k-means.
        labels = np.zeros(n_claims, dtype=np.int64)
        effective_k = 1
    else:
        km = KMeans(n_clusters=effective_k, n_init=kmeans_n_init, random_state=random_state)
        labels = km.fit_predict(claim_matrix)

    doc_vecs = np.zeros((n, effective_k), dtype=np.float32)
    for doc_idx, cluster_id in zip(doc_of_claim, labels):
        doc_vecs[doc_idx, cluster_id] += 1.0

    if label_mode == "binary":
        doc_vecs = (doc_vecs > 0).astype(np.float32)
    else:  # "scaled"
        row_sums = doc_vecs.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-9] = 1.0
        doc_vecs = doc_vecs / row_sums

    return doc_vecs, labels, doc_of_claim, claim_ids


def _doc_sim_matrix(doc_vecs):
    """Cosine similarity between doc cluster-membership vectors. An
    all-zero row (doc with no claims in the shards) stays zero-similarity
    to everyone, including itself."""
    norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    normalized = doc_vecs / norms
    return normalized @ normalized.T


def _print_cluster_summary(qid, n_clusters, labels):
    if labels is None:
        print(f"[kmeans_dd] qid={qid}: no claims found in this pool, nothing clustered")
        return
    sizes = np.bincount(labels, minlength=n_clusters)
    print(f"[kmeans_dd] qid={qid}: {len(labels)} claim(s) clustered into {n_clusters} cluster(s); "
          f"cluster sizes={sizes.tolist()}")


def _print_doc_vectors(list_docids, doc_vecs, label_mode, topn=10):
    n = min(topn, len(list_docids))
    print(f"[kmeans_dd] top-{n} doc cluster-membership vectors (label_mode={label_mode}):")
    for docid, vec in zip(list_docids[:n], doc_vecs[:n]):
        nz = np.nonzero(vec)[0]
        vals = [round(float(vec[j]), 3) for j in nz]
        print(f"  {str(docid)[:14]:<14} clusters={nz.tolist()} values={vals}")


def _print_sim_matrix(list_docids, sim, topn=10):
    """Print the top-N x top-N doc-doc similarity submatrix (hits are already
    rank-ordered by base relevance, so the first `topn` are the top-N)."""
    n = min(topn, len(list_docids))
    ids = [str(d)[:12] for d in list_docids[:n]]
    header = " " * 14 + "".join(f"{c:>8}" for c in ids)
    print(f"[kmeans_dd] top-{n} x top-{n} cluster-cosine doc-doc matrix:")
    print(header)
    for i in range(n):
        row = "".join(f"{sim[i, j]:>8.3f}" for j in range(n))
        print(f"{ids[i]:<14}{row}")


def _select(
    hits,
    claim_reps_by_id,
    rows_by_parent,
    k,
    lambda_mult,
    mode,
    n_clusters,
    label_mode,
    kmeans_n_init,
    random_state,
    qid=None,
):
    if len(hits) <= 1:
        return hits
    if mode not in ("subtract", "add"):
        raise ValueError(f"mode must be 'subtract' (MMR) or 'add' (claim-echo boost), got {mode!r}")
    sign = -1.0 if mode == "subtract" else 1.0

    list_docids = [h.docid for h in hits]
    relevance = np.asarray([h.score for h in hits], dtype=np.float32)

    doc_vecs, labels, _, _ = _doc_cluster_vectors(
        list_docids=list_docids,
        claim_reps_by_id=claim_reps_by_id,
        rows_by_parent=rows_by_parent,
        n_clusters=n_clusters,
        label_mode=label_mode,
        kmeans_n_init=kmeans_n_init,
        random_state=random_state,
    )
    _print_cluster_summary(qid, doc_vecs.shape[1], labels)
    _print_doc_vectors(list_docids, doc_vecs, label_mode, topn=10)

    sim = _doc_sim_matrix(doc_vecs)
    _print_sim_matrix(list_docids, sim, topn=10)

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
    label_mode: str = "binary",
    kmeans_n_init: int = 10,
    random_state: int = 42,
) -> List[Result]:
    logger.info("kmeans-dd: mode=%s, label_mode=%s, n_clusters=%d, base relevance from run file %s, "
                "pool k=%d, lambda=%.2f, claim_reps=%s",
                mode, label_mode, n_clusters, run_file, k, lambda_mult, claim_reps)
    base_run = load_run(run_file, k=k)
    claim_corpus = load_corpus(corpus)

    needed_docids = {docid for pool in base_run.values() for docid, _ in pool}
    logger.info("kmeans-dd: %d unique pooled docid(s) across %d topic(s) need claim embeddings",
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
        logger.info("kmeans-dd: qid=%s pooled %d documents, %d claims found in embedding shards", qid, len(pool), n_claims)

        outputs[i].hits = hits
        outputs[i].evidences = _select(
            hits, claim_reps_by_id, rows_by_parent, k, lambda_mult, mode=mode,
            n_clusters=n_clusters, label_mode=label_mode,
            kmeans_n_init=kmeans_n_init, random_state=random_state, qid=qid,
        )

    return outputs
