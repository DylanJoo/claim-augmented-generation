"""
DBSCAN variant of cc_kmeans.py: same "cluster the claims, describe every doc
as a vector over cluster ids, greedily pick docs by alpha-nDCG-style coverage
gain" pipeline, but the claim-topic clusters come from DBSCAN, so the number
of clusters is discovered per topic instead of being specified up front. See
cc_kmeans.py's module docstring for the shared mechanism (label_mode,
top_m core fit, lambda_mult blend, discount_floor, fixed normalizer); only
what differs is documented here.

Clustering: DBSCAN over the core docs' claim embeddings, cosine distance
(embeddings are L2-normalized, distance = 1 - dot). Two knobs:

  - eps: neighbourhood radius in cosine distance. Either fixed (`eps`) or
    adaptive per topic (`eps_quantile`, which overrides `eps`): eps is set
    to that quantile of every claim's distance to its `min_samples`-th
    nearest claim (the k-distance-plot heuristic, automated). A fixed eps
    doesn't transfer well across topics whose claim pools differ in density.
  - min_samples: claims needed inside eps to form a dense core point. No
    min_cluster_size -- a very common topic yields many near-duplicate
    claims, and those should be able to form one big cluster, which DBSCAN
    allows.

Noise (DBSCAN label -1) is handled by `noise_mode`:

  - "single" (default): every noise claim goes into ONE shared extra
    "noise" cluster. Docs whose claims are idiosyncratic outliers all
    collapse onto that one column, so they can earn (and then be discounted
    for) that one cluster's coverage only once, instead of each looking like
    its own novel topic. This is the denoising behaviour.
  - "drop": noise claims are ignored; they contribute nothing to doc_vecs.
  - "singleton": each noise claim is its own cluster, i.e. every outlier
    claim counts as a unique topic.

Docs beyond top_m (the tail) can't be .predict()ed -- DBSCAN has none -- so
each tail claim takes the label of its nearest DBSCAN core sample if that
sample is within eps. Otherwise it is noise: in "single" mode it joins the
noise cluster, in "singleton" mode it joins the nearest noise claim's
cluster if that is within eps, and in "drop" mode it is ignored.

If no cluster survives at all (everything dropped), _select returns the base
relevance order unchanged rather than a tie-broken arbitrary order.
"""
import copy
import logging
from typing import List

import numpy as np
from sklearn.cluster import DBSCAN

from retrieval.cc_kmeans import _load_claim_reps, _rows_by_parent
from utils import Result, Hit, load_run

logger = logging.getLogger(__name__)

NOISE_MODES = ("single", "drop", "singleton")


def _normalize(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return x / norms


def _resolve_eps(dist, eps, eps_quantile, min_samples):
    if eps_quantile is None:
        return eps
    kth = min(min_samples, dist.shape[0]) - 1
    # row-wise sort includes the self-distance (0) at position 0, matching
    # DBSCAN's convention that min_samples counts the point itself.
    kth_dist = np.sort(dist, axis=1)[:, kth]
    return float(np.quantile(kth_dist, eps_quantile))


def _dbscan_doc_vectors(
    list_docids,
    claim_reps_by_id,
    rows_by_parent,
    top_m=None,
    label_mode="binary",
    eps=0.3,
    eps_quantile=None,
    min_samples=5,
    noise_mode="single",
):
    """Returns (doc_vecs [n_docs, n_clusters], info dict or None -- None only
    when the core has no claims in the shards). info has: eps (used),
    n_clusters (real DBSCAN clusters, excluding any noise column), n_noise,
    n_core_claims."""
    if label_mode not in ("binary", "scaled"):
        raise ValueError(f"label_mode must be 'binary' or 'scaled', got {label_mode!r}")
    if noise_mode not in NOISE_MODES:
        raise ValueError(f"noise_mode must be one of {NOISE_MODES}, got {noise_mode!r}")

    n = len(list_docids)
    if top_m is None or top_m > n:
        top_m = n

    core_doc_of_claim, core_claim_vecs = [], []
    for doc_idx, docid in enumerate(list_docids[:top_m]):
        for cid in (rows_by_parent.get(docid) or []):
            if cid in claim_reps_by_id:
                core_claim_vecs.append(claim_reps_by_id[cid])
                core_doc_of_claim.append(doc_idx)

    if not core_claim_vecs:
        print(f"[cc_dbscan] no claims found among the top-{top_m} core doc(s); "
              f"returning all-zero vectors for the whole pool")
        return np.zeros((n, 0), dtype=np.float32), None

    core_x = _normalize(np.stack(core_claim_vecs))
    dist = np.clip(1.0 - core_x @ core_x.T, 0.0, 2.0)
    used_eps = _resolve_eps(dist, eps, eps_quantile, min_samples)
    db = DBSCAN(eps=used_eps, min_samples=min_samples, metric="precomputed").fit(dist)
    labels = db.labels_
    n_real = int(labels.max()) + 1 if (labels >= 0).any() else 0
    noise_idx = np.flatnonzero(labels < 0)
    core_sample_idx = db.core_sample_indices_

    # Final cluster id per core claim (-1 = ignored) and the id space size.
    final = labels.copy()
    if noise_mode == "single":
        n_ids = n_real + 1 if len(noise_idx) else n_real
        final[noise_idx] = n_real
    elif noise_mode == "singleton":
        n_ids = n_real + len(noise_idx)
        final[noise_idx] = n_real + np.arange(len(noise_idx))
    else:  # drop
        n_ids = n_real

    counts = np.zeros((n, n_ids), dtype=np.float32)
    for doc_idx, cid in zip(core_doc_of_claim, final):
        if cid >= 0:
            counts[doc_idx, cid] += 1.0

    # Tail docs: nearest DBSCAN core sample within eps, else treat as noise.
    for doc_idx in range(top_m, n):
        ids = [c for c in (rows_by_parent.get(list_docids[doc_idx]) or []) if c in claim_reps_by_id]
        if not ids or n_ids == 0:
            continue
        tail_x = _normalize(np.stack([claim_reps_by_id[c] for c in ids]))
        for row in range(tail_x.shape[0]):
            cid = -1
            if len(core_sample_idx):
                d = 1.0 - core_x[core_sample_idx] @ tail_x[row]
                j = int(np.argmin(d))
                if d[j] <= used_eps:
                    cid = int(labels[core_sample_idx[j]])
            if cid < 0:
                if noise_mode == "single" and len(noise_idx):
                    cid = n_real
                elif noise_mode == "singleton" and len(noise_idx):
                    d = 1.0 - core_x[noise_idx] @ tail_x[row]
                    j = int(np.argmin(d))
                    if d[j] <= used_eps:
                        cid = int(final[noise_idx[j]])
            if cid >= 0:
                counts[doc_idx, cid] += 1.0

    if label_mode == "binary":
        doc_vecs = (counts > 0).astype(np.float32)
    else:
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-9] = 1.0
        doc_vecs = counts / row_sums

    info = {"eps": used_eps, "n_clusters": n_real, "n_noise": int(len(noise_idx)),
            "n_core_claims": int(len(labels)), "labels": final}
    return doc_vecs, info


def _print_cluster_summary(qid, doc_vecs, info, noise_mode):
    if info is None:
        print(f"[cc_dbscan] qid={qid}: no claims found in this pool, nothing clustered")
        return
    labels = info["labels"]
    sizes = np.bincount(labels[labels >= 0], minlength=doc_vecs.shape[1])
    print(f"[cc_dbscan] qid={qid}: eps={info['eps']:.3f}, {info['n_core_claims']} core claim(s) -> "
          f"{info['n_clusters']} cluster(s) + {info['n_noise']} noise claim(s) "
          f"(noise_mode={noise_mode}); vector width={doc_vecs.shape[1]}; "
          f"top sizes={sorted(sizes.tolist(), reverse=True)[:15]}")


def _select(
    hits,
    claim_reps_by_id,
    rows_by_parent,
    k,
    top_m=None,
    label_mode="binary",
    eps=0.3,
    eps_quantile=None,
    min_samples=5,
    noise_mode="single",
    alpha=0.5,
    lambda_mult=0.0,
    discount_floor=0.0,
    qid=None,
):
    if len(hits) <= 1:
        return hits

    relevance = np.asarray([h.score for h in hits], dtype=np.float32)
    list_docids = [h.docid for h in hits]  # already relevance-sorted, see utils.load_run

    doc_vecs, info = _dbscan_doc_vectors(
        list_docids, claim_reps_by_id, rows_by_parent, top_m=top_m, label_mode=label_mode,
        eps=eps, eps_quantile=eps_quantile, min_samples=min_samples, noise_mode=noise_mode,
    )
    _print_cluster_summary(qid, doc_vecs, info, noise_mode)

    effective_k = doc_vecs.shape[1]
    n_select = min(k, len(hits))
    if effective_k == 0:
        return hits[:n_select]

    # Same greedy alpha-nDCG-style loop as cc_kmeans._select, including the
    # fixed effective_k normalizer (see cc_kmeans's module docstring).
    touched = doc_vecs > 0
    covered_count = np.zeros(effective_k, dtype=np.float32)
    selected, selected_scores = [], []
    for _ in range(n_select):
        discount = np.maximum(np.power(1.0 - alpha, covered_count), discount_floor)
        gain_norm = (doc_vecs * discount).sum(axis=1) / effective_k
        scores = lambda_mult * relevance + (1.0 - lambda_mult) * gain_norm
        scores[selected] = -np.inf
        pick = int(np.argmax(scores))
        selected_scores.append(float(scores[pick]))
        selected.append(pick)
        covered_count += touched[pick]

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
    top_m: int = None,
    label_mode: str = "binary",
    eps: float = 0.3,
    eps_quantile: float = None,
    min_samples: int = 5,
    noise_mode: str = "single",
    alpha: float = 0.5,
    lambda_mult: float = 0.0,
    discount_floor: float = 0.0,
) -> List[Result]:
    logger.info("cc-dbscan: eps=%s, eps_quantile=%s, min_samples=%d, noise_mode=%s, top_m=%s, "
                "run file %s, pool k=%d, alpha=%.2f, lambda_mult=%.2f, claim_reps=%s",
                eps, eps_quantile, min_samples, noise_mode, top_m, run_file, k, alpha,
                lambda_mult, claim_reps)
    base_run = load_run(run_file, k=k)

    needed_docids = {docid for pool in base_run.values() for docid, _ in pool}
    claim_reps_by_id = _load_claim_reps(claim_reps, needed_docids)
    rows_by_parent = _rows_by_parent(claim_reps_by_id)

    outputs = copy.deepcopy(inputs)
    for i, inp in enumerate(inputs):
        qid = str(inp.topic["qid"])
        pool = base_run.get(qid, [])
        hits = [
            Hit(docid=docid, score=score, rank=rank, content_dict={"text": None, "title": None})
            for rank, (docid, score) in enumerate(pool, start=1)
        ]
        outputs[i].hits = hits
        outputs[i].evidences = _select(
            hits, claim_reps_by_id, rows_by_parent, k,
            top_m=top_m, label_mode=label_mode, eps=eps, eps_quantile=eps_quantile,
            min_samples=min_samples, noise_mode=noise_mode, alpha=alpha,
            lambda_mult=lambda_mult, discount_floor=discount_floor, qid=qid,
        )
    return outputs
