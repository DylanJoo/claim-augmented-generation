"""
Rather than scoring exact claim-pair similarity (cc_dense.py's maxsim/mean),
every claim among the topic's pooled documents is first clustered with
k-means into `n_clusters` groups, fit fresh per topic (a cluster id is only
meaningful within one topic's own pool). Each pooled doc d is then
re-described as a vector over cluster ids ("multi-hot", since a doc can have
claims landing in several clusters):

  - label_mode="binary" (default): v_d[j] = 1 if d has >=1 claim in cluster
    j, else 0.
  - label_mode="scaled": v_d[j] = (# of d's claims in cluster j) / (# of d's
    claims total) -- a within-doc distribution over clusters, summing to 1.

A doc with zero claims in the shards gets the all-zero vector (it is *not*
padded into the k-means fit with a dummy zero-vector claim -- doing so would
inject a fake "no claim" cluster and skew the fit for every doc that does
have real claims). sim(d, d') is cosine similarity between these
cluster-membership vectors. Rationale: maxsim/mean tie diversity to exact
claim-pair similarity, which can be noisy claim-by-claim; clustering first
collapses that into a shared, topic-specific vocabulary of "claim topics"
every pooled doc is scored against, letting diversity be reasoned about in
terms of topic coverage rather than pairwise echoes.

top_m -- the "relevant core" fit -- controls WHICH docs' claims k-means is
actually fit on:

  - top_m=None (default): fit on every pooled doc's claims, exactly as the
    original cc_dense.py agg="kmeans" behaved.
  - top_m=<int>: `hits` arrive already sorted by base relevance (descending
    -- see utils.load_run); only the first top_m docs (the docs most likely
    to actually be relevant) are used to FIT k-means. Every doc's claims --
    core AND the long tail beyond top_m alike -- are then assigned cluster
    ids: core docs reuse the labels fit_predict already computed, tail docs
    are scored with km.predict() against those same fitted centroids. This
    matters because a pool is typically k=1000 docs deep while only ~15-20
    are actually relevant (checked directly against the neuclir1/ragtime1
    qrels' subtopic-id column) -- fitting on the whole pool means the fit is
    done on claims that are ~98% noise, which can dilute the topic buckets
    the whole method is trying to recover. top_m is a hard cutoff on rank,
    not a relevance-score threshold; pick a number comfortably above the
    qrels' observed ~15-20 true subtopics-per-topic (e.g. 50-100) to leave
    headroom for retrieval error while still cutting the pool down before
    fitting. Passing top_m >= len(pool) degenerates to the top_m=None case
    exactly (the tail-predict loop below is simply empty), so this is a
    strict generalization, not a separate code path.

Selection greedily maximizes a lambda_mult blend of base relevance and
per-round coverage gain, the same tradeoff construction cc_dense.py's
_select uses for its MMR subtract/add modes:

    score(d) = lambda_mult * relevance(d)
             + (1 - lambda_mult) * coverage_gain(d) / max_d' coverage_gain(d')

where coverage_gain(d) is

    sum_j doc_vecs[d, j] * (1 - alpha) ** covered_count[j]

the same marginal-gain construction rac_eval_ub.py's greedy_alpha_ndcg_oracle
uses to build alpha-nDCG's own ideal gain vector (Clarke et al., 2008) --
here with k-means clusters standing in for that oracle's ground-truth
subtopics, and doc_vecs (binary presence, or within-doc fraction under
label_mode="scaled") standing in for its J(d, s). covered_count[j] is how
many already-picked docs touch cluster j (presence, not the scaled weight),
so a cluster gets increasingly discounted the more it's already been
covered, rather than only being compared to the single closest already-
picked doc the way MMR's max-similarity penalty does. coverage_gain is
divided by n_clusters -- a fixed, round-invariant normalizer -- so it sits
on the same roughly-[0, 1] scale as relevance before the two are blended,
without swamping a lambda_mult blend against relevance scores that are
typically close to 1. This must NOT be each round's own max achievable
gain: dividing by that instead would map the current round's winning doc
to exactly 1.0 every round by construction, destroying the
decreasing-score signal a TREC run file's score column needs (and, at
lambda_mult=0.0, tying every selected doc's score at 1.0 outright).

lambda_mult=0.0 (default) is a true no-op reproducing the original pure-
coverage selection bit-for-bit: dividing every round's gain by a positive
per-round constant doesn't change its argmax, and relevance's contribution
is exactly zero. lambda_mult=1.0 ignores clusters entirely and picks by
base relevance alone, matching cc_dense's convention for the same value.
This was added because pure coverage (lambda_mult=0.0) can and does demote
the single most relevant doc out of rank 1 whenever it doesn't happen to
touch the most claim clusters -- costing StRecall@1 outright, since nothing
in the pure-coverage objective ever looks at relevance. Blending in
relevance protects the top rank(s) the same way cc_dense's lambda_mult
does, without discarding the coverage mechanism for later ranks.
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
        logger.info("cc-kmeans: [%d/%d] loaded shard %s, %d claim vector(s) matched so far",
                    i, len(files), fpath, len(reps_by_id))
    return reps_by_id

def _rows_by_parent(claim_reps_by_id):
    rows_by_parent = defaultdict(list)
    for claim_docid in claim_reps_by_id:
        parent_id = claim_docid.rsplit("#", 1)[0]
        rows_by_parent[parent_id].append(claim_docid)
    return rows_by_parent


def _kmeans_doc_vectors(
    list_docids,
    claim_reps_by_id,
    rows_by_parent,
    n_clusters,
    top_m=None,
    label_mode="binary",
    kmeans_n_init=10,
):
    """Fit k-means on the claims of the top `top_m` docs only (by base
    relevance -- `list_docids` must already be sorted that way), then assign
    every pooled doc's claims to those fitted centroids. top_m=None (or
    top_m >= len(list_docids)) fits on the whole pool -- see module
    docstring for the full mechanism, and for label_mode semantics.

    Returns (doc_vecs [n_docs, effective_k], core_labels [n_core_claims] or
    None -- None only when the core itself has no claims in the shards).
    """
    if label_mode not in ("binary", "scaled"):
        raise ValueError(f"label_mode must be 'binary' or 'scaled', got {label_mode!r}")

    n = len(list_docids)
    if top_m is None or top_m > n:
        top_m = n
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
        print(f"[cc_kmeans] {len(missing_docs)}/{n} pooled doc(s) have no claims in embedding "
              f"shards; their cluster vector is all-zero, e.g. {missing_docs[:3]!r}")

    if not core_claim_vecs:
        print(f"[cc_kmeans] no claims found among the top-{top_m} core doc(s); "
              f"returning all-zero vectors for the whole pool")
        return np.zeros((n, 0), dtype=np.float32), None

    core_claim_matrix = np.stack(core_claim_vecs).astype(np.float32)
    n_core_claims = core_claim_matrix.shape[0]

    effective_k = min(n_clusters, n_core_claims)
    if effective_k < n_clusters:
        print(f"[cc_kmeans] only {n_core_claims} claim(s) available among the top-{top_m} core "
              f"doc(s); clamping n_clusters {n_clusters} -> {effective_k}")

    km = KMeans(n_clusters=effective_k, n_init=kmeans_n_init, random_state=_KMEANS_RANDOM_STATE)
    core_labels = km.fit_predict(core_claim_matrix)

    # 2. Score every doc's claims against the core's centroids: core docs
    # reuse the labels fit_predict already computed, tail docs (beyond
    # top_m, if any) are assigned via .predict() into that same fitted
    # space. When top_m covers the whole pool, this loop is simply empty.
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

    if label_mode == "binary":
        doc_vecs = (counts > 0).astype(np.float32)
    else:  # "scaled"
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-9] = 1.0
        doc_vecs = counts / row_sums

    return doc_vecs, core_labels


def _print_cluster_summary(qid, n_clusters, labels):
    if labels is None:
        print(f"[cc_kmeans] qid={qid}: no claims found in this pool, nothing clustered")
        return
    sizes = np.bincount(labels, minlength=n_clusters)
    print(f"[cc_kmeans] qid={qid}: {len(labels)} claim(s) clustered into {n_clusters} cluster(s); "
          f"cluster sizes={sizes.tolist()}")


def _print_cluster_assignment(list_docids, doc_vecs, agg, topn=10):
    """Print the top-N docs' cluster-membership vectors (rows: docs, cols:
    cluster ids) -- which claim-topic cluster(s) each doc touches, and how
    strongly (label_mode="binary" -> 0/1, "scaled" -> within-doc fraction)."""
    n = min(topn, len(list_docids))
    k = doc_vecs.shape[1]
    ids = [str(d)[:12] for d in list_docids[:n]]
    header = " " * 14 + "".join(f"c{j:<5}" for j in range(k))
    print(f"[cc_kmeans] top-{n} doc x {k}-cluster ({agg}) assignment:")
    print(header)
    for i in range(n):
        row = "".join(f"{doc_vecs[i, j]:>6.2f}" for j in range(k))
        print(f"{ids[i]:<14}{row}")


def _select(
    hits,
    claim_reps_by_id,
    rows_by_parent,
    k,
    n_clusters,
    top_m=None,
    label_mode="binary",
    kmeans_n_init=10,
    alpha=0.5,
    lambda_mult=0.0,
    discount_floor=0.0,
    qid=None,
):
    """lambda_mult (default 0.0, see module docstring's relevance-blend
    section): 0.0 reproduces the original pure-coverage selection exactly;
    > 0.0 blends in base relevance so rank 1 (and later ranks) can't be
    handed to a doc that only wins on cluster coverage."""
    if len(hits) <= 1:
        return hits

    relevance = np.asarray([h.score for h in hits], dtype=np.float32)
    list_docids = [h.docid for h in hits]  # already relevance-sorted, see utils.load_run

    doc_vecs, labels = _kmeans_doc_vectors(
        list_docids=list_docids,
        claim_reps_by_id=claim_reps_by_id,
        rows_by_parent=rows_by_parent,
        n_clusters=n_clusters,
        top_m=top_m,
        label_mode=label_mode,
        kmeans_n_init=kmeans_n_init,
    )
    agg_label = "kmeans" if (top_m is None or top_m >= len(hits)) else f"kmeans-core(top{top_m})"
    _print_cluster_summary(qid, doc_vecs.shape[1], labels)
    _print_cluster_assignment(list_docids, doc_vecs, agg=agg_label, topn=10)

    n_select = min(k, len(hits))
    selected = []
    selected_scores = []

    # Same greedy construction as rac_eval_ub.py's greedy_alpha_ndcg_oracle
    # (Clarke et al. 2008's alpha-nDCG gain), with k-means cluster ids
    # standing in for that oracle's ground-truth subtopics and doc_vecs
    # (binary presence, or within-doc fraction under label_mode="scaled")
    # standing in for its J(d, s), blended against base relevance by
    # lambda_mult (see module docstring and _select's own docstring).
    touched = doc_vecs > 0
    covered_count = np.zeros(doc_vecs.shape[1], dtype=np.float32)
    # Fixed (round-invariant) normalizer: the most any single doc's gain
    # could be this topic, if it touched every cluster and none were
    # covered yet. Dividing by the CURRENT round's own max instead (i.e.
    # gain.max() computed fresh each iteration) would map the winning doc's
    # score to exactly 1.0 in every round by construction -- destroying the
    # decreasing-score signal entirely and, for lambda_mult=0.0, tying
    # every selected doc's score at 1.0 (silent bug caught by comparing
    # eval numbers against pre-refactor output, not by the selection-order
    # diff alone -- diffing the docid/rank columns can't see a broken score
    # column when the argmax sequence is untouched).
    effective_k = doc_vecs.shape[1]
    for _ in range(n_select):
        # discount_floor: a covered cluster never drops below this weight, so
        # overlap with already-selected (i.e. likely relevant) docs keeps
        # earning a little credit instead of only being penalized. 0.0 is a
        # no-op ((1-alpha)**n >= 0 always).
        discount = np.maximum(np.power(1.0 - alpha, covered_count), discount_floor)
        gain = (doc_vecs * discount).sum(axis=1)
        gain_norm = gain / effective_k if effective_k > 0 else gain
        scores = lambda_mult * relevance + (1.0 - lambda_mult) * gain_norm
        scores[selected] = -np.inf
        pick = int(np.argmax(scores))
        selected_scores.append(float(scores[pick]))
        selected.append(pick)
        covered_count += touched[pick]

    # lambda_mult=0.0's pure coverage_gain is monotone submodular (the
    # (1-alpha)**covered_count discount only ever grows), so its raw scores
    # are non-increasing by construction -- but per-round normalization by
    # gain_max, and any lambda_mult > 0 blend against relevance (which
    # doesn't shrink round to round the way coverage_gain does), aren't
    # guaranteed to preserve that ordering. Re-sorting descending keeps
    # every actually-computed magnitude but reassigns it by rank, so
    # downstream TREC tooling (which sorts by the score column) can't
    # silently discard the greedy order the loop above just computed --
    # same fix cc_dense.py's mode="add" needed for the same reason.
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
    n_clusters: int = 20,
    top_m: int = None,
    label_mode: str = "binary",
    kmeans_n_init: int = 10,
    alpha: float = 0.5,
    lambda_mult: float = 0.0,
    discount_floor: float = 0.0,
) -> List[Result]:
    """lambda_mult (default 0.0, no-op -- see module docstring and
    _select's docstring): relevance/coverage tradeoff, same convention as
    cc_dense.py's lambda_mult (1.0 = pure relevance, 0.0 = pure coverage)."""
    logger.info("cc-kmeans: n_clusters=%d, top_m=%s, base relevance from run file %s, "
                "pool k=%d, alpha=%.2f, lambda_mult=%.2f, claim_reps=%s",
                n_clusters, top_m, run_file, k, alpha, lambda_mult, claim_reps)
    base_run = load_run(run_file, k=k)
    claim_corpus = {}  # corpus text unused downstream; skip loading to save memory

    needed_docids = {docid for pool in base_run.values() for docid, _ in pool}
    logger.info("cc-kmeans: %d unique pooled docid(s) across %d topic(s) need claim embeddings",
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
        logger.info("cc-kmeans: qid=%s pooled %d documents, %d claims found in embedding shards",
                    qid, len(pool), n_claims)

        outputs[i].hits = hits
        outputs[i].evidences = _select(
            hits, claim_reps_by_id, rows_by_parent, k,
            n_clusters=n_clusters, top_m=top_m, label_mode=label_mode,
            kmeans_n_init=kmeans_n_init, alpha=alpha, lambda_mult=lambda_mult,
            discount_floor=discount_floor, qid=qid,
        )

    return outputs
