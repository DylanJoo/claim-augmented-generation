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
"""
import copy
import glob
import logging
import pickle
from collections import defaultdict
from typing import List

import numpy as np

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

def _print_sim_matrix(list_docids, sim, topn=10):
    """Print the top-N x top-N doc-doc similarity submatrix (hits are already
    rank-ordered by base relevance, so the first `topn` are the top-N)."""
    n = min(topn, len(list_docids))
    ids = [str(d)[:12] for d in list_docids[:n]]
    header = " " * 14 + "".join(f"{c:>8}" for c in ids)
    print(f"[cc_dense] top-{n} x top-{n} claim-MaxSim doc-doc matrix:")
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
    agg
):
    if len(hits) <= 1:
        return hits
    if mode not in ("subtract", "add"):
        raise ValueError(f"mode must be 'subtract' (MMR) or 'add' (claim-echo boost), got {mode!r}")
    sign = -1.0 if mode == "subtract" else 1.0

    relevance = np.asarray([h.score for h in hits], dtype=np.float32)

    sim = _claim_aggsim_matrix(
        list_docids=[h.docid for h in hits],
        claim_reps_by_id=claim_reps_by_id,
        rows_by_parent=rows_by_parent,
        dim=dim,
        agg=agg
    )
    _print_sim_matrix([h.docid for h in hits], sim, topn=10)

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
) -> List[Result]:
    logger.info("cc-dense: mode=%s, agg=%s, base relevance from run file %s, pool k=%d, lambda=%.2f, claim_reps=%s",
                mode, agg, run_file, k, lambda_mult, claim_reps)
    base_run = load_run(run_file, k=k)
    claim_corpus = load_corpus(corpus)

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
        )

    return outputs
