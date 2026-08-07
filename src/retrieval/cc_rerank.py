"""
cc-rerank (Claim-Claim rerank) on top of a doc-level run file.

Greedily re-ranks a pooled doc list by combining a base relevance score
with a claim-level MaxSim signal against the docs already selected. For a
pair of pooled documents,

    sim(d, d') = sum_{c in d} max_{c' in d'} sim_claim(c, c')

i.e. the MaxSim aggregation ColBERT uses for late-interaction token
embeddings, applied here to claims via lexical BM25 instead of dense
vectors. mmr.py concatenates all of a document's claims into one
bag-of-words vector before comparing documents -- that blurs claims
together, so two docs that each contain one near-duplicate claim (but are
otherwise unrelated) can look dissimilar overall. Comparing claim-to-claim
and taking the max keeps that overlap visible.

`mode` controls how this signal combines with relevance at each greedy
step:

  - "subtract" (MMR): score = lambda * relevance - (1 - lambda) * max_sim
    Treats claim overlap with already-picked docs as redundancy and
    penalizes it, pushing the selection toward diverse claims.

  - "add" (claim-echo boost): score = lambda * relevance + (1 - lambda) * max_sim
    Treats claim overlap with already-picked docs as corroborating
    evidence of relevance and rewards it -- closer to pseudo-relevance
    feedback than to MMR. Note max_sim_to_selected is exactly zero on the
    first pick (nothing is selected yet), so mode only affects rounds 2+.

The base relevance signal is read directly from a pre-computed doc-level
TREC run file; per-doc claims for the diversity signal come from the
document corpus's "statements" field, keyed by docid.
"""
import copy
import logging
from typing import List

import bm25s
import numpy as np

from utils import Result, Hit, load_run, load_corpus

logger = logging.getLogger(__name__)


def _normalize(values):
    values = np.asarray(values, dtype=np.float32)
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-9:
        return np.ones_like(values)
    return (values - lo) / (hi - lo)


def _claim_aggsim_matrix(pool_docids, claims_by_doc, stopwords, stemmer, k1, b):
    """Doc-doc similarity via claim-level MaxSim, row-normalized to [0, 1].

    Builds one local BM25 index over every pooled doc's claims, scores every
    claim against every other claim, then aggregates the claim x claim matrix
    to doc x doc using ColBERT-style MaxSim: for each claim c in d, take its
    max similarity to any claim c' in d', then sum those per-claim maxes,

        sim(d, d') = sum_{c in d} max_{c' in d'} sim_claim(c, c')
    """
    n = len(pool_docids)
    doc_of_claim = []
    texts = []
    for doc_idx, docid in enumerate(pool_docids):
        doc_claims = claims_by_doc.get(docid) or [""]
        for claim_text in doc_claims:
            texts.append(claim_text)
            doc_of_claim.append(doc_idx)
    doc_of_claim = np.asarray(doc_of_claim)

    tokens = bm25s.tokenize(texts, stopwords=stopwords, stemmer=stemmer, show_progress=False)
    local = bm25s.BM25(k1=k1, b=b)
    local.index(tokens, show_progress=False)

    n_claims = len(texts)
    claim_sim = np.zeros((n_claims, n_claims), dtype=np.float32)
    # fill the matrix with claim-claim scores
    for i in range(n_claims):
        if len(tokens.ids[i]) == 0:
            print(f"[cc_rerank] empty token list for claim {i} (doc={pool_docids[doc_of_claim[i]]!r}, text={texts[i]!r}); leaving similarity row as zeros")
            continue
        claim_sim[i] = local.get_scores(tokens.ids[i])

    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        rows = claim_sim[doc_of_claim == i]
        for j in range(n):
            sim[i, j] = rows[:, doc_of_claim == j].max(axis=1).sum()

    row_max = sim.max(axis=1, keepdims=True)
    row_max[row_max < 1e-9] = 1.0
    return sim / row_max


def _print_sim_matrix(pool_docids, sim, topn=10):
    """Print the top-N x top-N doc-doc similarity submatrix (hits are already
    rank-ordered by base relevance, so the first `topn` are the top-N)."""
    n = min(topn, len(pool_docids))
    ids = [str(d)[:12] for d in pool_docids[:n]]
    header = " " * 14 + "".join(f"{c:>8}" for c in ids)
    print(f"[cc_rerank] top-{n} x top-{n} claim-MaxSim doc-doc matrix:")
    print(header)
    for i in range(n):
        row = "".join(f"{sim[i, j]:>8.3f}" for j in range(n))
        print(f"{ids[i]:<14}{row}")


def _select(hits, claims_by_doc, k, lambda_mult, mode, stopwords, stemmer, k1, b):
    if len(hits) <= 1:
        return hits
    if mode not in ("subtract", "add"):
        raise ValueError(f"mode must be 'subtract' (MMR) or 'add' (claim-echo boost), got {mode!r}")
    sign = -1.0 if mode == "subtract" else 1.0

    pool_docids = [h.docid for h in hits]
    relevance = _normalize([h.score for h in hits]) # NOTE: This is optional?
    sim = _claim_aggsim_matrix(pool_docids, claims_by_doc, stopwords, stemmer, k1, b)
    _print_sim_matrix(pool_docids, sim, topn=10)

    n = len(hits)
    n_select = min(k, n)
    selected = []
    picked_scores = []
    max_sim_to_selected = np.zeros(n, dtype=np.float32)

    for _ in range(n_select):
        scores = lambda_mult * relevance + sign * (1 - lambda_mult) * max_sim_to_selected
        scores[selected] = -np.inf
        pick = int(np.argmax(scores))
        picked_scores.append(float(scores[pick]))
        selected.append(pick)
        max_sim_to_selected = np.maximum(max_sim_to_selected, sim[pick])

    # In "subtract" mode picked_scores is non-increasing by construction
    # (max_sim_to_selected only grows, so each round's best achievable score can
    # only shrink), which keeps the written score monotonic with rank -- downstream
    # TREC eval tools sort by score, not rank. In "add" mode this monotonicity is
    # not guaranteed (a later pick's boost can exceed an earlier round's score), so
    # scores are re-flattened to be strictly rank-decreasing before writing out.
    if mode == "add" and len(picked_scores) > 1:
        picked_scores = list(np.minimum.accumulate(picked_scores))

    return [
        Hit(docid=hits[idx].docid, score=picked_scores[rank - 1], rank=rank, content_dict=hits[idx].content_dict)
        for rank, idx in enumerate(selected, start=1)
    ]


def run(
    inputs: List[Result],
    run_file: str,
    corpus: List[str],
    k: int = 1000,
    lambda_mult: float = 0.9,
    mode: str = "subtract",
    stopwords: str = "en",
    stemmer_name: str = None,
    local_k1: float = 1.2,
    local_b: float = 0.5,
) -> List[Result]:
    stemmer = None
    if stemmer_name == "snowball":
        import Stemmer
        stemmer = Stemmer.Stemmer("english")

    logger.info("cc-rerank: mode=%s, base relevance from run file %s, pool k=%d, lambda=%.2f", mode, run_file, k, lambda_mult)
    base_run = load_run(run_file, k=k) # this is the results based on document
    claim_corpus = load_corpus(corpus)

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
                    # "text": claim_corpus.get(docid, {}).get("text", ""),
                    "text": claim_corpus.get(docid, {}).get("statements"),
                    "title": claim_corpus.get(docid, {}).get("title"),
                },
            )
            for rank, (docid, score) in enumerate(pool, start=1)
        ]
        claims_by_doc = {docid: claim_corpus.get(docid, {}).get("statements") or [] for docid, _ in pool}

        n_claims = sum(len(c) for c in claims_by_doc.values())
        logger.info("cc-rerank: qid=%s pooled %d documents, %d claims", qid, len(pool), n_claims)

        outputs[i].hits = hits
        outputs[i].evidences = _select(
            hits,
            claims_by_doc,
            k,
            lambda_mult,
            mode=mode,
            stopwords=stopwords,
            stemmer=stemmer,
            k1=local_k1,
            b=local_b,
        )

    return outputs
