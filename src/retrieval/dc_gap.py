"""
dc-gap (Doc-Claim Gap) re-ranking on top of a doc-level run file.

Unlike mmr.py (whole-document similarity) and cc_mmr.py (claim-claim MaxSim),
this is not a relevance/diversity trade-off and has no lambda knob. Instead,
each already-selected document's full text is used as a query against every
individual claim of every remaining candidate document:

    score(D_i, c) = BM25(text(D_i), c)   for c in claims(D_c)

Each candidate D_c is scored against a selected document D_i by pooling
*all* of D_c's claim scores into a single bridge value (its best-matching
claim) and a single novelty value (its worst-matching claim):

    bridge(D_i, D_c)  = max_{c in claims(D_c)} score(D_i, c)
    novelty(D_i, D_c) = min_{c in claims(D_c)} score(D_i, c)

The bridge and novelty values need not come from the same claim -- one
claim of D_c may drive the bridge value while a different claim drives the
novelty value. As documents are selected, both quantities are tracked as
running extrema over the selected set:

    running_bridge(D_c)  = max_i bridge(D_i, D_c)
    running_novelty(D_c) = min_i novelty(D_i, D_c)

and D_c's selection score is the gap between them:

    gap(D_c) = running_bridge(D_c) - running_novelty(D_c)

A large gap means D_c has a claim that closely tracks something already
selected (bridge, coherence with the picked set) while also having a claim
that doesn't resemble at least one other selected document (novelty, so it
isn't just a rehash of everything already covered).

Each side (bridge, novelty) uses a single operator (max, min respectively)
consistently across both the claims-of-D_c dimension and the selected-docs
dimension, so aggregation order doesn't matter within a side. What must not
happen is mixing the two operators in a nested manner -- e.g. min_i [ max_c
score(D_i, c) ] is a different quantity from max_c [ min_i score(D_i, c) ],
since min and max don't commute across levels; that mismatch is why bridge
and novelty are each computed with one operator end-to-end rather than
aggregated in the opposite order partway through.

The first pick has no selected set to compare against, so it falls back to
pure base relevance (same as rank 1 of the input run). The second pick has
exactly one selected document, so running_bridge and running_novelty are
identical and the gap is trivially zero; that round instead uses
running_bridge directly (plain resemblance to the one selected document).

The base relevance signal is read directly from a pre-computed doc-level
TREC run file; per-doc claims and full text for the doc-claim scoring come
from the document corpus's "statements" and "text" fields, keyed by docid.
"""
import copy
import logging
from typing import List

import bm25s
import bm25s.tokenization
import numpy as np

from utils import Result, Hit, load_run, load_corpus

logger = logging.getLogger(__name__)


def _normalize(values):
    values = np.asarray(values, dtype=np.float32)
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-9:
        return np.ones_like(values)
    return (values - lo) / (hi - lo)


def _doc_to_claim_scores(pool_docids, claims_by_doc, fulltext_by_doc, stopwords, stemmer, k1, b):
    """Doc(query)-to-claim(candidate) BM25 scores, one column per pooled claim.

    Builds one local BM25 index over every pooled doc's claims, then scores
    each pooled doc's full text against every individual claim:

        score(D_i, c) = BM25(text(D_i), c)

    Returns the raw (n_docs x n_claims) score matrix, unaggregated, plus
    doc_of_claim mapping each claim column to its owning candidate doc.
    Aggregation to doc level is intentionally left to the caller (see module
    docstring for why doing it here, before selection, would conflate the
    bridge and novelty terms of the gap).

    A single Tokenizer is shared between the claim index and the doc
    queries so that query token IDs resolve against the same vocabulary the
    index was built with. Tokenizing them independently (e.g. two separate
    bm25s.tokenize calls) assigns different IDs to the same word in each
    call, producing out-of-vocabulary token IDs at query time.
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

    tokenizer = bm25s.tokenization.Tokenizer(stopwords=stopwords, stemmer=stemmer)
    claim_tokens = tokenizer.tokenize(texts, update_vocab=True, return_as="tuple", show_progress=False)
    index = bm25s.BM25(k1=k1, b=b)
    index.index(claim_tokens, show_progress=False)

    query_texts = [fulltext_by_doc.get(docid) or "" for docid in pool_docids]
    query_ids = tokenizer.tokenize(query_texts, update_vocab=False, return_as="ids", show_progress=False)

    n_claims = len(texts)
    doc_claim_scores = np.zeros((n, n_claims), dtype=np.float32)
    for i in range(n):
        if len(query_ids[i]) == 0:
            print(f"[dc_gap] empty token list for doc {i} (docid={pool_docids[i]!r}); leaving score row as zeros")
            continue
        doc_claim_scores[i] = index.get_scores(query_ids[i])

    return doc_claim_scores, doc_of_claim


def _claims_to_doc_extrema(claim_values, claim_idx_by_doc):
    """Aggregate a per-claim value array to one (max, min) pair per candidate
    doc, over that doc's own claims only. Each side uses a single operator
    throughout (max for bridge, min for novelty), so it stays well-defined
    regardless of whether you aggregate over claims first or over selected
    docs first -- only *mixing* max and min in a nested fashion is order
    dependent, not repeated use of the same operator."""
    doc_max = np.empty(len(claim_idx_by_doc), dtype=np.float32)
    doc_min = np.empty(len(claim_idx_by_doc), dtype=np.float32)
    for j, idxs in enumerate(claim_idx_by_doc):
        doc_max[j] = claim_values[idxs].max() if len(idxs) else -np.inf
        doc_min[j] = claim_values[idxs].min() if len(idxs) else np.inf
    return doc_max, doc_min


def _print_bridge_matrix(pool_docids, bridge_matrix, topn=10):
    """Print the top-N x top-N doc-doc bridge submatrix (hits are already
    rank-ordered by base relevance, so the first `topn` are the top-N).
    bridge[i, j] = max_{c in claims(D_j)} score(D_i, c); asymmetric: row =
    query doc, column = candidate doc."""
    n = min(topn, len(pool_docids))
    ids = [str(d)[:12] for d in pool_docids[:n]]
    header = " " * 14 + "".join(f"{c:>8}" for c in ids)
    print(f"[dc_gap] top-{n} x top-{n} doc(query)-claim(candidate) bridge matrix:")
    print(header)
    for i in range(n):
        row = "".join(f"{bridge_matrix[i, j]:>8.3f}" for j in range(n))
        print(f"{ids[i]:<14}{row}")


def _gap_select(hits, claims_by_doc, fulltext_by_doc, k, stopwords, stemmer, k1, b):
    if len(hits) <= 1:
        return hits

    pool_docids = [h.docid for h in hits]
    relevance = _normalize([h.score for h in hits])
    doc_claim_scores, doc_of_claim = _doc_to_claim_scores(
        pool_docids, claims_by_doc, fulltext_by_doc, stopwords, stemmer, k1, b
    )

    n = len(hits)
    claim_idx_by_doc = [np.where(doc_of_claim == j)[0] for j in range(n)]

    # bridge_matrix[i, j] = max_{c in claims(D_j)} score(D_i, c)
    # novelty_matrix[i, j] = min_{c in claims(D_j)} score(D_i, c)
    # both pooled over every claim belonging to D_j at once, so the claim
    # driving the bridge value and the claim driving the novelty value need
    # not be the same claim
    bridge_matrix = np.zeros((n, n), dtype=np.float32)
    novelty_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        bridge_matrix[i], novelty_matrix[i] = _claims_to_doc_extrema(doc_claim_scores[i], claim_idx_by_doc)
    _print_bridge_matrix(pool_docids, bridge_matrix, topn=10)

    n_select = min(k, n)
    selected = []
    picked_scores = []
    running_bridge = np.full(n, -np.inf, dtype=np.float32)
    running_novelty = np.full(n, np.inf, dtype=np.float32)

    # pick 1: no selected set yet to compare against -- pure base relevance
    first = int(np.argmax(relevance))
    selected.append(first)
    picked_scores.append(float(relevance[first]))
    running_bridge = np.maximum(running_bridge, bridge_matrix[first])
    running_novelty = np.minimum(running_novelty, novelty_matrix[first])

    for _ in range(1, n_select):
        if len(selected) == 1:
            # only one selected doc so far -> bridge == novelty, gap is
            # trivially zero; use the plain bridge score instead
            scores = running_bridge.copy()
        else:
            scores = running_bridge - running_novelty
        scores[selected] = -np.inf
        pick = int(np.argmax(scores))
        picked_scores.append(float(scores[pick]))
        selected.append(pick)
        running_bridge = np.maximum(running_bridge, bridge_matrix[pick])
        running_novelty = np.minimum(running_novelty, novelty_matrix[pick])

    # unlike cc_mmr's penalty-based score (monotonic by construction since
    # its running max-sim-to-selected only grows), this is a reward-based
    # score and isn't guaranteed non-increasing across ranks -- clamp it so
    # downstream TREC eval tools, which sort by score rather than the rank
    # column, don't reorder hits relative to the greedy pick order
    written_scores = []
    ceiling = np.inf
    for s in picked_scores:
        ceiling = min(ceiling, s)
        written_scores.append(ceiling)

    return [
        Hit(docid=hits[idx].docid, score=written_scores[rank - 1], rank=rank, content_dict=hits[idx].content_dict)
        for rank, idx in enumerate(selected, start=1)
    ]


def run(
    inputs: List[Result],
    run_file: str,
    corpus: List[str],
    k: int = 1000,
    stopwords: str = "en",
    stemmer_name: str = None,
    local_k1: float = 1.2,
    local_b: float = 0.5,
) -> List[Result]:
    stemmer = None
    if stemmer_name == "snowball":
        import Stemmer
        stemmer = Stemmer.Stemmer("english")

    logger.info("dc-gap: base relevance from run file %s, pool k=%d", run_file, k)
    base_run = load_run(run_file, k=k)  # this is the results based on document
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
                    "doc-text": claim_corpus.get(docid, {}).get("text"),
                    "claim-text": claim_corpus.get(docid, {}).get("statements"),
                    "title": claim_corpus.get(docid, {}).get("title"),
                },
            )
            for rank, (docid, score) in enumerate(pool, start=1)
        ]
        claims_by_doc = {docid: claim_corpus.get(docid, {}).get("statements") or [] for docid, _ in pool}
        doc_by_doc = {docid: claim_corpus.get(docid, {}).get("text") or "" for docid, _ in pool}

        n_claims = sum(len(c) for c in claims_by_doc.values())
        logger.info("dc-gap: qid=%s pooled %d documents, %d claims", qid, len(pool), n_claims)

        outputs[i].hits = hits
        outputs[i].evidences = _gap_select(
            hits,
            claims_by_doc,
            doc_by_doc,
            k,
            stopwords=stopwords,
            stemmer=stemmer,
            k1=local_k1,
            b=local_b,
        )

    return outputs
