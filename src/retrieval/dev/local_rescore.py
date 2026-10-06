"""
Doc-level global retrieval -> claim expansion -> local reindex with
term-discounting + doc-level soft gating -> iterative claim selection.

Stage 1:   doc-level global retrieval (concat-claims index) -> top-k_doc docs, s(q,d).
Stage 1.5: (optional, if dedup_threshold is set) collapse near-duplicate docs that
           share most of their claims (e.g. wire-service reposts), keeping only the
           one with the most claims per cluster -- see _dedup_similar_docs.
Stage 1.6: expand each surviving doc into its individual claims (claim-level index)
           -> the "subset corpus".
Stage 2:   local reindex over the expanded claims (in-memory bm25s).
Stage 3:   iterative greedy selection over the local index:
             - a per-claim `weight_mask` derived from the claim's parent doc's
               global d_score (min-max normalized, floored at `doc_floor`) acts as
               a soft multiplicative "globally relevant enough" gate -- not a
               hard prefilter, low-scoring docs are dampened, not excluded.
             - after picking a claim each round, its unique terms' contribution
               is discounted directly in bm25s's internal CSC score data
               (indptr is keyed by token id: indptr[tok]:indptr[tok+1] gives
               every claim's precomputed tf*idf*norm contribution for that
               term), so subsequent rounds are scored almost entirely by
               novel/uncovered terms -- a live, in-place SumBasic-style
               coverage discount, no reindexing needed.
"""
import copy
import json
import logging
from collections import defaultdict
from typing import List

import bm25s
import numpy as np

from utils import Result, Hit

logger = logging.getLogger(__name__)


def _load_bm25(index_dir, mmap=True, load_corpus=True):
    retriever = bm25s.BM25.load(index_dir, load_corpus=load_corpus, mmap=mmap)
    with open(f"{index_dir}/docids.json") as f:
        docids = json.load(f)
    return retriever, docids


def _retrieve_docs(doc_retriever, doc_docids, query_tokens, k_doc):
    """Doc-level global retrieval (concatenated-claims index) -> s(q, d)."""
    results, scores = doc_retriever.retrieve(query_tokens, k=k_doc, show_progress=False)
    doc_hits = []
    for rank, (doc, score) in enumerate(zip(results[0], scores[0]), start=1):
        parent_id = doc_docids[doc["id"]]
        doc_hits.append({"docid": parent_id, "text": doc["text"], "d_score": float(score), "d_rank": rank})
    return doc_hits


def _scan_claim_indices(claim_docids, retrieved_parents):
    """Find which claim-index positions belong to the retrieved parent docs.

    NOTE: does a linear scan over every claim docid to find matches for the
    retrieved parents. Fine per-topic at prototype scale; a production version
    should precompute a parent_id -> claim_indices side-index once at indexing
    time instead of rescanning per query.
    """
    claim_indices_by_parent = defaultdict(list)
    for idx, docid in enumerate(claim_docids):
        parent = docid.rsplit("#", 1)[0]
        if parent in retrieved_parents:
            claim_indices_by_parent[parent].append(idx)
    return claim_indices_by_parent


def _dedup_similar_docs(doc_hits, claim_counts, stopwords, stemmer, k1, b, threshold):
    """Collapse near-duplicate documents (e.g. wire-service reposts) that share
    most of their claims, keeping only the one with the most claims per cluster.

    Two docs are linked if EITHER direction's BM25 self-similarity (one doc's
    concatenated-claims text used as a query against the local pool) exceeds
    `threshold` -- this catches both mutual near-duplicates and the case where
    a shorter doc's claims are a redundant subset of a longer one. `threshold`
    is a raw BM25 score, so its right value depends on corpus/query length --
    tune empirically per corpus rather than trusting the default as-is.
    """
    n = len(doc_hits)
    if n <= 1:
        return doc_hits

    texts = [h["text"] for h in doc_hits]
    tokens = bm25s.tokenize(texts, stopwords=stopwords, stemmer=stemmer, show_progress=False)
    local = bm25s.BM25(k1=k1, b=b)
    local.index(tokens, show_progress=False)

    parent_of = list(range(n))

    def find(x):
        while parent_of[x] != x:
            parent_of[x] = parent_of[parent_of[x]]
            x = parent_of[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent_of[ry] = rx

    for i in range(n):
        scores_i = local.get_scores(tokens.ids[i])  # doc i's claims as a query against the whole pool
        for j in range(n):
            if i != j and scores_i[j] > threshold:
                union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    survivors = []
    for members in clusters.values():
        best = max(members, key=lambda i: claim_counts.get(doc_hits[i]["docid"], 0))
        survivors.append(doc_hits[best])
        if len(members) > 1:
            dropped = [doc_hits[i]["docid"] for i in members if i != best]
            logger.info(
                "Deduped cluster: kept %s (%d claims), dropped %s",
                doc_hits[best]["docid"], claim_counts.get(doc_hits[best]["docid"], 0), dropped,
            )

    survivors.sort(key=lambda h: h["d_rank"])
    return survivors


def _fetch_claim_texts(claim_retriever, claim_docids, claim_indices_by_parent, doc_hit_by_id):
    subset = []
    for parent, idxs in claim_indices_by_parent.items():
        d_hit = doc_hit_by_id[parent]
        for idx in idxs:
            text = claim_retriever.corpus[idx]["text"]
            subset.append({
                "docid": claim_docids[idx],
                "parent": parent,
                "text": text,
                "d_score": d_hit["d_score"],
                "d_rank": d_hit["d_rank"],
            })
    return subset


def _iterative_select(subset, query_tokens, k1, b, stopwords, stemmer, doc_floor, discount, n_iters):
    local_texts = [h["text"] for h in subset]
    local_tokens = bm25s.tokenize(local_texts, stopwords=stopwords, stemmer=stemmer, show_progress=False)
    local_retriever = bm25s.BM25(k1=k1, b=b)
    local_retriever.index(local_tokens, show_progress=False)

    d_scores = np.array([h["d_score"] for h in subset], dtype=np.float32)
    d_min, d_max = d_scores.min(), d_scores.max()
    doc_relevance_weight = doc_floor + (1 - doc_floor) * (d_scores - d_min) / (d_max - d_min + 1e-9)
    exclusion_mask = np.ones(len(subset), dtype=np.float32)

    indptr = local_retriever.scores["indptr"]
    data = local_retriever.scores["data"]

    selected = []
    for r in range(1, min(n_iters, len(subset)) + 1):
        weight_mask = doc_relevance_weight * exclusion_mask
        results, scores = local_retriever.retrieve(
            query_tokens, k=1, weight_mask=weight_mask, show_progress=False
        )
        pick_idx = int(results[0][0]["id"] if isinstance(results[0][0], dict) else results[0][0])
        pick_score = float(scores[0][0])
        h = subset[pick_idx]
        selected.append(Hit(
            docid=h["docid"],
            score=pick_score,
            rank=r,
            content_dict={"text": h["text"], "title": None},
        ))

        # discount this claim's terms in-place so future rounds favor novel terms
        for tok_id in set(local_tokens.ids[pick_idx]):
            start, end = indptr[tok_id], indptr[tok_id + 1]
            data[start:end] *= discount

        exclusion_mask[pick_idx] = 0.0

    return selected


def run(
    inputs: List[Result],
    doc_index: str,
    claim_index: str,
    k_doc: int = 50,
    doc_floor: float = 0.3,
    discount: float = 0.3,
    n_iters: int = 15,
    k1: float = 1.2,
    b: float = 0.5,
    stopwords: str = "en",
    stemmer_name: str = None,
    dedup_threshold: float = None,
) -> List[Result]:
    outputs = copy.deepcopy(inputs)

    stemmer = None
    if stemmer_name == "snowball":
        import Stemmer
        stemmer = Stemmer.Stemmer("english")

    logger.info("Loading doc-level index from %s", doc_index)
    doc_retriever, doc_docids = _load_bm25(doc_index)
    logger.info("Loading claim-level index from %s", claim_index)
    claim_retriever, claim_docids = _load_bm25(claim_index)

    for i, inp in enumerate(inputs):
        query = inp.topic["query"]
        query_tokens = bm25s.tokenize([query], stopwords=stopwords, stemmer=stemmer)

        doc_hits = _retrieve_docs(doc_retriever, doc_docids, query_tokens, k_doc)

        retrieved_parents = {h["docid"] for h in doc_hits}
        claim_indices_by_parent = _scan_claim_indices(claim_docids, retrieved_parents)
        claim_counts = {parent: len(idxs) for parent, idxs in claim_indices_by_parent.items()}

        if dedup_threshold is not None:
            doc_hits = _dedup_similar_docs(
                doc_hits, claim_counts,
                stopwords=stopwords, stemmer=stemmer, k1=k1, b=b,
                threshold=dedup_threshold,
            )
            survivors = {h["docid"] for h in doc_hits}
            claim_indices_by_parent = {
                p: idxs for p, idxs in claim_indices_by_parent.items() if p in survivors
            }

        doc_hit_by_id = {h["docid"]: h for h in doc_hits}
        subset = _fetch_claim_texts(claim_retriever, claim_docids, claim_indices_by_parent, doc_hit_by_id)

        selected = _iterative_select(
            subset, query_tokens,
            k1=k1, b=b, stopwords=stopwords, stemmer=stemmer,
            doc_floor=doc_floor, discount=discount, n_iters=n_iters,
        )

        outputs[i].hits = selected
        outputs[i].evidences = selected

    return outputs
