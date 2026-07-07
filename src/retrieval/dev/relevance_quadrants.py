"""
Classify retrieved documents by (doc-relevance, claim-relevance).

Given a topic's doc-level retrieval hits and their expanded claims (the same
`doc_hits` / `subset` produced by local_rescore.py's Stage 1 / Stage 1.6),
buckets each document into one of four cases:

  relevant_both : d_score above doc_threshold AND has >=1 claim above
                  claim_threshold -- safe, well-supported documents.
  claims_only   : NOT doc-relevant but has a standout claim -- the "hidden
                  useful claim in a seemingly similar/unremarkable document"
                  case local_rescore.py's iterative selection targets. The
                  size of this bucket is a direct read on whether local
                  rescoring is earning its keep.
  doc_only      : doc-relevant but no claim stands out -- ambiguous: either
                  the doc is only tangentially on-topic, or claim
                  decontextualization missed the part that actually matters
                  (a possible extraction-quality signal, not just relevance).
  neither       : discard, not relevant by either signal.

"Claims_relevant" uses MAX claim score per doc, not sum/count -- the question
here is binary ("does this doc have at least one standout claim"), not "how
many" (that's a separate multiplicity question, handled elsewhere).

This is a read-only diagnostic layer: it does not change local_rescore.py's
selection behavior, just reports where documents land.
"""
import logging
from collections import defaultdict
from typing import List

import bm25s
import numpy as np

logger = logging.getLogger(__name__)


def _raw_claim_scores(subset, query_tokens, k1, b, stopwords, stemmer):
    """Score every expanded claim against the query with a fresh local index
    (no term-discounting, no doc-relevance gating -- just the plain BM25
    score each claim would get on its own)."""
    local_texts = [h["text"] for h in subset]
    local_tokens = bm25s.tokenize(local_texts, stopwords=stopwords, stemmer=stemmer, show_progress=False)
    local_retriever = bm25s.BM25(k1=k1, b=b)
    local_retriever.index(local_tokens, show_progress=False)

    results, scores = local_retriever.retrieve(query_tokens, k=len(local_texts), show_progress=False)
    raw = np.zeros(len(local_texts), dtype=np.float32)
    for pos, score in zip(results[0], scores[0]):
        idx = int(pos["id"] if isinstance(pos, dict) else pos)
        raw[idx] = float(score)
    return raw


def classify(
    doc_hits: List[dict],
    subset: List[dict],
    query_tokens,
    doc_threshold: float,
    claim_threshold: float,
    k1: float = 1.2,
    b: float = 0.5,
    stopwords: str = "en",
    stemmer=None,
):
    """Bucket `doc_hits` by (doc-relevance, claim-relevance).

    `doc_hits` and `subset` are the outputs of local_rescore.py's
    `_retrieve_docs` and `_fetch_claim_texts` (or `run`'s internals) for a
    single topic. Both thresholds are raw BM25 scores -- tune empirically per
    corpus, same caveat as local_rescore.py's `dedup_threshold`.

    Returns (buckets, details):
      buckets: dict[str, list[str]] -- docid lists per bucket name.
      details: dict[str, dict] -- per-docid d_score/max_claim_score/bucket.
    """
    raw_scores = _raw_claim_scores(subset, query_tokens, k1, b, stopwords, stemmer)

    max_claim_score_by_parent = defaultdict(float)
    for h, score in zip(subset, raw_scores):
        if score > max_claim_score_by_parent[h["parent"]]:
            max_claim_score_by_parent[h["parent"]] = float(score)

    buckets = defaultdict(list)
    details = {}
    for h in doc_hits:
        docid = h["docid"]
        max_claim_score = max_claim_score_by_parent.get(docid, 0.0)
        doc_rel = h["d_score"] > doc_threshold
        claim_rel = max_claim_score > claim_threshold

        if doc_rel and claim_rel:
            bucket = "relevant_both"
        elif claim_rel and not doc_rel:
            bucket = "claims_only"
        elif doc_rel and not claim_rel:
            bucket = "doc_only"
        else:
            bucket = "neither"

        buckets[bucket].append(docid)
        details[docid] = {
            "d_score": h["d_score"],
            "max_claim_score": max_claim_score,
            "bucket": bucket,
        }

    return dict(buckets), details
