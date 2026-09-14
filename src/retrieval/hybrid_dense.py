"""
Hybrid retrieval, dense embedding variant: claim-level dense retrieval as
the base signal, combined with doc-level dense retrieval, fused at the
parent-doc level.

Reuses retrieval.search_dense.run for both legs (shard loading/streaming,
faiss search, and claim -> parent-doc fusion within a leg) and
retrieval.hybrid's _combine for the cross-leg alpha blend, so the
combination step (min-max normalize each leg, then
alpha * claim + (1 - alpha) * doc) is identical to the BM25 hybrid --
only where each leg's scores come from differs (dense dot-product search
against tevatron embedding shards instead of BM25 over a bm25s index).
"""
import copy
import logging
from collections import defaultdict
from typing import List

from retrieval import search_dense
from utils import Result, Hit

logger = logging.getLogger(__name__)

def _normalize(hits):
    """Min-max normalize hit scores to [0, 1] (no-op if <2 hits or all-equal)."""
    if len(hits) < 2:
        return hits
    scores = [h.score for h in hits]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return hits
    return [Hit(docid=h.docid, score=(h.score - lo) / (hi - lo), rank=h.rank, content_dict=h.content_dict)
            for h in hits]

def _combine(claim_hits, aux_hits, alpha=0.5, normalize=True):
    """Combine two parent-doc-level ranked lists into one, keyed by docid."""
    if normalize:
        claim_hits = _normalize(claim_hits)
        aux_hits = _normalize(aux_hits)

    contrib = defaultdict(list)  # docid -> [(weighted_score, rank), ...]
    content = {}
    for h in claim_hits:
        contrib[h.docid].append((alpha * h.score, h.rank))
        content[h.docid] = h.content_dict
    for h in aux_hits:
        contrib[h.docid].append(((1 - alpha) * h.score, h.rank))
        content.setdefault(h.docid, h.content_dict)

    fused = {}
    for docid, items in contrib.items():
        fused[docid] = sum(score for score, _ in items)
    fused = dict(sorted(fused.items(), key=lambda x: x[1], reverse=True))

    return [
        Hit(docid=docid, score=score, rank=rank, content_dict=content[docid])
        for rank, (docid, score) in enumerate(fused.items(), start=1)
    ]

def run(
    inputs: List[Result],
    query_reps: str,
    claim_shard_groups: List[str],
    doc_passage_reps: str,
    corpus_paths: List[str] = None,
    k_claim: int = 1000,
    k_doc: int = 1000,
    claim_fusion: str = "sum",
    alpha: float = 0.5,
    normalize: bool = True,
) -> List[Result]:
    """Claim-level dense retrieval (base signal) combined with doc-level
    dense retrieval (aux signal).

    `alpha` weights the claim-level leg (1 - alpha weights the doc leg);
    with `normalize=True` both legs' scores are min-max scaled to [0, 1]
    first so the weighting is meaningful across the two legs' differently
    distributed cosine-similarity scores (claim-level scores are
    fused/summed over a variable number of claims per doc, doc-level
    scores are a single dot product).
    """
    logger.info("Hybrid-dense leg 1/2: claim-level dense retrieval (%d shard group(s))", len(claim_shard_groups))
    claim_results = search_dense.run(
        inputs,
        query_reps=query_reps,
        passage_reps_groups=claim_shard_groups,
        corpus_paths=corpus_paths,
        k=k_claim,
        fusion=claim_fusion,
    )
    logger.info("Hybrid-dense leg 2/2: doc-level dense retrieval (%s)", doc_passage_reps)
    doc_results = search_dense.run(
        inputs,
        query_reps=query_reps,
        passage_reps=doc_passage_reps,
        corpus_paths=corpus_paths,
        k=k_doc,
    )

    outputs = copy.deepcopy(inputs)
    for i, (claim_res, doc_res) in enumerate(zip(claim_results, doc_results)):
        outputs[i].hits = claim_res.hits + doc_res.hits
        outputs[i].evidences = _combine(
            claim_res.evidences, doc_res.evidences,
            alpha=alpha, normalize=normalize,
        )

    return outputs
