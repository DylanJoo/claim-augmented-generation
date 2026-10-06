"""
Hybrid retrieval: claim-level retrieval as the base signal, combined with a
second, coarser signal -- either doc-level or concat-claims-level retrieval
-- fused at the parent-doc level.

Reuses retrieval.search.run for both legs (index loading, tokenization,
per-query BM25 retrieval, and claim -> parent-doc fusion within a leg); this
module only combines the two resulting doc-level evidence lists.
"""
import copy
import logging
from collections import defaultdict
from typing import List

from retrieval import search
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
    claim_index: str,
    aux_index: str,
    k_claim: int = 1000,
    k_aux: int = 1000,
    claim_fusion: str = "sum",
    aux_fusion: str = "sum",
    alpha: float = 0.5,
    normalize: bool = True,
    stopwords: str = "en",
    stemmer: str = None,
) -> List[Result]:
    """Claim-level retrieval (base signal) combined with a coarser aux signal
    (doc-level or concat-claims-level retrieval over `aux_index`).

    `alpha` weights the claim-level leg (1 - alpha weights the aux leg); with
    `normalize=True` both legs' scores are min-max scaled to [0, 1] first so
    the weighting is meaningful across differently-scaled BM25 indices.
    """
    logger.info("Hybrid leg 1/2: claim-level retrieval (%s)", claim_index)
    claim_results = search.run(
            inputs, 
            index=claim_index, 
            k=k_claim,
            stopwords=stopwords, 
            stemmer=stemmer, 
            fusion=claim_fusion
    )
    logger.info("Hybrid leg 2/2: auxiliary retrieval (%s)", aux_index)
    aux_results = search.run(
            inputs, 
            index=aux_index, 
            k=k_aux,
            stopwords=stopwords, 
            stemmer=stemmer
    )

    outputs = copy.deepcopy(inputs)
    for i, (claim_res, aux_res) in enumerate(zip(claim_results, aux_results)):
        outputs[i].hits = claim_res.hits + aux_res.hits
        outputs[i].evidences = _combine(
            claim_res.evidences, aux_res.evidences,
            alpha=alpha, normalize=normalize,
        )

    return outputs
