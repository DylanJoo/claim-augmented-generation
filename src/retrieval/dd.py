"""
MMR (Maximal Marginal Relevance) re-ranking on top of a doc-level run file.

The base relevance signal is read directly from a pre-computed TREC run file
(doc-level, one score per docid) instead of running BM25 retrieval here --
only the document corpus is needed to fetch text for the diversity signal.
This module greedily re-orders that pool by MMR:

    argmax_d  lambda * relevance(d)  -  (1 - lambda) * max_sim(d, selected)

`relevance(d)` is the min-max normalized score taken from the run file.
`sim(d, d')` is a lexical BM25 self-similarity computed by building a small
in-memory bm25s index over just the pool's texts and querying it with each
pooled doc's own tokens -- the same trick used in dev/local_rescore.py's
near-duplicate detection (_dedup_similar_docs). Keeping the diversity signal
BM25-based avoids adding a dense-embedding dependency and stays consistent
with the rest of this codebase.
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


def _similarity_matrix(texts, stopwords, stemmer, k1, b):
    """Pairwise lexical similarity among pooled docs, row-normalized to [0, 1].
    Each retrieved documents are considered as query and calculate scores with every other.
    """
    n = len(texts)
    tokens = bm25s.tokenize(texts, stopwords=stopwords, stemmer=stemmer, show_progress=False)
    local = bm25s.BM25(k1=k1, b=b)
    local.index(tokens, show_progress=False)

    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sim[i] = local.get_scores(tokens.ids[i])

    row_max = sim.max(axis=1, keepdims=True)
    row_max[row_max < 1e-9] = 1.0
    return sim / row_max


def _mmr_select(hits, k, lambda_mult, stopwords, stemmer, k1, b):
    if len(hits) <= 1:
        return hits

    texts = [ ( h.content_dict.get("title") + " " + h.content_dict.get("text") ) or "" for h in hits]
    relevance = _normalize([h.score for h in hits])
    sim = _similarity_matrix(texts, stopwords, stemmer, k1, b)

    n = len(hits)
    n_select = min(k, n)
    selected = []
    picked_scores = []
    max_sim_to_selected = np.zeros(n, dtype=np.float32)

    for _ in range(n_select):
        mmr_scores = lambda_mult * relevance - (1 - lambda_mult) * max_sim_to_selected
        mmr_scores[selected] = -np.inf
        pick = int(np.argmax(mmr_scores))
        picked_scores.append(float(mmr_scores[pick]))
        selected.append(pick)
        max_sim_to_selected = np.maximum(max_sim_to_selected, sim[pick])

    # picked_scores is non-increasing by construction (max_sim_to_selected only grows,
    # so each round's best achievable score can only shrink) -- this keeps the written
    # score monotonic with rank, which downstream TREC eval tools sort by, not rank.
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
    stopwords: str = "en",
    stemmer_name: str = None,
    local_k1: float = 1.2,
    local_b: float = 0.5,
) -> List[Result]:
    stemmer = None
    if stemmer_name == "snowball":
        import Stemmer
        stemmer = Stemmer.Stemmer("english")

    logger.info("MMR: base relevance from run file %s, pool k=%d, lambda=%.2f", run_file, k, lambda_mult)
    base_run = load_run(run_file, k=k)
    doc_corpus = load_corpus(corpus)

    outputs = copy.deepcopy(inputs)
    for i, inp in enumerate(inputs):
        qid = str(inp.topic["qid"])
        hits = [
            Hit(
                docid=docid,
                score=score,
                rank=rank,
                content_dict={
                    "text": doc_corpus.get(docid, {}).get("text", ""),
                    "title": doc_corpus.get(docid, {}).get("title"),
                },
            )
            for rank, (docid, score) in enumerate(base_run.get(qid, []), start=1)
        ]
        outputs[i].hits = hits
        outputs[i].evidences = _mmr_select(
            hits, 
            k, 
            lambda_mult,
            stopwords=stopwords, 
            stemmer=stemmer, 
            k1=local_k1, 
            b=local_b,
        )

    return outputs
