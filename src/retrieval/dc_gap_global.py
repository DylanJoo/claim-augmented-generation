"""
dc-gap (Doc-Claim Gap) re-ranking on top of a doc-level run file -- global
claim-index variant.

Identical selection logic to dc_gap.py (see that module's docstring for the
full bridge/novelty/gap explanation). The only difference is where
score(D_i, c) = BM25(text(D_i), c) comes from:

  dc_gap.py (local)
    Builds a fresh BM25 index per topic over *only* the claims of the
    pooled documents. IDF / avg claim length are local to that ~k-document
    pool, and rebuilt from scratch every topic.

  dc_gap_global.py (this module)
    Loads a single pre-built BM25 index covering *every* claim in the whole
    corpus (built via indexing.py --claim-level, e.g.
    scripts/index/neuclir1-claims.sh), once, up front. D_i's full text is
    queried against that global index, and the result is then restricted to
    the pooled candidates' claims. IDF / avg claim length reflect the full
    collection instead of the pool.

Caveat: the two are only a fair "local pool stats vs. global corpus stats"
comparison if the query-side tokenization here (stopwords/stemmer) matches
whatever the global index was actually built with. bm25s silently drops
query tokens that aren't in the index's vocabulary, so e.g. querying with a
snowball stemmer against an index built without one will look like
near-zero overlap everywhere -- a tokenization mismatch, not a genuine
global-vs-local-statistics difference. Check how the target index was built
(its indexing script's --stopwords/--stemmer flags) before comparing runs.
"""
import copy
import json
import logging
import os
from collections import defaultdict
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


def _load_global_claim_index(index_dir):
    """Load a prebuilt claim-level bm25s index (see indexing.py --claim-level).

    Returns (retriever, rows_by_parent) where rows_by_parent maps a parent
    document id to the list of row indices -- in the retriever's internal
    corpus order -- of that document's claims ("{parent_id}#{i}" docids).
    """
    retriever = bm25s.BM25.load(index_dir, load_corpus=False)
    docids_path = os.path.join(index_dir, "docids.json")
    with open(docids_path, encoding="utf-8") as f:
        docids = json.load(f)

    rows_by_parent = defaultdict(list)
    for row, claim_docid in enumerate(docids):
        parent_id = claim_docid.rsplit("#", 1)[0]
        rows_by_parent[parent_id].append(row)

    logger.info("Loaded global claim index with %d claims (%d parent docs) from %s",
                len(docids), len(rows_by_parent), index_dir)
    return retriever, rows_by_parent


def _doc_to_claim_scores(pool_docids, fulltext_by_doc, retriever, rows_by_parent, stopwords, stemmer):
    """Doc(query)-to-claim(candidate) BM25 scores, one column per pooled claim.

    Same (n_docs x n_claims) / doc_of_claim contract as dc_gap.py's
    _doc_to_claim_scores, but scores come from the global index: each pooled
    doc's full text is scored against *every* claim in the whole corpus via
    retriever.get_scores, then the columns are restricted to the pooled
    candidates' own claims. Pooled docs with no claims recorded in the
    global index contribute zero columns (handled downstream the same way
    dc_gap.py's local index would treat an empty-claims doc).
    """
    n = len(pool_docids)
    doc_of_claim = []
    global_row_of_claim = []
    for doc_idx, docid in enumerate(pool_docids):
        rows = rows_by_parent.get(docid) or []
        for row in rows:
            global_row_of_claim.append(row)
            doc_of_claim.append(doc_idx)
    doc_of_claim = np.asarray(doc_of_claim, dtype=np.int64)
    global_row_of_claim = np.asarray(global_row_of_claim, dtype=np.int64)

    query_texts = [fulltext_by_doc.get(docid) or "" for docid in pool_docids]
    query_tokens = bm25s.tokenize(
        query_texts, stopwords=stopwords, stemmer=stemmer, return_ids=False, show_progress=False
    )

    n_claims = len(global_row_of_claim)
    doc_claim_scores = np.zeros((n, n_claims), dtype=np.float32)
    for i in range(n):
        toks = query_tokens[i]
        if len(toks) == 0:
            print(f"[dc_gap_global] empty token list for doc {i} (docid={pool_docids[i]!r}); leaving score row as zeros")
            continue
        full_scores = retriever.get_scores(toks)  # scored against the ENTIRE global claim corpus
        doc_claim_scores[i] = full_scores[global_row_of_claim]

    return doc_claim_scores, doc_of_claim


def _claims_to_doc_extrema(claim_values, claim_idx_by_doc):
    """Aggregate a per-claim value array to one (max, min) pair per candidate
    doc, over that doc's own claims only. See dc_gap.py for why max/min must
    each be applied consistently end-to-end rather than mixed."""
    doc_max = np.empty(len(claim_idx_by_doc), dtype=np.float32)
    doc_min = np.empty(len(claim_idx_by_doc), dtype=np.float32)
    for j, idxs in enumerate(claim_idx_by_doc):
        doc_max[j] = claim_values[idxs].max() if len(idxs) else -np.inf
        doc_min[j] = claim_values[idxs].min() if len(idxs) else np.inf
    return doc_max, doc_min


def _print_bridge_matrix(pool_docids, bridge_matrix, topn=10):
    """Print the top-N x top-N doc-doc bridge submatrix (hits are already
    rank-ordered by base relevance, so the first `topn` are the top-N)."""
    n = min(topn, len(pool_docids))
    ids = [str(d)[:12] for d in pool_docids[:n]]
    header = " " * 14 + "".join(f"{c:>8}" for c in ids)
    print(f"[dc_gap_global] top-{n} x top-{n} doc(query)-claim(candidate) bridge matrix:")
    print(header)
    for i in range(n):
        row = "".join(f"{bridge_matrix[i, j]:>8.3f}" for j in range(n))
        print(f"{ids[i]:<14}{row}")


def _gap_select(hits, fulltext_by_doc, retriever, rows_by_parent, k, stopwords, stemmer):
    if len(hits) <= 1:
        return hits

    pool_docids = [h.docid for h in hits]
    relevance = _normalize([h.score for h in hits])
    doc_claim_scores, doc_of_claim = _doc_to_claim_scores(
        pool_docids, fulltext_by_doc, retriever, rows_by_parent, stopwords, stemmer
    )

    n = len(hits)
    claim_idx_by_doc = [np.where(doc_of_claim == j)[0] for j in range(n)]

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

    first = int(np.argmax(relevance))
    selected.append(first)
    picked_scores.append(float(relevance[first]))
    running_bridge = np.maximum(running_bridge, bridge_matrix[first])
    running_novelty = np.minimum(running_novelty, novelty_matrix[first])

    for _ in range(1, n_select):
        if len(selected) == 1:
            scores = running_bridge.copy()
        else:
            scores = running_bridge - running_novelty
        scores[selected] = -np.inf
        pick = int(np.argmax(scores))
        picked_scores.append(float(scores[pick]))
        selected.append(pick)
        running_bridge = np.maximum(running_bridge, bridge_matrix[pick])
        running_novelty = np.minimum(running_novelty, novelty_matrix[pick])

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
    claim_index: str,
    k: int = 1000,
    stopwords: str = "en",
    stemmer_name: str = None,
) -> List[Result]:
    stemmer = None
    if stemmer_name == "snowball":
        import Stemmer
        stemmer = Stemmer.Stemmer("english")

    logger.info("dc-gap-global: base relevance from run file %s, pool k=%d, claim index=%s",
                run_file, k, claim_index)
    base_run = load_run(run_file, k=k)
    corpus_meta = load_corpus(corpus)  # only needed for doc full text + display fields
    retriever, rows_by_parent = _load_global_claim_index(claim_index)

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
                    "doc-text": corpus_meta.get(docid, {}).get("text"),
                    "claim-text": corpus_meta.get(docid, {}).get("statements"),
                    "title": corpus_meta.get(docid, {}).get("title"),
                },
            )
            for rank, (docid, score) in enumerate(pool, start=1)
        ]
        doc_by_doc = {docid: corpus_meta.get(docid, {}).get("text") or "" for docid, _ in pool}

        n_claims_in_pool = sum(len(rows_by_parent.get(docid, [])) for docid, _ in pool)
        logger.info("dc-gap-global: qid=%s pooled %d documents, %d claims found in global index",
                    qid, len(pool), n_claims_in_pool)

        outputs[i].hits = hits
        outputs[i].evidences = _gap_select(
            hits,
            doc_by_doc,
            retriever,
            rows_by_parent,
            k,
            stopwords=stopwords,
            stemmer=stemmer,
        )

    return outputs
