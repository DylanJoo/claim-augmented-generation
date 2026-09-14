"""
dc-gap (Doc-Claim Gap) re-ranking on top of a doc-level run file -- dense
embedding variant.

Identical selection logic to dc_gap.py (see that module's docstring for the
full bridge/novelty/gap explanation). The only difference is where
score(D_i, c) comes from:

    score(D_i, c) = dot(embed(D_i), embed(c))

Both sides are read from pre-computed tevatron embedding shards (see
scripts/dense-index/) instead of being scored by BM25 at rerank time: D_i's
side from the doc-level shards (one vector per whole document), c's side
from the claim-level shards (one vector per claim, docid
"{parent_id}#{i}"). 

Caveat: this trades a lexical-overlap signal for a semantic-embedding one,
and the two don't behave the same way for novelty. BM25 gives an honest 0
for no lexical overlap; two semantically unrelated claims still often land
at a moderately positive cosine similarity, so novelty (min score) may not
spread out as much here as it does under BM25, and the resulting
bridge-novelty gap may be more compressed.

Only a "global" style lookup is implemented -- there is no local/global
distinction the way dc_gap.py vs dc_gap_global.py have one, because dense
embeddings are frozen at encode time, not corpus-statistics-dependent the
way BM25's IDF and average length are. There is no notion of a "local pool"
embedding to rebuild per topic the way dc_gap.py rebuilds a local BM25
index.

include_query (dc_gap.py's option to prefix D_i's text with the topic query
before scoring) is intentionally not supported here: unlike the BM25
variants, where that is a free string-concat-then-tokenize operation, doing
it with dense embeddings would require running the actual encoder model at
rerank time instead of a plain vector lookup -- a much heavier runtime
dependency this module does not take on.
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

def _load_reps(reps_path, needed_ids, id_transform=None):
    files = sorted(glob.glob(reps_path))
    if not files:
        raise FileNotFoundError(f"No passage rep shards matched: {reps_path}")

    reps_by_id = {}
    for i, fpath in enumerate(files, 1):
        reps, lookup = _pickle_load(fpath)
        for vec, repid in zip(reps, lookup):
            key = id_transform(repid) if id_transform else repid
            if key in needed_ids:
                reps_by_id[repid] = vec.copy()
        logger.info("dc-gap-dense: [%d/%d] loaded shard %s, %d/%d needed id(s) matched so far",
                    i, len(files), fpath, len(reps_by_id), len(needed_ids))
    return reps_by_id

def _load_query_reps(query_reps_path):
    reps, lookup = _pickle_load(query_reps_path)
    return reps, {str(qid): i for i, qid in enumerate(lookup)}

def _rows_by_parent(claim_reps_by_id):
    rows_by_parent = defaultdict(list)
    for claim_docid in claim_reps_by_id:
        parent_id = claim_docid.rsplit("#", 1)[0]
        rows_by_parent[parent_id].append(claim_docid)
    return rows_by_parent


def _doc_to_claim_scores(
    list_docids, 
    list_relevance,
    doc_reps_by_id, 
    claim_reps_by_id, 
    rows_by_parent, 
    dim,
    query_vec=None, 
    claim_filter="none",
    claims_per_doc=5, 
    claim_sim_threshold=0.0,
    scale_topn_by_relevance=False, 
):

    doc_of_claim = []
    claim_vecs = []

    if claim_filter != "none":
        values = np.asarray(list_relevance, dtype=np.float32)
        lo, hi = values.min(), values.max()
        list_relevance = np.ones_like(values) if hi - lo < 1e-9 else (values - lo) / (hi - lo)

    for doc_idx, (docid, relevance) in enumerate(zip(list_docids, list_relevance)):

        doc_relevance = float(relevance)
        claim_ids = rows_by_parent.get(docid) or []
        cids = [id for id in claim_ids if id in claim_reps_by_id]

        # 1a. Use all the claim ids
        if claim_filter == "none":
            claim_ids = cids

        # 1b. Filter off-topic claims based on query-claim similarity
        else:
            sims = np.asarray(
                [float(np.dot(query_vec, claim_reps_by_id[cid])) for cid in cids], 
                dtype=np.float32
            )
            orders = np.argsort(sims)[::-1]  # descending by query-claim similarity

            if claim_filter == "topn":
                n = claims_per_doc
                if scale_topn_by_relevance:
                    n = max(1, round(claims_per_doc * doc_relevance))
                keep = orders[:n]
            elif claim_filter == "threshold":
                keep = orders[sims[orders] >= claim_sim_threshold]
                if len(keep) == 0:
                    keep = orders[:1]  # never zero out a doc entirely -- keep its single best claim
            # if claim_filter == "under-relevance": # NOTE: maybe s(q,d) > or < s(q,c) can be a filter?
            else:
                raise ValueError(f"Unknown claim_filter: {claim_filter!r}")

            claim_ids = [cids[i] for i in keep]

        vecs = [claim_reps_by_id[id] for id in claim_ids]
        if not vecs:
            vecs = [np.zeros(dim, dtype=np.float32)]
        for vec in vecs:
            claim_vecs.append(vec)
            doc_of_claim.append(doc_idx)

    doc_of_claim = np.asarray(doc_of_claim, dtype=np.int64) # the array for claim-to-doc mapping
    claim_matrix = (
        np.stack(claim_vecs).astype(np.float32) if claim_vecs else np.zeros((0, dim), dtype=np.float32)
    )

    # 2 collect document vector
    n = len(list_docids)
    missing_doc = []
    doc_matrix = np.zeros((n, dim), dtype=np.float32)
    for i, docid in enumerate(list_docids):
        vec = doc_reps_by_id.get(docid)
        if vec is None:
            missing_doc.append(docid)
            continue
        doc_matrix[i] = vec
    if missing_doc:
        print(f"[dc_gap_dense] {len(missing_doc)} pooled doc(s) missing from doc-level shards; "
              f"leaving their score rows as zeros, e.g. {missing_doc[:3]!r}")

    # 3 calculate full document-claim scores
    doc_claim_scores = (
        doc_matrix @ claim_matrix.T if claim_matrix.shape[0] else np.zeros((n, 0), dtype=np.float32)
    )
    return doc_claim_scores, doc_of_claim


def _gap_select(
    hits, 
    doc_reps_by_id, 
    claim_reps_by_id, 
    rows_by_parent, 
    dim, 
    k,
    query_vec=None, 
    claim_filter="none",
    claims_per_doc=5, 
    scale_topn_by_relevance=False, 
    claim_sim_threshold=0.0
):
    if len(hits) <= 1:
        return hits

    # Obtain (1) a big document-to-claim matrix (with pre-filter if applicable)
    # and also (2) a claim-to-doc mapping
    doc_claim_scores, doc_of_claim = _doc_to_claim_scores(
        list_docids=[h.docid for h in hits],
        list_relevance=[h.score for h in hits],
        doc_reps_by_id=doc_reps_by_id, 
        claim_reps_by_id=claim_reps_by_id,
        rows_by_parent=rows_by_parent, 
        dim=dim,
        query_vec=query_vec, 
        claim_filter=claim_filter,
        claims_per_doc=claims_per_doc, 
        scale_topn_by_relevance=scale_topn_by_relevance,
        claim_sim_threshold=claim_sim_threshold,
    )

    n = len(hits)
    # NOTE: this is a list of list. Each list represents where the corresponding claims are.
    claim_idx_by_doc = [np.where(doc_of_claim == j)[0] for j in range(n)]

    # Obtain (1) the max doc-to-claim matrix, and (2) the min doc-to-claim matrix
    bridge_matrix = np.zeros((n, n), dtype=np.float32)
    novelty_matrix = np.zeros((n, n), dtype=np.float32)

    ## row-i represents doc
    for i in range(n):
        doc_max = np.empty(len(claim_idx_by_doc), dtype=np.float32)
        doc_min = np.empty(len(claim_idx_by_doc), dtype=np.float32)
        ## col-j represents claims in that i-th doc
        for j, idxs in enumerate(claim_idx_by_doc):
            doc_max[j] = doc_claim_scores[i][idxs].max() if len(idxs) else -np.inf
            doc_min[j] = doc_claim_scores[i][idxs].min() if len(idxs) else np.inf

        bridge_matrix[i], novelty_matrix[i] = doc_max, doc_min # bridge mean high doc-to-claim connect

    # Greedy selection
    n_select = min(k, n)
    selected = []
    selected_scores = []
    running_bridge = np.full(n, -np.inf, dtype=np.float32)
    running_novelty = np.full(n, np.inf, dtype=np.float32)

    ## The first one is pure relevance-based
    # TODO: maybe we can have another selection strategy to get the first one
    first = int(np.argmax([h.score for h in hits]))
    selected.append(first)
    selected_scores.append(float([h.score for h in hits][first]))
    running_bridge = np.maximum(running_bridge, bridge_matrix[first])
    running_novelty = np.minimum(running_novelty, novelty_matrix[first])

    ## The second to k Strategy: (a) largest-gap-first
    for _ in range(1, n_select):
        scores = running_bridge - running_novelty
        scores[selected] = -np.inf
        pick = int(np.argmax(scores))
        selected_scores.append(float(scores[pick]))
        selected.append(pick)
        running_bridge = np.maximum(running_bridge, bridge_matrix[pick])
        running_novelty = np.minimum(running_novelty, novelty_matrix[pick])

    # Finally assign scores
    written_scores = [0.0]
    for s in selected_scores[1:]:
        written_scores.append(written_scores[-1] - s)

    return [
        Hit(docid=hits[idx].docid, score=written_scores[rank - 1], rank=rank, content_dict=hits[idx].content_dict)
        for rank, idx in enumerate(selected, start=1)
    ]


def run(
    inputs: List[Result],
    run_file: str,
    corpus: List[str],
    doc_reps: str,
    claim_reps: str,
    k: int = 1000,
    query_reps: str = None,
    claim_filter: str = "none",
    claims_per_doc: int = 5,
    scale_topn_by_relevance: bool = False,
    claim_sim_threshold: float = 0.0,
) -> List[Result]:
    """query_reps/claim_filter: optional query-claim similarity filter on
    each pooled doc's claims before bridge/novelty scoring (see
    _select_claims_for_doc). claim_filter="none" (default) keeps every
    claim found in the shards, matching pre-existing behavior; query_reps
    is required for "topn"/"threshold"."""
    if claim_filter != "none" and not query_reps:
        raise ValueError(f"claim_filter={claim_filter!r} requires query_reps")

    logger.info("dc-gap-dense: base relevance from run file %s, pool k=%d, doc_reps=%s, claim_reps=%s",
                run_file, k, doc_reps, claim_reps)
    base_run = load_run(run_file, k=k)
    claim_corpus = load_corpus(corpus)  # only needed for display fields (title/text/statements)

    # Union of pooled docids across every topic in this run -- the only ids
    # dc-gap will ever score, and typically a tiny fraction of the full
    needed_docids = {docid for pool in base_run.values() for docid, _ in pool}
    logger.info("dc-gap-dense: %d unique pooled docid(s) across %d topic(s) need embeddings",
                len(needed_docids), len(base_run))

    # Only load document vectors that are in the base run's docid
    doc_reps_by_id = _load_reps(doc_reps, needed_docids)
    dim = next(iter(doc_reps_by_id.values())).shape[0]

    # Only load claim vectors that are in the base run's docid
    claim_reps_by_id = _load_reps(
        claim_reps, needed_docids,
        id_transform=lambda claim_docid: claim_docid.rsplit("#", 1)[0],
    )

    ## collect the claim list by docid
    rows_by_parent = _rows_by_parent(claim_reps_by_id) # chance the format of {docid: [claim id...]

    q_reps, q_pos = (None, {})
    if query_reps:
        q_reps, q_pos = _load_query_reps(query_reps)
        logger.info("dc-gap-dense: loaded %d query embedding(s) from %s for claim_filter=%r",
                    len(q_pos), query_reps, claim_filter)

    outputs = copy.deepcopy(inputs)
    for i, inp in enumerate(inputs):
        qid = str(inp.topic["qid"])
        pool = base_run.get(qid, [])

        query_vec = None
        if claim_filter != "none":
            query_vec = q_reps[q_pos[qid]]

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

        n_claims = sum(len(rows_by_parent.get(docid, [])) for docid, _ in pool)
        logger.info("dc-gap-dense: qid=%s pooled %d documents, %d claims found in embedding shards",
                    qid, len(pool), n_claims)

        outputs[i].hits = hits
        outputs[i].evidences = _gap_select(
            hits, doc_reps_by_id, claim_reps_by_id, rows_by_parent, dim, k,
            query_vec=query_vec, 
            claim_filter=claim_filter, 
            claims_per_doc=claims_per_doc,
            scale_topn_by_relevance=scale_topn_by_relevance, 
            claim_sim_threshold=claim_sim_threshold,
        )

    return outputs
