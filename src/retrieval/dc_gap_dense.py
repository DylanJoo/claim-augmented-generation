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
"{parent_id}#{i}"). Embeddings are encoded with --normalize, so the dot
product is already cosine similarity in [-1, 1] and needs no query-length
renormalization -- unlike the BM25 variants' _normalize, which exists
specifically to cancel out BM25 raw-score inflation from longer queries
(see dc_gap.py's _doc_to_claim_scores docstring). Reusing that per-row
min-max here would stretch already-comparable, bounded scores to fill
[0, 1] on every row, manufacturing apparent signal out of noise, so this
module skips it and works with raw cosine similarity throughout.

Caveat: this trades a lexical-overlap signal for a semantic-embedding one,
and the two don't behave the same way for novelty. BM25 gives an honest 0
for no lexical overlap; two semantically unrelated claims still often land
at a moderately positive cosine similarity, so novelty (min score) may not
spread out as much here as it does under BM25, and the resulting
bridge-novelty gap may be more compressed. Check the printed score
matrices (see _print_score_matrices) before trusting the picks.

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


def _normalize(values):
    values = np.asarray(values, dtype=np.float32)
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-9:
        return np.ones_like(values)
    return (values - lo) / (hi - lo)


def _pickle_load(path):
    """Same shard format tevatron.retriever.driver.search reads/writes: a
    (reps, lookup) tuple, reps as an array of embeddings and lookup as the
    parallel list of ids. Mirrors search_dense.py's _pickle_load."""
    with open(path, "rb") as f:
        reps, lookup = pickle.load(f)
    return np.asarray(reps), lookup


def _load_reps(passage_reps, desc, needed_ids, id_transform=None):
    """Load only the embeddings dc-gap actually needs (needed_ids) from
    shards matching passage_reps, filtering row-by-row as each shard is
    read rather than materializing the whole shard set into one dict first.

    This matters at real corpus scale: a claim-level embedding set can run
    into the hundreds of GB (e.g. neuclir1's is ~439GB across shards),
    while dc-gap only ever needs vectors for the pooled candidates of the
    run file(s) actually being re-ranked -- typically a small fraction of
    the full corpus. Unlike search_dense.py's _load_flat_from_files (which
    builds a faiss index for approximate nearest-neighbor search over an
    entire shard group), dc-gap needs exact vectors for a known id set, so
    filtering during the read and keeping a flat dict is enough -- no index
    build, and no need to hold an unfiltered shard's ids in memory past the
    loop that scans it (only the still-large `reps` array from
    np.asarray(reps) in _pickle_load is unavoidable per shard, since the
    pickle format bundles a whole shard as one object).

    id_transform, when given, maps a shard row's raw docid to the key
    tested against needed_ids (e.g. a claim's "{parent}#{i}" id stripped
    down to its parent id) without changing the key the vector is stored
    under.
    """
    files = sorted(glob.glob(passage_reps))
    if not files:
        raise FileNotFoundError(f"No passage rep shards matched: {passage_reps}")

    reps_by_id = {}
    for fpath in files:
        reps, lookup = _pickle_load(fpath)
        for vec, docid in zip(reps, lookup):
            key = id_transform(docid) if id_transform else docid
            if key in needed_ids:
                reps_by_id[docid] = vec
    logger.info("Loaded %d/%d needed %s embeddings from %d shard(s) matching %s",
                len(reps_by_id), len(needed_ids), desc, len(files), passage_reps)
    return reps_by_id


def _rows_by_parent(claim_reps_by_id):
    """parent docid -> list of that doc's claim ids, recovered from the
    claim embedding shards' own "{parent_id}#{i}" ids -- same convention
    dc_gap_global.py's _load_global_claim_index uses for its BM25 index."""
    rows_by_parent = defaultdict(list)
    for claim_docid in claim_reps_by_id:
        parent_id = claim_docid.rsplit("#", 1)[0]
        rows_by_parent[parent_id].append(claim_docid)
    return rows_by_parent


def _doc_to_claim_scores(pool_docids, doc_reps_by_id, claim_reps_by_id, rows_by_parent, dim):
    """Doc(query)-to-claim(candidate) dense scores, one column per pooled
    claim. Same (n_docs x n_claims) / doc_of_claim contract as dc_gap.py's
    _doc_to_claim_scores, but scores come from a single matrix multiply of
    precomputed embeddings instead of a BM25 query:

        score(D_i, c) = dot(doc_emb[D_i], claim_emb[c])

    Pooled docs missing from the doc-level shards contribute a zero row.
    A pooled doc with *no claims at all* in the claim-level shards gets one
    synthetic zero-vector claim instead of no claims -- mirroring dc_gap.py's
    `claims_by_doc.get(docid) or [""]` fallback, which keeps an empty-claims
    doc scoring a real (near-zero) value rather than dropping out of the
    per-doc max/min entirely. Without this, _claims_to_doc_extrema's max-of-
    nothing/min-of-nothing (-inf/+inf) for that doc's column collides with
    the -inf used elsewhere to mask already-selected docs, and argmax's
    tie-break can then re-pick an already-selected doc instead of this one.
    """
    doc_of_claim = []
    claim_vecs = []
    for doc_idx, docid in enumerate(pool_docids):
        claim_docids = rows_by_parent.get(docid) or []
        vecs = [claim_reps_by_id[cid] for cid in claim_docids if cid in claim_reps_by_id]
        if not vecs:
            vecs = [np.zeros(dim, dtype=np.float32)]
        for vec in vecs:
            claim_vecs.append(vec)
            doc_of_claim.append(doc_idx)
    doc_of_claim = np.asarray(doc_of_claim, dtype=np.int64)
    claim_matrix = (
        np.stack(claim_vecs).astype(np.float32) if claim_vecs else np.zeros((0, dim), dtype=np.float32)
    )

    n = len(pool_docids)
    missing_doc = []
    doc_matrix = np.zeros((n, dim), dtype=np.float32)
    for i, docid in enumerate(pool_docids):
        vec = doc_reps_by_id.get(docid)
        if vec is None:
            missing_doc.append(docid)
            continue
        doc_matrix[i] = vec
    if missing_doc:
        print(f"[dc_gap_dense] {len(missing_doc)} pooled doc(s) missing from doc-level shards; "
              f"leaving their score rows as zeros, e.g. {missing_doc[:3]!r}")

    doc_claim_scores = (
        doc_matrix @ claim_matrix.T if claim_matrix.shape[0] else np.zeros((n, 0), dtype=np.float32)
    )
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


def _print_score_matrices(pool_docids, bridge_matrix, novelty_matrix, topn=5):
    """Print the top-N x top-N doc-doc bridge, novelty, and gap submatrices,
    same layout as dc_gap.py's _print_score_matrices."""
    n = min(topn, len(pool_docids))
    ids = [str(d)[:12] for d in pool_docids[:n]]
    header = " " * 14 + "".join(f"{c:>8}" for c in ids)
    gap_matrix = bridge_matrix - novelty_matrix
    for name, matrix in (("bridge", bridge_matrix), ("novelty", novelty_matrix), ("gap", gap_matrix)):
        print(f"[dc_gap_dense] top-{n} x top-{n} doc(query)-claim(candidate) {name} matrix:")
        print(header)
        for i in range(n):
            row = "".join(f"{matrix[i, j]:>8.3f}" for j in range(n))
            print(f"{ids[i]:<14}{row}")


def _gap_select(hits, doc_reps_by_id, claim_reps_by_id, rows_by_parent, dim, k):
    if len(hits) <= 1:
        return hits

    pool_docids = [h.docid for h in hits]
    relevance = _normalize([h.score for h in hits])
    doc_claim_scores, doc_of_claim = _doc_to_claim_scores(
        pool_docids, doc_reps_by_id, claim_reps_by_id, rows_by_parent, dim
    )

    n = len(hits)
    claim_idx_by_doc = [np.where(doc_of_claim == j)[0] for j in range(n)]

    # bridge_matrix[i, j] = max_{c in claims(D_j)} score(D_i, c)
    # novelty_matrix[i, j] = min_{c in claims(D_j)} score(D_i, c)
    bridge_matrix = np.zeros((n, n), dtype=np.float32)
    novelty_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        bridge_matrix[i], novelty_matrix[i] = _claims_to_doc_extrema(doc_claim_scores[i], claim_idx_by_doc)
    _print_score_matrices(pool_docids, bridge_matrix, novelty_matrix, topn=5)

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
        scores = running_bridge - running_novelty
        scores[selected] = -np.inf
        pick = int(np.argmax(scores))
        picked_scores.append(float(scores[pick]))
        selected.append(pick)
        running_bridge = np.maximum(running_bridge, bridge_matrix[pick])
        running_novelty = np.minimum(running_novelty, novelty_matrix[pick])

    # running_bridge[j] >= running_novelty[j] always (per selected i,
    # bridge_matrix[i, j] >= novelty_matrix[i, j] since max >= min over the
    # same claim set; taking max_i on one side and min_i on the other only
    # widens that), so each round's gap is >= 0 regardless of the sign of
    # the underlying cosine similarities -- anchoring rank 1 at 0 and
    # walking later ranks down by their own picked_scores stays
    # non-increasing the same way it does in dc_gap.py.
    written_scores = [0.0]
    for s in picked_scores[1:]:
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
) -> List[Result]:
    logger.info("dc-gap-dense: base relevance from run file %s, pool k=%d, doc_reps=%s, claim_reps=%s",
                run_file, k, doc_reps, claim_reps)
    base_run = load_run(run_file, k=k)
    claim_corpus = load_corpus(corpus)  # only needed for display fields (title/text/statements)

    # Union of pooled docids across every topic in this run -- the only ids
    # dc-gap will ever score, and typically a tiny fraction of the full
    # embedding corpus (see _load_reps for why that distinction matters).
    needed_docids = {docid for pool in base_run.values() for docid, _ in pool}
    logger.info("dc-gap-dense: %d unique pooled docid(s) across %d topic(s) need embeddings",
                len(needed_docids), len(base_run))

    doc_reps_by_id = _load_reps(doc_reps, desc="doc-level", needed_ids=needed_docids)
    claim_reps_by_id = _load_reps(
        claim_reps, desc="claim-level", needed_ids=needed_docids,
        id_transform=lambda claim_docid: claim_docid.rsplit("#", 1)[0],
    )
    rows_by_parent = _rows_by_parent(claim_reps_by_id)
    dim = next(iter(doc_reps_by_id.values())).shape[0]

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

        n_claims = sum(len(rows_by_parent.get(docid, [])) for docid, _ in pool)
        logger.info("dc-gap-dense: qid=%s pooled %d documents, %d claims found in embedding shards",
                    qid, len(pool), n_claims)

        outputs[i].hits = hits
        outputs[i].evidences = _gap_select(
            hits, doc_reps_by_id, claim_reps_by_id, rows_by_parent, dim, k,
        )

    return outputs
