import copy
import glob
import logging
import pickle
from typing import List

import numpy as np

from utils import Result, Hit, load_run, load_corpus

logger = logging.getLogger(__name__)


def _pickle_load(path):
    with open(path, "rb") as f:
        reps, lookup = pickle.load(f)
    return np.asarray(reps), lookup

def _load_reps(doc_reps_path, needed_docids):
    files = sorted(glob.glob(doc_reps_path))
    if not files:
        raise FileNotFoundError(f"No passage rep shards matched: {doc_reps_path}")

    reps_by_id = {}
    for i, fpath in enumerate(files, 1):
        reps, lookup = _pickle_load(fpath)
        for vec, repid in zip(reps, lookup):
            if repid in needed_docids:
                reps_by_id[repid] = vec.copy()
        del reps, lookup
        logger.info("dd-dense: [%d/%d] loaded shard %s, %d/%d needed id(s) matched so far",
                    i, len(files), fpath, len(reps_by_id), len(needed_docids))
    return reps_by_id

def _doc_matrix(list_docids, doc_reps_by_id, dim):
    n = len(list_docids)
    missing = []
    doc_matrix = np.zeros((n, dim), dtype=np.float32)
    for i, docid in enumerate(list_docids):
        vec = doc_reps_by_id.get(docid)
        if vec is None:
            missing.append(docid)
            continue
        doc_matrix[i] = vec
    if missing:
        print(f"[dd_dense] {len(missing)} pooled doc(s) missing from doc-level shards; "
              f"leaving their score rows as zeros, e.g. {missing[:3]!r}")

    return doc_matrix @ doc_matrix.T


def _mmr_select(
    hits, 
    doc_reps_by_id, 
    dim, 
    k,
    lambda_mult
):
    if len(hits) <= 1:
        return hits

    relevance = np.asarray([h.score for h in hits], dtype=np.float32)
    sim = _doc_matrix(
        list_docids=[h.docid for h in hits],
        doc_reps_by_id=doc_reps_by_id, 
        dim=dim
    )

    n = len(hits)
    n_select = min(k, n)
    selected = []
    selected_scores = []
    max_sim_to_selected = np.zeros(n, dtype=np.float32)

    for _ in range(n_select):
        scores = lambda_mult * relevance - (1 - lambda_mult) * max_sim_to_selected
        scores[selected] = -np.inf
        pick = int(np.argmax(scores))
        selected_scores.append(float(scores[pick]))
        selected.append(pick)
        max_sim_to_selected = np.maximum(max_sim_to_selected, sim[pick])

    # selected_scores is non-increasing by construction (max_sim_to_selected only grows,
    # so each round's best achievable score can only shrink) -- this keeps the written
    # score monotonic with rank, which downstream TREC eval tools sort by, not rank.
    return [
        Hit(docid=hits[idx].docid, score=selected_scores[rank - 1], rank=rank, content_dict=hits[idx].content_dict)
        for rank, idx in enumerate(selected, start=1)
    ]


def run(
    inputs: List[Result],
    run_file: str,
    corpus: List[str],
    doc_reps: str,
    k: int = 1000,
    lambda_mult: float = 0.9,
    mode: str = "subtract", # mmr always does subtract
    agg: str = None, # No aggregation needed
) -> List[Result]:
    logger.info("dd-dense: base relevance from run file %s, pool k=%d, lambda=%.2f, doc_reps=%s",
                run_file, k, lambda_mult, doc_reps)
    base_run = load_run(run_file, k=k)
    doc_corpus = load_corpus(corpus)

    needed_docids = {docid for pool in base_run.values() for docid, _ in pool}
    logger.info("dd-dense: %d unique pooled docid(s) across %d topic(s) need doc embeddings",
                len(needed_docids), len(base_run))

    doc_reps_by_id = _load_reps(doc_reps, needed_docids)
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
                    "text": doc_corpus.get(docid, {}).get("text", ""),
                    "title": doc_corpus.get(docid, {}).get("title"),
                },
            )
            for rank, (docid, score) in enumerate(pool, start=1)
        ]
        outputs[i].hits = hits
        outputs[i].evidences = _mmr_select(
            hits, doc_reps_by_id, dim, k, lambda_mult
        )

    return outputs
