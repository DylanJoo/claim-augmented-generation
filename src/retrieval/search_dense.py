import faiss
import glob
import logging
import copy
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import List

from utils import Result, Hit, load_corpus

logger = logging.getLogger(__name__)


def _pickle_load(path):
    """Same shard format tevatron.retriever.driver.search reads/writes: a
    (reps, lookup) tuple, reps as an array of embeddings and lookup as the
    parallel list of ids."""
    with open(path, "rb") as f:
        reps, lookup = pickle.load(f)
    return np.asarray(reps), lookup


def _load_flat_from_files(index_files, desc="Loading shards into index"):
    """Merge a list of tevatron embedding shard files into one flat
    inner-product faiss index (tevatron normalizes embeddings at encode
    time, so IP search is cosine similarity) -- same index-building mechanics
    as tevatron.retriever.driver.search.main()/FaissFlatSearcher.

    Shards are loaded and added to the index one at a time (never all held
    in memory together), so peak memory is one shard's reps plus the index
    being built, not every shard's reps at once.
    """
    index = None
    look_up = []
    for f in tqdm(index_files, desc=desc, leave=False):
        p_reps, p_lookup = _pickle_load(f)
        if index is None:
            index = faiss.IndexFlatIP(p_reps.shape[1])
        index.add(p_reps)
        look_up += p_lookup
        del p_reps
    return index, look_up


def _load_index(passage_reps):
    """Load every shard matching passage_reps into a single flat index.
    Fine for doc-level scale (small corpus, loaded once); see
    _resolve_groups/_run_grouped for the claim-level streaming path.
    """
    index_files = sorted(glob.glob(passage_reps))
    if not index_files:
        raise FileNotFoundError(f"No passage rep shards matched: {passage_reps}")

    index, look_up = _load_flat_from_files(index_files, desc="Loading claim shards into index")
    logger.info("Loaded dense index with %d claims from %d shard(s)", len(look_up), len(index_files))
    return index, look_up


def _resolve_groups(passage_reps_groups):
    """Each glob pattern in passage_reps_groups expands to one shard group
    (e.g. one language's worth of claim shards) -- one load/search/drop unit
    for _run_grouped, so peak memory is one group's worth of claims rather
    than the whole corpus."""
    groups = []
    for pattern in passage_reps_groups:
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No passage rep shards matched group pattern: {pattern}")
        groups.append(files)
    return groups


def _build_queries(topic, subquestions):
    main_query = topic["query"]
    if not subquestions:
        return [main_query]
    return [main_query] + list(subquestions)


def _query_keys(inp):
    qid = inp.topic["qid"]
    queries = _build_queries(inp.topic, inp.subquestions)
    return [qid if qi == 0 else f"{qid}#s{qi - 1}" for qi in range(len(queries))]


def _fuse(temp, hits, strategy="sum"):
    """Return fused_hits at parent-doc level.

    Mirrors the logic in search.py so results are interchangeable.
    """
    is_claim_level = any("#" in h.docid for h in hits)

    fusion = {}
    for docid, items in temp.items():
        if strategy == "rrf":
            fusion[docid] = sum(1 / rank for _, rank in items)
        elif strategy == "max":
            fusion[docid] = max(score for score, _ in items)
        elif strategy == "first":
            fusion[docid] = items[0][0]
        else:  # sum
            fusion[docid] = sum(score for score, _ in items)
    fusion = dict(sorted(fusion.items(), key=lambda x: x[1], reverse=True))

    doc_hits = defaultdict(dict)
    for h in hits:
        doc_hits[h.docid.split("#")[0]][h.docid] = h

    fused_hits = []
    for rank, (parent, fused_score) in enumerate(fusion.items(), start=1):
        if parent not in doc_hits:
            continue
        entries = list(doc_hits[parent].values())
        if is_claim_level:
            combined_text = " ".join(h.content_dict["text"] for h in entries if h.content_dict.get("text"))
        else:
            combined_text = entries[0].content_dict.get("text")
        fused_hits.append(Hit(
            docid=parent,
            score=fused_score,
            rank=rank,
            content_dict={"text": combined_text, "title": entries[0].content_dict.get("title")}
        ))

    return fused_hits


def _finalize(outputs, i, inp, candidates_by_key, corpus, k, fusion):
    """candidates_by_key: qkey -> list of (score, docid), already the
    top-k for that query. Builds .hits/.evidences the same way for both
    the single-index and grouped-streaming search paths."""
    temp = defaultdict(list)
    hits = []
    for qkey in _query_keys(inp):
        for rank, (score, docid) in enumerate(candidates_by_key.get(qkey, []), start=1):
            hits.append(Hit(
                docid=docid,
                score=score,
                rank=rank,
                content_dict={
                    'text': corpus.get(docid, {}).get('text'),
                    'title': None
                }
            ))
            temp[docid.split("#")[0]].append((score, rank))

    # For pure doc-level retrieval, .hits and .evidences would give you the same stuff
    # For claim-level retrieval, .hits is the retrieved claims while .evidence would
    # return the aggregated/fused doc-level retrieval results.
    outputs[i].hits = hits
    outputs[i].evidences = _fuse(temp, hits, strategy=fusion)

    if any("#" in h.docid for h in hits):
        n_claims, n_parents = len(hits), len(outputs[i].evidences)
        logger.debug(
            "[%s] claim-level: %d claim hits -> %d unique parent docs (requested k=%d)",
            inp.topic.get("request_id", i), n_claims, n_parents, k,
        )


def _run_single(inputs, index, look_up, q_reps, q_pos, corpus, k, fusion):
    """Original behavior: one index already holds every shard. Used for
    doc-level retrieval, where the whole corpus fits in memory at once."""
    outputs = copy.deepcopy(inputs)
    for i, inp in tqdm(enumerate(inputs), desc="Retrieving", total=len(inputs)):
        candidates_by_key = {}
        # NOTE: Since the subqueries have to be encoded together.
        # We need to trace the sqid.
        # Unlike BM25s that can return text, we need to load corpus
        for qkey in _query_keys(inp):
            if qkey not in q_pos:
                continue
            q_vec = q_reps[q_pos[qkey]: q_pos[qkey] + 1]
            scores, indices = index.search(q_vec, k)
            candidates_by_key[qkey] = [
                (float(score), look_up[idx]) for idx, score in zip(indices[0], scores[0]) if idx >= 0
            ]
        _finalize(outputs, i, inp, candidates_by_key, corpus, k, fusion)
    return outputs


def _run_grouped(inputs, groups, q_reps, q_pos, corpus, k, fusion):
    """Claim-level streaming path: one shard group (e.g. one language) is
    loaded, searched against every query, and dropped before the next
    group loads, so peak memory is one group's worth of claims rather than
    the whole corpus. Each group already returns its own top-k per query;
    taking the top-k of the union across groups is the exact global top-k
    (any candidate ranked beyond k within its own group is beaten by k
    other candidates in that same group, so it can't be in the global
    top-k either) -- no approximation from splitting the search this way.
    """
    outputs = copy.deepcopy(inputs)
    query_keys = [_query_keys(inp) for inp in inputs]
    accumulator = [defaultdict(list) for _ in inputs]

    # TODO: I believe this section can be resued from run_single
    for files in tqdm(groups, desc="Streaming shard groups"):
        index, look_up = _load_flat_from_files(files, desc=f"Loading group ({len(files)} shard(s))")
        for i, keys in enumerate(query_keys):
            for qkey in keys:
                if qkey not in q_pos:
                    continue
                q_vec = q_reps[q_pos[qkey]: q_pos[qkey] + 1]
                scores, indices = index.search(q_vec, k)
                accumulator[i][qkey].extend(
                    (float(score), look_up[idx]) for idx, score in zip(indices[0], scores[0]) if idx >= 0
                )
        del index, look_up

    for i, inp in enumerate(inputs):
        candidates_by_key = {
            qkey: sorted(cands, key=lambda x: x[0], reverse=True)[:k]
            for qkey, cands in accumulator[i].items()
        }
        _finalize(outputs, i, inp, candidates_by_key, corpus, k, fusion)
    return outputs


def run(
    inputs: List[Result],
    query_reps,
    passage_reps=None,
    passage_reps_groups=None,
    corpus_paths=None,
    k=100,
    fusion="sum", # combines scores across queries (main + sub-questions) and across claims, same as search.py
):
    """passage_reps: single glob, all matching shards loaded into one index
    (doc-level scale). passage_reps_groups: list of globs, one shard group
    (e.g. one language) streamed at a time (claim-level scale). Exactly one
    of the two must be given.
    """
    if bool(passage_reps) == bool(passage_reps_groups):
        raise ValueError("run() takes exactly one of passage_reps or passage_reps_groups")

    q_reps, q_lookup = _pickle_load(query_reps)
    q_pos = {qid: i for i, qid in enumerate(q_lookup)}
    corpus = load_corpus(corpus_paths) if corpus_paths else {}

    if passage_reps_groups:
        groups = _resolve_groups(passage_reps_groups)
        logger.info("Streaming %d shard group(s) for claim-level retrieval", len(groups))
        return _run_grouped(inputs, groups, q_reps, q_pos, corpus, k, fusion)

    index, look_up = _load_index(passage_reps)
    return _run_single(inputs, index, look_up, q_reps, q_pos, corpus, k, fusion)
