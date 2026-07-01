import json
import logging
import copy
from tqdm import tqdm
from collections import defaultdict
from typing import List

from pyserini.search.lucene import LuceneSearcher

from utils import Result, Hit

logger = logging.getLogger(__name__)


def _load_index(index_dir, language=None, k1=0.9, b=0.4):
    searcher = LuceneSearcher(index_dir)
    if language:
        searcher.set_language(language)
    searcher.set_bm25(k1=k1, b=b)
    logger.info("Loaded Lucene index from %s (BM25 k1=%s, b=%s)", index_dir, k1, b)
    return searcher


def _build_queries(topic, subquestions):
    main_query = topic["query"]
    if not subquestions:
        return [main_query]
    return [main_query + sq for sq in subquestions]


def _fuse(temp, hits, strategy="sum"):
    """Fuse claim-level hits up to parent-doc level.

    Mirrors the logic in search.py so results are interchangeable.
    """
    is_claim_level = any("#" in h.docid for h in hits)
    if not is_claim_level:
        return hits

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

    doc_claims = defaultdict(list)
    for h in hits:
        doc_claims[h.docid.split("#")[0]].append(h)

    fused_hits = []
    for rank, (parent, fused_score) in enumerate(fusion.items(), start=1):
        if parent not in doc_claims:
            continue
        claims = doc_claims[parent]
        combined_text = " ".join(
            h.content_dict["text"] for h in claims if h.content_dict.get("text")
        )
        fused_hits.append(Hit(
            docid=parent,
            score=fused_score,
            rank=rank,
            content_dict={"text": combined_text, "title": claims[0].content_dict.get("title")},
        ))

    return fused_hits


def run(
    inputs: List[Result],
    index,
    k=100,
    language=None,
    fusion="sum",
    k1=0.9,
    b=0.4,
):
    outputs = copy.deepcopy(inputs)
    searcher = _load_index(index, language, k1=k1, b=b)

    for i, inp in tqdm(enumerate(inputs), desc="Retrieving", total=len(inputs)):
        queries = _build_queries(inp.topic, inp.subquestions)
        temp = defaultdict(list)
        hits = []

        for qtext in queries:
            raw_hits = searcher.search(qtext, k=k)

            for rank, h in enumerate(raw_hits, start=1):
                text = None # Skip the saving because this is just the testing
                hits.append(Hit(
                    docid=h.docid,
                    score=h.score,
                    rank=rank,
                    content_dict={"text": text, "title": None},
                ))
                temp[h.docid.split("#")[0]].append((h.score, rank))

        outputs[i].hits = hits
        outputs[i].evidences = _fuse(temp, hits, strategy=fusion)

    return outputs
