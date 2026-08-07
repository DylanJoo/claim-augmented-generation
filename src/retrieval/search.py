import bm25s
import json
import logging
import os
import copy
from tqdm import tqdm
from collections import defaultdict
from typing import List

from utils import Result, Hit

logger = logging.getLogger(__name__)


def _load_index(index_dir, stemmer_name=None):
    retriever = bm25s.BM25.load(index_dir, load_corpus=True)
    docids_path = os.path.join(index_dir, "docids.json")
    with open(docids_path, encoding="utf-8") as f:
        docids = json.load(f)
    stemmer = None
    if stemmer_name == "snowball":
        import Stemmer
        stemmer = Stemmer.Stemmer("english")
    logger.info("Loaded index with %d documents from %s", len(docids), index_dir)
    return retriever, docids, stemmer


def _build_queries(topic, subquestions):
    """Original query plus each sub-question fired as its own independent
    retrieval pass. Sub-questions are written by rewrite.py to already carry
    "comprehensive context" on their own, so gluing the (generic) main query
    text onto them would just dilute their specific vocabulary with the same
    broad terms every other sub-question -- and the main-query -- already
    contributes."""
    main_query = topic["query"]
    if not subquestions:
        return [main_query]
    return [main_query] + list(subquestions)


def _fuse(temp, hits, strategy="sum"):
    """Return fused_hits at parent-doc level.

    `temp` already carries every (score, rank) a parent doc earned across
    *all* fired queries (plain query + each sub-question) and, for a
    claim-level index, across all of that doc's claims too -- both axes
    land in the same list, so this always has to fuse, even for a doc-level
    index with a single query where every parent has exactly one entry.

    Claim-level index: fuse scores via strategy and concatenate claim texts
    per doc. Doc-level index: fuse scores via strategy; text is identical
    across duplicate hits of the same doc (one per query it matched), so it
    is taken once rather than concatenated.
    """
    is_claim_level = any("#" in h.docid for h in hits)

    # Compute fused score per parent doc.
    # Under the same docid, reconstruct the scores.
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

    # Group hits by parent doc, deduped by full docid so a doc/claim matched
    # by multiple queries doesn't get its text concatenated with itself.
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


def run(
    inputs: List[Result],
    index,
    k=100,
    stopwords="en",
    stemmer=None,
    fusion="sum", # combines scores across queries (main + sub-questions) and, for claim-level indices, across claims too
):
    outputs = copy.deepcopy(inputs)
    retriever, docids, _stemmer = _load_index(index, stemmer)
    n_docs = len(docids)
    n_parent_docs_per_query = []

    for i, inp in tqdm(enumerate(inputs), desc="Retrieving", total=len(inputs)):
        queries = _build_queries(inp.topic, inp.subquestions)
        temp = defaultdict(list)
        hits = []

        for qtext in queries:
            query_tokens = bm25s.tokenize([qtext], stopwords=stopwords, stemmer=_stemmer)
            results, scores = retriever.retrieve(query_tokens, k=min(k, n_docs))

            for rank, (doc, score) in enumerate(zip(results[0], scores[0]), start=1):
                corpus_idx = doc["id"]
                docid = docids[corpus_idx]
                hits.append(Hit(
                    docid=docid,
                    score=score,
                    rank=rank,
                    content_dict={
                        'text': doc['text'],
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
            n_parent_docs_per_query.append(n_parents)
            logger.debug(
                "[%s] claim-level: %d claim hits -> %d unique parent docs (requested k=%d)",
                inp.topic.get("request_id", i), n_claims, n_parents, k,
            )

    return outputs
