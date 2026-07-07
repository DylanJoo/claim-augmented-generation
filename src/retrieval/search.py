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
    main_query = topic["query"]
    if not subquestions:
        return [main_query]
    return [main_query + sq for sq in subquestions]


def _fuse(temp, hits, strategy="sum"):
    """Return (evidences, fused_hits) at parent-doc level.

    Doc-level index: hits already have parent docids — sort by score, no aggregation. N fusion
    Claim-level index: fuse scores via strategy and concatenate claim texts per doc.
    """
    is_claim_level = any("#" in h.docid for h in hits)

    if not is_claim_level:
        return hits

    # Compute fused score per parent doc
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

    # Group claim hits by parent doc and concatenate texts
    doc_claims = defaultdict(list)
    for h in hits:
        doc_claims[h.docid.split("#")[0]].append(h)

    fused_hits = []
    for rank, (parent, fused_score) in enumerate(fusion.items(), start=1):
        if parent not in doc_claims:
            continue
        claims = doc_claims[parent]
        combined_text = " ".join(h.content_dict["text"] for h in claims if h.content_dict.get("text"))
        fused_hits.append(Hit(
            docid=parent,
            score=fused_score,
            rank=rank,
            content_dict={"text": combined_text, "title": claims[0].content_dict.get("title")}
        ))

    return fused_hits


def run(
    inputs: List[Result],
    index,
    k=100,
    stopwords="en",
    stemmer=None,
    fusion="sum", # only claim-level needs fusion
):
    outputs = copy.deepcopy(inputs)
    retriever, docids, _stemmer = _load_index(index, stemmer)
    n_docs = len(docids)

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

        outputs[i].hits = hits
        outputs[i].evidences = _fuse(temp, hits, strategy=fusion)

    return outputs
