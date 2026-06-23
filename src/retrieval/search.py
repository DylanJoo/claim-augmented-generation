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


def _fuse(temp, strategy="sum"):
    """Aggregate per-query (score, rank) pairs into a single score per parent doc."""
    fusion = {}
    for docid, items in temp.items():
        if strategy == "rrf":
            fusion[docid] = sum(1 / rank for _, rank in items)
        elif strategy == "max":
            fusion[docid] = max(score for score, _ in items)
        else:  # sum
            fusion[docid] = sum(score for score, _ in items)
    return dict(sorted(fusion.items(), key=lambda x: x[1], reverse=True))


def run(
    inputs: List[Result],
    index,
    k=100,
    stopwords="en",
    stemmer=None,
    fusion="sum",
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

            for rank, (corpus_idx, score) in enumerate(zip(results[0], scores[0]), start=1):
                hits.append(Hit(
                    docid=docids[corpus_idx],
                    score=score,
                    rank=rank,
                    content_dict={
                        'text': retriever.corpus[corpus_idx]['text'],
                        'title': None
                    }
                ))
                temp[docids[corpus_idx].split("#")[0]].append((score, rank))

        outputs[i].hits = hits
        outputs[i].evidences = _fuse(temp, strategy=fusion)

    return outputs
