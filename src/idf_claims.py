"""Retrieve the top-k claims for a neuclir1 query and show each claim's
top-10 highest-idf tokens (idf computed from the claims index).

Edit the variables below, then run: python src/idf_claims.py
"""
import json
import os
import sys

import bm25s
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_topics

claim_index = os.path.join(os.environ["HOME"], "scratch/neuclir1/claims.bm25s")
topics_path = os.path.join(os.environ["HOME"], "claim-augmented-generation/data/neuclir2024.topics.test.jsonl")
qid = "300"
stopwords = "en"
stemmer_name = "snowball"
k = 10


def idf_array_from_df(df_array, N, method):
    df = df_array.astype(np.float64)
    if method == "robertson":
        return np.log((N - df + 0.5) / (df + 0.5))
    if method == "atire":
        return np.log(N / df)
    if method == "bm25l":
        return np.log((N + 1) / (df + 0.5))
    if method in ("bm25+", "bm25-"):
        return np.log((N + 1) / df)
    return np.log(1 + (N - df + 0.5) / (df + 0.5))  # lucene (bm25s default)


def load_idf(index_dir):
    with open(os.path.join(index_dir, "params.index.json")) as f:
        params = json.load(f)
    with open(os.path.join(index_dir, "vocab.index.json")) as f:
        vocab_dict = json.load(f)
    indptr = np.load(os.path.join(index_dir, "indptr.csc.index.npy"), mmap_mode="r")
    df_array = np.diff(np.asarray(indptr, dtype=np.int64))
    idf_array = idf_array_from_df(df_array, params["num_docs"], params.get("idf_method", "lucene"))
    return vocab_dict, idf_array


stemmer = None
if stemmer_name == "snowball":
    import Stemmer
    stemmer = Stemmer.Stemmer("english")

topics = {t["qid"]: t for t in load_topics(topics_path)}
query = topics[qid]["query"]

claim_vocab, claim_idf = load_idf(claim_index)
retriever = bm25s.BM25.load(claim_index, load_corpus=True)
with open(os.path.join(claim_index, "docids.json"), encoding="utf-8") as f:
    docids = json.load(f)

query_tokens = bm25s.tokenize([query], stopwords=stopwords, stemmer=stemmer)
results, scores = retriever.retrieve(query_tokens, k=k)

print(f"=== qid {qid}: {query[:80]}... ===")
for rank, (doc, score) in enumerate(zip(results[0], scores[0]), start=1):
    corpus_idx = doc["id"]
    docid = docids[corpus_idx]
    text = doc["text"]
    tokens = set(bm25s.tokenize([text], stopwords=stopwords, stemmer=stemmer,
                                 return_ids=False, show_progress=False)[0])

    claim_top = sorted(
        ((tok, claim_idf[claim_vocab[tok]]) for tok in tokens if tok in claim_vocab),
        key=lambda x: x[1], reverse=True,
    )[:10]

    print(f"\n  --- rank {rank} docid={docid} score={score:.4f} ---")
    print(f"  claim: {text[:120]}")
    print(f"  {'token':<25}{'idf':>10}")
    for tok, val in claim_top:
        print(f"  {tok:<25}{val:>10.3f}")
