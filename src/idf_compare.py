"""Compare a token's idf across two bm25s indexes (e.g. documents vs claims),
restricted to the tokens found in the top-k docs of a run file.

Edit the variables below, then run: python src/idf.py
"""
import json
import math
import os

import bm25s
import numpy as np

doc_index = os.path.join(os.environ["HOME"], "scratch/neuclir1/documents.bm25s")
claim_index = os.path.join(os.environ["HOME"], "scratch/neuclir1/claims.bm25s")
run_file = os.path.join(os.environ["HOME"], "claim-augmented-generation/runs/run.neuclir1.documents.bm25.txt")
stopwords = "en"
stemmer_name = "snowball"
qid = None  # None = every qid in the run file, or set e.g. "300"
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


def load_run(path):
    runs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            q, _, docid, rank, score, _ = line.split()
            runs.setdefault(q, []).append((int(rank), docid, float(score)))
    for q in runs:
        runs[q].sort(key=lambda x: x[0])
    return runs


stemmer = None
if stemmer_name == "snowball":
    import Stemmer
    stemmer = Stemmer.Stemmer("english")

doc_vocab, doc_idf = load_idf(doc_index)
claim_vocab, claim_idf = load_idf(claim_index)

retriever = bm25s.BM25.load(doc_index, load_corpus=True)
with open(os.path.join(doc_index, "docids.json"), encoding="utf-8") as f:
    docids = json.load(f)
docid_to_idx = {d: i for i, d in enumerate(docids)}

runs = load_run(run_file)
qids = [qid] if qid else list(runs.keys())

for q in qids:
    print(f"\n=== qid {q} (top-{k}) ===")
    for rank, docid, score in runs[q][:k]:
        corpus_idx = docid_to_idx.get(docid)
        if corpus_idx is None:
            print(f"  [rank {rank}] {docid} not found in {doc_index}")
            continue
        text = retriever.corpus[corpus_idx]["text"]
        tokens = set(bm25s.tokenize([text], stopwords=stopwords, stemmer=stemmer,
                                     return_ids=False, show_progress=False)[0])

        doc_top = sorted(
            ((tok, doc_idf[doc_vocab[tok]]) for tok in tokens if tok in doc_vocab),
            key=lambda x: x[1], reverse=True,
        )[:10]
        claim_top = sorted(
            ((tok, claim_idf[claim_vocab[tok]]) for tok in tokens if tok in claim_vocab),
            key=lambda x: x[1], reverse=True,
        )[:10]

        print(f"\n  --- rank {rank} docid={docid} score={score:.4f} ---")
        print(f"  {'doc top-10':<25}{'idf':>10}    {'claim top-10':<25}{'idf':>10}")
        for i in range(10):
            d_tok, d_val = doc_top[i] if i < len(doc_top) else (None, None)
            c_tok, c_val = claim_top[i] if i < len(claim_top) else (None, None)
            d_str = f"{d_tok:<25}{d_val:>10.3f}" if d_tok is not None else f"{'None':<25}{'':>10}"
            c_str = f"{c_tok:<25}{c_val:>10.3f}" if c_tok is not None else f"{'None':<25}{'':>10}"
            print(f"  {d_str}    {c_str}")
