"""Inspect statistics of a NeuCLIR1 bm25s index: vocab size, doc count, and
per-term document frequency / idf for a sample of topic queries.

idf is not stored by bm25s, so it is recomputed here from document frequency
using the same formula bm25s used at index time (read from params.index.json).

Usage:
    python src/dev/index_stats.py \
        --index $HOME/scratch/neuclir1/documents.bm25s \
        --topics data/neuclir2024.topics.test.jsonl \
        --num-topics 5
"""
import argparse
import json
import logging
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import load_topics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def idf_array_from_df(df_array, N, method):
    """Vectorized idf, mirroring bm25s's per-method formulas (bm25s/scoring.py)."""
    import numpy as np
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


def load_selected_topics(path, qids, num_topics):
    topics = load_topics(path)
    if qids:
        wanted = {str(q) for q in qids}
        topics = [t for t in topics if str(t.get("qid")) in wanted]
    else:
        topics = topics[:num_topics]
    return topics


def analyze_bm25s(index_dir, topics, top_k, skip_global, stopwords, stemmer_name):
    import numpy as np
    import bm25s

    with open(os.path.join(index_dir, "params.index.json")) as f:
        params = json.load(f)
    with open(os.path.join(index_dir, "vocab.index.json")) as f:
        vocab_dict = json.load(f)

    N = params["num_docs"]
    idf_method = params.get("idf_method", "lucene")
    vocab_size = len(vocab_dict)
    logger.info(
        "bm25s index at %s: %d docs, %d vocab terms, k1=%s b=%s idf_method=%s",
        index_dir, N, vocab_size, params.get("k1"), params.get("b"), idf_method,
    )

    # CSC matrix is stored with columns = vocab terms, rows = docs, so
    # per-term document frequency is just the column nnz count (indptr diff).
    indptr = np.load(os.path.join(index_dir, "indptr.csc.index.npy"), mmap_mode="r")
    if len(indptr) - 1 != vocab_size:
        logger.warning(
            "indptr length-1 (%d) != vocab size (%d) - this bm25s version may store "
            "the score matrix with a different orientation; df/idf below may be wrong.",
            len(indptr) - 1, vocab_size,
        )
    df_array = np.diff(np.asarray(indptr, dtype=np.int64))
    idf_array = idf_array_from_df(df_array, N, idf_method)

    stemmer = None
    if stemmer_name == "snowball":
        import Stemmer
        stemmer = Stemmer.Stemmer("english")

    if not skip_global:
        id_to_token = {i: t for t, i in vocab_dict.items()}
        rarest = np.argsort(idf_array)[::-1][:top_k]
        commonest = np.argsort(idf_array)[:top_k]
        print(f"\n=== global vocab stats ({vocab_size} terms, {N} docs) ===")
        print("rarest (highest idf) terms:")
        for i in rarest:
            i = int(i)
            print(f"  {id_to_token[i]:<25} df={df_array[i]:>10} idf={idf_array[i]:.3f}")
        print("most common (lowest idf) terms:")
        for i in commonest:
            i = int(i)
            print(f"  {id_to_token[i]:<25} df={df_array[i]:>10} idf={idf_array[i]:.3f}")

    for topic in topics:
        qid = topic.get("qid", "?")
        tokens = bm25s.tokenize(
            [topic["query"]], stopwords=stopwords, stemmer=stemmer,
            return_ids=False, show_progress=False,
        )[0]

        seen, rows = set(), []
        for tok in tokens:
            if tok in seen:
                continue
            seen.add(tok)
            tid = vocab_dict.get(tok)
            if tid is None:
                rows.append((tok, None, None))
            else:
                rows.append((tok, int(df_array[tid]), float(idf_array[tid])))
        rows.sort(key=lambda r: (r[2] is None, -(r[2] or 0)))

        n_oov = sum(1 for r in rows if r[1] is None)
        print(f"\n=== topic {qid} ({len(rows)} unique terms, {n_oov} OOV) ===")
        print(f"{'term':<25}{'df':>10}{'idf':>10}")
        for tok, df, idf in rows:
            print(f"{tok:<25}{'OOV':>10}{'':>10}" if df is None else f"{tok:<25}{df:>10}{idf:>10.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect vocab size, doc count, and per-term df/idf of a NeuCLIR1 bm25s index."
    )
    parser.add_argument("--index", required=True, help="Path to a bm25s index dir")
    parser.add_argument("--topics", default="data/neuclir2024.topics.test.jsonl")
    parser.add_argument("--qids", nargs="*", default=None,
                        help="Specific topic qids to inspect (default: first --num-topics topics)")
    parser.add_argument("--num-topics", type=int, default=5)
    parser.add_argument("--stopwords", default="en",
                        help="Stopword list used at indexing time (must match indexing.py args)")
    parser.add_argument("--stemmer", default=None, choices=[None, "snowball"],
                        help="Stemmer used at indexing time (must match indexing.py args)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Rarest/most-common terms to show in the global vocab scan")
    parser.add_argument("--skip-global", action="store_true",
                        help="Skip the full-vocabulary rarest/most-common term scan")
    args = parser.parse_args()

    topics = load_selected_topics(args.topics, args.qids, args.num_topics)
    logger.info("Inspecting %d topic(s): %s", len(topics), [t.get("qid") for t in topics])

    analyze_bm25s(args.index, topics, args.top_k, args.skip_global, args.stopwords, args.stemmer)


if __name__ == "__main__":
    main()
