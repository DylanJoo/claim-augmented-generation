"""
Relevance-quadrant report — for each topic, buckets the retrieved documents by
(doc-relevance, claim-relevance) using src/retrieval/dev/relevance_quadrants.py.
Read-only diagnostic: prints bucket counts and example docids, does not write
a run file or change any selection behavior.

Reuses local_rescore.py's doc-retrieval / claim-expansion helpers so this
stays a thin analysis layer on top of the same Stage 1 / Stage 1.6 logic.

Usage:
    python pipeline/run_relevance_quadrants.py \
        --topics <topics.jsonl> \
        --doc-index   <path/to/concat-claims.bm25s> \
        --claim-index <path/to/claims.bm25s> \
        --doc-threshold <float> --claim-threshold <float> \
        [--k-doc 50] [--k1 1.2] [--b 0.5] [--stopwords en] [--stemmer snowball] \
        [--examples 5]
"""

import argparse
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src", "retrieval", "dev"))

import bm25s

import local_rescore
import relevance_quadrants
from utils import load_topics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Doc-relevance x claim-relevance quadrant report")
    parser.add_argument("--topics", required=True,
                        help="JSONL file with topics; each line must have 'qid' and 'query'")
    parser.add_argument("--doc-index", required=True,
                        help="Directory of the pre-built doc-level (concat-claims) bm25s index")
    parser.add_argument("--claim-index", required=True,
                        help="Directory of the pre-built claim-level bm25s index")
    parser.add_argument("--doc-threshold", type=float, required=True,
                        help="Raw BM25 score above which a document counts as doc-relevant")
    parser.add_argument("--claim-threshold", type=float, required=True,
                        help="Raw BM25 score above which a document's best claim counts as claim-relevant")
    parser.add_argument("--k-doc", type=int, default=50,
                        help="Number of documents to retrieve at the doc level (default: 50)")
    parser.add_argument("--k1", type=float, default=1.2,
                        help="BM25 k1 for the in-memory local reindex (default: 1.2)")
    parser.add_argument("--b", type=float, default=0.5,
                        help="BM25 b for the in-memory local reindex (default: 0.5)")
    parser.add_argument("--stopwords", default="en",
                        help="Stopword list passed to bm25s.tokenize (default: en)")
    parser.add_argument("--stemmer", default=None,
                        help="Stemmer to use, e.g. 'snowball' (default: none -- see run_local_rescore.py's "
                             "note on why the global indices' own build-time tokenization has none either)")
    parser.add_argument("--examples", type=int, default=5,
                        help="Number of example docids to print per bucket (default: 5)")
    args = parser.parse_args()

    stemmer = None
    if args.stemmer == "snowball":
        import Stemmer
        stemmer = Stemmer.Stemmer("english")

    topics = load_topics(args.topics)
    logger.info("Loaded %d topic(s) from %s", len(topics), args.topics)

    logger.info("Loading doc-level index from %s", args.doc_index)
    doc_retriever, doc_docids = local_rescore._load_bm25(args.doc_index)
    logger.info("Loading claim-level index from %s", args.claim_index)
    claim_retriever, claim_docids = local_rescore._load_bm25(args.claim_index)

    for topic in topics:
        qid = topic["qid"]
        query = topic["query"]
        query_tokens = bm25s.tokenize([query], stopwords=args.stopwords, stemmer=stemmer)

        doc_hits = local_rescore._retrieve_docs(doc_retriever, doc_docids, query_tokens, args.k_doc)

        retrieved_parents = {h["docid"] for h in doc_hits}
        claim_indices_by_parent = local_rescore._scan_claim_indices(claim_docids, retrieved_parents)
        doc_hit_by_id = {h["docid"]: h for h in doc_hits}
        subset = local_rescore._fetch_claim_texts(claim_retriever, claim_docids, claim_indices_by_parent, doc_hit_by_id)

        buckets, details = relevance_quadrants.classify(
            doc_hits, subset, query_tokens,
            doc_threshold=args.doc_threshold,
            claim_threshold=args.claim_threshold,
            k1=args.k1, b=args.b, stopwords=args.stopwords, stemmer=stemmer,
        )

        print(f"\n=== Topic {qid}: {query[:80]!r}... ===")
        for name in ["relevant_both", "claims_only", "doc_only", "neither"]:
            docids = buckets.get(name, [])
            print(f"  {name:15s} n={len(docids)}")
            for docid in docids[: args.examples]:
                d = details[docid]
                print(f"      {docid}  d_score={d['d_score']:.2f}  max_claim_score={d['max_claim_score']:.2f}")


if __name__ == "__main__":
    main()
