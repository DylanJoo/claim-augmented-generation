"""
Local rescore retrieval demo — doc-level global retrieval (concat-claims index)
+ claim expansion (claim-level index) + local in-memory reindex with
term-discounting and a doc-level soft relevance gate, then iterative greedy
claim selection. Writes TREC-format output. See src/retrieval/local_rescore.py
for the full mechanism.

Usage:
    python pipeline/run_local_rescore.py \
        --topics <topics.jsonl> \
        --doc-index   <path/to/concat-claims.bm25s> \
        --claim-index <path/to/claims.bm25s> \
        --output <results.txt> \
        [--k-doc 50] [--doc-floor 0.3] [--discount 0.3] [--n-iters 15] \
        [--k1 1.2] [--b 0.5] [--stopwords en] [--stemmer snowball] \
        [--tag local-rescore]
"""

import argparse
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src", "retrieval", "dev"))

import local_rescore
from utils import Result, load_topics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def write_trec(results, output_path, tag):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for result in results:
            qid = result.topic["qid"]
            for hit in result.evidences:
                out.write(f"{qid} Q0 {hit.docid} {hit.rank} {hit.score:.6f} {tag}\n")
            logger.info("topic %s: wrote %d selected claim(s)", qid, len(result.evidences))
    logger.info("Done. Results saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Local-rescore retrieval over bm25s indices")
    parser.add_argument("--topics", required=True,
                        help="JSONL file with topics; each line must have 'qid' and 'query'")
    parser.add_argument("--doc-index", required=True,
                        help="Directory of the pre-built doc-level (concat-claims) bm25s index")
    parser.add_argument("--claim-index", required=True,
                        help="Directory of the pre-built claim-level bm25s index")
    parser.add_argument("--output", required=True,
                        help="Output file path (TREC run format)")
    parser.add_argument("--k-doc", type=int, default=50,
                        help="Number of documents to retrieve at the doc level before expansion (default: 50)")
    parser.add_argument("--doc-floor", type=float, default=0.3,
                        help="Soft floor for the doc-relevance weight_mask, in [0,1] (default: 0.3)")
    parser.add_argument("--discount", type=float, default=0.3,
                        help="Factor applied to a selected claim's term contributions each round (default: 0.3)")
    parser.add_argument("--n-iters", type=int, default=15,
                        help="Number of claims to greedily select per topic (default: 15)")
    parser.add_argument("--k1", type=float, default=1.2,
                        help="BM25 k1 for the in-memory local reindex (default: 1.2)")
    parser.add_argument("--b", type=float, default=0.5,
                        help="BM25 b for the in-memory local reindex (default: 0.5)")
    parser.add_argument("--stopwords", default="en",
                        help="Stopword list passed to bm25s.tokenize (default: en)")
    parser.add_argument("--stemmer", default=None,
                        help="Stemmer to use, e.g. 'snowball' (default: none -- matches how the "
                             "global indices were actually built; the indexing scripts under "
                             "scripts/index/ don't pass --stemmer either)")
    parser.add_argument("--tag", default="local-rescore",
                        help="Run tag written in the TREC output (default: local-rescore)")
    parser.add_argument("--dedup-threshold", type=float, default=None,
                        help="If set, collapse near-duplicate docs (e.g. wire-service reposts) whose "
                             "concat-claims BM25 self-similarity exceeds this raw score, keeping only "
                             "the doc with the most claims per cluster. Unset by default (no dedup); "
                             "this is a raw BM25 score so tune it empirically per corpus.")
    args = parser.parse_args()

    topics = load_topics(args.topics)
    logger.info("Loaded %d topic(s) from %s", len(topics), args.topics)

    inputs = [Result(topic=t, subquestions=[]) for t in topics]
    results = local_rescore.run(
        inputs,
        doc_index=args.doc_index,
        claim_index=args.claim_index,
        k_doc=args.k_doc,
        doc_floor=args.doc_floor,
        discount=args.discount,
        n_iters=args.n_iters,
        k1=args.k1,
        b=args.b,
        stopwords=args.stopwords,
        stemmer_name=args.stemmer,
        dedup_threshold=args.dedup_threshold,
    )

    write_trec(results, args.output, args.tag)


if __name__ == "__main__":
    main()
