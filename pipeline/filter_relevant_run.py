"""
Relevant-only sanity-check filter -- takes a doc-level TREC run file and a
qrel, and writes a new run file containing only the hits that are judged
relevant, in their original (base-run) rank/score order.

This is a diagnostic tool, not a retrieval method: feeding this filtered run
back into a diversity reranker (run_dd.py / run_cc.py / run_cc_kmeans.py
as --run-file) means every candidate in the pool is already relevant, so
relevance can no longer explain differences in StRecall@k -- only the order
the reranker chooses among relevant docs can. See runs/sanity-relevant-only/.

Usage:
    python pipeline/filter_relevant_run.py \
        --run-file runs/run.neuclir1.documents.Qwen3-Embedding-0.6B.txt \
        --qrel $HOME/trec2026/data/neuclir/neuclir24-test-request.qrel \
        --output runs/sanity-relevant-only/base/run.neuclir1.documents.Qwen3-Embedding-0.6B.relevant-only.txt \
        [--threshold 1] [--tag qwen3-embed-0.6b:doc:relevant-only]
"""

import argparse
import logging
import os
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_run(path):
    """Parse a TREC run file -> {qid: [(docid, score), ...]}, in file order."""
    run = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, docid, score = parts[0], parts[2], float(parts[4])
            run[qid].append((docid, score))
    return run


def load_relevant_docids(path, threshold):
    """Parse a TREC qrel file -> {qid: {docid, ...}} for rel >= threshold."""
    relevant = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _iteration, docid, rel = line.split()
            if int(rel) >= threshold:
                relevant[qid].add(docid)
    return relevant


def main():
    parser = argparse.ArgumentParser(
        description="Filter a doc-level TREC run down to judged-relevant docs only, "
                     "preserving the base run's rank/score order")
    parser.add_argument("--run-file", required=True,
                        help="Pre-computed doc-level TREC run file to filter")
    parser.add_argument("--qrel", required=True,
                        help="TREC qrel file (qid iteration docid rel)")
    parser.add_argument("--output", required=True,
                        help="Output file path (TREC run format)")
    parser.add_argument("--threshold", type=int, default=1,
                        help="Minimum qrel relevance grade to keep (default: 1)")
    parser.add_argument("--tag", default="relevant-only",
                        help="Run tag written in the TREC output (default: relevant-only)")
    args = parser.parse_args()

    run = load_run(args.run_file)
    relevant = load_relevant_docids(args.qrel, args.threshold)
    logger.info("Loaded base run with %d topics from %s", len(run), args.run_file)
    logger.info("Loaded qrel with %d topics from %s", len(relevant), args.qrel)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for qid, hits in run.items():
            kept = [(docid, score) for docid, score in hits if docid in relevant.get(qid, ())]
            for rank, (docid, score) in enumerate(kept, start=1):
                out.write(f"{qid} Q0 {docid} {rank} {score:.6f} {args.tag}\n")
            logger.info(
                "topic %s: kept %d/%d relevant hits (qrel has %d relevant docs total)",
                qid, len(kept), len(hits), len(relevant.get(qid, ())),
            )
    logger.info("Done. Filtered run saved to %s", args.output)


if __name__ == "__main__":
    main()
