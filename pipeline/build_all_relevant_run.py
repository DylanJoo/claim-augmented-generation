"""
All-relevant sanity-check run -- writes a doc-level TREC run containing
*every* judged-relevant doc in the qrel (not just the ones a base run
happened to retrieve, cf. pipeline/filter_relevant_run.py), each scored by
the dense query-doc inner product from tevatron embedding shards.

Like filter_relevant_run.py this is a diagnostic, not a retrieval method:
fed to a diversity reranker as --run-file, every candidate is relevant and
the pool is the full relevant set, so StRecall@k differences come only from
the order the reranker picks among relevant docs. See runs/sanity-all-relevant/.

Usage:
    python pipeline/build_all_relevant_run.py \
        --qrel $HOME/trec2026/data/neuclir/neuclir24-test-request.qrel \
        --query_reps $HOME/scratch/neuclir1/Qwen3-Embedding-0.6B/queries_emb/queries_emb.pkl \
        --passage_reps "$HOME/scratch/neuclir1/Qwen3-Embedding-0.6B/docs_emb/docs_emb.*.pkl" \
        --output runs/sanity-all-relevant/base/run.neuclir1.documents.Qwen3-Embedding-0.6B.all-relevant.txt \
        [--threshold 1] [--tag qwen3-embed-0.6b:doc:all-relevant]
"""

import argparse
import glob
import logging
import os
import pickle
from collections import defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_relevant_docids(path, threshold):
    """Parse a TREC qrel file -> {qid: {docid, ...}} for rel >= threshold."""
    relevant = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _iteration, docid, rel = line.split()
            if int(rel) >= threshold:
                relevant[qid].add(docid)
    return relevant


def load_reps(path):
    """Tevatron shard: a (reps, lookup) tuple."""
    with open(path, "rb") as f:
        reps, lookup = pickle.load(f)
    return np.asarray(reps, dtype=np.float32), lookup


def main():
    parser = argparse.ArgumentParser(
        description="Build a doc-level TREC run of every judged-relevant doc, "
                     "scored by dense query-doc similarity")
    parser.add_argument("--qrel", required=True,
                        help="TREC qrel file (qid iteration docid rel)")
    parser.add_argument("--query_reps", required=True,
                        help="Tevatron query embeddings pkl (a (reps, lookup) tuple)")
    parser.add_argument("--passage_reps", required=True,
                        help="Glob pattern matching tevatron doc embedding shard pkl(s)")
    parser.add_argument("--output", required=True,
                        help="Output file path (TREC run format)")
    parser.add_argument("--threshold", type=int, default=1,
                        help="Minimum qrel relevance grade to keep (default: 1)")
    parser.add_argument("--tag", default="all-relevant",
                        help="Run tag written in the TREC output (default: all-relevant)")
    args = parser.parse_args()

    relevant = load_relevant_docids(args.qrel, args.threshold)
    logger.info("Loaded qrel with %d topics from %s", len(relevant), args.qrel)

    q_reps, q_lookup = load_reps(args.query_reps)
    q_pos = {qid: i for i, qid in enumerate(q_lookup)}

    # docid -> list of qids it is relevant for, so each shard is scanned once
    wanted = defaultdict(list)
    for qid, docids in relevant.items():
        for docid in docids:
            wanted[docid].append(qid)

    scores = defaultdict(dict)  # qid -> {docid: score}
    shard_files = sorted(glob.glob(args.passage_reps))
    if not shard_files:
        parser.error(f"no shards match {args.passage_reps}")
    for path in shard_files:
        p_reps, p_lookup = load_reps(path)
        for i, docid in enumerate(p_lookup):
            for qid in wanted.get(docid, ()):
                if qid in q_pos:
                    scores[qid][docid] = float(p_reps[i] @ q_reps[q_pos[qid]])
        logger.info("Scanned %s", path)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for qid in sorted(relevant, key=lambda q: (q not in q_pos, q)):
            if qid not in q_pos:
                logger.warning("topic %s: in qrel but no query embedding, skipped", qid)
                continue
            ranked = sorted(scores[qid].items(), key=lambda x: -x[1])
            for rank, (docid, score) in enumerate(ranked, start=1):
                out.write(f"{qid} Q0 {docid} {rank} {score:.6f} {args.tag}\n")
            logger.info("topic %s: wrote %d/%d relevant docs (rest have no embedding)",
                        qid, len(ranked), len(relevant[qid]))
    logger.info("Done. All-relevant run saved to %s", args.output)


if __name__ == "__main__":
    main()
