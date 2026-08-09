"""
Dense retrieval demo -- loads tevatron query/claim embedding shards directly
(same pickle shard format and faiss flat inner-product index tevatron's own
driver.search uses) and fuses claim-level hits into document-level TREC
output.

Encoding happens upstream via tevatron (see scripts/dense-index/); this
script does the search itself rather than shelling out to
tevatron.retriever.driver.search. Docids carry a '#<i>' suffix; this
aggregates them to document level, same as pipeline/run_bm25.py does for
claim-level bm25s indices.

Usage:
    python pipeline/run_dense.py \
        --topics data/neuclir2024.topics.test.jsonl \
        --query_reps <queries_emb.pkl> \
        --passage_reps <'claims_emb.*.pkl'> \
        --corpus <claims_flat/*.claims.jsonl.gz> \
        --output results.txt \
        [--k 1000] [--tag dense-claim] [--fusion sum|rrf|max|first]
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from retrieval import search_dense
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
            logger.info("topic %s: wrote %d hits of (document) evidence", qid, len(result.evidences))
    logger.info("Done. Results saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Dense retrieval over tevatron embedding shards")
    parser.add_argument("--topics", required=True,
                        help="JSONL file with topics; each line must have 'qid' and 'query'")
    parser.add_argument("--query_reps", required=True,
                        help="Tevatron query embeddings pkl (a (reps, lookup) tuple)")
    parser.add_argument("--passage_reps", default=None,
                        help="Glob pattern matching tevatron claim embedding shard pkl(s); "
                             "loaded all at once into one index (doc-level scale)")
    parser.add_argument("--shard_groups", nargs="+", default=None,
                        help="One glob pattern per shard group (e.g. one per language); each group is "
                             "loaded, searched, and dropped before the next loads, so peak memory is one "
                             "group's worth of claims rather than the whole corpus (claim-level scale). "
                             "Mutually exclusive with --passage_reps.")
    parser.add_argument("--corpus", nargs="+", default=None,
                        help="Flattened claims corpus file(s)/glob(s), for claim text")
    parser.add_argument("--output", required=True,
                        help="Output file path (TREC run format)")
    parser.add_argument("--k", type=int, default=1000,
                        help="Number of claim hits to fuse per query (default: 1000)")
    parser.add_argument("--tag", default="dense",
                        help="Run tag written in the TREC output (default: dense)")
    parser.add_argument("--fusion", default="sum",
                        choices=["sum", "rrf", "max", "first"],
                        help="Score fusion strategy for claim-level runs (default: sum)")
    args = parser.parse_args()

    if bool(args.passage_reps) == bool(args.shard_groups):
        parser.error("specify exactly one of --passage_reps or --shard_groups")

    topics = load_topics(args.topics)
    logger.info("Loaded %d topic(s) from %s", len(topics), args.topics)

    inputs = [Result(topic=t, subquestions=t.get("subquestions", [])) for t in topics]
    results = search_dense.run(
        inputs,
        query_reps=args.query_reps,
        passage_reps=args.passage_reps,
        passage_reps_groups=args.shard_groups,
        corpus_paths=args.corpus,
        k=args.k,
        fusion=args.fusion,
    )

    write_trec(results, args.output, args.tag)


if __name__ == "__main__":
    main()
