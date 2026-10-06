"""
Query decomposition — LLM breaks each topic's request into standalone
sub-questions, written back out as a topics file with a "subquestions"
field added. Decoupled from retrieval (run_bm25.py etc.) so the expensive
GPU generation step runs once and every downstream retrieval experiment
just reads the cached subquestions off disk.

Usage:
    python pipeline/run_decompose.py \
        --topics data/neuclir2024.topics.test.jsonl \
        --output data/neuclir2024.topics.test.subq.jsonl \
        [--n-questions 10] [--model Qwen/Qwen3-8B]
"""
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from retrieval import rewrite
from utils import load_topics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LLM query decomposition into sub-questions")
    parser.add_argument("--topics", required=True,
                        help="JSONL file with topics; each line must have 'qid', 'query', 'meta.background'")
    parser.add_argument("--output", required=True,
                        help="Output path: same topics, each with a 'subquestions' field added")
    parser.add_argument("--n-questions", type=int, default=10,
                        help="Number of sub-questions to generate per topic (default: 10)")
    parser.add_argument("--model", default="Qwen/Qwen3-8B",
                        help="HF model id for decomposition (default: Qwen/Qwen3-8B)")
    args = parser.parse_args()

    topics = load_topics(args.topics)
    logger.info("Loaded %d topic(s) from %s", len(topics), args.topics)

    results = rewrite.run(topics, n_questions=args.n_questions, model_name_or_path=args.model)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for result in results:
            topic_with_subq = {**result.topic, "subquestions": result.subquestions}
            out.write(json.dumps(topic_with_subq, ensure_ascii=False) + "\n")
            logger.info("qid %s: %d subquestions", result.topic["qid"], len(result.subquestions))

    logger.info("Done. Topics with subquestions saved to %s", args.output)


if __name__ == "__main__":
    main()
