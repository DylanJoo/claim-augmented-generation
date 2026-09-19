"""
Ground-truth-aware oracle reranking, reported as an upper-bound metric.

Given a run file's top-k2 pool per topic (k2 defaults to 100 -- deliberately
smaller than the 1000-doc pools dd.py/cc.py/dc_gap.py rerank over, so
this reads as "how much headroom is left in the part of the pool a reranker
would realistically operate on"), this greedily reorders each query's pool
using the subtopic-level ground truth qrel directly, then scores that oracle
ordering with the same metrics rac_eval.py reports (StRecall@1..10).

The result is not a method -- it cheats by reading the test qrel -- so it is
never a run you'd submit. It's a ceiling: run it next to the real system's
rac_eval.py output on the same run file to see how much of the top-k2 pool's
diversity potential the actual reranker is capturing versus leaving on the
table.

Oracle: at each step pick the pool doc covering the most subtopics not yet
covered by any already-picked doc (classic greedy maximum coverage). This is
a monotonic submodular objective, so a single greedy sequence is near-optimal
at every prefix length simultaneously -- one oracle run is truncated at every
cutoff @1..@10.

Usage:
    python -m src.evaluator.rac_eval_ub \
        --run runs/run.neuclir1.dc-gap-doc.txt \
        --qrel $HOME/trec2026/data/neuclir/neuclir24-test-request.qrel \
        --k2 100
"""
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
import argparse
from collections import defaultdict

import numpy as np
import ir_measures
from ir_measures import StRecall

try:
    from .rac_eval import load_run_or_qrel, load_diversity_qrel, print_markdown_row
except ImportError:
    from rac_eval import load_run_or_qrel, load_diversity_qrel, print_markdown_row


def build_subtopic_map(div_qrel):
    """query_id -> {doc_id: set(subtopic_ids)}, positive-relevance rows only."""
    subtopic_map = defaultdict(lambda: defaultdict(set))
    for row in div_qrel.itertuples(index=False):
        if row.relevance > 0:
            subtopic_map[row.query_id][row.doc_id].add(row.iteration)
    return subtopic_map


def greedy_strecall_oracle(pool, doc_subtopics):
    covered = set()
    remaining = list(pool)
    order = []
    while remaining:
        best_idx, best_gain = 0, -1
        for idx, docid in enumerate(remaining):
            gain = len(doc_subtopics.get(docid, set()) - covered)
            if gain > best_gain:
                best_idx, best_gain = idx, gain
        pick = remaining.pop(best_idx)
        order.append(pick)
        covered.update(doc_subtopics.get(pick, ()))
    return order


def _to_run_scores(order):
    n = len(order)
    return {docid: float(n - rank) for rank, docid in enumerate(order)}


def oracle_upper_bound(run, div_qrel):
    """run: qid -> {docid: score}, already loaded with topk=k2 (see main())."""
    subtopic_map = build_subtopic_map(div_qrel)
    metrics_used = [StRecall@k for k in range(1, 11)]

    oracle_run = {}
    for qid, docs in run.items():
        doc_subtopics = subtopic_map.get(qid, {})
        oracle_run[qid] = _to_run_scores(greedy_strecall_oracle(list(docs.keys()), doc_subtopics))

    outputs = defaultdict(list)
    for metric in ir_measures.iter_calc(metrics_used, div_qrel, oracle_run):
        outputs[str(metric.measure)].append(metric.value)
    return outputs, metrics_used


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--qrel", type=str, required=True)
    parser.add_argument("--k2", type=int, default=100, help="pool depth reranked by the oracle")
    args = parser.parse_args()

    run = load_run_or_qrel(args.run, topk=args.k2)
    div_qrel = load_diversity_qrel(args.qrel)

    missing_qids = [qid for qid in run if qid not in set(div_qrel['query_id'])]
    if missing_qids:
        run = {k: v for k, v in run.items() if k not in missing_qids}
        logger.warning(f"Missing qrel for {len(missing_qids)} topics; evaluating on {len(run)}")
    div_qrel = div_qrel[div_qrel['query_id'].isin(run.keys())]

    outputs, metrics_used = oracle_upper_bound(run, div_qrel)

    run_name = args.run.rsplit('/', 1)[-1]
    keys = [str(m) for m in metrics_used]
    columns = [f"oracle{args.k2}_{key}" for key in keys]
    values = ["{:.4f}".format(np.mean(outputs[key])) for key in keys]

    print_markdown_row(["Run"] + columns, run_name, values)
