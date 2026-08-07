"""
Ground-truth-aware oracle reranking, reported as an upper-bound metric.

Given a run file's top-k2 pool per topic (k2 defaults to 100 -- deliberately
smaller than the 1000-doc pools mmr.py/cc_rerank.py/dc_gap.py rerank over, so
this reads as "how much headroom is left in the part of the pool a reranker
would realistically operate on"), this greedily reorders each query's pool
using the subtopic-level ground truth qrel directly, then scores that oracle
ordering with the same ir_measures diversity metrics rac_eval.py reports.

The result is not a method -- it cheats by reading the test qrel -- so it is
never a run you'd submit. It's a ceiling: run it next to the real system's
rac_eval.py output on the same run file to see how much of the top-100 pool's
diversity potential the actual reranker (mmr/cc_rerank/dc_gap/...) is
capturing versus leaving on the table.

There is no single oracle ordering because alpha_nDCG and StRecall reward
different things -- alpha_nDCG rank-discounts and diminishes repeat coverage
of a subtopic, StRecall just wants distinct-subtopic coverage inside a
cutoff with no discount. So two greedy sequences are built per query, one
per objective, each using that metric's own marginal-gain definition:

  alpha_nDCG oracle: at each step pick the pool doc maximizing
      sum_{s in subtopics(d)} (1 - alpha) ** covered_count[s]
  where covered_count[s] is how many already-picked docs are relevant to
  subtopic s. This is the same greedy construction ir_measures/trec_eval
  uses internally to build the ideal gain vector for alpha-nDCG's own
  normalizer (Clarke et al., 2008) -- here it's materialized as an actual
  run instead of a hidden scalar.

  StRecall oracle: at each step pick the pool doc covering the most
  subtopics not yet covered by any already-picked doc (classic greedy
  maximum coverage).

Both are monotonic submodular objectives, so a single greedy sequence is
near-optimal at every prefix length simultaneously -- one alpha_nDCG oracle
run is truncated at @10/@20, one StRecall oracle run at @1/@10/@20, rather
than building a separate oracle per cutoff.

Usage:
    python -m src.evaluator.rac_eval_ub \
        --run runs/run.neuclir1.dc-gap-doc.txt \
        --qrel $HOME/trec2026/data/neuclir/neuclir24-test-request.qrel \
        --k2 100 [--alpha 0.5]
"""
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
import argparse
from collections import defaultdict

import numpy as np
import ir_measures
from ir_measures import alpha_nDCG, StRecall

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


def greedy_alpha_ndcg_oracle(pool, doc_subtopics, alpha=0.5):
    covered_count = defaultdict(int)
    remaining = list(pool)
    order = []
    while remaining:
        best_idx, best_gain = 0, -1.0
        for idx, docid in enumerate(remaining):
            gain = sum((1 - alpha) ** covered_count[s] for s in doc_subtopics.get(docid, ()))
            if gain > best_gain:
                best_idx, best_gain = idx, gain
        pick = remaining.pop(best_idx)
        order.append(pick)
        for s in doc_subtopics.get(pick, ()):
            covered_count[s] += 1
    return order


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


def oracle_upper_bound(run, div_qrel, k2=100, alpha=0.5):
    """run: qid -> {docid: score}, already loaded with topk=k2 (see main())."""
    subtopic_map = build_subtopic_map(div_qrel)
    alpha_measures = [alpha_nDCG(alpha=alpha)@10, alpha_nDCG(alpha=alpha)@20]
    strecall_measures = [StRecall@1, StRecall@10, StRecall@20]

    alpha_run, strecall_run = {}, {}
    for qid, docs in run.items():
        pool = list(docs.keys())
        doc_subtopics = subtopic_map.get(qid, {})
        alpha_run[qid] = _to_run_scores(greedy_alpha_ndcg_oracle(pool, doc_subtopics, alpha))
        strecall_run[qid] = _to_run_scores(greedy_strecall_oracle(pool, doc_subtopics))

    outputs = defaultdict(list)
    for metric in ir_measures.iter_calc(alpha_measures, div_qrel, alpha_run):
        outputs[str(metric.measure)].append(metric.value)
    for metric in ir_measures.iter_calc(strecall_measures, div_qrel, strecall_run):
        outputs[str(metric.measure)].append(metric.value)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--qrel", type=str, required=True)
    parser.add_argument("--k2", type=int, default=100, help="pool depth reranked by the oracle")
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha_nDCG novelty discount")
    args = parser.parse_args()

    run = load_run_or_qrel(args.run, topk=args.k2)
    div_qrel = load_diversity_qrel(args.qrel)

    missing_qids = [qid for qid in run if qid not in set(div_qrel['query_id'])]
    if missing_qids:
        run = {k: v for k, v in run.items() if k not in missing_qids}
        logger.warning(f"Missing qrel for {len(missing_qids)} topics; evaluating on {len(run)}")
    div_qrel = div_qrel[div_qrel['query_id'].isin(run.keys())]

    outputs = oracle_upper_bound(run, div_qrel, k2=args.k2, alpha=args.alpha)

    run_name = args.run.rsplit('/', 1)[-1]
    keys = sorted(outputs.keys())
    columns = [f"oracle{args.k2}_{key}" for key in keys]
    values = ["{:.4f}".format(np.mean(outputs[key])) for key in keys]

    print_markdown_row(["Run"] + columns, run_name, values)
