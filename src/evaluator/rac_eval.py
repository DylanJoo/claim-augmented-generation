import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
import json
import argparse
import numpy as np
from collections import defaultdict
import ir_measures
from ir_measures import Metric, MAP, nDCG, P, alpha_nDCG, StRecall
import pandas as pd

EVAL_CUTOFFS = (10, 20)

def eval_metrics():
    return [StRecall@k for k in EVAL_CUTOFFS] + [alpha_nDCG@k for k in EVAL_CUTOFFS]

def load_run_or_qrel(path, topk=1000, threshold=1):
    run_dict = defaultdict(dict)
    with open(path, "r") as f:
        for line in f:
            try:
                qid, _, docid, rank, score, _ = line.strip().split()
                if int(rank) <= topk:
                    run_dict[qid][docid] = float(score)
            except ValueError:
                qid, iteration, docid, rel = line.strip().split()
                if int(rel) >= threshold:
                    run_dict[qid][docid] = int(rel)
    return run_dict

def load_diversity_qrel(path):
    df = pd.read_csv(path, sep=r'\s+', names=['query_id', 'iteration', 'doc_id', 'relevance'])
    df['query_id'] = df['query_id'].astype(str)
    df['doc_id'] = df['doc_id'].astype(str)
    return df

def markdown_columns(metrics):
    """Column names for metrics; a metric sharing the previous one's name is shortened to '@k'."""
    columns, prev = [], None
    for m in metrics:
        name, _, cutoff = str(m).rpartition('@')
        if not name:
            name, cutoff = str(m), None
        columns.append(f"@{cutoff}" if name == prev and cutoff else str(m))
        prev = name
    return columns

def print_markdown_header(columns):
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))

def print_markdown_row(columns, run_name, values):
    print("| " + " | ".join([run_name] + values) + " |")


def rac_eval(run, qrel, div_qrel, tau=3, metrics=None):
    outputs = defaultdict(list)

    metrics_used = metrics if metrics is not None else eval_metrics()
    for metric in ir_measures.iter_calc(metrics_used, div_qrel, run):
        outputs[str(metric.measure)].append(metric.value)

    return outputs, metrics_used


def main(metrics=None):
    """CLI shared by rac_eval variants; a variant only needs to pass its own metrics list."""
    metrics_used = metrics if metrics is not None else eval_metrics()

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, nargs="+", required=True)
    parser.add_argument("--qrel", type=str, required=True)
    parser.add_argument("--tau", type=int, default=3)
    parser.add_argument("--no_header", action="store_true")
    args = parser.parse_args()

    full_qrel = load_run_or_qrel(args.qrel, threshold=1)
    full_div_qrel = load_diversity_qrel(args.qrel)

    keys = [str(m) for m in metrics_used]
    if not args.no_header:
        print_markdown_header(["Run"] + markdown_columns(metrics_used))

    for run_path in args.run:
        run = load_run_or_qrel(run_path, topk=1000)
        qrel, div_qrel = full_qrel, full_div_qrel

        missing_qids = [qid for qid in qrel if qid not in run]
        if missing_qids:
            qrel = {k: v for k, v in qrel.items() if k in run}
            div_qrel = div_qrel[div_qrel['query_id'].isin(run.keys())]
            logger.warning(f"{run_path}: missing results for {len(missing_qids)} topics; evaluating on {len(qrel)}")

        outputs, _ = rac_eval(
            run=run,
            qrel=qrel,
            div_qrel=div_qrel,
            tau=args.tau,
            metrics=metrics_used,
        )

        run_name = run_path.rsplit('/', 1)[-1]
        values = ["{:.4f}".format(np.mean(outputs[key])) for key in keys]
        print_markdown_row(["Run"] + keys, run_name, values)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
