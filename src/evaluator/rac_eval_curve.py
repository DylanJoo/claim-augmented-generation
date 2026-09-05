"""
alpha-nDCG / StRecall at every cutoff from 1 to --max_cutoff, in tidy
long-format CSV (run,metric,cutoff,value) -- one row per (run, metric,
cutoff) triple, so plotting a metric-vs-cutoff curve per run in matplotlib
is a groupby/pivot away instead of a table full of fixed @k columns.

Usage:
    python -m src.evaluator.rac_eval_curve \
        --run runs/run.ragtime1.documents.bm25.txt \
        --qrel $HOME/trec2026/data/ragtime1/ragtime25-test-request.qrel \
        --out results/curve-ragtime1.csv

Re-running with the same --out appends more rows (e.g. once per run file
in a loop); pass --overwrite to start the CSV fresh instead.

Matplotlib-side sketch once the CSV has multiple runs appended:
    import pandas as pd
    df = pd.read_csv("results/curve-ragtime1.csv")
    for run_name, g in df[df.metric == "alpha_nDCG"].groupby("run"):
        plt.plot(g.cutoff, g.value, label=run_name)
"""
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
import csv
import argparse

import ir_measures
from ir_measures import alpha_nDCG, StRecall

try:
    from .rac_eval import load_run_or_qrel, load_diversity_qrel
except ImportError:
    from rac_eval import load_run_or_qrel, load_diversity_qrel


METRIC_FACTORIES = {
    "alpha_nDCG": alpha_nDCG,
    "StRecall": StRecall,
}


def build_measures(max_cutoff, metric_names):
    measures = []
    for name in metric_names:
        factory = METRIC_FACTORIES[name]
        measures.extend(factory @ k for k in range(1, max_cutoff + 1))
    return measures


def rac_eval_curve(run, div_qrel, max_cutoff=20, metric_names=("alpha_nDCG", "StRecall")):
    """Returns rows: list of (metric_name, cutoff, value), one per cutoff per metric."""
    measures = build_measures(max_cutoff, metric_names)
    aggregated = ir_measures.calc_aggregate(measures, div_qrel, run)

    rows = []
    for measure, value in aggregated.items():
        name, cutoff = str(measure).split("@")
        rows.append((name, int(cutoff), value))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def write_rows(out_path, run_name, rows, overwrite=False):
    write_header = overwrite or not os.path.exists(out_path)
    mode = "w" if overwrite else "a"
    with open(out_path, mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["run", "metric", "cutoff", "value"])
        for metric_name, cutoff, value in rows:
            writer.writerow([run_name, metric_name, cutoff, f"{value:.6f}"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--qrel", type=str, required=True)
    parser.add_argument("--out", type=str, default=None,
                         help="CSV path to append rows to; if omitted, rows are printed to stdout as CSV")
    parser.add_argument("--overwrite", action="store_true", default=False,
                         help="Start --out fresh instead of appending")
    parser.add_argument("--max_cutoff", type=int, default=20)
    parser.add_argument("--metrics", nargs="+", choices=list(METRIC_FACTORIES.keys()),
                         default=list(METRIC_FACTORIES.keys()))
    args = parser.parse_args()

    run = load_run_or_qrel(args.run, topk=1000)
    qrel = load_run_or_qrel(args.qrel, threshold=1)
    div_qrel = load_diversity_qrel(args.qrel)

    missing_qids = [qid for qid in qrel if qid not in run]
    if missing_qids:
        div_qrel = div_qrel[div_qrel['query_id'].isin(run.keys())]
        logger.warning(f"Missing results for {len(missing_qids)} topics; evaluating on {len(qrel) - len(missing_qids)}")

    rows = rac_eval_curve(run, div_qrel, max_cutoff=args.max_cutoff, metric_names=args.metrics)

    run_name = args.run.rsplit('/', 1)[-1]
    if args.out:
        write_rows(args.out, run_name, rows, overwrite=args.overwrite)
        logger.info(f"Appended {len(rows)} rows for {run_name} to {args.out}")
    else:
        writer = csv.writer(__import__("sys").stdout)
        writer.writerow(["run", "metric", "cutoff", "value"])
        for metric_name, cutoff, value in rows:
            writer.writerow([run_name, metric_name, cutoff, f"{value:.6f}"])
