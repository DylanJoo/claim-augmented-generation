"""
Build one wide results table (one row per run) for paper tables/figures.

Columns:
    dataset, encoder, base_run, run,
    subtopic@1 .. subtopic@20      (StRecall, i.e. subtopic recall)
    alpha-nDCG@1 .. alpha-nDCG@20

Scoring reuses src.evaluator.rac_eval_curve, so numbers match the existing
RESULT-*.md tables at @10/@20.

Usage (from repo root, on a compute node):
    python results/build_eval_table.py --out results/eval-table.csv --workers 16

Colab side:
    df = pd.read_csv("https://raw.githubusercontent.com/DylanJoo/claim-augmented-generation/main/results/eval-table.csv")
"""
import os
import sys
import csv
import glob
import argparse
import logging
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.evaluator.rac_eval import load_run_or_qrel, load_diversity_qrel
from src.evaluator.rac_eval_curve import rac_eval_curve

logging.getLogger().setLevel(logging.WARNING)

QRELS = {
    "neuclir1": os.path.expanduser("~/trec2026/data/neuclir/neuclir24-test-request.qrel"),
    "ragtime1": os.path.expanduser("~/trec2026/data/ragtime1/ragtime25-test-request.qrel"),
}

# Searched in order; the first copy of a duplicated filename wins.
# Excluded on purpose: runs/sanity-relevant-only (pool = qrel-relevant docs only,
# inflated scores) and runs/ragtime2 (no qrels yet).
RUN_GLOBS = [
    "runs/run.*.txt",
    "runs/neuclir1/run.*.txt",
    "runs/neuclir1/claim-based-scoring/run.*.txt",
    "runs/ragtime1/run.*.txt",
    "runs/ragtime1/claim-based-scoring/run.*.txt",
    "runs/claims-aggregation/run.*.txt",
]

ENCODERS = ["Qwen3-Embedding-0.6B", "modernbert-base.cover-5k", "bm25"]
METRICS = [("StRecall", "subtopic"), ("alpha_nDCG", "alpha-nDCG")]


def parse_name(name):
    """run.<dataset>.<source>.<encoder>[.<method...>].txt -> (dataset, encoder, base_run)."""
    parts = name.split(".")
    dataset, source = parts[1], parts[2]
    encoder = next((e for e in ENCODERS if f".{e}." in name or name.endswith(f".{e}.txt")), "")
    # Reranked runs point to the first-stage document run they rerank;
    # first-stage runs (claims-k*, hybrid, concat-claims, bm25 docs) are their own base.
    if source == "documents" and encoder:
        base_run = f"run.{dataset}.documents.{encoder}.txt"
    else:
        base_run = name
    return dataset, encoder, base_run


def collect_runs():
    seen, runs = set(), []
    for pattern in RUN_GLOBS:
        for path in sorted(glob.glob(os.path.join(REPO, pattern))):
            name = os.path.basename(path)
            if name in seen or name.split(".")[1] not in QRELS:
                continue
            seen.add(name)
            runs.append(path)
    return runs


_DIV_QRELS = {}


def evaluate(path):
    name = os.path.basename(path)
    dataset, encoder, base_run = parse_name(name)
    if dataset not in _DIV_QRELS:
        _DIV_QRELS[dataset] = load_diversity_qrel(QRELS[dataset])
    div_qrel = _DIV_QRELS[dataset]

    run = load_run_or_qrel(path, topk=1000)
    div_qrel = div_qrel[div_qrel["query_id"].isin(run.keys())]
    values = {(m, k): v for m, k, v in rac_eval_curve(run, div_qrel, max_cutoff=MAX_CUTOFF)}

    row = [dataset, encoder, base_run, name]
    for metric, _ in METRICS:
        row += [f"{values[(metric, k)]:.4f}" for k in range(1, MAX_CUTOFF + 1)]
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(REPO, "results/eval-table.csv"))
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--max_cutoff", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N runs (smoke test)")
    args = parser.parse_args()
    MAX_CUTOFF = args.max_cutoff

    runs = collect_runs()[:args.limit]
    print(f"evaluating {len(runs)} runs with {args.workers} workers", file=sys.stderr)

    header = ["dataset", "encoder", "base_run", "run"]
    for _, label in METRICS:
        header += [f"{label}@{k}" for k in range(1, MAX_CUTOFF + 1)]

    with Pool(args.workers) as pool:
        rows = []
        for i, row in enumerate(pool.imap(evaluate, runs), 1):
            rows.append(row)
            if i % 50 == 0:
                print(f"  {i}/{len(runs)}", file=sys.stderr)

    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}", file=sys.stderr)
