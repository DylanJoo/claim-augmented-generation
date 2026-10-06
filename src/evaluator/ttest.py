"""
Paired t-test between base and target runs, per metric, over topics.

Per topic, every metric is computed for both runs with ir_measures; the
per-topic scores are compared with scipy.stats.ttest_rel (two-sided). Only
topics judged in the qrel AND present in both runs are used.

Edit the settings in the `if __name__ == "__main__":` block at the bottom, then
    python src/evaluator/ttest.py
"""
import os

import ir_measures
import numpy as np
import pandas as pd
from scipy import stats

try:
    from .rac_eval import load_diversity_qrel, load_run_or_qrel
except ImportError:
    from rac_eval import load_diversity_qrel, load_run_or_qrel

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DEFAULT_METRICS = ["alpha_nDCG@10", "alpha_nDCG@20", "StRecall@10", "StRecall@20"]

QRELS = {
    "neuclir1": "~/trec2026/data/neuclir/neuclir24-test-request.qrel",
    "ragtime1": "~/trec2026/data/ragtime1/ragtime25-test-request.qrel",
}

def _measures(metrics):
    """Accept ir_measures objects or strings like 'alpha_nDCG@10'."""
    return [ir_measures.parse_measure(m) if isinstance(m, str) else m for m in metrics]


def per_topic_scores(run, qrel, metrics=DEFAULT_METRICS, topk=1000):
    """DataFrame indexed by query_id, one column per metric.

    `run` is a path or a {qid: {docid: score}} dict; `qrel` is a path, a
    collection key from QRELS, or an already-loaded qrel DataFrame.
    """
    if isinstance(run, str):
        run = load_run_or_qrel(os.path.join(REPO, os.path.expanduser(run)), topk=topk)
    if isinstance(qrel, str):
        qrel = load_diversity_qrel(os.path.expanduser(QRELS.get(qrel, qrel)))
    qrel = qrel[qrel["query_id"].isin(run.keys())]

    measures = _measures(metrics)
    rows = {}
    for m in ir_measures.iter_calc(measures, qrel, run):
        rows.setdefault(m.query_id, {})[str(m.measure)] = m.value
    return pd.DataFrame.from_dict(rows, orient="index")[[str(m) for m in measures]].sort_index()


def paired_ttest(run_a, run_b, qrel, metrics=DEFAULT_METRICS, alpha=0.05, topk=1000):
    """Paired two-sided t-test of run_b vs run_a for each metric.

    Returns a DataFrame (one row per metric) with mean A, mean B, diff (B - A),
    t, p, wins/losses/ties of B over A, and n topics. `sig` marks p < alpha.
    """
    if isinstance(qrel, str):
        qrel = load_diversity_qrel(os.path.expanduser(QRELS.get(qrel, qrel)))
    a = per_topic_scores(run_a, qrel, metrics, topk)
    b = per_topic_scores(run_b, qrel, metrics, topk)
    common = a.index.intersection(b.index)
    dropped = len(a.index.union(b.index)) - len(common)
    if dropped:
        print(f"[paired_ttest] {dropped} topic(s) not in both runs, using {len(common)} shared topics")
    a, b = a.loc[common], b.loc[common]

    out = []
    for col in a.columns:
        x, y = a[col].to_numpy(), b[col].to_numpy()
        diff = y - x
        t, p = stats.ttest_rel(y, x) if np.any(diff != 0) else (0.0, 1.0)
        out.append({
            "metric": col,
            "A": x.mean(),
            "B": y.mean(),
            "diff": diff.mean(),
            "t": t,
            "p": p,
            "sig": p < alpha,
            "win": int((diff > 0).sum()),
            "loss": int((diff < 0).sum()),
            "tie": int((diff == 0).sum()),
            "n": len(diff),
        })
    return pd.DataFrame(out).set_index("metric")


if __name__ == "__main__":
    # ---- settings ------------------------------------------------------------
    SYSTEMS = ["neuclir1", "ragtime1"]
    METRICS = ["alpha_nDCG@10", "alpha_nDCG@20", "StRecall@10", "StRecall@20"]
    ALPHA = 0.05

    # paths relative to repo root; {sys} is filled with each entry of SYSTEMS
    DICE_CFG = "cckmeans.top20-k50-binary.alpha-0.5.lambda-0.6"
    RUNS = {
        "qwen3-doc": "runs/run.{sys}.documents.Qwen3-Embedding-0.6B.txt",
        "modernbert-doc": "runs/run.{sys}.documents.modernbert-base.cover-5k.txt",
        "qwen3-hybrid": "runs/hybrid/run.{sys}.hybrid-claim-rrf-doc.Qwen3-Embedding-0.6B.alpha-0.8.txt",
        "modernbert-hybrid": "runs/hybrid/run.{sys}.hybrid-claim-rrf-doc.modernbert-base.cover-5k.alpha-0.8.txt",
        "qwen3-cckmeans": f"runs/cckmeans/run.{{sys}}.documents.Qwen3-Embedding-0.6B.{DICE_CFG}.txt",
        "modernbert-cckmeans": f"runs/cckmeans/run.{{sys}}.documents.modernbert-base.cover-5k.{DICE_CFG}.txt",
        "qwen3-lancer": "runs/relrerank/run.{sys}.documents.Qwen3-Embedding-0.6B.autollmrerank-70b-lancer.txt",
        "modernbert-lancer": "runs/relrerank/run.{sys}.documents.modernbert-base.cover-5k.autollmrerank-70b-lancer.txt",
        "qwen3-rankgpt": "runs/relrerank/run.{sys}.documents.Qwen3-Embedding-0.6B.autollmrerank-70b-rankgpt.txt",
        "modernbert-rankgpt": "runs/relrerank/run.{sys}.documents.modernbert-base.cover-5k.autollmrerank-70b-rankgpt.txt",
        "qwen3-dice": f"runs/dice/run.{{sys}}.hybrid-a0.8.Qwen3-Embedding-0.6B.{DICE_CFG}.txt",
        "modernbert-dice": f"runs/dice/run.{{sys}}.hybrid-a0.8.modernbert-base.cover-5k.{DICE_CFG}.txt",
    }
    # (base, target) pairs to test; names are keys of RUNS
    PAIRS = [
        ("qwen3-doc", "qwen3-hybrid"),
        ("qwen3-doc", "qwen3-cckmeans"),
        ("qwen3-doc", "qwen3-dice"),
        ("qwen3-hybrid", "qwen3-cckmeans"),
        ("qwen3-hybrid", "qwen3-dice"),
        ("qwen3-cckmeans", "qwen3-dice"),
        ("qwen3-lancer", "qwen3-dice"),
        ("qwen3-rankgpt", "qwen3-dice"),
        ("modernbert-doc", "modernbert-hybrid"),
        ("modernbert-doc", "modernbert-cckmeans"),
        ("modernbert-doc", "modernbert-dice"),
        ("modernbert-hybrid", "modernbert-cckmeans"),
        ("modernbert-hybrid", "modernbert-dice"),
        ("modernbert-cckmeans", "modernbert-dice"),
        ("modernbert-lancer", "modernbert-dice"),
        ("modernbert-rankgpt", "modernbert-dice"),
    ]
    # --------------------------------------------------------------------------

    dfs = []
    for sys in SYSTEMS:
        qrel = load_diversity_qrel(os.path.expanduser(QRELS[sys]))
        for base, target in PAIRS:
            df = paired_ttest(RUNS[base].format(sys=sys), RUNS[target].format(sys=sys),
                              qrel, METRICS, ALPHA)
            dfs.append(df.assign(sys=sys, base=base, target=target))
    result = pd.concat(dfs).reset_index().set_index(["sys", "base", "target", "metric"])

    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 250,
                           "display.max_columns", None, "display.max_rows", None):
        print(result)
