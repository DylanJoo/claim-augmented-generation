"""
Does a reranked run beat its base run? -- paired significance test.

Built for the relevant-only sanity check (runs/sanity-relevant-only/, see
pipeline/filter_relevant_run.py): every doc in those pools is already judged
relevant, so any difference between a reranked run and the base run comes from
the order the reranker picks, not from relevance. It works on any base/run pair
though.

Per topic, the chosen metric is computed for the base run and for each
candidate; the per-topic differences are tested with
  - a paired bootstrap 95% CI of the mean difference,
  - a paired sign-flip permutation test (two-sided p), Holm-corrected across all
    candidates in the call (you are usually scanning many runs and keeping the
    best, which inflates false positives otherwise),
  - wins / losses / ties over topics.
Verdict: BEATS = Holm p < --alpha and mean diff > 0; WORSE = same, diff < 0;
otherwise n.s. Only topics present in the base AND the candidate run (and the
qrel) are used; each candidate is compared on the topics it shares with base.

Usage (from repo root):
    python -m src.evaluator.sanity_vs_base \
        --base runs/sanity-relevant-only/base/run.ragtime1.documents.Qwen3-Embedding-0.6B.relevant-only.txt \
        --runs "runs/sanity-relevant-only/reranked/run.ragtime1*.txt" \
        --qrel $HOME/trec2026/data/ragtime1/ragtime25-test-request.qrel \
        [--metric alpha_nDCG@10] [--alpha 0.05] [--n-boot 10000]
"""
import argparse
import glob
import logging
import os

import ir_measures
import numpy as np
from ir_measures import alpha_nDCG, StRecall

from src.evaluator.rac_eval import load_diversity_qrel, load_run_or_qrel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

METRICS = {"alpha_nDCG@10": alpha_nDCG @ 10, "alpha_nDCG@20": alpha_nDCG @ 20,
           "StRecall@10": StRecall @ 10, "StRecall@20": StRecall @ 20}


def per_topic(run_path, div_qrel):
    """{measure_name: {qid: value}} for all four metrics."""
    run = load_run_or_qrel(run_path, topk=1000)
    qrel = div_qrel[div_qrel["query_id"].isin(run.keys())]
    out = {name: {} for name in METRICS}
    by_measure = {str(m): name for name, m in METRICS.items()}
    for m in ir_measures.iter_calc(list(METRICS.values()), qrel, run):
        out[by_measure[str(m.measure)]][m.query_id] = m.value
    return out


def paired_test(diff, n_boot, rng):
    """diff: per-topic (candidate - base). Returns (mean, ci_lo, ci_hi, p)."""
    n = len(diff)
    obs = diff.mean()
    boots = diff[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    signs = rng.choice([-1.0, 1.0], size=(n_boot, n))
    perm = (signs * diff).mean(axis=1)
    p = (np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (n_boot + 1)
    return obs, lo, hi, p


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="run files or globs to compare against --base")
    ap.add_argument("--qrel", required=True)
    ap.add_argument("--metric", default="alpha_nDCG@10", choices=list(METRICS))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    div_qrel = load_diversity_qrel(args.qrel)
    paths = sorted({p for pat in args.runs for p in glob.glob(pat) if p.endswith(".txt")} - {args.base})  # skip .stagelog/.tmp
    if not paths:
        raise SystemExit(f"no runs matched {args.runs}")
    base = per_topic(args.base, div_qrel)
    rng = np.random.default_rng(args.seed)

    rows = []
    for path in paths:
        cand = per_topic(path, div_qrel)
        qids = sorted(set(base[args.metric]) & set(cand[args.metric]))
        if len(qids) < 5:
            logger.warning("skipping %s: only %d shared topic(s)", path, len(qids))
            continue
        diff = np.array([cand[args.metric][q] - base[args.metric][q] for q in qids])
        mean, lo, hi, p = paired_test(diff, args.n_boot, rng)
        rows.append(dict(name=os.path.basename(path), n=len(qids), mean=mean, lo=lo, hi=hi, p=p,
                         wins=int((diff > 1e-9).sum()), losses=int((diff < -1e-9).sum()),
                         cand={k: float(np.mean([cand[k][q] for q in qids])) for k in METRICS},
                         base={k: float(np.mean([base[k][q] for q in qids])) for k in METRICS}))

    padj = holm(np.array([r["p"] for r in rows]))
    for r, pa in zip(rows, padj):
        r["p_holm"] = pa
        r["verdict"] = ("BEATS" if r["mean"] > 0 else "WORSE") if pa < args.alpha else "n.s."
    rows.sort(key=lambda r: -r["mean"])

    b = rows[0]["base"]
    print(f"\nbase: {os.path.basename(args.base)}")
    print(f"      " + "  ".join(f"{k}={b[k]:.4f}" for k in METRICS) + f"   (on the first candidate's {rows[0]['n']} topics)")
    print(f"\n{args.metric}: candidate - base, paired over topics; {len(rows)} candidate(s), Holm-corrected p, alpha={args.alpha}\n")
    print(f"| Run | n | {args.metric} | diff | 95% CI | p (Holm) | W/L | verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['name']} | {r['n']} | {r['cand'][args.metric]:.4f} | {r['mean']:+.4f} | "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['p_holm']:.3f} | {r['wins']}/{r['losses']} | {r['verdict']} |")
    print("\nother metrics (candidate means):")
    print("| Run | " + " | ".join(METRICS) + " |")
    print("|---|" + "---|" * len(METRICS))
    for r in rows:
        print(f"| {r['name']} | " + " | ".join(f"{r['cand'][k]:.4f}" for k in METRICS) + " |")
    n_beat = sum(r["verdict"] == "BEATS" for r in rows)
    print(f"\n{n_beat}/{len(rows)} candidate(s) significantly beat the base run.")


if __name__ == "__main__":
    main()
