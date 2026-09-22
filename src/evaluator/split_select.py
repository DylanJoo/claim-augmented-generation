"""
Selection-bias check for a pool of candidate runs vs a base run.

Repeatedly: split the topics at random into a SELECT half and a HELD-OUT half,
pick the candidate with the best mean metric on SELECT, and record that pick's
(candidate - base) difference on HELD-OUT. Averaged over many splits this is an
honest estimate of "tune on some topics, then deploy on new ones" -- unlike
reporting the best run on the same topics it was chosen on.

Usage (from repo root):
    python -m src.evaluator.split_select --base runs/run.ragtime1.documents.Qwen3-Embedding-0.6B.txt \
        --runs "runs/ragtime1/run.ragtime1*cckmeansw.20-*-binary.alpha-*" --qrel <qrel> [--metric alpha_nDCG@10]
"""
import argparse
import glob
import os
from collections import Counter

import numpy as np

from src.evaluator.rac_eval import load_diversity_qrel
from src.evaluator.sanity_vs_base import METRICS, per_topic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--qrel", required=True)
    ap.add_argument("--metric", default="alpha_nDCG@10", choices=list(METRICS))
    ap.add_argument("--n-splits", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dq = load_diversity_qrel(args.qrel)
    paths = sorted({p for pat in args.runs for p in glob.glob(pat) if p.endswith(".txt")} - {args.base})
    base = per_topic(args.base, dq)[args.metric]
    cands = {os.path.basename(p): per_topic(p, dq)[args.metric] for p in paths}
    qids = sorted(set(base).intersection(*[set(c) for c in cands.values()]))
    names = list(cands)
    diff = np.array([[cands[n][q] - base[q] for q in qids] for n in names])  # [n_runs, n_topics]
    print(f"{len(names)} candidate(s), {len(qids)} topics, metric {args.metric}")

    rng = np.random.default_rng(args.seed)
    held, picks, oracle_gap = [], Counter(), []
    for _ in range(args.n_splits):
        perm = rng.permutation(len(qids))
        sel, out = perm[: len(qids) // 2], perm[len(qids) // 2:]
        best = int(np.argmax(diff[:, sel].mean(axis=1)))
        held.append(diff[best, out].mean())
        picks[names[best]] += 1
    held = np.array(held)
    print(f"\nheld-out diff of the SELECT-half winner vs base: mean {held.mean():+.4f}, "
          f"2.5-97.5% of splits [{np.percentile(held, 2.5):+.4f}, {np.percentile(held, 97.5):+.4f}], "
          f"positive in {100 * (held > 0).mean():.0f}% of splits")
    full_best = int(np.argmax(diff.mean(axis=1)))
    print(f"(for contrast, best run on ALL topics: {names[full_best]} = {diff[full_best].mean():+.4f} -- optimistic)")
    print("\nmost frequently selected:")
    for n, c in picks.most_common(5):
        print(f"  {100 * c / args.n_splits:4.0f}%  {n}")


if __name__ == "__main__":
    main()
