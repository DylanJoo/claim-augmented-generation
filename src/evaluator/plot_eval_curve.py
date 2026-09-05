"""
Render the alpha_nDCG / StRecall vs. cutoff curves from rac_eval_curve.py's
CSV output as PNGs, so results are easy to skim on GitHub instead of reading
raw numbers.

Only the top-N runs (ranked by a chosen metric at a chosen cutoff -- default
alpha_nDCG@10) are plotted; with 30-40 runs per system the full legend is
unreadable.

Usage:
    python -m src.evaluator.plot_eval_curve \
        --csv results/curve-ragtime1.csv \
        --out results/plots/ragtime1.png \
        --top_n 10 --rank_metric alpha_nDCG --rank_cutoff 10
"""
import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def top_runs(df, rank_metric, rank_cutoff, top_n):
    ranking = df[(df.metric == rank_metric) & (df.cutoff == rank_cutoff)]
    ranking = ranking.drop_duplicates(subset="run").sort_values("value", ascending=False)
    return ranking["run"].head(top_n).tolist()


def plot_curves(df, runs, title, out_path):
    metrics = sorted(df.metric.unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5), squeeze=False)
    axes = axes[0]

    colors = plt.cm.tab10.colors + plt.cm.tab20.colors
    run_color = {run: colors[i % len(colors)] for i, run in enumerate(runs)}

    for ax, metric in zip(axes, metrics):
        sub = df[df.metric == metric]
        for run in runs:
            g = sub[sub.run == run].sort_values("cutoff")
            ax.plot(g.cutoff, g.value, marker="o", markersize=3,
                     label=run, color=run_color[run])
        ax.set_xlabel("cutoff")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.set_xticks(range(1, df.cutoff.max() + 1, 2))
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.15 - 0.03 * len(runs)))
    fig.suptitle(title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--rank_metric", type=str, default="alpha_nDCG")
    parser.add_argument("--rank_cutoff", type=int, default=10)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv).drop_duplicates(subset=["run", "metric", "cutoff"])
    runs = top_runs(df, args.rank_metric, args.rank_cutoff, args.top_n)
    title = args.title or f"{os.path.basename(args.csv)} -- top {args.top_n} by {args.rank_metric}@{args.rank_cutoff}"
    plot_curves(df, runs, title, args.out)
