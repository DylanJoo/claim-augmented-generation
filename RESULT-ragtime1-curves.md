# ragtime1: alpha_nDCG / StRecall vs. cutoff (1-20)

Companion to [RESULT-ragtime1.md](RESULT-ragtime1.md), which has the raw @1/@10/@20
numbers for a quick read. This shows the full cutoff curve for the top 10 runs
ranked by alpha_nDCG@10, so trends across the ranking (not just three fixed
points) are visible at a glance.

![ragtime1 top-10 runs by alpha_nDCG@10, cutoffs 1-20](results/plots/ragtime1.png)

Regenerate after new runs land:
```
bash scripts/run_eval_curve.sh ragtime1
python -m src.evaluator.plot_eval_curve \
    --csv results/curve-ragtime1.csv --out results/plots/ragtime1.png \
    --title "ragtime1: top 10 runs by alpha_nDCG@10"
```
