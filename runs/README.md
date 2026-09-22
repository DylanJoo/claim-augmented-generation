# cc-kmeans hyperparameter search log (RAGTime1, Qwen3-Embedding-0.6B claims)

Record of the hyperparameter combinations tried with `cc_kmeans_weighted`
(`src/retrieval/cc_kmeans_weighted.py`, sweeps via `pipeline/sweep_cc_kmeans_weighted.py`,
Slurm scripts `scripts/dense-retrieve/Qwen3-Embedding-0.6B/temp/search-ragtime1-cc-kmeans-weighted*.sh`).
No metric values here on purpose -- this is a map of the search space, for planning future searches.

Status of each combination:
- **kept** = run file still present, was not worse than its base run on alpha_nDCG@10.
- **removed** = run was worse than its base run on alpha_nDCG@10 and was deleted (regenerate with the sweep script if needed).

Caveat: evaluation covers only the 34 topics with qrels, so small differences are noise. Compare with
`python -m src.evaluator.sanity_vs_base` (paired bootstrap, Holm) and `python -m src.evaluator.split_select`
(select on half the topics, test on the other half), not with raw table means.

## Setup common to all runs
- Base run: `runs/run.ragtime1.documents.Qwen3-Embedding-0.6B.txt`; reranked pool = its top-100 (`--k 100`).
- Full-pool runs: `runs/ragtime1/run.ragtime1.documents.Qwen3-Embedding-0.6B.cckmeansw.<config>.alpha-A.floor-0.0.lambda-L.txt`
  (`floor-0.0` is a legacy filename token; the discount floor was removed).
- Sanity runs (pool = only qrel-relevant docs): `runs/sanity-relevant-only/reranked/...relevant-only.cckmeansw.<config>...`.
- `<config>` = `<stages>-<label mode>[-idf][.softmax-<tau>][.trig-<fraction>]`, stages written `top_m-k` joined by `_`
  (e.g. `20-50_40-100` = fit on top-20 with k=50, then re-cluster on top-40 with k=100).
- Fixed everywhere: unweighted-by-default k-means, `n_init` 10, binary labels unless stated.

## Hyperparameters explored

### 1. Core size (`top_m`) and number of clusters (`k`), single stage, unweighted
| core:k | lambda values tried | status |
|---|---|---|
| 10:37 | 0.3 0.4 0.5 0.6 0.7 | kept |
| 20:75 | 0.3 0.4 0.5 0.6 0.7 | kept |
| 20:100 | 0.3 0.4 0.5 0.6 0.7 | kept |
| 30:110 | 0.3 0.4 0.5 0.6 0.7 | kept |
| 40:150 | 0.3 0.4 0.5 0.6 0.7 | kept |
| 5:19 | 0.4 0.5 0.6 0.7 | kept for lambda 0.6 / 0.7; lambda 0.4 / 0.5 removed |
| 8:30 | 0.4 0.5 0.6 0.7 | kept |
| 10:50 (denser k) | 0.4 0.5 0.6 0.7 | kept |
| 12:45 | 0.4 0.5 0.6 0.7 | kept |
| 15:56 | 0.4 0.5 0.6 0.7 | kept |
| 15:75 (denser k) | 0.4 0.5 0.6 0.7 | kept |
| 25:93 | 0.4 0.5 0.6 0.7 | kept |
| 20:50 | 0.5 | kept |
| 100:50 (whole pool) | 0.5 | kept |
| 20:20 | 0.5 | removed |
| 100:20 (whole pool) | 0.5 | removed |

Rule of thumb behind the k values: about 3.5 claims per cluster (top-10 ~130 claims, top-20 ~260, top-30 ~390, top-40 ~510),
i.e. k about 3.7 x core. The `10:50` and `15:75` runs use a denser k of 5 x core.
The core-size family (cores 5 to 30, lambda 0.4-0.7, alpha 0.5) was swept in round 6
(`search-ragtime1-cc-kmeans-weighted6.sh`) to check whether a smaller core helps subtopic recall (StRecall).

### 2. alpha (novelty discount `(1-alpha)^covered_count`)
- alpha 0.3 and 0.7 at lambda 0.5 for cores 10:37, 20:75, 20:100, 30:110, 40:150 -- kept. All other runs use alpha 0.5.

### 3. Relevance-weighted fit (`--kmeans-rel-tau`, softmax of doc score / tau, applied to the fit core)
- 100:50 with tau 0.02, 0.05, 0.1 -- kept
- 20:50 with tau 0.005, 0.01, 0.02, 0.05 -- kept
- 20:75 with tau 0.005, 0.01, 0.02, 0.05 -- kept
- 100:50 with rank weighting `1/log2(rank+1)` (option since removed from the code) -- kept
- lambda 0.5 throughout.

### 4. Staged re-clustering (`--kmeans-stages`, from-scratch refit on a larger core when `trigger` of the clusters are touched)
| stages (top_m:k) | trigger | lambda | status |
|---|---|---|---|
| 20:50, 40:100 | 0.3 0.5 0.7 0.8 0.9 1.0 | 0.5 | kept |
| 20:40, 40:80 | 0.9 1.0 | 0.5 | kept |
| 20:50, 40:100, 80:200 | 0.9 | 0.5 | kept |
| 20:30, 40:75 / 20:30, 40:100 | 1.0 | 0.5 | kept |
| 20:20, 40:50 / 20:20, 40:100 | 1.0 | 0.5 | removed |
| 20:75, 40:150 | 0.5 | 0.3 0.5 | kept |
| 20:100, 40:150 | 0.5 | 0.3 0.5 | kept |
| 20:100, 100:100 | 0.5 | 0.3 0.5 | kept |
| 20:50, 40:100 with softmax tau 0.05 | 0.9 1.0 | 0.5 | kept |
| 20:50, 40:100:0.1 (stage-2 tau 0.1) with softmax tau 0.05 | 0.9 | 0.5 | kept |

### 5. Cluster weighting (label mode x idf reweight) on 20:75 and 20:100, alpha 0.5
- binary (plain) -- kept (lambda 0.3, 0.5)
- binary + idf -- lambda 0.3 and 0.5 kept, except `20:75 idf lambda 0.3` removed
- centroid (label weighted by query-centroid cosine) -- kept (lambda 0.3, 0.5)
- centroid + idf -- kept (lambda 0.3, 0.5)
- centroid_count was tried earlier (old runs, since deleted from the working tree), not in this search.

### 6. All-relevant sanity pool (`runs/sanity-relevant-only/`)
- 20:50, 20:75, 20:100, 20:150 unweighted, alpha 0.5, lambda 0.0 / 0.5 / 0.8.
- lambda 0.0 (pure coverage, no relevance blend) for 20:50, 20:75, 20:100 -- removed; everything else kept.

## What this suggests for future searches (qualitative)
- Dimensions that mattered: **number of clusters relative to the core's claim count** (too few clusters was the only clearly bad
  region), and **lambda in the middle range** (pure coverage, lambda 0, was the bad end). Core size 20 was preferred over 10 / 30 / 40; cores 5-8 were not useful; at small cores (10, 15) a denser k (about 5 x core) was better than 3.7 x core.
- Dimensions that looked flat within noise: alpha, softmax weighting, staged re-clustering (trigger, stage sizes), idf and
  query-centroid cluster weights. Not worth more grid points unless there is a new angle.
- Reasonable next grid points: k between 75 and 100 in finer steps; k/core ratio (3.7 vs 5 vs higher) at cores 15-25;
  lambda 0.4-0.5 in steps of 0.05; a claim-level (claim-to-query) weight instead of a doc-level one; warm-started instead of
  from-scratch staging; independent topics (NeuCLIR, more RAGTime topics) to confirm the chosen region generalizes.
- Search protocol: pick on half the topics and confirm on the other half (`split_select`) instead of picking the best row on all 34.
