# Run files

TREC-format run files (`qid Q0 docid rank score tag`) for the two test collections:

| dataset token | collection | topics |
|---|---|---|
| `ragtime1` | RAGTime 2025 | `data/ragtime2025.topics.test.jsonl` |
| `neuclir1` | NeuCLIR 2024 | `data/neuclir2024.topics.test.jsonl` |

Two dense encoders appear in almost every family (the `<model>` token):

| model token | encoder |
|---|---|
| `Qwen3-Embedding-0.6B` | Qwen/Qwen3-Embedding-0.6B |
| `modernbert-base.cover-5k` | DylanJHJ/modernbert-base.cover-5k |

Every family below is complete for both datasets x both models unless noted.
Scripts live under `scripts/dense-retrieve/<model>/` (shown for Qwen3; the ModernBERT folder mirrors it).

## Naming convention

```
run.<dataset>.<unit>.<model>[.<method>[.<hyperparams>]].txt
```

- `<unit>`: `documents` (doc-level retrieval / reranking of doc runs), `claims` / `claims-k<K>` (claim-level
  retrieval aggregated to docs), `concat-claims` (BM25 over each doc's claims concatenated), `hybrid-claim-doc`.
- Hyperparameters are dot-separated `name-value` tokens (`alpha-0.5`, `lambda-0.6`).

## Layout

```
runs/
├── run.<dataset>.documents.<model>.txt         first-stage dense doc retrieval (base runs)
├── run.<dataset>.documents.bm25.txt            first-stage BM25 doc retrieval
├── run.<dataset>.concat-claims.bm25.txt        BM25 over concatenated claims
├── claims-agg/                                 claim-level dense retrieval -> doc runs
│   └── backup/                                 older claims-agg runs (superseded)
├── hybrid/                                     claim-level + doc-level score fusion
├── claim-based-scoring/                        cc-dense / dd-dense (MMR-style) rerankers of the top-100
├── cckmeans/                                   cc-kmeans (cluster-coverage) reranker of the top-100
├── relrerank/                                  LLM relevance rerankers of the top-100
└── sanity-relevant-only/
    ├── base/                                   base runs filtered to qrel-relevant docs only
    └── reranked/                               cc-kmeans applied to those relevant-only pools
```

## 1. First-stage runs (top level, 8 files)

| file | produced by |
|---|---|
| `run.<dataset>.documents.<model>.txt` | `pipeline/run_dense.py`, `search-<dataset>-docs.sh` (k = 1000) |
| `run.<dataset>.documents.bm25.txt` | `scripts/retrieve/<dataset>-documents.sh` |
| `run.<dataset>.concat-claims.bm25.txt` | `scripts/retrieve/<dataset>-concat-claims.sh` (index: `concat-claims.bm25s`) |

The dense `documents` runs are the **base runs** that every reranker below consumes (reranked pool = their top-100).

## 2. `claims-agg/` -- claim retrieval aggregated to documents (68 files + 44 in `backup/`)

Dense retrieval over the claim index, then claim scores fused into doc scores
(`pipeline/run_dense.py --fusion`, script `search-<dataset>-claims.sh`).

| pattern | values |
|---|---|
| `run.<dataset>.claims-k<K>.<model>.<fusion>.txt` | K in {100, 200, 300, 400, 500, 750, 1000, 2000}; fusion in {`sum`, `rrf`} |
| `run.<dataset>.claims.<model>.max.txt` | fusion `max` (doc score = best claim), K = 200 |

`K` = number of claims retrieved per query before aggregating. `backup/` holds an earlier version
(K in {100, 200, 500, 750, 1000, 1500, 2000}; not every K for every model) and is not used for reporting.

## 3. `hybrid/` -- claim + doc fusion (20 files)

`run.<dataset>.hybrid-claim-doc.<model>.alpha-<A>.txt`, A in {0.1, 0.2, 0.3, 0.4, 0.5}.
`pipeline/run_hybrid_dense.py` (`search-<dataset>-hybrid.sh`): score = A x claim leg (claims-k1000, `sum` fusion)
+ (1 - A) x doc leg (top-1000), after normalization.

## 4. `claim-based-scoring/` -- greedy similarity rerankers (176 files)

Rerank the top-100 of the base run, blending base relevance with a similarity-to-selected term
(`--lambda-mult` = weight on base relevance). Scripts: `divrerank/search-<dataset>-{cc,dd}-dense.sh`,
plus the Slurm-array sweep `modernbert-base.cover-5k/divrerank/sweep-dense-cc-dd.sh`.

| pattern | method |
|---|---|
| `run.<dataset>.documents.<model>.ccmaxsim-<mode>.lambda-<L>.txt` | `run_cc_dense.py --agg maxsim`: claim-claim MaxSim to already-selected docs |
| `run.<dataset>.documents.<model>.dd-<mode>.lambda-<L>.txt` | `run_dd_dense.py`: doc-doc embedding similarity to already-selected docs |

- `<mode>`: `subtract` = MMR (penalize overlap), `add` = treat overlap as corroborating evidence (boost).
- L in {0.0, 0.1, ..., 1.0} (11 values). L = 1.0 reproduces the base ranking.
- 2 methods x 2 modes x 11 lambdas x 2 datasets x 2 models = 176.

## 5. `cckmeans/` -- cluster-coverage reranker (216 files)

`run.<dataset>.documents.<model>.cckmeans.top<M>-k<C>-<label>.alpha-<A>.lambda-<L>.txt`

`pipeline/run_cc_kmeans.py`, script `divrerank/search-<dataset>-cc-kmeans.sh`. Claims of the top-M docs are
clustered with k-means into C clusters (the rest of the top-100 is assigned by `.predict()`); documents are
then greedily selected to maximize `L x base relevance + (1 - L) x alpha-nDCG-style cluster-coverage gain`.

| token | meaning | values in this folder |
|---|---|---|
| `top<M>` | docs used to fit k-means (`--kmeans-top-m`) | 20 |
| `k<C>` | number of clusters (`--kmeans-n-clusters`) | 50, 75, 100 |
| `<label>` | doc-cluster membership (`--kmeans-label-mode`) | `binary` |
| `alpha-<A>` | novelty discount `(1-A)^times_covered` | 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 |
| `lambda-<L>` | weight on base relevance (`--lambda-mult`) | 0.5, 0.6, 0.7 |

3 k x 6 alpha x 3 lambda = 54 per dataset/model; x 2 x 2 = 216.

## 6. `relrerank/` -- LLM relevance reranking (16 files)

`run.<dataset>.documents.<model>.autollmrerank-70b-<method>.txt`

`pipeline/run_rerank_llm.py` (APRIL `ModularReranker`) with Llama-3.3-70B-Instruct over the top-100,
script `scripts/reranknig/<dataset>-rerank-llm.sh`. `<method>` in {`point`, `rankgpt`, `umbrela`, `lancer`}.

## 7. `sanity-relevant-only/` -- oracle pool sanity check (4 + 216 files)

Tests whether a reranker can order an all-relevant pool better than the base ranking.

| folder | pattern | produced by |
|---|---|---|
| `base/` | `run.<dataset>.documents.<model>.relevant-only.txt` | `pipeline/filter_relevant_run.py` via `scripts/dense-retrieve/create_sanity_runs.sh`: base run with non-relevant docs removed (original order kept) |
| `reranked/` | `run.<dataset>.documents.<model>.relevant-only.cckmeans.top<M>-k<C>-<label>.alpha-<A>.lambda-<L>.txt` | `divrerank/search-<dataset>-cc-kmeans-sanity.sh`, same grid as `cckmeans/` (reranks top-20) |

## Evaluation caveat

Only topics with qrels are evaluated (34 on RAGTime1), so small differences are noise. Compare runs with
`python -m src.evaluator.sanity_vs_base` (paired bootstrap, Holm; `scripts/run_sanity_vs_base.sh`) and select
configurations on half the topics / confirm on the other half rather than taking the best row on all topics. The earlier `cckmeansw` (weighted / staged k-means) search log that used to live here is in git history.
