# claim-based-rag

This repository contains scripts for building a **claim-based retrieval pipeline** for report-generation tasks (NeuCLIR/RAGTIME style settings).

## Repository layout

```
pipeline/          # entry-point scripts
  run.py           # standard pipeline (local BM25)
  run_ss.py        # remote RRF-fusion retrieval variant
  run_oracle.py    # oracle/ceiling variant (qrel-based filtering)

src/               # pipeline stage modules
  query_expansion.py      # stage 1 – sub-question generation
  retrieval/
    bm25.py               # stage 2a – Pyserini/Lucene BM25
    remote.py             # stage 2b – remote RRF-fusion HTTP service
  filtering/
    llm.py                # stage 3a – LLM relevance scoring
    oracle.py             # stage 3b – ground-truth (qrel) oracle filter
  clustering.py           # stage 4 – FAISS clustering
  generation/
    reflect_bfs.py        # stage 5a – BFS iterative claim selection
    reflect_dfs.py        # stage 5b – DFS iterative claim selection
    refine.py             # stage 6 – optional claim reranking
    greedy.py             # greedy baseline generation
  llm/
    vllm.py               # local vLLM backend
    litellm.py            # LiteLLM backend (remote/API models)
  utils.py                # Result dataclass + submission export helpers

data_prep/         # corpus preprocessing
  decontext/
    neuclir.py            # decontextualize NeuCLIR corpus
    ragtime.py            # decontextualize RAGTIME corpus
    biogen.py             # decontextualize BioGen corpus
    prompts.py            # shared LLM prompts for decontextualization
    utils.py
  index/
    flatten.py            # flatten statement-level JSONL for Lucene indexing
  merge_offload.py        # merge offloaded LLM batch outputs

slurm/             # SLURM job scripts
  decontext_neuclir.sh
  decontext_ragtime.sh
  decontext_biogen.sh
  index_neuclir.sh

analysis/          # evaluation and leaderboard utilities
  report_analysis.py
  upload_leaderboard.sh

legacy/            # deprecated helpers (kept for reference)
  litellm_api.py
  vllm_api.py
  post_retrieval_wo_pooling.py
  prompt_unused.py

submissions/       # JSONL output files (one per experiment run)
```

## Pipeline overview

The pipeline runs in six stages. Entry points are the scripts in `pipeline/`.

### Stage 1 — Query expansion (`src/query_expansion.py`)

Expands each report request into `n_questions` standalone sub-questions using **Qwen3-8B** (via vLLM). Each sub-question is tagged with `<q>…</q>` and parsed into a `Result` object.

### Stage 2 — Retrieval

Two backends are available:

- **`src/retrieval/bm25.py`** — Pyserini `LuceneSearcher` over the flattened statement index. Queries are the concatenation of the problem statement and each sub-question. Per-sub-question hits are fused with reciprocal-rank fusion (RRF).
- **`src/retrieval/remote.py`** — Sends queries to a remote RRF-fusion HTTP service (`rrf(plaidx+lsr+qwen)-neuclir`). Used by `pipeline/run_ss.py`.

### Stage 3 — Filtering

- **`src/filtering/llm.py`** — Scores each retrieved document with **Qwen3-8B** (Yes/No log-prob). Documents below `rel_threshold` are pruned.
- **`src/filtering/oracle.py`** — Keeps only documents present in the ground-truth qrels. Used by `pipeline/run_oracle.py` for ceiling analysis.

### Stage 4 — Clustering (`src/clustering.py`)

Embeds sub-questions and retrieved statements with **Qwen3-Embedding-8B** (via vLLM). Runs k-means (`n_centroids`) on statement embeddings, then FAISS nearest-neighbor search (`n_neighbors`) per centroid. Supports:

- `allsep` — include all statements from a retrieved document (not just the hit statement)
- `addadjacent` — concatenate adjacent statements
- `instruct` — prepend the problem statement as an instruction prefix

### Stage 5 — Generation

Iteratively selects novel and insightful claims from clusters:

- **`src/generation/reflect_bfs.py`** — BFS order: visits each cluster once per sweep, repeats until `max_char_limit` is reached.
- **`src/generation/reflect_dfs.py`** — DFS order: exhausts one cluster before moving to the next.

At each step the LLM compares a candidate claim against the already-collected set and picks the most novel and insightful item.

### Stage 6 — Refinement (optional, `src/generation/refine.py`)

Reorders the collected claims by relevance before final export.

### Export (`src/utils.py`)

`convert_result_to_submission()` writes a JSONL file to `submissions/` with metadata (`run_id`, `topic_id`, `team_id`, `task`) and the ordered response claims.

## Data preparation and indexing

### 1. Decontextualize corpus

Scripts in `data_prep/decontext/` split each document into standalone atomic statements using an LLM. Supported corpora:

| Script | Corpus |
|---|---|
| `data_prep/decontext/neuclir.py` | NeuCLIR (zho/rus/fas, MT or original) |
| `data_prep/decontext/ragtime.py` | RAGTIME |
| `data_prep/decontext/biogen.py` | BioGen |

SLURM wrappers: `slurm/decontext_neuclir.sh`, `slurm/decontext_ragtime.sh`, `slurm/decontext_biogen.sh`.

The `--offload` flag writes LLM inputs to disk for batch API processing; `data_prep/merge_offload.py` merges the returned outputs back.

### 2. Flatten for indexing (`data_prep/index/flatten.py`)

Converts statement-level JSONL (one document → many statements) into a flat JSONL where each row is a single statement with id `{docid}#{i}`.

### 3. Build Lucene index

`slurm/index_neuclir.sh` calls `pyserini.index.lucene` over the flattened JSONL produced by the previous step.

## Running the pipeline

Each `pipeline/run*.py` script has a **Hyperparameters** block at the top. Stage code blocks are commented/uncommented to resume from a saved checkpoint (`temp*.pkl`).

Typical workflow:

```
# 1. Edit hyperparameters in pipeline/run.py
# 2. Uncomment stages 1–4, run to build checkpoints
python pipeline/run.py

# 3. Comment out stages 1–4, uncomment stage 5 and resume
python pipeline/run.py

# 4. (Optional) uncomment stage 6 to rerank
python pipeline/run.py
```

Intermediate checkpoints are saved as pickle files at paths hard-coded in the script (e.g. `/exp/jhueiju/temp*.pkl`). Final submissions land in `submissions/`.

## Key hyperparameters

| Parameter | Description |
|---|---|
| `n_questions` | Number of sub-questions per topic |
| `k` | Candidates retrieved per sub-question |
| `rel_threshold` | LLM relevance score cutoff (0–1) |
| `n_centroids` | k-means clusters for the claim pool |
| `n_neighbors` | Nearest neighbors per centroid |
| `sim_threshold` | FAISS similarity cutoff for neighbor validity |
| `encode_type` | Embedding mode: `allsep`, `addadjacent`, `instruct-*` |
| `max_char_limit` | Character budget for the final response |

## Outputs

- Intermediate checkpoints: `temp*.pkl` at script-configured external paths
- Final submissions: `submissions/decontext-{version}.{encode_type}.*.jsonl`
- Leaderboard upload: `analysis/upload_leaderboard.sh`
