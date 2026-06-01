# claim-based-rag

This repository contains scripts for building a **claim-based retrieval pipeline** for report-generation tasks (NeuCLIR/RAGTIME style settings).

## Pipeline overview

The pipeline is implemented in `src/pipeline.py` (and variant scripts `src/pipeline_ss.py`, `src/pipeline_cheat.py`).

Main stages:

1. **Pre-retrieval (`src/pre_retrieval.py`)**
   - Expands each request into multiple standalone sub-questions with Qwen3-8B.

2. **Retrieval (`src/retrieval.py` or `src/retrieval_ss.py`)**
   - Retrieves statement-level hits from a Lucene index (Pyserini) or remote service.
   - Aggregates evidence with reciprocal-rank-style fusion.

3. **Filtering (`src/filtering.py`)**
   - Uses LLM relevance scoring to prune non-useful evidence.

4. **Post-retrieval clustering (`src/post_retrieval.py`)**
   - Embeds sub-questions and retrieved statements (Qwen3 Embedding).
   - Builds FAISS-based neighbor clusters for claim selection.

5. **Claim aggregation (`src/generate_reflection_bfs.py` / `src/generate_reflection_dfs.py`)**
   - Iteratively selects novel and insightful claims from clusters.

6. **Optional claim reranking (`src/generate_refine.py`)**
   - Reorders selected claims by relevance.

7. **Submission export (`src/utils.py`)**
   - Writes JSONL output with metadata + response claims to `submissions/`.

## Data preparation and indexing

### Decontextualize corpus

Legacy preprocessing scripts exist in `legacy/`:

- `legacy/neuclir_corpus.py`
- `legacy/ragtime_corpus.py`
- `legacy/convert_to_jsonl.py`

These scripts produce statement-level JSONL data used for indexing/retrieval.

### Build Lucene index

SLURM examples are under `slurm/`:

- `slurm/decontext_neuclir.sh`
- `slurm/decontext_ragtime.sh`
- `slurm/index_neuclir.sh`

`index_neuclir.sh` uses `pyserini.index.lucene` over flattened JSONL statements.

## Running the pipeline

The current pipeline scripts are configured as experiment runners with hard-coded paths and stage toggles (comment/uncomment blocks and intermediate `temp*.pkl` files).

Typical usage pattern:

1. Load topics JSONL.
2. Run stages 1→4 and save intermediate pickle files.
3. Run claim aggregation (stage 5).
4. Optionally run reranking/refinement.
5. Export final JSONL submission in `submissions/`.

## Outputs

- Intermediate artifacts: script-configured absolute paths such as `/exp/jhueiju/temp4.pkl` and `/exp/jhueiju/temp5.pkl`
- Final submissions: `submissions/*.jsonl`
- Optional leaderboard upload helper: `upload_leaderboard.sh`
