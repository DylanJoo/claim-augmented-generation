#!/bin/bash
#SBATCH --job-name=search-ragtime1-dc-gap-grid
#SBATCH --output=logs/%x-%a.out
#SBATCH --error=logs/%x-%a.err
#SBATCH --partition=small
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --account=project_465002532
#SBATCH --array=0-12

# ENV
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

cd $HOME/claim-augmented-generation

# NOTE: doc_reps/claim_reps point at the Qwen3-Embedding-0.6B shards built by
# scripts/dense-index/Qwen3-Embedding-0.6B/ragtime1-encode-{docs,claims}.sh.
# dc_gap_dense.py filters each shard down to just the docids pooled by
# run-file as it streams them in, so peak memory stays bounded by one raw
# shard rather than the full claims_emb corpus -- see
# src/retrieval/dc_gap_dense.py's _load_reps.
MODEL_NAME=Qwen3-Embedding-0.6B
EMB_ROOT=$HOME/scratch/ragtime1/${MODEL_NAME}

# Full grid over --claim-filter and its mode-specific params, see
# src/retrieval/dc_gap_dense.py's _doc_to_claim_scores docstring. Each array
# task runs one combo. Index -> filter:claims_per_doc:scale:threshold;
# claims_per_doc/scale only apply to "topn", threshold only to "threshold".
COMBOS=(
  "none:0:0:0.0"
  "topn:5:0:0.0"
  "topn:5:1:0.0"
  "topn:10:0:0.0"
  "topn:10:1:0.0"
  "topn:50:0:0.0"
  "topn:50:1:0.0"
  "topn:100:0:0.0"
  "topn:100:1:0.0"
  "threshold:0:0:0.2"
  "threshold:0:0:0.4"
  "threshold:0:0:0.6"
  "threshold:0:0:0.8"
)

IFS=':' read -r CLAIM_FILTER CLAIMS_PER_DOC SCALE THRESHOLD <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

EXTRA_ARGS=()
TAG_SUFFIX="$CLAIM_FILTER"
if [ "$CLAIM_FILTER" = "topn" ]; then
    EXTRA_ARGS+=(--claims-per-doc "$CLAIMS_PER_DOC")
    TAG_SUFFIX="topn-cpd${CLAIMS_PER_DOC}"
    if [ "$SCALE" = "1" ]; then
        EXTRA_ARGS+=(--scale-topn-by-relevance)
        TAG_SUFFIX="${TAG_SUFFIX}-scaled"
    fi
elif [ "$CLAIM_FILTER" = "threshold" ]; then
    EXTRA_ARGS+=(--claim-sim-threshold "$THRESHOLD")
    TAG_SUFFIX="threshold-t${THRESHOLD}"
fi

python pipeline/run_dc_gap_dense.py \
    --topics data/ragtime2025.topics.test.jsonl \
    --run-file runs/run.ragtime1.documents.Qwen3-Embedding-0.6B.txt \
    --corpus  "$HOME/scratch/ragtime1/*.processed-claims.jsonl.gz" \
    --doc-reps "$EMB_ROOT/docs_emb/docs_emb.*.pkl" \
    --claim-reps "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
    --query-reps "$EMB_ROOT/queries_emb/queries_emb.pkl" \
    --claim-filter "$CLAIM_FILTER" \
    "${EXTRA_ARGS[@]}" \
    --output "runs/ragtime1/run.ragtime1.documents.Qwen3-Embedding-0.6B.dc-gap.${TAG_SUFFIX}.txt" \
    --k 100 \
    --tag "dc-gap-dense-doc-${TAG_SUFFIX}"
