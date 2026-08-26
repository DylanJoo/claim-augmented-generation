#!/bin/sh
#SBATCH --job-name=search-neuclir1-dc-gap
#SBATCH --output=logs/search-neuclir1-dc-gap.out
#SBATCH --error=logs/search-neuclir1-dc-gap.err
#SBATCH --partition=small
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --account=project_465002438

# ENV
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

cd $HOME/claim-augmented-generation

# NOTE: doc_reps/claim_reps point at the Qwen3-Embedding-0.6B shards built by
# scripts/dense-index/Qwen3-Embedding-0.6B/neuclir1-encode-{docs,claims}.sh.
# dc_gap_dense.py filters each shard down to just the docids pooled by
# run-file as it streams them in, so peak memory stays bounded by one raw
# shard (~41GB for the largest claims_emb shard) rather than the full
# ~439GB claims_emb corpus -- see src/retrieval/dc_gap_dense.py's _load_reps.
MODEL_NAME=Qwen3-Embedding-0.6B
EMB_ROOT=$HOME/scratch/neuclir1/${MODEL_NAME}

python pipeline/run_dc_gap_dense.py \
    --topics data/neuclir2024.topics.test.jsonl \
    --run-file runs/neuclir1/run.neuclir1.documents.bm25.txt \
    --corpus  "$HOME/scratch/neuclir1/*.processed-claims.jsonl.gz" \
    --doc-reps "$EMB_ROOT/docs_emb/docs_emb.*.pkl" \
    --claim-reps "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
    --output runs/neuclir1/run.neuclir1.dc-gap-dense-doc.${MODEL_NAME}.txt \
    --k 1000 \
    --tag dc-gap-dense-doc
