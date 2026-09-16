#!/bin/sh
#SBATCH --job-name=search-ragtime1-cc-dense
#SBATCH --output=logs/search-ragtime1-cc-dense.out
#SBATCH --error=logs/search-ragtime1-cc-dense.err
#SBATCH --partition=small
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --account=project_465002532

# ENV
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

cd $HOME/claim-augmented-generation

# NOTE: claim_reps point at the modernbert-base.cover-5k shards built by
# scripts/dense-index/modernbert-base.cover-5k/ragtime1-encode-claims.sh.
# cc_dense.py filters each shard down to just the docids pooled by
# run-file as it streams them in, so peak memory stays bounded by one raw
# shard rather than the full claims_emb corpus -- see
# src/retrieval/cc_dense.py's _load_claim_reps.
#
# run-file is modernbert's own dense doc retrieval (not BM25), and the
# AGG/MODE/LAMBDA sweep mirrors the Qwen3-Embedding-0.6B ragtime1 grid in
# scripts/dense-retrieve/Qwen3-Embedding-0.6B/temp/search-ragtime1-cc-dense.sh.
MODEL_NAME=modernbert-base.cover-5k
EMB_ROOT=$HOME/scratch/ragtime1/${MODEL_NAME}

for AGG in maxsim mean; do
for MODE in add subtract; do
for LAMBDA in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    python pipeline/run_cc_dense.py \
        --topics data/ragtime2025.topics.test.jsonl \
        --run-file runs/run.ragtime1.documents.${MODEL_NAME}.txt \
        --corpus  "$HOME/scratch/ragtime1/*.processed-claims.jsonl.gz" \
        --claim-reps "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
        --output runs/ragtime1/run.ragtime1.documents.${MODEL_NAME}.cc${AGG}-${MODE}.lambda-${LAMBDA}.txt \
        --k 100 \
        --lambda-mult ${LAMBDA} \
        --mode $MODE \
        --agg $AGG \
        --tag doc-cc${AGG}-${MODE}-l${LAMBDA}
done
done
done
