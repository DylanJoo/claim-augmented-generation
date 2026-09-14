#!/bin/sh
#SBATCH --job-name=search-ragtime1-dd-dense
#SBATCH --output=logs/search-ragtime1-dd-dense.out
#SBATCH --error=logs/search-ragtime1-dd-dense.err
#SBATCH --partition=small
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --account=project_465002532

# ENV
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

cd $HOME/claim-augmented-generation

# NOTE: doc_reps point at the modernbert-base.cover-5k shards built by
# scripts/dense-index/modernbert-base.cover-5k/ragtime1-encode-docs.sh.
# dd_dense.py filters each shard down to just the docids pooled by
# run-file as it streams them in, so peak memory stays bounded by one raw
# shard rather than the full docs_emb corpus -- see src/retrieval/dd_dense.py's _load_doc_reps.
#
# run-file is modernbert's own dense doc retrieval (not BM25), and the
# LAMBDA sweep mirrors the Qwen3-Embedding-0.6B ragtime1 grid in
# scripts/dense-retrieve/Qwen3-Embedding-0.6B/temp/search-ragtime1-dd-dense.sh.
MODEL_NAME=modernbert-base.cover-5k
EMB_ROOT=$HOME/scratch/ragtime1/${MODEL_NAME}

for LAMBDA in 0.5 0.7 0.9; do
    python pipeline/run_dd_dense.py \
        --topics data/ragtime2025.topics.test.jsonl \
        --run-file runs/run.ragtime1.documents.${MODEL_NAME}.txt \
        --corpus  "$HOME/scratch/ragtime1/*.processed-claims.jsonl.gz" \
        --doc-reps "$EMB_ROOT/docs_emb/docs_emb.*.pkl" \
        --output runs/ragtime1/run.ragtime1.documents.${MODEL_NAME}.dd-dense.lambda-${LAMBDA}.txt \
        --k 1000 \
        --lambda-mult ${LAMBDA} \
        --tag dd-dense-doc-l${LAMBDA}
done
