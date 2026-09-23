#!/bin/sh
#SBATCH --job-name=search-ragtime1-cc-dense
#SBATCH --output=logs/search-ragtime1-cc-dense.out
#SBATCH --error=logs/search-ragtime1-cc-dense.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=48:00:00

# ENV
source ~/.bashrc
initconda
conda activate basic

cd $HOME/claim-augmented-generation

MODEL_NAME=modernbert-base.cover-5k
EMB_ROOT=$HOME/scratch/ragtime1/${MODEL_NAME}

for AGG in maxsim; do
for MODE in add subtract; do
for LAMBDA in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    python pipeline/run_cc_dense.py \
        --topics data/ragtime2025.topics.test.jsonl \
        --run-file runs/run.ragtime1.documents.${MODEL_NAME}.txt \
        --corpus  "$HOME/scratch/ragtime1/*.processed-claims.jsonl.gz" \
        --claim-reps "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
        --output runs/ragtime1/claim-based-scoring/run.ragtime1.documents.${MODEL_NAME}.cc${AGG}-${MODE}.lambda-${LAMBDA}.txt \
        --k 100 \
        --lambda-mult ${LAMBDA} \
        --mode ${MODE} \
        --agg $AGG \
        --tag doc-cc${AGG}-${MODE}-l${LAMBDA}
done
done
done
