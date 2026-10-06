#!/bin/sh
#SBATCH --job-name=search-neuclir1-cc-dense
#SBATCH --output=logs/search-neuclir1-cc-dense.out
#SBATCH --error=logs/search-neuclir1-cc-dense.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=24:00:00

# ENV
source ~/.bashrc
initconda
conda activate basic

cd $HOME/claim-augmented-generation

MODEL_NAME=Qwen3-Embedding-0.6B
EMB_ROOT=$HOME/scratch/neuclir1/${MODEL_NAME}

for AGG in maxsim; do
for MODE in add subtract; do
for LAMBDA in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    python pipeline/run_cc_dense.py \
        --topics data/neuclir2024.topics.test.jsonl \
        --run-file runs/run.neuclir1.documents.${MODEL_NAME}.txt \
        --corpus  "$HOME/scratch/neuclir1/*.processed-claims.jsonl.gz" \
        --claim-reps "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
        --output runs/neuclir1/claim-based-scoring/run.neuclir1.documents.${MODEL_NAME}.cc${AGG}-${MODE}.lambda-${LAMBDA}.txt \
        --k 100 \
        --lambda-mult ${LAMBDA} \
        --mode ${MODE} \
        --agg $AGG \
        --tag doc-cc${AGG}-${MODE}-l${LAMBDA}
done
done
done
