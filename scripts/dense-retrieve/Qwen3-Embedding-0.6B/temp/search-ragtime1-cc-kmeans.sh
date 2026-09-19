#!/bin/sh
#SBATCH --job-name=search-ragtime1-cc-kmeans
#SBATCH --output=logs/search-ragtime1-cc-kmeans.out
#SBATCH --error=logs/search-ragtime1-cc-kmeans.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00

# ENV
source ~/.bashrc
initconda
conda activate basic

cd $HOME/claim-augmented-generation

MODEL_NAME=Qwen3-Embedding-0.6B
EMB_ROOT=$HOME/scratch/ragtime1/${MODEL_NAME}

# Focused sweep, optimized for alpha-nDCG@10 -- 54 runs (existing outputs skipped)
LABEL_MODE=binary
ALPHA=0.3
for TOP_M in 20 40; do
for N_CLUSTERS in 30 50 75; do
for FLOOR in 0.0 0.1 0.3; do
for LAMBDA in 0.5 0.6 0.7; do
    OUT=runs/ragtime1/run.ragtime1.documents.Qwen3-Embedding-0.6B.cckmeans.top${TOP_M}-k${N_CLUSTERS}-${LABEL_MODE}.alpha-${ALPHA}.floor-${FLOOR}.lambda-${LAMBDA}.txt
    [ -s "$OUT" ] && { echo "skip $OUT"; continue; }
    python pipeline/run_cc_kmeans.py \
        --topics data/ragtime2025.topics.test.jsonl \
        --run-file runs/run.ragtime1.documents.Qwen3-Embedding-0.6B.txt \
        --corpus  "$HOME/scratch/ragtime1/*.processed-claims.jsonl.gz" \
        --claim-reps "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
        --output "$OUT" \
        --k 100 \
        --alpha ${ALPHA} \
        --discount-floor ${FLOOR} \
        --lambda-mult ${LAMBDA} \
        --kmeans-top-m ${TOP_M} \
        --kmeans-n-clusters ${N_CLUSTERS} \
        --kmeans-label-mode ${LABEL_MODE} \
        --tag doc-cckmeans-top${TOP_M}-k${N_CLUSTERS}-a${ALPHA}-f${FLOOR}-l${LAMBDA}
done
done
done
done
