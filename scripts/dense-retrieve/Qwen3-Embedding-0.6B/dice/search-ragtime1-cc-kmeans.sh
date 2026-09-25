#!/bin/sh
#SBATCH --job-name=search-ragtime1-cc-kmeans
#SBATCH --output=logs/search-ragtime1-dice.out
#SBATCH --error=logs/search-ragtime1-dice.err
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

LABEL_MODE=binary
for TOP_M in 20; do
for ALPHA in 0.2 0.3 0.4 0.5 0.6 0.7;do
for N_CLUSTERS in 50 75 100; do
for LAMBDA in 0.5 0.6 0.7; do
    OUT=runs/dice/run.ragtime1.hybrid-a0.3.${MODEL_NAME}.cckmeans.top${TOP_M}-k${N_CLUSTERS}-${LABEL_MODE}.alpha-${ALPHA}.lambda-${LAMBDA}.txt
    [ -s "$OUT" ] && { echo "skip $OUT"; continue; }
    python pipeline/run_cc_kmeans.py \
        --topics data/ragtime2025.topics.test.jsonl \
        --run-file runs/hybrid/run.ragtime1.hybrid-claim-doc.${MODEL_NAME}.alpha-0.3.txt \
        --corpus  "$HOME/scratch/ragtime1/*.processed-claims.jsonl.gz" \
        --claim-reps "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
        --output "$OUT" \
        --k 100 \
        --alpha ${ALPHA} \
        --lambda-mult ${LAMBDA} \
        --kmeans-top-m ${TOP_M} \
        --kmeans-n-clusters ${N_CLUSTERS} \
        --kmeans-label-mode ${LABEL_MODE} \
        --tag hybrid-cckmeans-top${TOP_M}-k${N_CLUSTERS}-a${ALPHA}-l${LAMBDA}
done
done
done
done
