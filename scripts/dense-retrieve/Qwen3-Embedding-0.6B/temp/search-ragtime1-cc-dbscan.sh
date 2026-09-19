#!/bin/sh
#SBATCH --job-name=search-ragtime1-cc-dbscan
#SBATCH --output=logs/search-ragtime1-cc-dbscan.out
#SBATCH --error=logs/search-ragtime1-cc-dbscan.err
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

# DBSCAN sweep -- 3 eps x 3 min_samples x 3 noise modes x 2 top_m x 2 lambda = 108 runs
# (existing outputs skipped). eps is cosine distance; check the per-topic
# "[cc_dbscan] qid=..." lines in the log to confirm the eps range yields a sane
# number of clusters / noise fraction before trusting the numbers.
LABEL_MODE=binary
ALPHA=0.3
FLOOR=0.0
for TOP_M in 20 40; do
for EPS in 0.15 0.25 0.35; do
for MIN_SAMPLES in 3 5 10; do
for NOISE in single drop singleton; do
for LAMBDA in 0.5 0.7; do
    OUT=runs/ragtime1/run.ragtime1.documents.Qwen3-Embedding-0.6B.ccdbscan.top${TOP_M}-eps${EPS}-ms${MIN_SAMPLES}-${NOISE}-${LABEL_MODE}.alpha-${ALPHA}.floor-${FLOOR}.lambda-${LAMBDA}.txt
    [ -s "$OUT" ] && { echo "skip $OUT"; continue; }
    python pipeline/run_cc_dbscan.py \
        --topics data/ragtime2025.topics.test.jsonl \
        --run-file runs/run.ragtime1.documents.Qwen3-Embedding-0.6B.txt \
        --corpus  "$HOME/scratch/ragtime1/*.processed-claims.jsonl.gz" \
        --claim-reps "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
        --output "$OUT" \
        --k 100 \
        --alpha ${ALPHA} \
        --discount-floor ${FLOOR} \
        --lambda-mult ${LAMBDA} \
        --dbscan-top-m ${TOP_M} \
        --dbscan-eps ${EPS} \
        --dbscan-min-samples ${MIN_SAMPLES} \
        --dbscan-noise-mode ${NOISE} \
        --dbscan-label-mode ${LABEL_MODE} \
        --tag doc-ccdbscan-top${TOP_M}-e${EPS}-m${MIN_SAMPLES}-${NOISE}-a${ALPHA}-l${LAMBDA}
done
done
done
done
done
