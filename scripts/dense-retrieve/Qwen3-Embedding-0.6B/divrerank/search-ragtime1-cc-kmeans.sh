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

# Base (no clustering) alpha-nDCG@10 = 0.5598 / @20 = 0.5843.  
# Format: top_m / k / alpha / lambda -> aNDCG@10, @20 | StRecall@10, @20
#   top20 k100 alpha0.5 lambda0.4 -> 0.6269, 0.6405 | 0.7194, 0.7748   (best @10)
#   top20 k100 alpha0.3 lambda0.4 -> 0.6256, 0.6418 | 0.7115, 0.7759
#   top20 k100 alpha0.9 lambda0.4 -> 0.6219, 0.6377 | 0.7116, 0.7715
#   top20 k75  alpha0.5 lambda0.5 -> 0.6219, 0.6351 | 0.7184, 0.7791
#   top20 k75  alpha0.3 lambda0.5 -> 0.6188, 0.6367 | 0.7125, 0.7808
#   top20 k100 alpha0.5 lambda0.5 -> 0.6183, 0.6297 | 0.7214, 0.7715   (best StRecall@10)
#   top20 k150 alpha0.3 lambda0.5 -> 0.6179, 0.6296 | 0.7208, 0.7698
#   top10 k75  alpha0.1 lambda0.6 -> 0.6204, 0.6418 | 0.7069, 0.7762
# Takeaways: top20 core, k=75-100, lambda 0.4-0.5, alpha 0.3-0.5. lambda 0.0 is bad (~0.52), lambda 0.3 is weaker (~0.57-0.61).

# Focused sweep, optimized for alpha-nDCG@10 -- 54 runs (existing outputs skipped)
LABEL_MODE=binary
for TOP_M in 20; do
for ALPHA in 0.2 0.3 0.4 0.5 0.6 0.7;do
for N_CLUSTERS in 50 75 100; do
for LAMBDA in 0.5 0.6 0.7; do
    OUT=runs/ragtime1/run.ragtime1.documents.Qwen3-Embedding-0.6B.cckmeans.top${TOP_M}-k${N_CLUSTERS}-${LABEL_MODE}.alpha-${ALPHA}.lambda-${LAMBDA}.txt
    [ -s "$OUT" ] && { echo "skip $OUT"; continue; }
    python pipeline/run_cc_kmeans.py \
        --topics data/ragtime2025.topics.test.jsonl \
        --run-file runs/run.ragtime1.documents.Qwen3-Embedding-0.6B.txt \
        --corpus  "$HOME/scratch/ragtime1/*.processed-claims.jsonl.gz" \
        --claim-reps "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
        --output "$OUT" \
        --k 100 \
        --alpha ${ALPHA} \
        --lambda-mult ${LAMBDA} \
        --kmeans-top-m ${TOP_M} \
        --kmeans-n-clusters ${N_CLUSTERS} \
        --kmeans-label-mode ${LABEL_MODE} \
        --tag doc-cckmeans-top${TOP_M}-k${N_CLUSTERS}-a${ALPHA}-l${LAMBDA}
done
done
done
done
