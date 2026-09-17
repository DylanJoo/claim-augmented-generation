#!/bin/sh
#SBATCH --job-name=search-ragtime1-cc-kmeans
#SBATCH --output=logs/search-ragtime1-cc-kmeans.out
#SBATCH --error=logs/search-ragtime1-cc-kmeans.err
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

MODEL_NAME=Qwen3-Embedding-0.6B
EMB_ROOT=$HOME/scratch/ragtime1/${MODEL_NAME}

for LABEL_MODE in binary;do
for N_CLUSTERS in 20 50; do
for ALPHA in 0.5 0.6 0.7 0.8 0.9; do
    python pipeline/run_cc_kmeans.py \
        --topics data/ragtime2025.topics.test.jsonl \
        --run-file runs/run.ragtime1.documents.Qwen3-Embedding-0.6B.txt \
        --corpus  "$HOME/scratch/ragtime1/*.processed-claims.jsonl.gz" \
        --claim-reps "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
        --output runs/ragtime1/run.ragtime1.documents.Qwen3-Embedding-0.6B.cckmeans.k${N_CLUSTERS}-${LABEL_MODE}.alpha-${ALPHA}.txt \
        --k 100 \
        --alpha ${ALPHA} \
        --kmeans-n-clusters ${N_CLUSTERS} \
        --kmeans-label-mode $LABEL_MODE \
        --tag doc-cckmeans-k${N_CLUSTERS}-${LABEL_MODE}-a${ALPHA}
done
done
done
