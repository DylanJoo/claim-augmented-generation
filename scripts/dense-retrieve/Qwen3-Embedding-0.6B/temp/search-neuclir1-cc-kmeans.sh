#!/bin/sh
#SBATCH --job-name=search-neuclir1-cc-kmeans
#SBATCH --output=logs/search-neuclir1-cc-kmeans.out
#SBATCH --error=logs/search-neuclir1-cc-kmeans.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00

# ENV
source ~/.bashrc
initconda
conda activate basic

cd $HOME/claim-augmented-generation

MODEL_NAME=Qwen3-Embedding-0.6B
EMB_ROOT=$HOME/scratch/neuclir1/${MODEL_NAME}

for LABEL_MODE in binary;do
for N_CLUSTERS in 20 50; do
for ALPHA in 0.5 0.6 0.7 0.8 0.9; do
    python pipeline/run_cc_kmeans.py \
        --topics data/neuclir2024.topics.test.jsonl \
        --run-file runs/run.neuclir1.documents.Qwen3-Embedding-0.6B.txt \
        --corpus  "$HOME/scratch/neuclir1/*.processed-claims.jsonl.gz" \
        --claim-reps "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
        --output runs/neuclir1/run.neuclir1.documents.Qwen3-Embedding-0.6B.cckmeans.k${N_CLUSTERS}-${LABEL_MODE}.alpha-${ALPHA}.txt \
        --k 100 \
        --alpha ${ALPHA} \
        --kmeans-n-clusters ${N_CLUSTERS} \
        --kmeans-label-mode $LABEL_MODE \
        --tag doc-cckmeans-k${N_CLUSTERS}-${LABEL_MODE}-a${ALPHA}
done
done
done
