#!/bin/sh
#SBATCH --job-name=search-neuclir1-cc-kmeans-core
#SBATCH --output=logs/search-neuclir1-cc-kmeans-core.out
#SBATCH --error=logs/search-neuclir1-cc-kmeans-core.err
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

# cc_kmeans.py's --kmeans-top-m fits k-means on a filtered "relevant core"
# instead of the whole pool, to keep the likely-irrelevant tail from
# diluting the cluster fit (see src/retrieval/cc_kmeans.py's docstring).
# top_m is swept around the qrels' observed ~15-20 true subtopics-per-topic
# (see the neuclir1/ragtime1 qrel subtopic-id analysis), with headroom for
# retrieval error; n_clusters/alpha reuse the same ranges validated in the
# cc-dense/cc-kmeans sweeps.

for LABEL_MODE in binary; do
for TOP_M in 20; do
for N_CLUSTERS in 50; do
for ALPHA in 0.2 0.3 0.7 0.8; do
    python pipeline/run_cc_kmeans.py \
        --topics data/neuclir2024.topics.test.jsonl \
        --run-file runs/run.neuclir1.documents.Qwen3-Embedding-0.6B.txt \
        --corpus  "$HOME/scratch/neuclir1/*.processed-claims.jsonl.gz" \
        --claim-reps "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
        --output runs/neuclir1/run.neuclir1.documents.Qwen3-Embedding-0.6B.cckmeans-core.top${TOP_M}-k${N_CLUSTERS}-${LABEL_MODE}.alpha-${ALPHA}.txt \
        --k 100 \
        --alpha ${ALPHA} \
        --kmeans-top-m ${TOP_M} \
        --kmeans-n-clusters ${N_CLUSTERS} \
        --kmeans-label-mode $LABEL_MODE \
        --tag doc-cckmeans-core-top${TOP_M}-k${N_CLUSTERS}-${LABEL_MODE}-a${ALPHA}
done
done
done
done
