#!/bin/sh
#SBATCH --job-name=search-neuclir1-cc-kmeans-core
#SBATCH --output=logs/search-neuclir1-cc-kmeans-core.out
#SBATCH --error=logs/search-neuclir1-cc-kmeans-core.err
#SBATCH --partition=small
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --account=project_465002532

# ENV
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

cd $HOME/claim-augmented-generation

MODEL_NAME=Qwen3-Embedding-0.6B
EMB_ROOT=$HOME/scratch/neuclir1/${MODEL_NAME}

# Unlike search-neuclir1-cc-kmeans.sh (which reranks only the top k=100 of
# the pool), this sweep keeps the full k=1000 pool -- the whole point of
# cc_kmeans_core.py is to fit k-means on a filtered "relevant core"
# (--kmeans-top-m docs) instead of letting the ~980 likely-irrelevant
# tail docs dilute the cluster fit, so it needs the full noisy pool present
# to actually test that. top_m is swept around the qrels' observed ~15-20
# true subtopics-per-topic (see the neuclir1/ragtime1 qrel subtopic-id
# analysis), with headroom for retrieval error; n_clusters/lambda reuse the
# same ranges validated in the cc-dense/cc-kmeans sweeps.
MODE=subtract
LABEL_MODE=scaled

for TOP_M in 5 10 20 50; do
for N_CLUSTERS in 20; do
for LAMBDA in 0.6 0.7; do
    python pipeline/run_cc_kmeans_core.py \
        --topics data/neuclir2024.topics.test.jsonl \
        --run-file runs/run.neuclir1.documents.Qwen3-Embedding-0.6B.txt \
        --corpus  "$HOME/scratch/neuclir1/*.processed-claims.jsonl.gz" \
        --claim-reps "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
        --output runs/neuclir1/run.neuclir1.documents.Qwen3-Embedding-0.6B.cckmeans-core-${MODE}.top${TOP_M}-k${N_CLUSTERS}-${LABEL_MODE}.lambda-${LAMBDA}.txt \
        --k 100 \
        --lambda-mult ${LAMBDA} \
        --mode $MODE \
        --kmeans-top-m ${TOP_M} \
        --kmeans-n-clusters ${N_CLUSTERS} \
        --kmeans-label-mode $LABEL_MODE \
        --tag doc-cckmeans-core-${MODE}-top${TOP_M}-k${N_CLUSTERS}-${LABEL_MODE}-l${LAMBDA}
done
done
done
