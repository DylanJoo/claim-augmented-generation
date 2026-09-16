#!/bin/sh
#SBATCH --job-name=search-neuclir1-cc-kmeans
#SBATCH --output=logs/search-neuclir1-cc-kmeans.out
#SBATCH --error=logs/search-neuclir1-cc-kmeans.err
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

# subtract-only sweep: "add" already behaves like maxsim/mean's echo-boost
# (see cc_dense.py), so the open question here is whether clustering claims
# into topic buckets rescues genuine MMR-style diversification, which raw
# --agg maxsim/mean subtract never managed. n_clusters is the parameter
# that matters most (too few = everything collides into one bucket, too
# many = degenerates back to raw claim-pair MaxSim), so it gets the wide
# sweep; lambda is held near the 0.6-0.7 sweet spot found in the dd/cc
# add-mode sweeps.
MODE=subtract

for LABEL_MODE in scaled;do
for N_CLUSTERS in 20 50; do
for LAMBDA in 0.9; do
    python pipeline/run_cc_dense.py \
        --topics data/neuclir2024.topics.test.jsonl \
        --run-file runs/run.neuclir1.documents.Qwen3-Embedding-0.6B.txt \
        --corpus  "$HOME/scratch/neuclir1/*.processed-claims.jsonl.gz" \
        --claim-reps "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
        --output runs/neuclir1/run.neuclir1.documents.Qwen3-Embedding-0.6B.cckmeans-${MODE}.k${N_CLUSTERS}-${LABEL_MODE}.lambda-${LAMBDA}.txt \
        --k 100 \
        --lambda-mult ${LAMBDA} \
        --mode $MODE \
        --agg kmeans \
        --kmeans-n-clusters ${N_CLUSTERS} \
        --kmeans-label-mode $LABEL_MODE \
        --tag doc-cckmeans-${MODE}-k${N_CLUSTERS}-${LABEL_MODE}-l${LAMBDA}
done
done
done
