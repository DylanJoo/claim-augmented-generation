#!/bin/sh
#SBATCH --job-name=search-neuclir1-cc-kmeans-sanity
#SBATCH --output=logs/search-neuclir1-cc-kmeans-sanity.out
#SBATCH --error=logs/search-neuclir1-cc-kmeans-sanity.err
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

# Sanity check: --run-file below is the qrel-filtered "relevant-only" pool
# (see pipeline/filter_relevant_run.py and runs/sanity-relevant-only/base/),
# not the raw Qwen3 top-1000. Every candidate here is already judged
# relevant, so relevance can no longer explain StRecall@k differences --
# only the order cc_kmeans (via --kmeans-top-m) chooses among relevant docs
# can. Compare against runs/sanity-relevant-only/base/run.neuclir1...relevant-only.txt
# (unreranked) to see the isolated effect of the diversity signal.
MODEL_NAME=Qwen3-Embedding-0.6B
EMB_ROOT=$HOME/scratch/neuclir1/${MODEL_NAME}

# Same sweep as search-neuclir1-cc-kmeans.sh (existing outputs skipped)
LABEL_MODE=binary
for TOP_M in 20; do
for ALPHA in 0.2 0.3 0.4 0.5 0.6 0.7;do
for N_CLUSTERS in 50 75 100; do
for LAMBDA in 0.5 0.6 0.7; do
    OUT=runs/sanity-relevant-only/reranked/run.neuclir1.documents.Qwen3-Embedding-0.6B.relevant-only.cckmeans.top${TOP_M}-k${N_CLUSTERS}-${LABEL_MODE}.alpha-${ALPHA}.lambda-${LAMBDA}.txt
    [ -s "$OUT" ] && { echo "skip $OUT"; continue; }
    python pipeline/run_cc_kmeans.py \
        --topics data/neuclir2024.topics.test.jsonl \
        --run-file runs/sanity-relevant-only/base/run.neuclir1.documents.Qwen3-Embedding-0.6B.relevant-only.txt \
        --corpus  "$HOME/scratch/neuclir1/*.processed-claims.jsonl.gz" \
        --claim-reps "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
        --output "$OUT" \
        --k 20 \
        --alpha ${ALPHA} \
        --lambda-mult ${LAMBDA} \
        --kmeans-top-m ${TOP_M} \
        --kmeans-n-clusters ${N_CLUSTERS} \
        --kmeans-label-mode ${LABEL_MODE} \
        --tag doc-cckmeans-top${TOP_M}-k${N_CLUSTERS}-a${ALPHA}-l${LAMBDA}-relevant-only
done
done
done
done
