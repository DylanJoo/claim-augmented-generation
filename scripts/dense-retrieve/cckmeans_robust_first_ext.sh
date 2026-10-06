#!/bin/sh
#SBATCH --job-name=cckmeans-robust-first
#SBATCH --output=logs/cckmeans-robust-first.%a.out
#SBATCH --error=logs/cckmeans-robust-first.%a.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --array=0

# ENV
source ~/.bashrc
initconda
conda activate basic

cd $HOME/claim-augmented-generation

# Robustness of cc-kmeans w.r.t. the first-stage run being reranked.
# One array task per first stage: 0 bm25   1 cover   2 qwen3
# Existing outputs are skipped.
FIRSTS="bm25 cover qwen3"
FIRST=$(echo $FIRSTS | cut -d' ' -f$((${SLURM_ARRAY_TASK_ID:-0} + 1)))

OUT_DIR=runs/robust-first
mkdir -p $OUT_DIR
LABEL_MODE=binary
TOP_M=20
N_CLUSTERS=50

for DATASET in ragtime1 neuclir1; do
    case $DATASET in
        ragtime1) TOPICS=data/ragtime2025.topics.test.jsonl ;;
        neuclir1) TOPICS=data/neuclir2024.topics.test.jsonl ;;
    esac
    case $FIRST in
        bm25)  RUN_FILE=runs/run.${DATASET}.documents.bm25.txt ;;
        cover) RUN_FILE=runs/run.${DATASET}.documents.modernbert-base.cover-5k.txt ;;
        qwen3) RUN_FILE=runs/run.${DATASET}.documents.Qwen3-Embedding-0.6B.txt ;;
    esac
    [ -s "$RUN_FILE" ] || { echo "missing $RUN_FILE"; continue; }
for EMB_TAG in qwen3; do
    case $EMB_TAG in
        cover) EMB_NAME=modernbert-base.cover-5k ;;
        qwen3) EMB_NAME=Qwen3-Embedding-0.6B ;;
    esac
for ALPHA in 0.2; do
for LAMBDA in 0.5; do
    OUT=$OUT_DIR/run.${DATASET}.first-${FIRST}.claims-${EMB_TAG}.cckmeans.top${TOP_M}-k${N_CLUSTERS}-${LABEL_MODE}.alpha-${ALPHA}.lambda-${LAMBDA}.txt
    [ -s "$OUT" ] && { echo "skip $OUT"; continue; }
    python pipeline/run_cc_kmeans.py \
        --topics $TOPICS \
        --run-file $RUN_FILE \
        --corpus  "$HOME/scratch/${DATASET}/*.processed-claims.jsonl.gz" \
        --claim-reps "$HOME/scratch/${DATASET}/${EMB_NAME}/claims_emb/claims_emb.*.pkl" \
        --output "$OUT" \
        --k 100 \
        --alpha ${ALPHA} \
        --lambda-mult ${LAMBDA} \
        --kmeans-top-m ${TOP_M} \
        --kmeans-n-clusters ${N_CLUSTERS} \
        --kmeans-label-mode ${LABEL_MODE} \
        --tag ${FIRST}-${EMB_TAG}-cckmeans-top${TOP_M}-k${N_CLUSTERS}-a${ALPHA}-l${LAMBDA}
done
done
done
done
