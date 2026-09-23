#!/bin/bash
#SBATCH --job-name=mb-dense-cc-dd
#SBATCH --output=logs/%x_%a.out
#SBATCH --error=logs/%x_%a.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-87

# modernbert-base.cover-5k rerun of the final Qwen3-Embedding-0.6B cc_dense /
# dd_dense grid (see Qwen3-Embedding-0.6B/divrerank/sweep-{cc,dd}-dense-subtract.sh
# and sweep-dense-lambda-boundary.sh): agg=maxsim only (agg=mean dropped),
# mode {add,subtract} x lambda_mult 0.0-1.0 x k=100, on neuclir1 and ragtime1.
# 88 tasks = 2 methods x 2 datasets x 2 modes x 11 lambdas.

source ~/.bashrc
initconda
conda activate basic

cd $HOME/claim-augmented-generation

METHODS=(cc dd)
DATASETS=(neuclir1 ragtime1)
MODES=(add subtract)
LAMBDAS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

ID=$SLURM_ARRAY_TASK_ID
LAMBDA=${LAMBDAS[$(( ID % 11 ))]};        ID=$(( ID / 11 ))
MODE=${MODES[$(( ID % 2 ))]};             ID=$(( ID / 2 ))
DATASET=${DATASETS[$(( ID % 2 ))]};       ID=$(( ID / 2 ))
METHOD=${METHODS[$ID]}

MODEL_NAME=modernbert-base.cover-5k
EMB_ROOT=$HOME/scratch/${DATASET}/${MODEL_NAME}
OUT_DIR=runs/${DATASET}/claim-based-scoring

if [ "$DATASET" = "neuclir1" ]; then
    TOPICS=data/neuclir2024.topics.test.jsonl
else
    TOPICS=data/ragtime2025.topics.test.jsonl
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: method=${METHOD} dataset=${DATASET} mode=${MODE} lambda=${LAMBDA}"

if [ "$METHOD" = "cc" ]; then
    python pipeline/run_cc_dense.py \
        --topics "$TOPICS" \
        --run-file runs/run.${DATASET}.documents.${MODEL_NAME}.txt \
        --corpus  "$HOME/scratch/${DATASET}/*.processed-claims.jsonl.gz" \
        --claim-reps "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
        --output ${OUT_DIR}/run.${DATASET}.documents.${MODEL_NAME}.ccmaxsim-${MODE}.lambda-${LAMBDA}.txt \
        --k 100 \
        --lambda-mult ${LAMBDA} \
        --mode ${MODE} \
        --agg maxsim \
        --tag doc-ccmaxsim-${MODE}-l${LAMBDA}
else
    python pipeline/run_dd_dense.py \
        --topics "$TOPICS" \
        --run-file runs/run.${DATASET}.documents.${MODEL_NAME}.txt \
        --corpus  "$HOME/scratch/${DATASET}/*.processed-claims.jsonl.gz" \
        --doc-reps "$EMB_ROOT/docs_emb/docs_emb.*.pkl" \
        --output ${OUT_DIR}/run.${DATASET}.documents.${MODEL_NAME}.dd-${MODE}.lambda-${LAMBDA}.txt \
        --k 100 \
        --lambda-mult ${LAMBDA} \
        --mode ${MODE} \
        --tag doc-dd-${MODE}-l${LAMBDA}
fi
