#!/bin/sh
#SBATCH --job-name=search-ragtime1-dd-dense-fix
#SBATCH --output=logs/search-ragtime1-dd-dense-fix.out
#SBATCH --error=logs/search-ragtime1-dd-dense-fix.err
#SBATCH --partition=small
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --account=project_465002532

# ENV
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

cd $HOME/claim-augmented-generation

MODEL_NAME=Qwen3-Embedding-0.6B
EMB_ROOT=$HOME/scratch/ragtime1/${MODEL_NAME}

# Redo the lambda-0.0/1.0 boundary runs, which were previously generated with
# the --k 1000 default instead of --k 100 like the rest of the dd-* sweep.
for CFG in "add 0.0" "subtract 0.0" "subtract 1.0"; do
    set -- $CFG
    MODE=$1
    LAMBDA=$2
    python pipeline/run_dd_dense.py \
        --topics data/ragtime2025.topics.test.jsonl \
        --run-file runs/run.ragtime1.documents.Qwen3-Embedding-0.6B.txt \
        --corpus  "$HOME/scratch/ragtime1/*.processed-claims.jsonl.gz" \
        --doc-reps "$EMB_ROOT/docs_emb/docs_emb.*.pkl" \
        --output runs/ragtime1/run.ragtime1.documents.Qwen3-Embedding-0.6B.dd-${MODE}.lambda-${LAMBDA}.txt \
        --k 100 \
        --lambda-mult ${LAMBDA} \
        --mode $MODE \
        --tag doc-dd-${MODE}-l${LAMBDA}
done
