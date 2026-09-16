#!/bin/sh
#SBATCH --job-name=search-neuclir1-dd-dense
#SBATCH --output=logs/search-neuclir1-dd-dense.out
#SBATCH --error=logs/search-neuclir1-dd-dense.err
#SBATCH --partition=small
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=project_465002532

# ENV
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

cd $HOME/claim-augmented-generation

MODEL_NAME=modernbert-base.cover-5k
EMB_ROOT=$HOME/scratch/neuclir1/${MODEL_NAME}

for MODE in add subtract; do
for LAMBDA in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    python pipeline/run_dd_dense.py \
        --topics data/neuclir2024.topics.test.jsonl \
        --run-file runs/run.neuclir1.documents.${MODEL_NAME}.txt \
        --corpus  "$HOME/scratch/neuclir1/*.processed-claims.jsonl.gz" \
        --doc-reps "$EMB_ROOT/docs_emb/docs_emb.*.pkl" \
        --output runs/neuclir1/run.neuclir1.documents.${MODEL_NAME}.dd-${MODE}.lambda-${LAMBDA}.txt \
        --k 100 \
        --lambda-mult ${LAMBDA} \
        --mode $MODE \
        --tag dd-doc-${MODE}-l${LAMBDA}
done
done
