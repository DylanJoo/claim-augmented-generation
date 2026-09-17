#!/bin/sh
#SBATCH --job-name=search-neuclir1-cc-dense
#SBATCH --output=logs/search-neuclir1-cc-dense-m.out
#SBATCH --error=logs/search-neuclir1-cc-dense-m.err
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

for AGG in maxsim;do
for MODE in add subtract;do
for LAMBDA in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    python pipeline/run_cc_dense.py \
        --topics data/neuclir2024.topics.test.jsonl \
        --run-file runs/run.neuclir1.documents.Qwen3-Embedding-0.6B.txt \
        --corpus  "$HOME/scratch/neuclir1/*.processed-claims.jsonl.gz" \
        --claim-reps "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
        --output runs/neuclir1/run.neuclir1.documents.Qwen3-Embedding-0.6B.cc${AGG}-${MODE}.lambda-${LAMBDA}.txt \
        --k 100 \
        --lambda-mult ${LAMBDA} \
        --mode $MODE \
        --agg $AGG \
        --tag doc-cc${AGG}-${MODE}-l${LAMBDA}
done
done
done
