#!/bin/bash -l
#SBATCH --job-name=search-docs
#SBATCH --output=logs/search-docs.out
#SBATCH --error=logs/search-docs.err
#SBATCH --partition=debug
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --account=project_465002438

# ENV
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

MODEL_NAME_OR_PATH=Qwen/Qwen3-Embedding-0.6B
MODEL_NAME=${MODEL_NAME_OR_PATH##*/}

EMB_ROOT=${HOME}/scratch/neuclir1/${MODEL_NAME}
query_dir=${EMB_ROOT}/queries_emb
passage_dir=${EMB_ROOT}/docs_emb

cd $HOME/claim-augmented-generation

singularity exec $SIF \
    python pipeline/run_dense.py \
    --topics data/neuclir2024.topics.test.jsonl \
    --query_reps $query_dir/queries_emb.pkl \
    --passage_reps "$passage_dir/docs_emb.*.pkl" \
    --output runs/run.neuclir1.documents.${MODEL_NAME}.txt \
    --k 1000 \
    --fusion sum \
    --tag dense-doc
    # --corpus "${HOME}/scratch/neuclir1/*.processed-claims.jsonl.gz" \
