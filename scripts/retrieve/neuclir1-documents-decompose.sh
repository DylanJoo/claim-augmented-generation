#!/bin/sh
#SBATCH --job-name=retrieve-d-decompose
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --cpus-per-task=16
#SBATCH --partition=small
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --account=project_465002532

# ENV
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

cd $HOME/claim-augmented-generation

# Requires data/neuclir2024.topics.test.subq.jsonl from
# scripts/retrieve/neuclir1-decompose.sh (run once, on a GPU node).
singularity exec $SIF \
python pipeline/run_bm25.py \
    --topics data/neuclir2024.topics.test.subq.jsonl \
    --index  $HOME/scratch/neuclir1/documents.bm25s \
    --output runs/run.neuclir1.documents-decompose.bm25.txt \
    --k 1000 \
    --stopwords en \
    --stemmer snowball \
    --fusion sum \
    --tag bm25-doc-decompose
