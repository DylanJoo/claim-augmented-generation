#!/bin/sh
#SBATCH --job-name=retrieve-d
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --cpus-per-task=16
#SBATCH --partition=debug
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH --account=project_465002532

# ENV
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

cd $HOME/claim-augmented-generation

singularity exec $SIF \
python pipeline/run_bm25.py \
    --topics data/ragtime2025.topics.test.jsonl \
    --index  $HOME/scratch/ragtime1/documents.bm25s \
    --output runs/run.ragtime1.documents.bm25.txt \
    --k 1000 \
    --stopwords en \
    --stemmer snowball \
    --tag bm25-doc
