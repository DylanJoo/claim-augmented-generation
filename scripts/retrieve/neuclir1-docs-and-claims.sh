#!/bin/sh
#SBATCH --job-name=retrieve-dc
#SBATCH --cpus-per-task=16
#SBATCH --partition cpu
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x.out

source ~/.bashrc
initconda
conda activate inference

cd $HOME/claim-augmented-generation

# Docids carry a '#<i>' suffix (claim-level + doc-augmented index).
# Strip the suffix before passing results to document-level evaluation.
python pipeline/run_bm25.py \
    --topics $HOME/scratch/neuclir1/topics/neuclir24-test-request.jsonl \
    --index  $HOME/scratch/neuclir1/docs-and-claims.bm25s \
    --output $HOME/scratch/neuclir1/runs/bm25-docs-and-claims.txt \
    --k 1000 \
    --stopwords en \
    --stemmer snowball \
    --tag bm25-doc-claim
