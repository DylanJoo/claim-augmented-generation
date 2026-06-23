#!/bin/sh
#SBATCH --job-name=retrieve-c
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

# Claim-level docids have the form '<parent_id>#<i>' in the output.
# Use trec_eval or strip the suffix before passing to downstream tools.
python pipeline/run_bm25.py \
    --topics $HOME/scratch/neuclir1/topics/neuclir24-test-request.jsonl \
    --index  $HOME/scratch/neuclir1/claims.bm25s \
    --output $HOME/scratch/neuclir1/runs/bm25-claims.txt \
    --k 1000 \
    --stopwords en \
    --stemmer snowball \
    --tag bm25-claim
