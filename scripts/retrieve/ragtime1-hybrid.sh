#!/bin/sh
#SBATCH --job-name=retrieve-hybrid
#SBATCH --cpus-per-task=16
#SBATCH --partition cpu
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x.out

source ~/.bashrc
initconda
conda activate inference

cd $HOME/claim-augmented-generation

for ALPHA in 0.3 0.5 0.7; do
    python pipeline/run_hybrid.py \
        --topics data/ragtime2025.topics.test.jsonl \
        --claim-index "$HOME/scratch/ragtime1/claims.bm25s" \
        --aux-index   "$HOME/scratch/ragtime1/concat-claims.bm25s" \
        --output runs/run.ragtime1.hybrid-claim-concat.alpha${ALPHA}.txt \
        --k-claim 1000 \
        --k-aux 1000 \
        --claim-fusion sum \
        --aux-fusion sum \
        --alpha ${ALPHA} \
        --combine sum \
        --stopwords en \
        --stemmer snowball \
        --tag hybrid-claim-concat-a${ALPHA}
done
