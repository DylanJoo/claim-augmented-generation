#!/bin/sh
#SBATCH --job-name=retrieve-h
#SBATCH --cpus-per-task=16
#SBATCH --partition cpu
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x.out

source ~/.bashrc
initconda
conda activate inference

cd $HOME/claim-augmented-generation

for ALPHA in 0.0 0.1 0.2 0.3 0.4 0.5; do
    python pipeline/run_hybrid.py \
        --topics data/neuclir2024.topics.test.jsonl \
        --claim-index "$HOME/scratch/neuclir1/claims.bm25s" \
        --aux-index   "$HOME/scratch/neuclir1/documents.bm25s" \
        --output runs/run.neuclir1.hybrid-claim-concat.alpha-${ALPHA}.txt \
        --k-claim 1000 \
        --k-aux 1000 \
        --claim-fusion sum \
        --alpha ${ALPHA} \
        --stopwords en \
        --stemmer snowball \
        --tag hybrid-claim-concat-a${ALPHA}
done
