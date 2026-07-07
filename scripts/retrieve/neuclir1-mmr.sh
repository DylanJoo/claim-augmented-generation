#!/bin/sh
#SBATCH --job-name=retrieve-mmr
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

for LAMBDA in 0.7 0.8 0.9 0.95 1.0; do
    python pipeline/run_mmr.py \
        --topics data/neuclir2024.topics.test.jsonl \
        --run-file runs/run.neuclir1.documents.bm25.txt \
        --corpus  "$HOME/scratch/neuclir1/*.processed-claims.jsonl.gz" \
        --output runs/run.neuclir1.mmr-doc.lambda-${LAMBDA}.txt \
        --k 1000 \
        --lambda-mult ${LAMBDA} \
        --stopwords en \
        --stemmer snowball \
        --tag mmr-doc-l${LAMBDA}
done
