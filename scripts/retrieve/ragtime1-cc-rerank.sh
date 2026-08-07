#!/bin/sh
#SBATCH --job-name=retrieve-cc-rerank
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

# MMR with claim-claim scores (mode=subtract, the default)
for LAMBDA in 0.5 0.7 0.9; do
    python pipeline/run_cc_rerank.py \
        --topics data/ragtime2025.topics.test.jsonl \
        --run-file runs/run.ragtime1.documents.bm25.txt \
        --corpus "$HOME/scratch/ragtime1/*.processed-claims.jsonl.gz" \
        --output runs/run.ragtime1.cc-rerank-mmr-doc.lambda-${LAMBDA}.txt \
        --k 1000 \
        --lambda-mult ${LAMBDA} \
        --stopwords en \
        --stemmer snowball \
        --tag cc-rerank-mmr-doc-l${LAMBDA}
done
