#!/bin/sh
#SBATCH --job-name=retrieve-hybrid
#SBATCH --output=logs/retrieve-hybrid.out
#SBATCH --error=logs/retrieve-hybrid.err
#SBATCH --cpus-per-task=16
#SBATCH --partition=debug
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --account=project_465002532

# ENV
module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

cd $HOME/claim-augmented-generation

for ALPHA in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7; do
    singularity exec $SIF \
        python pipeline/run_hybrid.py \
        --topics data/ragtime2025.topics.test.jsonl \
        --claim-index "$HOME/scratch/ragtime1/claims.bm25s" \
        --aux-index   "$HOME/scratch/ragtime1/documents.bm25s" \
        --output runs/run.ragtime1.hybrid-claim-concat.alpha${ALPHA}.txt \
        --k-claim 1000 \
        --k-aux 1000 \
        --claim-fusion sum \
        --alpha ${ALPHA} \
        --combine sum \
        --stopwords en \
        --stemmer snowball \
        --tag hybrid-claim-concat-a${ALPHA}
done
