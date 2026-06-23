#!/bin/sh
#SBATCH --job-name=retrieve-d
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

python pipeline/run_bm25.py \
    --topics data/neuclir2024.topics.test.jsonl \
    --index  $HOME/scratch/neuclir1/documents.bm25s \
    --output runs/run.documents.bm25.txt \
    --k 1000 \
    --stopwords en \
    --stemmer snowball \
    --tag bm25-doc
