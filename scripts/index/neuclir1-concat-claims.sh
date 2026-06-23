#!/bin/sh
#SBATCH --job-name=index-cc
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x.out

source ~/.bashrc
initconda
conda activate inference

cd $HOME/claim-augmented-generation
python src/retrieval/indexing.py \
    --input $HOME/scratch/neuclir1/*.processed-claims.jsonl.gz \
    --index $HOME/scratch/neuclir1/concat-claims.bm25s \
    --concat-claims --include-title --k1 1.2 --b 0.75
