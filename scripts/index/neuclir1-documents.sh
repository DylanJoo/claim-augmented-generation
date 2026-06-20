#!/bin/sh
#SBATCH --job-name=index-d
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
    --input $HOME/scratch/neuclir1/*.processed.jsonl.gz \
    --index $HOME/scratch/neuclir1/documents.bm25s \
    --include-title --k1 1.2 --b 0.75
