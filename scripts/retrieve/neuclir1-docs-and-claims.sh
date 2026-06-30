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

python pipeline/run_bm25_pyserini.py \
    --topics data/neuclir2024.topics.test.jsonl \
    --index  $HOME/scratch/neuclir1/docs-and-claims.lucene \
    --output runs/runs.doc-claims.bm25.txt \
    --k 1000 --k1 1.2 --b 0.75 --tag bm25-doc-claim
