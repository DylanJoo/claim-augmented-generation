#!/bin/sh
#SBATCH --job-name=retrieve-c
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

for k in 1000 1500 2000;do
for FUSION in sum rrf max first; do
    python pipeline/run_bm25.py \
        --topics data/neuclir2024.topics.test.jsonl \
        --index  $HOME/scratch/neuclir1/claims.bm25s \
        --output runs/run.claims-k$k.bm25.${FUSION}.txt \
        --k $k \
        --stopwords en \
        --stemmer snowball \
        --fusion ${FUSION} \
        --tag bm25-claim-${FUSION}
done
done
