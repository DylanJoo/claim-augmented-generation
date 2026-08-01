#!/bin/sh
#SBATCH --job-name=index-dc
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x.out

source ~/.bashrc
initconda
conda activate inference

cd $HOME/claim-augmented-generation

COLLECTION=$HOME/scratch/neuclir1/docs-and-claims-collection
INDEX=$HOME/scratch/neuclir1/docs-and-claims.lucene

python src/retrieval/indexing_pyserini.py \
    --input $HOME/scratch/neuclir1/*.processed-claims.jsonl.gz \
    --output $COLLECTION \
    --include-title

python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input $COLLECTION \
    --index $INDEX \
    --generator DefaultLuceneDocumentGenerator \
    --threads 32 \
    --storeRaw
