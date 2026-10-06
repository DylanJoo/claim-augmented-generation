#!/bin/sh
#SBATCH --job-name=iter-rag
#SBATCH --cpus-per-task=8
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.out

source ~/.bashrc
initconda
conda activate inference

cd $HOME/claim-augmented-generation

# Run iterative RAG against the document-level BM25 results.
# To use claim-level results instead, swap --run-file to bm25-claims.txt.

python pipeline/run_iterative_rag.py \
    --topics   $HOME/scratch/neuclir1/topics/neuclir24-test-request.jsonl \
    --run-file $HOME/scratch/neuclir1/runs/bm25-documents.txt \
    --corpus   $HOME/scratch/neuclir1/*.processed.jsonl.gz \
    --output   $HOME/scratch/neuclir1/runs/iterative-rag.jsonl \
    --topk 50 \
    --n-rounds 3 \
    --docs-per-round 5 \
    --max-chars 2000 \
    --model llama3.3-70b-instruct \
    --base-url http://10.162.95.158:4000/v1 \
    --max-tokens 1024
