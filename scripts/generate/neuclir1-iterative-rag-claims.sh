#!/bin/sh
#SBATCH --job-name=iter-rag-c
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

# Iterative RAG using the claim-level BM25 index.
# Each retrieved unit is an individual claim (parent_id#i); the corpus is
# loaded by expanding the 'statements' field of each claims JSONL document.
# Bump --docs-per-round relative to the document run since claims are short.

python pipeline/run_iterative_rag.py \
    --topics      $HOME/scratch/neuclir1/topics/neuclir24-test-request.jsonl \
    --run-file    $HOME/scratch/neuclir1/runs/bm25-claims.txt \
    --corpus      $HOME/scratch/neuclir1/claims/*.jsonl \
    --output      $HOME/scratch/neuclir1/runs/iterative-rag-claims.jsonl \
    --claim-level \
    --topk 150 \
    --n-rounds 3 \
    --docs-per-round 15 \
    --max-chars 2000 \
    --model llama3.3-70b-instruct \
    --base-url http://10.162.95.158:4000/v1 \
    --max-tokens 1024
