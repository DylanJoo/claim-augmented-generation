#!/bin/sh
#SBATCH --job-name=analyze-topk-relevance-neuclir1
#SBATCH --output=logs/analyze-topk-relevance-neuclir1.out
#SBATCH --error=logs/analyze-topk-relevance-neuclir1.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=6:00:00

source ~/.bashrc
initconda
conda activate basic

cd $HOME/claim-augmented-generation

MODEL_NAME=Qwen3-Embedding-0.6B
EMB_ROOT=$HOME/scratch/neuclir1/${MODEL_NAME}

python pipeline/analyze_topk_relevance.py \
    --queries-emb "$EMB_ROOT/queries_emb/queries_emb.pkl" \
    --docs-emb "$EMB_ROOT/docs_emb/docs_emb.*.pkl" \
    --claims-emb "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
    --corpus "$HOME/scratch/neuclir1/*.processed-claims.jsonl.gz" \
    --runs \
        base=runs/sanity-relevant-only/base/run.neuclir1.documents.Qwen3-Embedding-0.6B.relevant-only.txt \
        cc_kmeans_core=runs/sanity-relevant-only/reranked/run.neuclir1.documents.Qwen3-Embedding-0.6B.relevant-only.cckmeans-core-subtract.top50-k20-scaled.lambda-0.9.txt \
        dd_dense=runs/sanity-relevant-only/reranked/run.neuclir1.documents.Qwen3-Embedding-0.6B.relevant-only.dd-dense.lambda-0.8.txt \
    --topk 3 \
    --output runs/sanity-relevant-only/reranked/topk-relevance-report.neuclir1.md
