#!/bin/sh
# Plain (non-SLURM) runner for a separate VM. Add SBATCH directives back if
# running this on the HPC cluster instead.

cd "$(dirname "$0")/../.."

python pipeline/dev/run_local_rescore.py \
    --topics data/ragtime2025.topics.test.jsonl \
    --doc-index   "$HOME/scratch/ragtime1/concat-claims.bm25s" \
    --claim-index "$HOME/scratch/ragtime1/claims.bm25s" \
    --output runs/run.ragtime1.local-rescore.txt \
    --k-doc 50 \
    --doc-floor 0.3 \
    --discount 0.3 \
    --n-iters 15 \
    --k1 1.2 \
    --b 0.5 \
    --stopwords en \
    --tag local-rescore
    # --dedup-threshold <tune-me>   # optional: collapse near-duplicate docs before claim expansion
