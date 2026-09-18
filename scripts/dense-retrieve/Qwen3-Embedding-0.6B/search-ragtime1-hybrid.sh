#!/bin/sh
#SBATCH --job-name=search-ragtime1-hybrid
#SBATCH --output=logs/search-ragtime1-hybrid.out
#SBATCH --error=logs/search-ragtime1-hybrid.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=512G
#SBATCH --time=1-00:00:00

# ENV
source ~/.bashrc
initconda
conda activate basic

cd $HOME/claim-augmented-generation

MODEL_NAME=Qwen3-Embedding-0.6B
EMB_ROOT=$HOME/scratch/ragtime1/${MODEL_NAME}

CLAIM_SHARD_GROUPS=()
for SHARD_FILE in "$EMB_ROOT"/claims_emb/claims_emb.*.pkl; do
    CLAIM_SHARD_GROUPS+=("$SHARD_FILE")
done

# for ALPHA in 0.1 0.2 0.3 0.4 0.5; do
for ALPHA in 0.5; do
    python pipeline/run_hybrid_dense.py \
        --topics data/ragtime2025.topics.test.jsonl \
        --query-reps "$EMB_ROOT/queries_emb/queries_emb.pkl" \
        --claim-shard-groups "${CLAIM_SHARD_GROUPS[@]}" \
        --doc-passage-reps "$EMB_ROOT/docs_emb/docs_emb.*.pkl" \
        --output runs/run.ragtime1.hybrid-claim-doc.${MODEL_NAME}.alpha-${ALPHA}.txt \
        --k-claim 1000 \
        --k-doc 1000 \
        --claim-fusion sum \
        --alpha ${ALPHA} \
        --tag hybrid-claim-doc-dense-a${ALPHA}
done
