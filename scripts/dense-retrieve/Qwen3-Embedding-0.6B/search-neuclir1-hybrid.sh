#!/bin/sh
#SBATCH --job-name=search-neuclir1-hybrid
#SBATCH --output=logs/search-neuclir1-hybrid.out
#SBATCH --error=logs/search-neuclir1-hybrid.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00

# ENV
source ~/.bashrc
initconda
conda activate basic

cd $HOME/claim-augmented-generation

MODEL_NAME=Qwen3-Embedding-0.6B
EMB_ROOT=$HOME/scratch/neuclir1/${MODEL_NAME}

CLAIM_SHARD_GROUPS=()
for SHARD_FILE in "$EMB_ROOT"/claims_emb/claims_emb.*.pkl; do
    CLAIM_SHARD_GROUPS+=("$SHARD_FILE")
done

for FUSION in sum;do
for ALPHA in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    python pipeline/run_hybrid_dense.py \
        --topics data/neuclir2024.topics.test.jsonl \
        --query-reps "$EMB_ROOT/queries_emb/queries_emb.pkl" \
        --claim-shard-groups "${CLAIM_SHARD_GROUPS[@]}" \
        --doc-passage-reps "$EMB_ROOT/docs_emb/docs_emb.*.pkl" \
        --output runs/hybrid/run.neuclir1.hybrid-claim-${FUSION}-doc.${MODEL_NAME}.alpha-${ALPHA}.txt \
        --k-claim 1000 \
        --k-doc 1000 \
        --claim-fusion ${FUSION} \
        --alpha ${ALPHA} \
        --tag hybrid-claim-${FUSION}-doc-a${ALPHA}
done
done
