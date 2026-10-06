#!/bin/sh
#SBATCH --job-name=search-ragtime1-hybrid
#SBATCH --output=logs/search-ragtime1-hybrid.out
#SBATCH --error=logs/search-ragtime1-hybrid.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00

# ENV
source ~/.bashrc
initconda
conda activate basic

cd $HOME/claim-augmented-generation
MODEL_NAME=modernbert-base.cover-5k
EMB_ROOT=$HOME/scratch/ragtime1/${MODEL_NAME}

LANGS=(arb-trans eng-docs rus-trans zho-trans)
CLAIM_SHARD_GROUPS=()
for LANG in "${LANGS[@]}"; do
    for SHARD_FILE in "$EMB_ROOT"/claims_emb/claims_emb.${LANG}-*.pkl; do
        CLAIM_SHARD_GROUPS+=("$SHARD_FILE")
    done
done

for FUSION in sum;do
for ALPHA in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    python pipeline/run_hybrid_dense.py \
        --topics data/ragtime2025.topics.test.jsonl \
        --query-reps "$EMB_ROOT/queries_emb/queries_emb.pkl" \
        --claim-shard-groups "${CLAIM_SHARD_GROUPS[@]}" \
        --doc-passage-reps "$EMB_ROOT/docs_emb/docs_emb.*.pkl" \
        --output runs/hybrid/run.ragtime1.hybrid-claim-${FUSION}-doc.${MODEL_NAME}.alpha-${ALPHA}.txt \
        --k-claim 1000 \
        --k-doc 1000 \
        --claim-fusion ${FUSION} \
        --alpha ${ALPHA} \
        --tag hybrid-claim-${FUSION}-doc-a${ALPHA}
done
done
