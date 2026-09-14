#!/bin/sh
#SBATCH --job-name=search-ragtime1-hybrid
#SBATCH --output=logs/search-ragtime1-hybrid.out
#SBATCH --error=logs/search-ragtime1-hybrid.err
#SBATCH --partition=small
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --account=project_465002532

# ENV
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

cd $HOME/claim-augmented-generation

# NOTE: mirrors scripts/dense-retrieve/modernbert-base.cover-5k/search-ragtime1-claims-dist.sh's
# shard-group construction (one group per shard FILE, streamed one at a time so peak
# memory stays bounded -- not one group per language: a language's shards can total
# 100-200G, which blew past even 512G) and scripts/retrieve/neuclir1-hybrid-claim-doc.sh's
# alpha sweep (0.0 = pure doc-level, 0.5 = evenly split).
MODEL_NAME=modernbert-base.cover-5k
EMB_ROOT=$HOME/scratch/ragtime1/${MODEL_NAME}

LANGS=(arb-trans eng-docs rus-trans zho-trans)
CLAIM_SHARD_GROUPS=()
for LANG in "${LANGS[@]}"; do
    for SHARD_FILE in "$EMB_ROOT"/claims_emb/claims_emb.${LANG}-*.pkl; do
        CLAIM_SHARD_GROUPS+=("$SHARD_FILE")
    done
done

# for ALPHA in 0.0 0.1 0.2 0.3 0.4 0.5; do
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
