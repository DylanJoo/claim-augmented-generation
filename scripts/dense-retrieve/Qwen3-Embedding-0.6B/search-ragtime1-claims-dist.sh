#!/bin/bash -l
#SBATCH --job-name=search-claims-dist
#SBATCH --output=logs/search-ragtime1-claims-dist.out
#SBATCH --error=logs/search-ragtime1-claims-dist.err
#SBATCH --partition=small
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --account=project_465002438

# ENV
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

MODEL_NAME_OR_PATH=Qwen/Qwen3-Embedding-0.6B
MODEL_NAME=${MODEL_NAME_OR_PATH##*/}

EMB_ROOT=${HOME}/scratch/ragtime1/${MODEL_NAME}
query_dir=${EMB_ROOT}/queries_emb
passage_dir=${EMB_ROOT}/claims_emb

LANGS=(arb-trans eng-docs rus-trans zho-trans)

SHARD_GROUPS=()
for LANG in "${LANGS[@]}"; do
    for SHARD_FILE in "$passage_dir"/claims_emb.${LANG}-*.pkl; do
        SHARD_GROUPS+=("$SHARD_FILE")
    done
done

cd $HOME/claim-augmented-generation

for k in 100 200 500 750 1000; do
for FUSION in sum rrf; do
singularity exec $SIF \
    python pipeline/run_dense.py \
    --topics data/ragtime2025.topics.test.jsonl \
    --query_reps $query_dir/queries_emb.pkl \
    --shard_groups "${SHARD_GROUPS[@]}" \
    --output runs/run.ragtime1.claims-k${k}.${MODEL_NAME}.${FUSION}.txt \
    --k $k \
    --fusion ${FUSION} \
    --tag dense-claim-${FUSION}
    done
done
