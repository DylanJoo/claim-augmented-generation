#!/bin/bash -l
#SBATCH --job-name=search-claims-dist
#SBATCH --output=logs/search-claims-dist.out
#SBATCH --error=logs/search-claims-dist.err
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

EMB_ROOT=${HOME}/scratch/neuclir1/${MODEL_NAME}
query_dir=${EMB_ROOT}/queries_emb
passage_dir=${EMB_ROOT}/claims_emb

LANGS=(fas rus zho)

SHARD_GROUPS=()
for LANG in "${LANGS[@]}"; do
    # lang-shard groups: one group per language
    # SHARD_GROUPS+=("$passage_dir/claims_emb.${LANG}-*.pkl")
    for SHARD_FILE in "$passage_dir"/claims_emb.${LANG}-*.pkl; do
        SHARD_GROUPS+=("$SHARD_FILE")
    done
done

cd $HOME/claim-augmented-generation

# for k in 1000 1500 2000; do
# for FUSION in sum rrf max first; do
for k in 1500; do
for FUSION in rrf; do
singularity exec $SIF \
    python pipeline/run_dense.py \
    --topics data/neuclir2024.topics.test.jsonl \
    --query_reps $query_dir/queries_emb.pkl \
    --shard_groups "${SHARD_GROUPS[@]}" \
    --output runs/run.neuclir1.claims-k${k}.${MODEL_NAME}.${FUSION}.txt \
    --k $k \
    --fusion ${FUSION} \
    --tag dense-claim-${FUSION}
    done
done
    # --corpus "${HOME}/scratch/neuclir1/claims_flat/*.jsonl.gz" \
