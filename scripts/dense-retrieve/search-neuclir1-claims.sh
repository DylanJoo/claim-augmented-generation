#!/bin/bash -l
#SBATCH --job-name=search-claims
#SBATCH --output=logs/search-claims.out
#SBATCH --error=logs/search-claims.err
#SBATCH --partition=debug
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --account=project_465002438

# ENV
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

MODEL_NAME_OR_PATH=Qwen/Qwen3-Embedding-0.6B
MODEL_NAME=${MODEL_NAME_OR_PATH##*/}

EMB_ROOT=${HOME}/scratch/neuclir1/${MODEL_NAME}
query_dir=${EMB_ROOT}/queries_emb
passage_dir=${EMB_ROOT}/claims_emb

cd $HOME/claim-augmented-generation

# Load + search the claim embedding shards and fuse claim-level hits into
# document-level TREC runs, mirroring the BM25 claims pipeline
# (scripts/retrieve/neuclir1-claims.sh). Index loading, dense (faiss) search,
# and claim -> parent-doc fusion all happen inside pipeline/run_dense.py
# (src/retrieval/search_dense.py) -- no separate tevatron search call.
for k in 1000 1500 2000; do
for FUSION in sum rrf max first; do
    singularity exec $SIF \
        python pipeline/run_dense.py \
        --topics data/neuclir2024.topics.test.jsonl \
        --query_reps $query_dir/queries_emb.pkl \
        --passage_reps "$passage_dir/claims_emb.*.pkl" \
        --output runs/run.neuclir1.claims-k$k.${MODEL_NAME}.${FUSION}.txt \
        --k $k \
        --fusion ${FUSION} \
        --tag dense-claim-${FUSION}
done
done
        # --corpus "${HOME}/scratch/neuclir1/claims_flat/*.jsonl.gz" \
