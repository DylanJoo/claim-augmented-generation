#!/bin/sh
#SBATCH --job-name=search-neuclir1-cc-kmeans-sanity-all
#SBATCH --output=logs/search-neuclir1-cc-kmeans-sanity-all.out
#SBATCH --error=logs/search-neuclir1-cc-kmeans-sanity-all.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00

# ENV
source ~/.bashrc
initconda
conda activate basic

cd $HOME/claim-augmented-generation

MODEL_NAME=Qwen3-Embedding-0.6B
EMB_ROOT=$HOME/scratch/neuclir1/${MODEL_NAME}
BASE=runs/sanity-all-relevant/base/run.neuclir1.documents.${MODEL_NAME}.all-relevant.txt

# Every judged-relevant doc in the qrel (not only the ones the base run
# retrieved, cf. sanity-relevant-only), scored by dense query-doc similarity
[ -s "$BASE" ] || python pipeline/build_all_relevant_run.py \
    --qrel $HOME/trec2026/data/neuclir/neuclir24-test-request.qrel \
    --query_reps $EMB_ROOT/queries_emb/queries_emb.pkl \
    --passage_reps "$EMB_ROOT/docs_emb/docs_emb.*.pkl" \
    --output $BASE \
    --tag qwen3-embed-0.6b:doc:all-relevant

LABEL_MODE=binary
for TOP_M in 20; do
for ALPHA in 0.4 0.5 0.6;do
for N_CLUSTERS in 50 75 100; do
for LAMBDA in 0.5 0.6 0.7; do
    OUT=runs/sanity-all-relevant/reranked/run.neuclir1.documents.Qwen3-Embedding-0.6B.all-relevant.cckmeans.top${TOP_M}-k${N_CLUSTERS}-${LABEL_MODE}.alpha-${ALPHA}.lambda-${LAMBDA}.txt
    [ -s "$OUT" ] && { echo "skip $OUT"; continue; }
    python pipeline/run_cc_kmeans.py \
        --topics data/neuclir2024.topics.test.jsonl \
        --run-file $BASE \
        --corpus  "$HOME/scratch/neuclir1/*.processed-claims.jsonl.gz" \
        --claim-reps "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
        --output "$OUT" \
        --k 100 \
        --alpha ${ALPHA} \
        --lambda-mult ${LAMBDA} \
        --kmeans-top-m ${TOP_M} \
        --kmeans-n-clusters ${N_CLUSTERS} \
        --kmeans-label-mode ${LABEL_MODE} \
        --tag doc-cckmeans-top${TOP_M}-k${N_CLUSTERS}-a${ALPHA}-l${LAMBDA}-all-relevant
done
done
done
done
