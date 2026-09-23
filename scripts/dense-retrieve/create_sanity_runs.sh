#!/bin/sh
#SBATCH --job-name=create-sanity-runs
#SBATCH --output=logs/create-sanity-runs.out
#SBATCH --error=logs/create-sanity-runs.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00

# ENV
source ~/.bashrc
initconda
conda activate basic

cd $HOME/claim-augmented-generation

# Build the qrel-filtered "relevant-only" base pools used by the
# */divrerank/search-*-cc-kmeans-sanity.sh experiments.
for DATASET in ragtime1 neuclir1; do
case $DATASET in
    ragtime1) QREL=$HOME/trec2026/data/ragtime1/ragtime25-test-request.qrel ;;
    neuclir1) QREL=$HOME/trec2026/data/neuclir/neuclir24-test-request.qrel ;;
esac

for MODEL_NAME in modernbert-base.cover-5k Qwen3-Embedding-0.6B; do
case $MODEL_NAME in
    modernbert-base.cover-5k) BASE_TAG=mdbert-cover-base:doc ;;
    Qwen3-Embedding-0.6B)     BASE_TAG=qwen3-embed-0.6b:doc ;;
esac
    BASE=runs/sanity-relevant-only/base/run.${DATASET}.documents.${MODEL_NAME}.relevant-only.txt
    [ -s "$BASE" ] && { echo "skip $BASE"; continue; }
    python pipeline/filter_relevant_run.py \
        --run-file runs/run.${DATASET}.documents.${MODEL_NAME}.txt \
        --qrel $QREL \
        --output $BASE \
        --tag ${BASE_TAG}:relevant-only
done
done
