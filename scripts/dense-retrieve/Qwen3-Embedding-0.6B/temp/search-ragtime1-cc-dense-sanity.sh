#!/bin/sh
#SBATCH --job-name=search-ragtime1-cc-dense-sanity
#SBATCH --output=logs/search-ragtime1-cc-dense-sanity.out
#SBATCH --error=logs/search-ragtime1-cc-dense-sanity.err
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

# Sanity check: --run-file below is the qrel-filtered "relevant-only" pool
# (see pipeline/filter_relevant_run.py and runs/sanity-relevant-only/base/),
# not the raw Qwen3 top-1000. Every candidate here is already judged
# relevant, so relevance can no longer explain StRecall@k differences --
# only the order cc_dense chooses among relevant docs can. Compare against
# runs/sanity-relevant-only/base/run.ragtime1...relevant-only.txt
# (unreranked) to see the isolated effect of the diversity signal.
MODEL_NAME=Qwen3-Embedding-0.6B
EMB_ROOT=$HOME/scratch/ragtime1/${MODEL_NAME}

for AGG in mean; do
for MODE in subtract; do
for LAMBDA in 0.6 0.7 0.8 0.9; do
    python pipeline/run_cc_dense.py \
        --topics data/ragtime2025.topics.test.jsonl \
        --run-file runs/sanity-relevant-only/base/run.ragtime1.documents.Qwen3-Embedding-0.6B.relevant-only.txt \
        --corpus  "$HOME/scratch/ragtime1/*.processed-claims.jsonl.gz" \
        --claim-reps "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
        --output runs/sanity-relevant-only/reranked/run.ragtime1.documents.Qwen3-Embedding-0.6B.relevant-only.cc${AGG}-${MODE}.lambda-${LAMBDA}.txt \
        --k 20 \
        --lambda-mult ${LAMBDA} \
        --mode $MODE \
        --agg $AGG \
        --tag doc-cc${AGG}-${MODE}-l${LAMBDA}-relevant-only
done
done
done
