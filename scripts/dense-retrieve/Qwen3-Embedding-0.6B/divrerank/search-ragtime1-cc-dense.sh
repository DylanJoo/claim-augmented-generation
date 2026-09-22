#!/bin/bash
#SBATCH --job-name=search-ragtime1-cc-dense
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00

# ENV
source ~/.bashrc
initconda
conda activate basic

cd $HOME/claim-augmented-generation

MODEL_NAME=Qwen3-Embedding-0.6B
EMB_ROOT=$HOME/scratch/ragtime1/${MODEL_NAME}
QUERY_REPS=${EMB_ROOT}/queries_emb/queries_emb.pkl

# Reduced lambda grid shared by all blocks below (was 0.1-0.9 step 0.1; cut
# to keep total run count sane now that more dimensions are swept).
LAMBDAS="0.8 0.9"

# --- Block 1: baseline AGG x MODE x LAMBDA sweep (claim_filter=none) --- (20 runs)
# for AGG in maxsim mean; do
# for MODE in add subtract; do
# for LAMBDA in $LAMBDAS; do
#     python pipeline/run_cc_dense.py \
#         --topics data/ragtime2025.topics.test.jsonl \
#         --run-file runs/run.ragtime1.documents.Qwen3-Embedding-0.6B.txt \
#         --corpus  "$HOME/scratch/ragtime1/*.processed-claims.jsonl.gz" \
#         --claim-reps "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
#         --output runs/ragtime1/run.ragtime1.documents.Qwen3-Embedding-0.6B.cc${AGG}-${MODE}.lambda-${LAMBDA}.txt \
#         --k 100 \
#         --lambda-mult ${LAMBDA} \
#         --mode ${MODE} \
#         --agg ${AGG} \
#         --tag doc-cc${AGG}-${MODE}-l${LAMBDA}
# done
# done
# done

# --- Block 2: document filter sweep, AGG/MODE fixed the same way --- (9 runs)
# Replaces the old per-doc "topn" claim_filter (removed from cc_dense.py --
# keeping only a document's N highest query-similarity claims was found to
# discard real nugget-matching claims alongside noise). A pooled document's
# query score is its best claim's query-cosine; the whole document is dropped
# if that is below --doc-threshold. On Qwen3 the per-doc scores have median
# ~0.55 (p10 ~0.47, p90 ~0.69), so 0.5/0.55/0.6 drop roughly 25%/50%/65%.
AGG=maxsim
MODE=subtract

for LAMBDA in $LAMBDAS; do
for THRESHOLD in 0.5 0.55 0.6; do
    python pipeline/run_cc_dense.py \
        --topics data/ragtime2025.topics.test.jsonl \
        --run-file runs/run.ragtime1.documents.Qwen3-Embedding-0.6B.txt \
        --corpus  "$HOME/scratch/ragtime1/*.processed-claims.jsonl.gz" \
        --claim-reps "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
        --output runs/ragtime1/run.ragtime1.documents.Qwen3-Embedding-0.6B.cc${AGG}-${MODE}.lambda-${LAMBDA}.docthr-${THRESHOLD}.txt \
        --k 100 \
        --lambda-mult ${LAMBDA} \
        --mode ${MODE} \
        --agg ${AGG} \
        --query-reps ${QUERY_REPS} \
        --doc-threshold ${THRESHOLD} \
        --tag doc-cc${AGG}-${MODE}-l${LAMBDA}-docthr${THRESHOLD}
done
done
