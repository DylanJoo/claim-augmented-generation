#!/bin/bash
#SBATCH --job-name=search-ragtime1-cc-dense
#SBATCH --output=logs/search-ragtime1-cc-dense_%A_%a.out
#SBATCH --error=logs/search-ragtime1-cc-dense_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --array=0-94%8

# ENV
source ~/.bashrc
initconda
conda activate basic

cd $HOME/claim-augmented-generation

MODEL_NAME=Qwen3-Embedding-0.6B
EMB_ROOT=$HOME/scratch/ragtime1/${MODEL_NAME}

# Reduced lambda grid shared by all three blocks below (was 0.1-0.9 step 0.1;
# cut to keep total job count sane now that two more dimensions are swept).
LAMBDAS=(0.0 0.5 0.6 0.7 1.0)

# Each array task runs exactly one config, indexed by SLURM_ARRAY_TASK_ID
# into the flattened grid built below (0-94, must match --array above).
EXTRA_ARGS=()
OUT_SUFFIX=()
TAG_SUFFIX=()

# --- Block 1: baseline AGG x MODE x LAMBDA sweep (claim_filter=none, novelty_weight=0) --- (20 configs)
for AGG in maxsim mean; do
for MODE in add subtract; do
for LAMBDA in "${LAMBDAS[@]}"; do
    EXTRA_ARGS+=("--lambda-mult ${LAMBDA} --mode ${MODE} --agg ${AGG}")
    OUT_SUFFIX+=("cc${AGG}-${MODE}.lambda-${LAMBDA}")
    TAG_SUFFIX+=("doc-cc${AGG}-${MODE}-l${LAMBDA}")
done
done
done

# --- Block 2: claim-coverage / novelty bonus sweep, AGG/MODE fixed at the
# baseline's best-performing combo (maxsim/subtract) --- (45 configs)
AGG=maxsim
MODE=subtract
for LAMBDA in "${LAMBDAS[@]}"; do
for NOVELTY in 0.1 0.3 0.5; do
for COVTHR in 0.7 0.8 0.9; do
    EXTRA_ARGS+=("--lambda-mult ${LAMBDA} --mode ${MODE} --agg ${AGG} --novelty-weight ${NOVELTY} --coverage-threshold ${COVTHR}")
    OUT_SUFFIX+=("cc${AGG}-${MODE}.lambda-${LAMBDA}.nov-${NOVELTY}.covthr-${COVTHR}")
    TAG_SUFFIX+=("doc-cc${AGG}-${MODE}-l${LAMBDA}-nov${NOVELTY}-covthr${COVTHR}")
done
done
done

# --- Block 3: query-claim filter sweep, AGG/MODE fixed the same way --- (30 configs)
# Replaces the old per-doc "topn" claim_filter (removed from cc_dense.py --
# keeping only a document's N highest query-similarity claims was found to
# discard real nugget-matching claims alongside noise, since a claim's own
# query-cosine doesn't reliably separate the two -- see src/retrieval/
# cc_dense.py's module docstring). The three modes below are its validated
# replacements: "abs_threshold"/"topic_percentile" are flat cosine cutoffs
# (global vs. resolved per topic); "neighbor_rescue" additionally lets a
# claim borrow its best same-pool neighbor's query-score before that cutoff,
# recovering real nuggets that score low only because of their own
# (often vocabulary-mismatched) phrasing.
AGG=maxsim
MODE=subtract
FILTER_CONFIGS=(
    "abs_threshold --claim-sim-threshold 0.30|absthr-0.30"
    "abs_threshold --claim-sim-threshold 0.35|absthr-0.35"
    "topic_percentile --topic-percentile 25|toppct-25"
    "topic_percentile --topic-percentile 40|toppct-40"
    "neighbor_rescue --neighbor-threshold 0.5 --claim-sim-threshold 0.45|nbresc-0.45"
    "neighbor_rescue --neighbor-threshold 0.5 --claim-sim-threshold 0.50|nbresc-0.50"
)
for LAMBDA in "${LAMBDAS[@]}"; do
for CFG in "${FILTER_CONFIGS[@]}"; do
    FILTER_ARGS="${CFG%%|*}"
    FILTER_TAG="${CFG##*|}"
    FILTER_MODE="${FILTER_ARGS%% *}"
    EXTRA_ARGS+=("--lambda-mult ${LAMBDA} --mode ${MODE} --agg ${AGG} --query-reps ${EMB_ROOT}/queries_emb/queries_emb.pkl --claim-filter ${FILTER_MODE} ${FILTER_ARGS#* }")
    OUT_SUFFIX+=("cc${AGG}-${MODE}.lambda-${LAMBDA}.${FILTER_TAG}")
    TAG_SUFFIX+=("doc-cc${AGG}-${MODE}-l${LAMBDA}-${FILTER_TAG}")
done
done

echo "Total configs: ${#EXTRA_ARGS[@]} (expect 95, array indices 0-94)"

IDX=${SLURM_ARRAY_TASK_ID}
if [ -z "$IDX" ] || [ "$IDX" -ge "${#EXTRA_ARGS[@]}" ]; then
    echo "SLURM_ARRAY_TASK_ID=${IDX} out of range for ${#EXTRA_ARGS[@]} configs -- submit with sbatch, not sh" >&2
    exit 1
fi

echo "Task ${IDX}: ${TAG_SUFFIX[$IDX]}"

python pipeline/run_cc_dense.py \
    --topics data/ragtime2025.topics.test.jsonl \
    --run-file runs/run.ragtime1.documents.Qwen3-Embedding-0.6B.txt \
    --corpus  "$HOME/scratch/ragtime1/*.processed-claims.jsonl.gz" \
    --claim-reps "$EMB_ROOT/claims_emb/claims_emb.*.pkl" \
    --output runs/ragtime1/run.ragtime1.documents.Qwen3-Embedding-0.6B.${OUT_SUFFIX[$IDX]}.txt \
    --k 100 \
    ${EXTRA_ARGS[$IDX]} \
    --tag ${TAG_SUFFIX[$IDX]}
