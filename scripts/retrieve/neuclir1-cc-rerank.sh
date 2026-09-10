#!/bin/sh
#SBATCH --job-name=retrieve-cc-rerank
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --cpus-per-task=16
#SBATCH --partition=small
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --account=project_465002532

# ENV
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

cd $HOME/claim-augmented-generation

# Now just use the max. 
# TODO: Maybe try portion of high sim?

# MMR with claim-claim scores (mode=subtract)
# Use sum of max claim-claim sim as redundancy penalty
# for LAMBDA in 0.7 0.8 0.9 1.0; do
#     python pipeline/run_cc.py \
#         --topics data/neuclir2024.topics.test.jsonl \
#         --run-file runs/run.neuclir1.documents.bm25.txt \
#         --corpus "$HOME/scratch/neuclir1/*.processed-claims.jsonl.gz" \
#         --output runs/run.neuclir1.cc-rerank-mmr-doc.lambda-${LAMBDA}.txt \
#         --k 1000 \
#         --lambda-mult ${LAMBDA} \
#         --mode 'subtract' \
#         --stopwords en \
#         --stemmer snowball \
#         --tag cc-rerank-mmr-doc-l${LAMBDA}
# done

# Claim-echo boost rerank (mode=add)
# Reweight the document relevance upward when claims echo already-selected docs.
for LAMBDA in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    singularity exec $SIF \
    python pipeline/run_cc.py \
        --topics data/neuclir2024.topics.test.jsonl \
        --run-file runs/run.neuclir1.documents.bm25.txt \
        --corpus "$HOME/scratch/neuclir1/*.processed-claims.jsonl.gz" \
        --output runs/run.neuclir1.cc-rerank-boost-doc.lambda-${LAMBDA}.txt \
        --k 1000 \
        --lambda-mult ${LAMBDA} \
        --mode 'add' \
        --stopwords en \
        --stemmer snowball \
        --tag cc-rerank-boost-doc-l${LAMBDA}
done
