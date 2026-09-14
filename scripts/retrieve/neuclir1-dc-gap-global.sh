#!/bin/sh
#SBATCH --job-name=retrieve-dc-gap-global
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --cpus-per-task=16
#SBATCH --partition=small
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --account=project_465002532

# ENV
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

cd $HOME/claim-augmented-generation

# NOTE: $HOME/scratch/neuclir1/claims.bm25s was built by
# scripts/index/neuclir1-claims.sh with no --stemmer flag (stopwords=en,
# stemmer=none). --stemmer is left unset below to match that vocabulary --
# bm25s silently drops query tokens that aren't in the index, so stemming
# here while the index isn't stemmed would look like near-zero overlap
# rather than a fair local-vs-global comparison against
# neuclir1-dc-gap.sh's --stemmer snowball run.
singularity exec $SIF \
python pipeline/run_dc_gap_global.py \
    --topics data/neuclir2024.topics.test.jsonl \
    --run-file runs/run.neuclir1.documents.bm25.txt \
    --corpus  "$HOME/scratch/neuclir1/*.processed-claims.jsonl.gz" \
    --claim-index "$HOME/scratch/neuclir1/claims.bm25s" \
    --output runs/run.neuclir1.dc-gap-global-doc.txt \
    --k 1000 \
    --stopwords en \
    --tag dc-gap-global-doc
