#!/bin/sh
#SBATCH --job-name=retrieve-h-doc
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --cpus-per-task=16
#SBATCH --partition=small
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --account=project_465002532

# ENV
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

cd $HOME/claim-augmented-generation

for ALPHA in 0.0 0.1 0.2 0.3 0.4 0.5; do
    singularity exec $SIF \
    python pipeline/run_hybrid.py \
        --topics data/neuclir2024.topics.test.jsonl \
        --claim-index "$HOME/scratch/neuclir1/claims.bm25s" \
        --aux-index   "$HOME/scratch/neuclir1/documents.bm25s" \
        --output runs/run.neuclir1.hybrid-claim-doc.alpha-${ALPHA}.txt \
        --k-claim 1000 \
        --k-aux 1000 \
        --claim-fusion sum \
        --alpha ${ALPHA} \
        --stopwords en \
        --stemmer snowball \
        --tag hybrid-claim-doc-a${ALPHA}
done

# TODO: maybe tring larger k for claims? --> so the derived parent docs could meet top 1000?
# but maybe this is not necessary because we only care about the top-ranking documents?
