#!/bin/sh
#SBATCH --job-name=index-claims
#SBATCH --output=logs/index-claims.out
#SBATCH --error=logs/index-claims.err
#SBATCH --cpus-per-task=16
#SBATCH --partition=small
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --account=project_465002532

# ENV
module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

cd $HOME/claim-augmented-generation
srun singularity exec $SIF \
    python src/retrieval/indexing.py \
    --input $HOME/scratch/ragtime1/*.processed-claims.jsonl.gz \
    --index $HOME/scratch/ragtime1/claims.bm25s \
    --claim-level --k1 1.2 --b 0.5
