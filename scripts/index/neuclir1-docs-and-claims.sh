#!/bin/sh
#SBATCH --job-name=index-dc
#SBATCH --cpus-per-task=32
#SBATCH --partition=largemem
#SBATCH --mem=1024G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.out
#SBATCH --account=project_465002532


cd $HOME/claim-augmented-generation
srun singularity exec $SIF \
    python src/retrieval/indexing.py \
    --input $HOME/scratch/neuclir1/*.processed-claims.jsonl.gz \
    --index $HOME/scratch/neuclir1/docs-and-claims.bm25s \
    --claim-level --doc-augmented \
    --include-title --k1 1.2 --b 0.75
