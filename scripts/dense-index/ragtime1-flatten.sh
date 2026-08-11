#!/bin/bash -l
#SBATCH --job-name=ragtime1-flatten
#SBATCH --output=logs/ragtime1-flatten.out.%a
#SBATCH --error=logs/ragtime1-flatten.err.%a
#SBATCH --partition=standard
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --account=project_465002438

# ENV
module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

LANGS=(
"arb-trans"
"eng-docs"
"rus-trans"
"zho-trans"
)
LANG=${LANGS[$SLURM_ARRAY_TASK_ID]}

FLAT_CLAIMS=${HOME}/scratch/ragtime1/claims_flat/${LANG}.claims.jsonl.gz
mkdir -p ${HOME}/scratch/ragtime1/claims_flat

cd $HOME/claim-augmented-generation

echo Flattening ragtime1 claims for $LANG
srun singularity exec $SIF \
    python -m src.retrieval.flatten_claims \
    --input ${HOME}/scratch/ragtime1/${LANG}.processed-claims.jsonl.gz \
    --output $FLAT_CLAIMS
