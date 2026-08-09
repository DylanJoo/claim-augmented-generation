#!/bin/bash -l
#SBATCH --job-name=neuclir-encode-claims
#SBATCH --output=logs/neuclir-enc.out.%a
#SBATCH --error=logs/neuclir-enc.err.%a
#SBATCH --partition=standard
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0,1,2
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --account=project_465002438

# ENV
module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

LANGS=(
"fas"
"rus"
"zho"
)
LANG=${LANGS[$SLURM_ARRAY_TASK_ID]}

CLAIMS_DIR=${HOME}/scratch/neuclir1/claims_flat
FLAT_CLAIMS=${CLAIMS_DIR}/${LANG}.flatten-claims.jsonl.gz
mkdir -p $CLAIMS_DIR

cd $HOME/claim-augmented-generation

echo Flattening NeuCLIR1 claims for $LANG
srun singularity exec $SIF \
    python -m src.retrieval.flatten_claims \
    --input ${HOME}/scratch/neuclir1/${LANG}.processed-claims.jsonl.gz \
    --output $FLAT_CLAIMS
