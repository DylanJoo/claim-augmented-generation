#!/bin/bash -l
#SBATCH --job-name=decontext-ragtime2-offload
#SBATCH --partition=small
#SBATCH --account=project_465002438
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err

module --force purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

cd $HOME/claim-augmented-generation

python -m data_prep.decontext.ragtime2 \
    --corpus_dir $HOME/scratch/ragtime2 \
    --output_dir $HOME/scratch/ragtime2/decontext-offload \
    --langs eng-docs rus-trans spa-trans zho-trans \
    --use_translation \
    --offload
