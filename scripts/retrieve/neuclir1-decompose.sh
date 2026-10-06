#!/bin/sh
#SBATCH --job-name=decompose
#SBATCH --output=logs/decompose.out
#SBATCH --error=logs/decompose.err
#SBATCH --partition=dev-g
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --account=project_465002438

# ENV 
module --force purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

cd $HOME/claim-augmented-generation

python pipeline/run_decompose.py \
    --topics data/neuclir2024.topics.test.jsonl \
    --output data/neuclir2024.topics.test.subq.jsonl \
    --n-questions 10 \
    --model meta-llama/Llama-3.3-70B-Instruct
