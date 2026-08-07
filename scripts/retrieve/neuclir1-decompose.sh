#!/bin/sh
#SBATCH --job-name=decompose
#SBATCH --cpus-per-task=8
#SBATCH --partition gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x.out

source ~/.bashrc
initconda
conda activate inference

cd $HOME/claim-augmented-generation

python pipeline/run_decompose.py \
    --topics data/neuclir2024.topics.test.jsonl \
    --output data/neuclir2024.topics.test.subq.jsonl \
    --n-questions 10 \
    --model Qwen/Qwen3-8B
