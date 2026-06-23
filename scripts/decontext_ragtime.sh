#!/bin/sh
#SBATCH --job-name=nug-all
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=120:00:00
#SBATCH --output=%x-%j.out

# Set-up the environment.
source ~/.bashrc
enter_conda
conda activate crc

# Start experiment
cd ~/decontextaulize/src
echo It is running on $lang task

python ragtime_corpus.py \
    --corpus_dir /exp/scale25/ragtime/docs \
    --output_dir /exp/jhueiju/ragtime \
    --langs mlir --use_translation \
    --offload

# python -m nuggetize.neuclir_corpus \
#     --corpus_dir /exp/scale25/neuclir/docs \
#     --output_dir /exp/jhueiju/neuclir \
#     --langs ${lang} --use_translation

