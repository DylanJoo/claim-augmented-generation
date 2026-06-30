#!/bin/sh
#SBATCH --job-name=nug-all
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --array=1-3%3
#SBATCH --ntasks-per-node=1
#SBATCH --time=120:00:00
#SBATCH --output=%x-%j.out

# Set-up the environment.
source ~/.bashrc
enter_conda
conda activate crc

MULTIJOBS=${HOME}/multilang.txt
lang=$(head -$SLURM_ARRAY_TASK_ID $MULTIJOBS | tail -1)

# Start experiment
cd ~/decontextaulize/src
echo It is running on $lang task

python -m nuggetize.neuclir_corpus \
    --corpus_dir /exp/scale25/neuclir/docs \
    --output_dir /exp/jhueiju/neuclir \
    --langs ${lang} --use_translation \
    --offload

# python -m nuggetize.neuclir_corpus \
#     --corpus_dir /exp/scale25/neuclir/docs \
#     --output_dir /exp/jhueiju/neuclir \
#     --langs ${lang} --use_translation

