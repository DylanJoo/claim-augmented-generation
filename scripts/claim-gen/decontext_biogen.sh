#!/bin/sh
#SBATCH --job-name=de
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
conda activate crux

# Start experiment
cd ~/decontextualize/legacy

python biogen_corpus.py --corpus_dir /home/hltcoe/jhueiju/temp/biogen --output_dir /home/hltcoe/jhueiju/temp/biogen 
