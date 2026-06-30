#!/bin/bash -l
#SBATCH --job-name=rac-eval
#SBATCH --output=logs/rac-eval.out
#SBATCH --error=logs/rac-eval.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=2:00:00

source ${HOME}/.bashrc
initconda
conda activate inference

cd ${HOME}/claim-augmented-generation/
for run in runs/run.*.txt; do
    python -m src.evaluator.rac_eval \
        --run $run \
        --qrel $HOME/trec2026/data/neuclir/neuclir24-test-request.qrel \
        --judge $HOME/trec2026/data/neuclir/neuclir24.ratings.human.jsonl
done
