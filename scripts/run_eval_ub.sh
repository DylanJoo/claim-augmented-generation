#!/bin/bash -l
#SBATCH --job-name=rac-eval-ub
#SBATCH --output=logs/rac-eval-ub.out
#SBATCH --error=logs/rac-eval-ub.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=2:00:00

source ${HOME}/.bashrc
initconda
conda activate inference

cd ${HOME}/claim-augmented-generation/
K2=100
echo "| UB(Run) | StRecall@1 | StRecall@10 | StRecall@20 | alpha_nDCG@10 | alpha_nDCG@20 |"
echo "|---|---|---|---|---|---|"
for run in runs/run.neuclir1*.txt; do
    python -m src.evaluator.rac_eval_ub \
        --run $run \
        --qrel $HOME/trec2026/data/neuclir/neuclir24-test-request.qrel \
        --k2 $K2
done
