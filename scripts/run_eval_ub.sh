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

K2=50
priority_runs="runs/run.neuclir1.documents.bm25.txt runs/run.neuclir1.doc-claims.bm25.txt runs/run.neuclir1.concat-claims.bm25.txt"

cd ${HOME}/claim-augmented-generation/
echo "| UB(Run) | StRecall@1 | StRecall@10 | StRecall@20 | alpha_nDCG@10 | alpha_nDCG@20 |"
echo "|---|---|---|---|---|---|"
for run in $priority_runs runs/run.neuclir1*.txt; do
    case " $seen " in
        *" $run "*) continue ;;
    esac
    seen="$seen $run"
    python -m src.evaluator.rac_eval_ub \
        --run $run \
        --qrel $HOME/trec2026/data/neuclir/neuclir24-test-request.qrel \
        --k2 $K2
done
