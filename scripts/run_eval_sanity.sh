#!/bin/bash -l
#SBATCH --job-name=rac-eval-sanity
#SBATCH --output=logs/rac-eval-sanity.out
#SBATCH --error=logs/rac-eval-sanity.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=2:00:00

# Evaluates the relevant-only sanity-check runs (see
# pipeline/filter_relevant_run.py and runs/sanity-relevant-only/) against the
# same qrels used by scripts/run_eval.sh. Because every candidate in these
# runs is already judged relevant, differences in StRecall@k across rows
# isolate the effect of the diversity reranker's ordering -- relevance can no
# longer explain them.

cd ${HOME}/claim-augmented-generation/

system=$1
if [ -z "$system" ]; then
    echo "usage: $0 <neuclir1|ragtime1>" >&2
    exit 1
fi

case "$system" in
    neuclir1)
        run_glob="runs/sanity-relevant-only/base/run.neuclir1*.txt runs/sanity-relevant-only/reranked/run.neuclir1*.txt"
        qrel="$HOME/trec2026/data/neuclir/neuclir24-test-request.qrel"
        ;;
    ragtime1)
        run_glob="runs/sanity-relevant-only/base/run.ragtime1*.txt runs/sanity-relevant-only/reranked/run.ragtime1*.txt"
        qrel="$HOME/trec2026/data/ragtime1/ragtime25-test-request.qrel"
        ;;
    *)
        echo "unknown system: $system (expected neuclir1|ragtime1)" >&2
        exit 1
        ;;
esac

echo "| Run | StRecall@1 | @2 | @3 | @4 | @5 | @6 | @7 | @8 | @9 | @10 |"
echo "|---|---|---|---|---|---|---|---|---|---|---|"
for run in $run_glob; do
    python -m src.evaluator.rac_eval \
        --run $run \
        --qrel $qrel
done
