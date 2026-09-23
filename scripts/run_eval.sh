#!/bin/bash -l
#SBATCH --job-name=rac-eval
#SBATCH --output=logs/rac-eval.out
#SBATCH --error=logs/rac-eval.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=2:00:00

cd ${HOME}/claim-augmented-generation/

system=$1
scope=$2
if [ -z "$system" ] || [ -z "$scope" ]; then
    echo "usage: $0 <neuclir1|ragtime1> <first|new|relrerank|sanity>" >&2
    exit 1
fi

case "$scope" in
    first|new|relrerank|sanity) ;;
    *)
        echo "unknown scope: $scope (expected first|new|relrerank|sanity)" >&2
        exit 1
        ;;
esac

case "$system" in
    neuclir1)
        first_glob="runs/run.neuclir1*.txt"
        new_glob="runs/neuclir1/run.neuclir1*.txt"
        sanity_glob="runs/sanity-relevant-only/base/run.neuclir1*.txt runs/sanity-relevant-only/reranked/run.neuclir1*.txt"
        qrel="$HOME/trec2026/data/neuclir/neuclir24-test-request.qrel"
        ;;
    ragtime1)
        first_glob="runs/run.ragtime1*.txt"
        new_glob="runs/ragtime1/run.ragtime1*.txt"
        sanity_glob="runs/sanity-relevant-only/base/run.ragtime1*.txt runs/sanity-relevant-only/reranked/run.ragtime1*.txt"
        qrel="$HOME/trec2026/data/ragtime1/ragtime25-test-request.qrel"
        ;;
    *)
        echo "unknown system: $system (expected neuclir1|ragtime1)" >&2
        exit 1
        ;;
esac

if [ "$scope" = "first" ]; then
    run_glob="$first_glob"
    header="runs"
elif [ "$scope" = "new" ]; then
    run_glob="$new_glob"
    header="runs"
else
    # sanity check: every candidate is judged relevant, so differences isolate
    # the reranker's ordering (see scripts/run_eval_sanity.sh)
    run_glob="$sanity_glob"
    header="sanity-relevant-only"
fi

out="results/${system}-${scope}.md"
mkdir -p "$(dirname "$out")"

# unmatched globs stay literal in bash; keep only files that exist
runs=()
for run in $run_glob; do
    [ -f "$run" ] && runs+=("$run")
done

{
    echo
    python -m src.evaluator.rac_eval \
        --run "${runs[@]}" \
        --qrel $qrel
} | tee "$out"

echo "wrote $out"
