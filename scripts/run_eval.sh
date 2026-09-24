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
scopes="first|cckmeans|claim-based-scoring|claims-agg|hybrid|relrerank|sanity"
if [ -z "$system" ] || [ -z "$scope" ]; then
    echo "usage: $0 <neuclir1|ragtime1> <$scopes>" >&2
    exit 1
fi

case "$system" in
    neuclir1) qrel="$HOME/trec2026/data/neuclir/neuclir24-test-request.qrel" ;;
    ragtime1) qrel="$HOME/trec2026/data/ragtime1/ragtime25-test-request.qrel" ;;
    *)
        echo "unknown system: $system (expected neuclir1|ragtime1)" >&2
        exit 1
        ;;
esac

case "$scope" in
    first)
        run_glob="runs/run.${system}*.txt"
        ;;
    cckmeans|claim-based-scoring|claims-agg|hybrid|relrerank)
        run_glob="runs/${scope}/run.${system}*.txt"
        ;;
    sanity)
        # sanity check: every candidate is judged relevant, so differences isolate
        # the reranker's ordering (see dense-retrieve/create_sanity_runs.sh)
        run_glob="runs/sanity-relevant-only/base/run.${system}*.txt runs/sanity-relevant-only/reranked/run.${system}*.txt"
        ;;
    *)
        echo "unknown scope: $scope (expected $scopes)" >&2
        exit 1
        ;;
esac

out="results/${system}-${scope}.md"
mkdir -p "$(dirname "$out")"

# unmatched globs stay literal in bash; keep only files that exist
runs=()
for run in $run_glob; do
    [ -f "$run" ] && runs+=("$run")
done

python -m src.evaluator.rac_eval \
    --run "${runs[@]}" \
    --qrel $qrel \
    | tee "$out"

echo "wrote $out"
