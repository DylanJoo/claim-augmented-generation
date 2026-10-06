#!/bin/bash -l
#SBATCH --job-name=rac-eval-full
#SBATCH --output=logs/rac-eval-full-%j.out
#SBATCH --error=logs/rac-eval-full-%j.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=2:00:00

# usage: run_eval_full.sh <neuclir1|ragtime1> [extra run paths/globs ...]
#
# Evaluates the BASELINES + SELECTED lists below (in that order), plus any
# extra paths/globs given on the command line. Entries are relative to the
# repo root, may be globs, and use {sys} as a placeholder for the system.

cd ${HOME}/claim-augmented-generation/

system=$1
if [ -z "$system" ]; then
    echo "usage: $0 <neuclir1|ragtime1> [extra run paths/globs ...]" >&2
    exit 1
fi
shift

case "$system" in
    neuclir1) qrel="$HOME/trec2026/data/neuclir/neuclir24-test-request.qrel" ;;
    ragtime1) qrel="$HOME/trec2026/data/ragtime1/ragtime25-test-request.qrel" ;;
    *)
        echo "unknown system: $system (expected neuclir1|ragtime1)" >&2
        exit 1
        ;;
esac

# baselines: first-stage retrievers and LLM rerankers on the dense runs
BASELINES=(
    "runs/run.{sys}.documents.bm25.txt"
    # "runs/run.{sys}.concat-claims.bm25.txt"
    # "runs/run.{sys}.documents.modernbert-base.cover-5k.txt"
    "runs/run.{sys}.documents.Qwen3-Embedding-0.6B.txt"
    # "runs/relrerank/run.{sys}.modernbert-base.cover-5k.autollmrerank-70b-lancer.txt"
    "runs/relrerank/run.{sys}.documents.Qwen3-Embedding-0.6B.autollmrerank-70b-lancer.txt"
)

# selected runs: add the configs worth a full eval here
SELECTED=(
    # DI
    "runs/hybrid/run.{sys}.hybrid-claim-doc.modernbert-base.cover-5k.alpha-0.3.txt"
    # DICE
    "runs/dice/run.{sys}.hybrid-a0.8.Qwen3-Embedding-0.6B.cckmeans.top20-k50-binary.alpha-0.2.lambda-0.5.txt"
)

out="results/${system}-full.md"
mkdir -p "$(dirname "$out")"

# expand {sys} and globs; unmatched globs stay literal in bash, so keep only
# files that exist and drop duplicates
seen=""
runs=()
for pattern in "${BASELINES[@]}" "${SELECTED[@]}" "$@"; do
    pattern="${pattern//\{sys\}/$system}"
    matched=0
    for run in $pattern; do
        [ -f "$run" ] || continue
        matched=1
        case " $seen " in
            *" $run "*) continue ;;
        esac
        seen="$seen $run"
        runs+=("$run")
    done
    [ "$matched" -eq 1 ] || echo "warning: no run matches $pattern" >&2
done

if [ "${#runs[@]}" -eq 0 ]; then
    echo "no runs to evaluate" >&2
    exit 1
fi
echo "evaluating ${#runs[@]} runs" >&2

python -m src.evaluator.rac_eval_full \
    --run "${runs[@]}" \
    --qrel $qrel | tee "$out"

echo "wrote $out"
