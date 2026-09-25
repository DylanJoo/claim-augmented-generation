#!/bin/bash -l
#SBATCH --job-name=rac-eval-full
#SBATCH --output=logs/rac-eval-full-%j.out
#SBATCH --error=logs/rac-eval-full-%j.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=2:00:00

cd ${HOME}/claim-augmented-generation/

system=$1
if [ -z "$system" ]; then
    echo "usage: $0 <neuclir1|ragtime1>" >&2
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

# scopes: first (runs/*.txt) + claim-based-scoring; dense baselines listed first
priority_runs="runs/run.${system}.documents.modernbert-base.cover-5k.txt runs/run.${system}.documents.Qwen3-Embedding-0.6B.txt"
run_glob="runs/run.${system}*.txt runs/claim-based-scoring/run.${system}*.txt"

out="results/${system}-full.md"
mkdir -p "$(dirname "$out")"

# unmatched globs stay literal in bash; keep only files that exist
seen=""
runs=()
for run in $priority_runs $run_glob; do
    case " $seen " in
        *" $run "*) continue ;;
    esac
    seen="$seen $run"
    [ -f "$run" ] || continue
    runs+=("$run")
done
python -m src.evaluator.rac_eval_full \
    --run "${runs[@]}" \
    --qrel $qrel | tee "$out"

echo "wrote $out"
