#!/bin/bash -l
#SBATCH --job-name=rac-eval-curve
#SBATCH --output=logs/rac-eval-curve.out
#SBATCH --error=logs/rac-eval-curve.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=2:00:00

cd ${HOME}/claim-augmented-generation/

system=$1
if [ -z "$system" ]; then
    echo "usage: $0 <neuclir1|ragtime1|ragtime2>" >&2
    exit 1
fi

case "$system" in
    neuclir1)
        run_glob="runs/run.neuclir1*.txt"
        priority_runs="runs/neuclir1/run.neuclir1.documents.bm25.txt runs/neuclir1/run.neuclir1.concat-claims.bm25.txt runs/neuclir1/run.neuclir1.documents.modernbert-base.cover-5k.txt runs/neuclir1/run.neuclir1.documents.Qwen3-Embedding-0.6B.txt"
        qrel="$HOME/trec2026/data/neuclir/neuclir24-test-request.qrel"
        ;;
    ragtime1)
        run_glob="runs/run.ragtime1*.txt"
        priority_runs="runs/ragtime1/run.ragtime1.documents.bm25.txt runs/ragtime1/run.ragtime1.concat-claims.bm25.txt runs/ragtime1/run.ragtime1.documents.modernbert-base.cover-5k.txt runs/ragtime1/run.ragtime1.documents.Qwen3-Embedding-0.6B.txt"
        qrel="$HOME/trec2026/data/ragtime1/ragtime25-test-request.qrel"
        ;;
    ragtime2)
        run_glob="runs/ragtime2/run.ragtime2*.txt"
        priority_runs="runs/ragtime2/run.ragtime2.documents.Qwen3-Embedding-0.6B.txt runs/ragtime2/run.ragtime2.documents.modernbert-base.cover-5k.txt"
        qrel="$HOME/trec2026/data/ragtime2/ragtime26-test-request.qrel"
        ;;
    *)
        echo "unknown system: $system (expected neuclir1|ragtime1|ragtime2)" >&2
        exit 1
        ;;
esac

out="results/curve-${system}.csv"
mkdir -p "$(dirname "$out")"
rm -f "$out"

for run in $priority_runs $run_glob; do
    case " $seen " in
        *" $run "*) continue ;;
    esac
    seen="$seen $run"
    [ -f "$run" ] || continue
    python -m src.evaluator.rac_eval_curve \
        --run $run \
        --qrel $qrel \
        --out "$out"
done

echo "wrote $out"
