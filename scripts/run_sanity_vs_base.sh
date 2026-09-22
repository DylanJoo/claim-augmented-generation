#!/bin/bash -l
#SBATCH --job-name=rac-sanity-vs-base
#SBATCH --output=logs/rac-sanity-vs-base.out
#SBATCH --error=logs/rac-sanity-vs-base.err
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=1:00:00

# Paired significance test of every relevant-only sanity-check reranked run
# against its base run (see src/evaluator/sanity_vs_base.py). Extra args are
# passed through, e.g. `--metric StRecall@10`.
#   sbatch scripts/run_sanity_vs_base.sh ragtime1 [--metric alpha_nDCG@20]

cd ${HOME}/claim-augmented-generation/
source ~/.bashrc
initconda
conda activate basic

system=$1; shift
case "$system" in
    neuclir1) qrel="$HOME/trec2026/data/neuclir/neuclir24-test-request.qrel" ;;
    ragtime1) qrel="$HOME/trec2026/data/ragtime1/ragtime25-test-request.qrel" ;;
    *) echo "usage: $0 <neuclir1|ragtime1> [sanity_vs_base args]" >&2; exit 1 ;;
esac

python -m src.evaluator.sanity_vs_base \
    --base runs/sanity-relevant-only/base/run.${system}.documents.Qwen3-Embedding-0.6B.relevant-only.txt \
    --runs "runs/sanity-relevant-only/reranked/run.${system}*.txt" \
    --qrel $qrel "$@"
