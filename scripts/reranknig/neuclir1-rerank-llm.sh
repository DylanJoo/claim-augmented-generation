#!/bin/bash -l
#SBATCH --job-name=neuclir1-rerank-llm
#SBATCH --partition=standard-g
#SBATCH --account=project_465002438
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err

module --force purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1
export VLLM_SKIP_P2P_CHECK=1

cd $HOME/claim-augmented-generation

MODEL=meta-llama/Llama-3.3-70B-Instruct
PORT=8000
endpoint="0.0.0.0:${PORT}"
LOG=vllm_server.log

METHODS=(point rankgpt umbrela)

INPUT_RUNS=(
    # runs/run.neuclir1.documents.modernbert-base.cover-5k.txt
    runs/run.neuclir1.documents.Qwen3-Embedding-0.6B.txt
)

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port "$PORT" \
    --enforce-eager \
    --max-model-len 81960 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 8 > $LOG 2>&1 &

vllm_pid=$!
trap 'kill "$vllm_pid" 2>/dev/null' EXIT

echo "waiting for server at ${endpoint}..."
until curl -s -o /dev/null "http://${endpoint}/v1/models"; do
    sleep 10
done
echo "server ready."

for METHOD in "${METHODS[@]}"; do
    for INPUT_RUN in "${INPUT_RUNS[@]}"; do
        OUTPUT_RUN="${INPUT_RUN%.txt}.autollmrerank-70b-${METHOD}.txt"

        srun singularity exec $SIF \
        python pipeline/run_rerank_llm.py \
            --topics data/neuclir2024.topics.test.jsonl \
            --run-file "$INPUT_RUN" \
            --corpus "$HOME/scratch/neuclir1/*.processed-claims.jsonl.gz" \
            --output "$OUTPUT_RUN" \
            --method $METHOD \
            --max_doc_length 1024 \
            --model $MODEL \
            --backend request \
            --base-url "http://${endpoint}/v1" \
            --k 100 \
            --tag rerank-${METHOD}-neuclir1
    done
done

kill "$vllm_pid" 2>/dev/null
wait "$vllm_pid" 2>/dev/null
echo "done."
