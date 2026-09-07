#!/bin/bash -l
#SBATCH --job-name=neuclir1-rerank-llm
#SBATCH --partition=dev-g
#SBATCH --account=project_465002438
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err

module --force purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
export HIP_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export VLLM_SKIP_P2P_CHECK=1

cd $HOME/claim-augmented-generation

MODEL=meta-llama/Llama-3.1-8B-Instruct
METHOD=rankgpt
PORT=8000
endpoint="0.0.0.0:${PORT}"
LOG=vllm_server.log

INPUT_RUN=runs/run.neuclir1.documents.modernbert-base.cover-5k.txt
OUTPUT_RUN=runs/run.neuclir1.documents.modernbert-base.cover-5k.rerank-${METHOD}.txt

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port "$PORT" \
    --enforce-eager \
    --max-model-len 30720 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1 > $LOG 2>&1 &

vllm_pid=$!
trap 'kill "$vllm_pid" 2>/dev/null' EXIT

echo "waiting for server at ${endpoint}..."
until curl -s -o /dev/null "http://${endpoint}/v1/models"; do
    sleep 10
done
echo "server ready."

srun singularity exec $SIF \
python pipeline/run_rerank_llm.py \
    --topics data/neuclir2024.topics.test.jsonl \
    --run-file "$INPUT_RUN" \
    --corpus "$HOME/scratch/neuclir1/*.processed-claims.jsonl.gz" \
    --output "$OUTPUT_RUN" \
    --method $METHOD \
    --max_doc_length 512 \
    --model $MODEL \
    --backend request \
    --base-url "http://${endpoint}/v1" \
    --k 100 \
    --tag rerank-${METHOD}-neuclir1

kill "$vllm_pid" 2>/dev/null
wait "$vllm_pid" 2>/dev/null
echo "done."
