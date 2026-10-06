#!/bin/bash -l
#SBATCH --job-name=ragtime2-vllm-decontext-test
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

LANGS=(eng-docs rus-trans spa-trans zho-trans)
NUM_SHARDS=10000
SHARD=0
MODEL=meta-llama/Llama-3.1-8B-Instruct
PORT=8000
endpoint="0.0.0.0:${PORT}"
LOG=vllm_server.log

INPUT_DIR=$HOME/scratch/ragtime2/decontext-offload
OUTPUT_DIR=$HOME/scratch/ragtime2/decontext-final
mkdir -p "$OUTPUT_DIR"

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port "$PORT" \
    --enforce-eager \
    --max-model-len 20480 \
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

for lang in "${LANGS[@]}"; do
    echo "[${lang}] shard ${SHARD}-of-${NUM_SHARDS} -> ${endpoint}"
    python -m data_prep.decontext.consume_offload \
        --input_file "$INPUT_DIR/nuggetized_corpus.${lang}.jsonl.gz" \
        --output_file "$OUTPUT_DIR/nuggetized_corpus.${lang}.shard${SHARD}-of-${NUM_SHARDS}.jsonl" \
        --endpoint "$endpoint" \
        --model "$MODEL" \
        --max_tokens 8196 \
        --shard_index "${SHARD}" \
        --num_shards "${NUM_SHARDS}"
done

kill "$vllm_pid" 2>/dev/null
wait "$vllm_pid" 2>/dev/null
echo "done."
