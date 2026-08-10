#!/bin/bash -l
#SBATCH --job-name=ragtime2-vllm-decontext
#SBATCH --partition=small-g
#SBATCH --account=project_465002438
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --gpus-per-node=8
#SBATCH --time=3-00:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%a.out
#SBATCH --error=logs/%x-%a.err

module --force purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

cd $HOME/claim-augmented-generation

LANGS=(eng-docs rus-trans spa-trans zho-trans)
NUM_SHARDS=4
MODEL=meta-llama/Llama-3.3-70B-Instruct
BASE_PORT=8000

lang=${LANGS[$(( SLURM_ARRAY_TASK_ID / NUM_SHARDS ))]}
shard=$(( SLURM_ARRAY_TASK_ID % NUM_SHARDS ))
port=$(( BASE_PORT + SLURM_ARRAY_TASK_ID ))
endpoint="0.0.0.0:${port}"
LOG=vllm_server.log

INPUT_DIR=$HOME/scratch/ragtime2/decontext-offload
OUTPUT_DIR=$HOME/scratch/ragtime2/decontext-final
mkdir -p "$OUTPUT_DIR"

echo "[task ${SLURM_ARRAY_TASK_ID}] lang=${lang} shard=${shard} -> ${endpoint}"
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port "$port" \
    --enforce-eager \
    --max-model-len 40960 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 8 > $LOG 2>&1 &

vllm_pid=$!
trap 'kill "$vllm_pid" 2>/dev/null' EXIT

echo "[task ${SLURM_ARRAY_TASK_ID}] waiting for server at ${endpoint}..."
until curl -s -o /dev/null "http://${endpoint}/v1/models"; do
    sleep 10
done
echo "[task ${SLURM_ARRAY_TASK_ID}] server process up, waiting for model to finish loading..."
until curl -s -o /dev/null -w "%{http_code}" \
        --max-time 60 \
        -X POST "http://${endpoint}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": 1}" \
        | grep -q '^200$'; do
    sleep 10
done
echo "[task ${SLURM_ARRAY_TASK_ID}] server ready."

# eng-docs prompts run ~2-3x longer than the translated variants (median ~4.8k
# vs ~1.8k chars), which was pushing some requests past the client timeout
# under high concurrency. Give that lang more time per request and less
# contention for decode steps.
if [[ "$lang" == "eng-docs" ]]; then
    max_concurrency=64
else
    max_concurrency=128
fi

python -m data_prep.decontext.consume_offload \
    --input_file "$INPUT_DIR/nuggetized_corpus.${lang}.jsonl.gz" \
    --output_file "$OUTPUT_DIR/nuggetized_corpus.${lang}.shard${shard}-of-${NUM_SHARDS}.jsonl" \
    --endpoint "$endpoint" \
    --model "$MODEL" \
    --max_tokens 8196 \
    --max_concurrency "${max_concurrency}" \
    --timeout 1800 \
    --shard_index "${shard}" \
    --num_shards "${NUM_SHARDS}"

kill "$vllm_pid" 2>/dev/null
wait "$vllm_pid" 2>/dev/null
echo "[task ${SLURM_ARRAY_TASK_ID}] done."
