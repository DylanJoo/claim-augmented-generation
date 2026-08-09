#!/bin/bash -l
#SBATCH --job-name=encode-claims
#SBATCH --output=logs/neuclir-enc-claims.out.%a
#SBATCH --error=logs/neuclir-enc-claims.err.%a
#SBATCH --partition=small-g
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --array=11
#SBATCH --time=2-00:00:00
#SBATCH --account=project_465002532

# ENV
module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

MODEL_NAME_OR_PATH=Qwen/Qwen3-Embedding-0.6B

LANGS=(
"fas"
"rus"
"zho"
)
NUM_SHARDS=5

LANG_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SHARDS ))
SHARD_ID=$(( SLURM_ARRAY_TASK_ID % NUM_SHARDS ))
LANG=${LANGS[$LANG_IDX]}

FLAT_CLAIMS=${HOME}/scratch/neuclir1/claims_flat/${LANG}.claims.jsonl.gz
output_dir=${HOME}/scratch/neuclir1/${MODEL_NAME_OR_PATH##*/}/claims_emb/
mkdir -p $output_dir

echo Encoding NeuCLIR1 claims $LANG shard $SHARD_ID
singularity exec $SIF  \
    python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --per_device_eval_batch_size 128 \
    --passage_max_len 512 \
    --pooling last --bf16 --normalize \
    --exclude_title \
    --padding_side left \
    --passage_prefix "" \
    --dataset_path $FLAT_CLAIMS \
    --encode_output_path $output_dir/claims_emb.${LANG}-${SHARD_ID}.pkl \
    --dataset_shard_index ${SHARD_ID} \
    --dataset_number_of_shards ${NUM_SHARDS}
