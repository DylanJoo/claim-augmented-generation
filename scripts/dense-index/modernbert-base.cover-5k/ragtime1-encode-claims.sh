#!/bin/bash -l
#SBATCH --job-name=ragtime1-encode-claims
#SBATCH --output=logs/ragtime1-enc-claims.out.%a
#SBATCH --error=logs/ragtime1-enc-claims.err.%a
#SBATCH --partition=small-g
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --array=0-19
#SBATCH --time=1-00:00:00
#SBATCH --account=project_465002532

# ENV
module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

MODEL_NAME_OR_PATH=DylanJHJ/modernbert-base.cover-5k

LANGS=(
"arb-trans"
"eng-docs"
"rus-trans"
"zho-trans"
)
NUM_SHARDS=5

LANG_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SHARDS ))
SHARD_ID=$(( SLURM_ARRAY_TASK_ID % NUM_SHARDS ))
LANG=${LANGS[$LANG_IDX]}

CLAIMS=${HOME}/scratch/ragtime1/claims_flat/${LANG}.claims.jsonl.gz
output_dir=${HOME}/scratch/ragtime1/${MODEL_NAME_OR_PATH##*/}/claims_emb/
mkdir -p $output_dir

echo Encoding ragtime1 claims $LANG shard $SHARD_ID
singularity exec $SIF  \
    python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --tokenizer_name answerdotai/ModernBERT-base \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --per_device_eval_batch_size 1024 \
    --passage_max_len 1024 \
    --pooling mean --bf16 --normalize \
    --passage_prefix "search_document: " \
    --dataset_path $CLAIMS \
    --encode_output_path $output_dir/claims_emb.${LANG}-${SHARD_ID}.pkl \
    --dataset_shard_index ${SHARD_ID} \
    --dataset_number_of_shards ${NUM_SHARDS}
