#!/bin/bash -l
#SBATCH --job-name=neuclir-encode
#SBATCH --output=logs/neuclir-encode.out.%a
#SBATCH --error=logs/neuclir-encode.err.%a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --array=0-2%3
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

# ENV
module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

MODEL_DIRS=DylanJHJ/modernbert-base.cover-5k

LANGS=(
"fas"
"rus"
"zho"
)
LANG=${LANGS[$SLURM_ARRAY_TASK_ID]}

output_dir=${HOME}/scratch/neuclir1/${model_dir##*/}
mkdir -p $output_dir

for SHARD_ID in 0 1 2 3 4;do
    echo Encoding NeuCLIR1 corpus $SHARD_ID
    python -m tevatron.retriever.driver.encode \
        --output_dir=temp \
        --tokenizer_name answerdotai/ModernBERT-base \
        --model_name_or_path $model_dir \
        --per_device_eval_batch_size 1024 \
        --passage_max_len 1024 \
        --pooling mean --bf16 --normalize  \
        --passage_prefix "search_document: " \
        --dataset_path ${HOME}/scratch/neuclir1/${LANG}.processed_output.jsonl.gz \
        --encode_output_path $output_dir/docs_emb.${LANG}-${SHARD_ID}.pkl \
        --dataset_shard_index ${SHARD_ID} \
        --dataset_number_of_shards 5
done
