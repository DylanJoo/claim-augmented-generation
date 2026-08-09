#!/bin/bash -l
#SBATCH --job-name=encode-q
#SBATCH --output=logs/neuclir-enc-q.out
#SBATCH --error=logs/neuclir-enc-q.err
#SBATCH --partition=dev-g
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --account=project_465002532

# ENV
module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

MODEL_NAME_OR_PATH=DylanJHJ/modernbert-base.cover-5k
output_dir=${HOME}/scratch/neuclir1/${MODEL_NAME_OR_PATH##*/}
mkdir -p $output_dir

topic_path=/home/dju/datasets/crux/crux-neuclir/topic/neuclir24-test-request.jsonl
echo "Encoding $topic_path"
singularity exec $SIF  \
    python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --tokenizer_name answerdotai/ModernBERT-base \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --pooling mean --normalize --bf16 \
    --query_prefix "search_query: " \
    --per_device_eval_batch_size 64 \
    --dataset_path data/neuclir2024.topics.test.jsonl \
    --encode_output_path $output_dir/queries_emb.pkl \
    --query_max_len 512 \
    --encode_is_query
