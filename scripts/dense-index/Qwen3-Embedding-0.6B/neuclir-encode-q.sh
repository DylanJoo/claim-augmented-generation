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
#SBATCH --time=02:00:00
#SBATCH --account=project_465002532

# ENV
module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

MODEL_NAME_OR_PATH=Qwen/Qwen3-Embedding-0.6B
output_dir=${HOME}/scratch/neuclir1/${MODEL_NAME_OR_PATH##*/}/queries_emb/
mkdir -p $output_dir

cd $HOME/claim-augmented-generation

echo Encoding NeuCLIR1 queries
singularity exec $SIF  \
    python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --per_device_eval_batch_size 128 \
    --query_max_len 512 \
    --pooling last --bf16 --normalize \
    --padding_side left \
    --encode_is_query \
    --query_prefix "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:" \
    --dataset_path data/neuclir2024.topics.test.jsonl \
    --encode_output_path $output_dir/queries_emb.pkl
