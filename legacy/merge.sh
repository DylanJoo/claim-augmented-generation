#!/bin/sh
#SBATCH --job-name=merge
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=384G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x.out

source ~/.bashrc
initconda
conda activate inference

cd $HOME/claim-augmented-generation


# for lang in fas rus zho;do
#     python legacy/merge.py \
#         --input $HOME/scratch/neuclir1/$lang.processed.jsonl.gz \
#         --claims $HOME/scratch/neuclir1/claims/neuclir-decontext.$lang.mt.jsonl \
#         --output $HOME/scratch/neuclir1/$lang.processed-claims.jsonl.gz
# done

for lang in eng-docs arb-trans rus-trans zho-trans;do
    python legacy/merge.py \
        --input $HOME/scratch/ragtime1/$lang.processed.jsonl.gz \
        --claims $HOME/scratch/ragtime1/claims/ragtime-decontext.mlir.mt.jsonl \
        --output $HOME/scratch/ragtime1/$lang.processed-claims.jsonl.gz
done
