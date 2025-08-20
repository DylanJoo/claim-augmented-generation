#!/bin/sh
#SBATCH --job-name=bm25_index
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=36:00:00
#SBATCH --output=%x-%j.out

# Set-up the environment.
# conda install -c conda-forge openjdk=21 maven -y # you might need this line if you haven't installed Java and Maven yet

source ~/.bashrc
enter_conda
conda activate basic

python -m pyserini.index.lucene \
    --collection JsonCollection \
    --fields title \
    --input /exp/scale25/artifacts/decontextualized_corpus/neuclir/flatten \
    --index /exp/scale25/artifacts/decontextualized_corpus/neuclir/index/flatten.mlir.mt.lucene \
    --generator DefaultLuceneDocumentGenerator \
    --threads 128 --storeRaw
