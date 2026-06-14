import json
import copy
from vllm import LLM

import numpy as np
import faiss
from tqdm import tqdm
from glob import glob
import torch
import gc

import pdb

def load_statements(path='/exp/scale25/artifacts/decontextualized_corpus/neuclir/neuclir-decontext.*.jsonl'):
    statements = {}
    for file in glob(path):
        with open(file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                docid = data.get('id')
                text = data.get('statements')
                statements[str(docid)] = text
    return statements

def add_instruct(claim: str, problem_statement: str) -> str:
    return f'Instruct: {problem_statement}\nQuery:{claim}'

def run(
    inputs, 
    n_centroids=5, 
    n_neighbors=10, 
    sim_threshold=0.5,
    add_all=False,
    concat_adjacent=False,
    add_condition=False,
    verbose=False
):
    outputs = copy.deepcopy(inputs)
    statements = load_statements()  

    # vllm
    llm = LLM(model="Qwen/Qwen3-Embedding-8B", task="embed")

    for i, input in tqdm(enumerate(inputs), desc='Clustering', total=len(inputs)):

        ## encode contexts
        ctx_sq = {}
        for j, subquestion in enumerate(input.subquestions):
            ctx_sq[f"sq#{j}"] = subquestion

        ctx_doc = {}
        for hits in input.hits:
            for sid, stext in hits.items():
                docid = sid.split('#')[0]

                # 1 include all statements
                # ctx_doc[docid] = "- ".join(statements[docid])

                # 2 only statement
                if concat_adjacent:
                    j = int(sid.split('#')[1])
                    if (j < len(statements[docid]) - 1) and (j > 0):
                        ctx_doc[sid] = statements[docid][j-1] + " ||| " + stext
                    else:
                        ctx_doc[sid] = "||| " + stext

                if add_all:
                    for k, stext in enumerate(statements[docid]):
                        ctx_doc[f"{docid}#{k}"] = stext

                if not add_all and not concat_adjacent:
                    ctx_doc[sid] = stext

            # 3 add condition
            if add_condition:
                for id in ctx_doc:
                    ctx_doc[id] = add_instruct(ctx_doc[id], input.topic['problem_statement'])

        context_list = list(ctx_sq.values()) + list(ctx_doc.values())
        outputs[i].pools = ctx_doc

        llm_outputs = llm.embed(context_list)
        embeddings = np.array([o.outputs.embedding for o in llm_outputs], dtype=np.float32)
        d = embeddings.shape[1]  # dimension of embeddings

        embeddings_sq = embeddings[:len(input.subquestions)]
        embeddings_doc = embeddings[len(input.subquestions):]
        # print(f"Subquestion embeddings shape: {embeddings_sq.shape}")
        print(f"Document embeddings shape: {embeddings_doc.shape}")

        ## Search for neighbors
        index = faiss.IndexFlatIP(d)
        index.add(embeddings_doc)
        if n_centroids == 0:
            D, I = index.search(embeddings_sq, n_neighbors)
        else:
            n_centroids = min(n_centroids, max(embeddings_doc.shape[0] // 10, 1))
            ## find cluster centroids
            kmeans = faiss.Kmeans(d, n_centroids, verbose=verbose)
            kmeans.train(embeddings_doc)
            embeddings_centroids = kmeans.centroids
            D, I = index.search(embeddings_centroids, n_neighbors)

        ## Log the clusters
        ctx_doc = {k: v.split("\nQuery:")[-1] for k, v in ctx_doc.items()}
        ctx_doc = {k: v.split("|||")[-1] for k, v in ctx_doc.items()}
        clusters = []
        for distance, knn in zip(D, I):
            valid = [d < sim_threshold for d in distance]
            # NOTE: fix this valud index
            if len(valid) >= 1:
                clusters.append({
                    list(ctx_doc.keys())[j]: list(ctx_doc.values())[j] \
                            for j in knn[valid]
                })

        if len(clusters) == 0:
            print(f"No clusters found for input {i} with sim_threshold {sim_threshold}.")
            exit(0)

        outputs[i].clusters = clusters
        print(f"Found {len(clusters)} clusters for input {input.topic['request_id']}")

        ## NOTE: Print neighbors
        if verbose:
            for j in range(I.shape[0]):
                key_sq = list(ctx_sq.keys())[j]
                print(f"\nCluster {j} neighbors")
                for rank, (k, dist) in enumerate(zip(I[j], D[j])):
                    key_doc = list(ctx_doc.keys())[k]
                    print(f" Index: {k} ({dist:.2f}): {ctx_doc[key_doc]}")

    ## Release ressource for indexing/clusters
    # import gc
    # import torch
    # from vllm.distributed.parallel_state import destroy_model_parallel
    #
    # destroy_model_parallel()
    # del llm.llm_engine
    # del llm
    # gc.collect()
    # torch.cuda.empty_cache()
    # import ray
    # ray.shutdown()

    return outputs
