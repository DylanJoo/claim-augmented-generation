import copy
from vllm import LLM
import numpy as np
import faiss
from tqdm import tqdm
import pdb
from glob import glob

def load_statements(path='/exp/scale25/artifacts/decontextualized_corpus/neuclir/decontext.*.jsonl'):
    statements = {}
    for file in glob(path):
        with open(file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                docid = data.get('id')
                text = data.get('statements')
                statements[str(docid)] = text
    return statements

def get_detailed_instruct(query: str) -> str:
    # TASK_DESCRIPTION = "Given a question, retrieve relevant claims that answer the question."
    TASK_DESCRIPTION = 'Given a web search query, retrieve relevant passages that answer the query'
    return f'Instruct: {TASK_DESCRIPTION}\nQuery:{query}'

def run(inputs, n_centroids=5, n_neighbors=10):
    outputs = copy.deepcopy(inputs)

    # vllm
    llm = LLM(model="Qwen/Qwen3-Embedding-8B", task="embed")

    for i, input in tqdm(enumerate(inputs), desc='Clustering', total=len(inputs)):

        ## encode contexts
        ctx_sq = {}
        for i, subquestion in enumerate(input.subquestions):
            ctx_sq[f"sq#{i}"] = get_detailed_instruct(subquestion)

        ctx_doc = {}
        for i, subquestion in enumerate(input.subquestions):
            for sid, stext in input.hits[i].items():
                ctx_doc[sid] = stext

        context_list = list(ctx_sq.values()) + list(ctx_doc.values())

        llm_outputs = llm.embed(context_list)
        embeddings = np.array([o.outputs.embedding for o in llm_outputs], dtype=np.float32)
        d = embeddings.shape[1]  # dimension of embeddings

        embeddings_sq = embeddings[:len(input.subquestions)]
        embeddings_doc = embeddings[len(input.subquestions):]
        print(f"Subquestion embeddings shape: {embeddings_sq.shape}")
        print(f"Document embeddings shape: {embeddings_doc.shape}")

        ## Search for neighbors
        index = faiss.IndexFlatL2(d)
        index.add(embeddings_doc)
        if n_centroids == 0:
            D, I = index.search(embeddings_sq, n_neighbors)
        else:
            ## find cluster centroids
            kmeans = faiss.Kmeans(d, n_centroids, niter=100, verbose=True)
            kmeans.train(embeddings_doc)
            embeddings_centroids = kmeans.centroids
            D, I = index.search(embeddings_sq, n_neighbors)

        # Print neighbors
        for i in range(I.shape[0]):
            key_sq = list(ctx_sq.keys())[i]
            print(f"\nCluster {i} neighbors")
            for rank, (j, dist) in enumerate(zip(I[i], D[i])):
                key_doc = list(ctx_doc.keys())[j]
                print(f" Index: {j} ({dist:.2f}): {ctx_doc[key_doc]}")
            breakpoint()
