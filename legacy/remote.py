import requests
from typing import List, Dict, Any
import os
import copy
import json
from tqdm import tqdm 
import pdb
from collections import defaultdict

def search_neuclir(
    query, 
    search_endpoint='http://10.162.95.158:5000', 
    service_name='rrf(plaidx+lsr+qwen)-neuclir', 
    limit=1000, 
    subset=None, **kwargs
):
    config = {"service": service_name, "query": str(query), "limit": limit, **kwargs}

    with requests.post(search_endpoint + "/query", json=config) as response:
        result = response.json()
        result = {k: v for k, v in sorted(result["result"].items(), key=lambda item: item[1], reverse=True)}
    return result

def run(
    inputs,
    index='/exp/scale25/artifacts/decontextualized_corpus/neuclir/index/flatten.mlir.mt.lucene',
    k1=0.5, b=0, k=100,
    batch_size=32,
):
    outputs = copy.deepcopy(inputs)

    for i, input in tqdm(enumerate(inputs), desc='Retrieving', total=len(inputs)):

        id = input.topic['qid']
        qtexts = input.subquestions
        qtexts = [input.topic['query'] + sq for sq in qtexts]
        qids = [str(i) for i in range(len(qtexts))]

        batch_hits = dict(zip(qids, [search_neuclir(query=query, limit=k) for query in qtexts]))

        # get documents for each hit
        outputs[i].hits = []
        rrf = defaultdict(int)
        for qid, hits in batch_hits.items():

            for rank, (docid, score) in enumerate(hits.items(), start=1):
                rrf[docid.split('#')[0]] += 1 / rank

            outputs[i].hits.append({docid: None for docid in hits.keys()})

        # reciproal rank fusion NOTE: truncate to smaller number?
        outputs[i].evidences = {k: v for k, v in sorted(rrf.items(), key=lambda x: x[1], reverse=True)}

        print(f"Total number of evidences for {id}: {len(outputs[i].evidences)}")

    return outputs  
