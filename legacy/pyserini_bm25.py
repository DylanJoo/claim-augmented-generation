from typing import List, Dict, Any
import os
import copy
import json
from tqdm import tqdm 
from crux.tools import batch_iterator
from pyserini.search.lucene import LuceneSearcher
from collections import defaultdict

def run(
    inputs,
    index='/exp/scale25/artifacts/decontextualized_corpus/neuclir/index/flatten.mlir.mt.lucene',
    k1=0.5, b=0, k=100,
    batch_size=32,
):
    outputs = copy.deepcopy(inputs)

    searcher = LuceneSearcher(index)
    searcher.set_bm25(k1=k1, b=b)

    for i, input in tqdm(enumerate(inputs), desc='Retrieving', total=len(inputs)):

        id = input.topic['request_id']
        qtexts = input.subquestions
        qids = [str(i) for i in range(len(qtexts))]

        batch_hits = searcher.batch_search(
            queries=[input.topic['problem_statement'] + sq for sq in qtexts], 
            qids=qids,
            threads=32,
            k=k,
            fields={'contents': 1.0, 'title': 1.0},
        )

        # get documents for each hit
        outputs[i].hits = []
        rrf = defaultdict(int)
        for _, hits in batch_hits.items():

            for rank, hit in enumerate(hits):
                rrf[hit.docid.split('#')[0]] += 1 / (rank + 1)

            outputs[i].hits.append({
                hit.docid: json.loads(hit.lucene_document.get('raw'))['contents']
            for _, hit in enumerate(hits)})

        # reciproal rank fusion NOTE: truncate to smaller number?
        outputs[i].evidences = {k: v for k, v in sorted(rrf.items(), key=lambda x: x[1], reverse=True)}

        print(f"Total number of evidences for {id}: {len(outputs[i].evidences)}")

    return outputs  
