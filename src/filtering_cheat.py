import json
from typing import List
import os
import copy

from tqdm import tqdm 
from crux.tools import load_run_or_qrel

def load_corpus(path='/exp/scale25/neuclir/docs/mlir.mt.jsonl'):
    corpus = {}

    with open(path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                docid = data.get('id', data.get('_id', ''))
                title = data.get('title', "").strip()
                text = data.get('contents', data.get('text', "")).strip()
                if title != "":
                    corpus[str(docid)] = f"[Title: {title}] {text}"
                else:
                    corpus[str(docid)] = text
            except:
                print(f"Error decoding JSON for line: {line.strip()}")
    return corpus

def run(inputs, rel_threshold=0.9, corpus_path=None):

    outputs = copy.deepcopy(inputs)
    corpus = load_corpus(corpus_path)
    qrel = load_run_or_qrel('/exp/scale25/neuclir/eval/qrel/neuclir24-test-request.qrel')

    # Prepare the input to the model
    for i, input in tqdm(enumerate(inputs), desc='Filtering information', total=len(inputs)):

        # Filter evidences based on ground truth
        outputs[i].evidences = {docid: 1.0 for docid in input.evidences if docid in qrel[input.topic['request_id']]}

        # Filter hits based on the filtered evidences
        for j in range(len(input.subquestions)):
            outputs[i].hits[j] = {\
                sid: stext for sid, stext in input.hits[j].items() \
                if sid.split('#')[0] in outputs[i].evidences
            }

        print(f"Filtered evidences from {len(input.evidences)} to {len(outputs[i].evidences)}\n")

    return outputs  
