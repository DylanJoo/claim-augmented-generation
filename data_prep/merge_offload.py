import os
import argparse
import json

from tqdm import tqdm
from pprint import pprint

def postprocess(text):
    try:
        x = text.strip().split('</s>')
        x = [xx.strip().replace('<s>', '').replace('</s>', '') for xx in x]
        return [xx for xx in x if xx not in ['', ' ', '\n']]
    except:
        return None

def parse(response_file, output_file=None):
    statements = {}

    with open(response_file + '.tbm', 'r') as fr, open(output_file, 'w') as fw:

        for line in tqdm(fr):
            item = json.loads(line.strip())
            docid = item.pop('id')
            response = item['response']['body']['choices'][0]['message']['content']
            statements = postprocess(response)
            item = {'id': docid, 'statements': statements}

            fw.write(json.dumps(item, ensure_ascii=False) + '\n')

# neuclir
# parse(
#     response_file='/home/hltcoe/jhueiju/temp/neuclir-offload/nuggetized_corpus.fas.mt.jsonl',
#     output_file='/home/hltcoe/jhueiju/temp/neuclir/decontext_corpus.fas.mt.jsonl',
# )

# ragtime
# /exp/ayates/scale25/batch-vllm/output/ragtime_nuggetized_corpus.mlir.mt_part-xa*/batch*.jsonl
parse(
    response_file='/home/hltcoe/jhueiju/temp/ragtime-offload/nuggetized_corpus.mlir.mt.jsonl',
    output_file='/home/hltcoe/jhueiju/temp/ragtime/decontext_corpus.mlir.mt.jsonl'
)

