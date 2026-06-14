import os
from tqdm import tqdm 
import json

lang='rus'

# load original corpus
corpus = {}
with open(f'/exp/scale25/neuclir/docs/{lang}.mt.jsonl', 'r') as f:
    for line in tqdm(f, desc="Loading original corpus"):
        try:
            data = json.loads(line)
            docid = data['id']
            corpus[docid] = data
        except:
            print(f"Error parsing line: {line.strip()}")
            continue

# write 
os.makedirs(f'/exp/scale25/artifacts/decontextualized_corpus/neuclir/flatten', exist_ok=True)
with open(f'/exp/scale25/artifacts/decontextualized_corpus/neuclir/decontext.{lang}.mt.jsonl', 'r') as fin, \
open(f'/exp/scale25/artifacts/decontextualized_corpus/neuclir/flatten/{lang}.mt.jsonl', 'w') as fout:

    for line in tqdm(fin):
        data = json.loads(line)
        docid = data['id']
        statements = data['statements']

        for i, statement in enumerate(statements):
            statement_id = f"{docid}#{i}"
            fout.write(json.dumps({
                "id": statement_id, 
                "lang": lang,
                "title": corpus[docid].get('title', ''),
                "contents": statement,
            }) + '\n')
