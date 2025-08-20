import os
import argparse
import json

from crux.tools import load_corpus

from decontext.utils import batch_iterator
from decontext.prompts import system_prompt, biogen_decontext_prompt
from tqdm import tqdm

def postprocess(text):
    try:
        x = text.strip().split('</s>')
        x = [xx.strip().replace('<s>', '').replace('</s>', '') for xx in x]
        return [xx for xx in x if xx not in ['', ' ', '\n']]
    except:
        return None

def main(args):
    corpus = load_corpus(f"{args.corpus_dir}/subset_corpus.jsonl")

    # get documents 
    if args.offload is False:
        llm = LLM(
            model="llama3.3-70b-instruct",
            temperature=0.0,
            top_p=1.0,
            logprobs=None,
            max_tokens=4096,
        )

    # inference and save file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"{args.corpus_dir}/biogen-decontext.jsonl"
    args.output_dir = args.output_dir + "-offload" if args.offload else args.output_dir

    # collect finished ids
    if os.path.exists(output_file):
        print(f"Found existing output file: {output_file}")
        with open(output_file, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                corpus.pop(item["id"])

    docids = list(corpus.keys())

    # writei output file
    with open(output_file, 'a') as f:

        for batch_docids in tqdm(batch_iterator(docids, size=96), desc=f"Processing corpus"):
            batch_texts = [corpus[docid]['text'] for docid in batch_docids]
            user_prompts = [biogen_decontext_prompt.format(text=text) \
                    for text in batch_texts]

            if args.offload:
                batch_messages = [[
                    {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": prompt}] 
                    for prompt in user_prompts
                ]

                for docid, messages in zip(batch_docids, batch_messages):
                    f.write(json.dumps(
                        {"id": docid, "messages": messages, "statements": "TOBEADDED."}, 
                        ensure_ascii=False
                    ) + '\n')
                continue

            # generate
            outputs = llm.inference_chat(user_prompts, system_prompt)
            outputs = [postprocess(output) for output in outputs]

            for docid, output in zip(batch_docids, outputs):
                if output is not None:
                    f.write(json.dumps({"id": docid, "statements": output}, ensure_ascii=False) + '\n')

    return 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_dir', type=str, default='/exp/scale25/neuclir/docs')
    parser.add_argument('--output_dir', type=str, default='/exp/jhueiju/neuclir')
    parser.add_argument('--offload', action='store_true', default=False)
    args = parser.parse_args()

    if args.offload is False:
        from litellm_api import LLM

    main(args)
