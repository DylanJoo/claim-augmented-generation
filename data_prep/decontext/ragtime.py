import os
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from .utils import batch_iterator
from .prompts import system_prompt, user_prompt_en_no_title
from tqdm import tqdm

def postprocess(text):
    try:
        x = text.strip().split('</s>')
        x = [xx.strip().replace('<s>', '').replace('</s>', '') for xx in x]
        return [xx for xx in x if xx not in ['', ' ', '\n']]
    except:
        return None

def main(args):

    corpus = {lang: {} for lang in args.langs}

    for lang in corpus:
        # get documents 
        file_name = f"{lang}.mt.jsonl" if args.use_translation else f"{lang}.jsonl"
        input_file = f"{args.corpus_dir}/{file_name}"
        with open(input_file, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    corpus[lang][item.pop('id')] = item
                except:
                    print(line)

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
        output_file = f"{args.output_dir}/nuggetized_corpus.{file_name}"

        # collect finished ids
        if os.path.exists(output_file):
            print(f"Found existing output file: {output_file}")
            with open(output_file, 'r') as f:
                for line in f:
                    item = json.loads(line.strip())
                    corpus[lang].pop(item["id"])

        args.output_dir = args.output_dir + "-offload" if args.offload else args.output_dir
        output_file = f"{args.output_dir}/nuggetized_corpus.{file_name}"
        docids = list(corpus[lang].keys())
        with open(output_file, 'a') as f:

            for batch_docids in tqdm(batch_iterator(docids, size=96), desc=f"Processing {lang} corpus"):

                batch_texts = [corpus[lang][docid]['text'] for docid in batch_docids]

                # prepare prompts
                if args.use_translation:
                    user_prompts = [user_prompt_en_no_title.format(lang=lang, text=text) \
                            for text in batch_texts]
                else:
                    lang_ = {'zho': 'Chinese', 'rus': 'Russian', 'fas': 'Persian'}[lang]
                    user_prompts = [user_prompt_en_no_title.format(lang=lang_, text=text) \
                            for text in batch_texts]

                # outputs
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
    parser.add_argument('--use_translation', action='store_true', default=False)
    parser.add_argument('--langs', type=str, nargs='+', default=['zho', 'rus', 'fas'])
    parser.add_argument('--offload', action='store_true', default=False)
    args = parser.parse_args()

    if args.offload is False:
        from llm.litellm import LLM

    main(args)
