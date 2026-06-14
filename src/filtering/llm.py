import json
from llm.vllm import LLM
from typing import List
import os
import copy

from tqdm import tqdm 

SYSTEM_MESSAGE = """\
You are an intellegence analyst in the Assessment team. You will be provided with a report request, which includes user background and problem statement. The report request will be used to assess the information usefulness of the given document. The information to assess is a document.

Specifically, your task is to judge the usefulness of the information, deciding whether the document is useful for the report request. You will be provided with a document that may contain useful information for the report request. If the information is relevant and also helpful for addressing the report request, your decision should be Yes, otherwise write No. Your output must be a single word: Yes or No. Do not provide explanations."""

USER_MESSAGE = """\
The report request is as follows:
- User Background: 
{user_background}
- Problem statement:
{problem_statement}

The document to be assessed is as follows:
{document}

Write final decision of assessment in one word, Yes or No. Do not provide any explanation or additional information."""

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

def run(inputs, rel_threshold=0.7, top_k=50, corpus_path=None):

    outputs = copy.deepcopy(inputs)
    corpus = load_corpus(corpus_path)

    # vllm # Qwen3: Temperature=0.7, TopP=0.8, TopK=20
    llm = LLM(model_name_or_path="Qwen/Qwen3-8B", 
              dtype="half",
              temperature=0.0,
              top_p=1.0,
              top_k=20,
              max_tokens=2,
              logprobs=20,
    )
    tokenizer = llm.tokenizer

    # Prepare the input to the model
    for i, input in tqdm(enumerate(inputs), desc='Filtering information', total=len(inputs)):
        messages = [[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": USER_MESSAGE.format(
                    user_background=input.topic["background"],
                    problem_statement=input.topic["problem_statement"],
                    document=" ".join(corpus[docid].split()[:512])  # Limit to first 512 words
                )}
        ] for docid in input.evidences]

        # Generate outputs
        llm_outputs = llm.generate(prompts=messages, is_message=True, use_prob=True)

        # Filter evidences based on scores
        outputs[i].evidences = {\
                docid: score for i, (docid, score) in enumerate(zip(input.evidences, llm_outputs)) \
                if (score >= rel_threshold) and (i < top_k)
        }

        # Filter hits based on the filtered evidences
        for j in range(len(input.subquestions)):
            outputs[i].hits[j] = {\
                sid: stext for sid, stext in input.hits[j].items() \
                if sid.split('#')[0] in outputs[i].evidences
            }

        print(f"Filtered evidences from {len(input.evidences)} to {len(outputs[i].evidences)}\n")

    return outputs  
