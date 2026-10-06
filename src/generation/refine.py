import json
from typing import List
import os
import copy
from tqdm import tqdm 
import re
import unicodedata
from llm.litellm import LLM

import pdb

# v3. Claim reranking
SYSTEM_MESSAGE = """\
You are an intelligence analyst in the Investigative team. You will be provided with a report request containing user background and problem statement. 
You will also be receive a set of collected information. Each item is labeled with an identifier in the format of [1], [2], etc.

Specifically, your task is to rank the information items above based on their how well they address the report request.
"""

USER_MESSAGE = """\
The Report Request is as follows:
- User Background: 
{user_background}
- Problem statement: 
{problem_statement}

The Collected Information is as follows:
{collected_information}

Rank the {n_items} information items aboce based on their relevance to the report request. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2], Only respond with the ranking results, do not say any word or explain."""

def response_format(responses: List[str]) -> str:
    information = ""
    for i, response in enumerate(responses, start=1):
        information += f"[{i}] {response['text']}\n"
    return information.strip()

def parse_response(response: str):
    matches = re.findall(r'\[(\d+)\]', response)
    indices = [int(num) - 1 for num in matches]
    return indices

def restore_all_indices(ranked_indices: List[int], n_items: int) -> List[int]:
    all_indices = [i for i in range(n_items) if i not in ranked_indices]
    return ranked_indices + all_indices

def run(inputs, max_char_limit=8000, max_sent_limit=30):
    outputs = copy.deepcopy(inputs)

    llm = LLM(model_name_or_path="Qwen/Qwen3-8B", max_tokens=128)

    # Prepare the input to the model
    for i, input in tqdm(enumerate(inputs), desc='Claim reranking', total=len(inputs)):

        ## collect the cluster information
        collected = []
        char_count = 0
        available = 999

        ## response so far NOTE: see if we need to do this in batch
        for curr_end in tqdm(range(len(input.responses), 0, -10), desc='Collecting responses'):
            curr_start = max(0, curr_end - 20)

            message = llm.tokenizer.apply_chat_template([
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": USER_MESSAGE.format(
                    user_background=input.topic["background"],
                    problem_statement=input.topic["problem_statement"],
                    collected_information=response_format(outputs[i].responses[curr_start:curr_end]),
                    n_items=len(outputs[i].responses[curr_start:curr_end]),
                )}],
                tokenize=False,
                add_generation_prompt=True,
            )

            ## generate outputs
            llm_output = llm.generate(message, is_message=True)[0]
            indices = parse_response(llm_output)
            indices_all = restore_all_indices(indices, len(outputs[i].responses[curr_start:curr_end]))

            for idx_original, idx in enumerate(indices_all):
                outputs[i].responses[curr_start + idx_original] = inputs[i].responses[curr_start + idx]

        ## Reranking finished
        char_count = 0
        sent_count = 0
        for response in outputs[i].responses:
            char_count += len(unicodedata.normalize('NFKC', response['text']))
            sent_count += 1
            if sent_count > max_sent_limit:
                break
            if char_count > max_char_limit:
                break
    return outputs
