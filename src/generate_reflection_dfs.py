import json
from typing import List
import os
import copy
from tqdm import tqdm 
import re
import unicodedata

import pdb

# v2. Select one
SYSTEM_MESSAGE = """\
You are an intelligence analyst in the Investigative team. You will be provided with a report request containing user background and problem statement. 
You will also be receive two sets of information:
- Collected Set: Verified information that has already been delivered to the user.
- Candidate Set: New findings from Retrieval Team. Each item is labeled with an identifier in the format of [1], [2], etc.

Specifically, your task is to: 
- Compare Candidate Set with Collected Set. 
- Identify which item in the Candidate Set is both:
  - Insightful: The information MUST be meaninngfully address report request. 
  - Novel: the information MUST have new details that are not present in the Collected Set.
"""

USER_MESSAGE = """\
The Report Request is as follows:
- User Background: 
{user_background}
- Problem statement: 
{problem_statement}

The Collected Set is as follows:
{collected_information}

The Candidate Set is as follows:
{candidate_information}

Identify which item in the Candidate Set is both novel to the Collected Set and is insightful with respect to the Report Request. You can ONLY select one information from Candidate Set. Write only the identifier of the information in this format: [number]. Write None if no items meet both criteria. Only write the identifier, do not write any other text."""

def cluster_format(cluster: List[str], limit=5) -> str:
    information = ""
    for i, (sid, stext) in enumerate(cluster.items(), start=1):
        if i <= limit:
            information += f"[{i}] {stext}\n"
    return information.strip()

def parse_response(response: str, n_max_citations: int = 3):
    matches = re.findall(r'\[(\d+)\]', response)
    indices = [int(num) - 1 for num in matches]
    clean_response = re.sub(r'\[\d+\]', '', response).strip()
    return clean_response, indices

def run(inputs, limit=5, max_char_limit=10000, use_litellm=False):
    if use_litellm:
        from llm.litellm_api import LLM
    else:
        from llm.vllm_back import LLM
    outputs = copy.deepcopy(inputs)

    llm = LLM(model_name_or_path="Qwen/Qwen3-8B", 
              dtype="half",
              temperature=0.0,
              top_p=1,
              max_tokens=32,
              logprobs=None,
    )

    # Prepare the input to the model
    for i, input in tqdm(enumerate(inputs), desc='Compile cluster', total=len(inputs)):

        ## collect the cluster information
        collected = []
        char_count = 0

        for j in range(len(input.clusters)):

            available = [len(cluster) for cluster in input.clusters]
            print(f"Char count: {char_count}, available: {available}")

            cluster = input.clusters[j]

            while len(cluster) > 0:

                ## get temp outputs
                response = "- " + "\n- ".join([r["text"] for r in outputs[i].responses])

                ## prepare prompts
                if use_litellm:
                    message = llm.tokenizer.apply_chat_template([
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": USER_MESSAGE.format(
                            user_background=input.topic["background"],
                            problem_statement=input.topic["problem_statement"],
                            collected_information="Nothing collected yet." if response == "" else response,
                            candidate_information=cluster_format(cluster, limit) if len(cluster) > 0 else "None"
                        )}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    llm_output = llm.generate(message)[0]
                else:
                    messages = [
                            {"role": "system", "content": SYSTEM_MESSAGE},
                            {"role": "user", "content": USER_MESSAGE.format(
                                user_background=input.topic["background"],
                                problem_statement=input.topic["problem_statement"],
                                collected_information="Nothing collected yet." if response == "" else response,
                                candidate_information=cluster_format(cluster, limit) if len(cluster) > 0 else "None"
                            )}
                    ]
                    llm_output = llm.generate(messages, is_message=True)[0]

                ## generate outputs
                _, indices = parse_response(llm_output)

                if 'none' in llm_output.lower():
                    cluster = {k: v for i_cluster, (k, v) in enumerate(cluster.items()) if i_cluster > limit}
                else:
                    sid = list(cluster.keys())[indices[0]]
                    stext = cluster.pop(sid) # remove the item from the cluster
                    docid = sid.split('#')[0]
                    if sid in collected:
                        continue

                    char_count += len(unicodedata.normalize('NFKC', stext.strip()))

                    if char_count <= max_char_limit: # NOTE: refine the iteration logic
                        outputs[i].responses.append({"text": stext.strip(), "citations": {docid: 1.0}})
                        collected.append(sid)
                    else:
                        break

            # replace the cluster
            inputs[i].clusters[j] = cluster

    return outputs
