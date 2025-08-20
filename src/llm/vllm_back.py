import torch
import math
import vllm
from typing import List
import numpy as np

class LLM:

    def __init__(
        self,
        model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
        temperature=0.0,
        top_p=1.0,
        logprobs=None,
        max_tokens=10,
        num_gpus=1, 
        dtype='half', 
        max_model_len=32768,
        gpu_memory_utilization=0.9, 
        task='generate',
        **kwargs
    ):
        self.model = vllm.LLM(
            model_name_or_path,
            dtype=dtype,
            enforce_eager=True,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_prefix_caching=True,
            task=task,
        )
        self.sampling_params = vllm.SamplingParams(
            temperature=temperature, 
            top_p=top_p,
            logprobs=logprobs,
            max_tokens=max_tokens, 
            min_tokens=1, 
        )
        self.tokenizer = self.model.get_tokenizer()
        yes_strings=[' Yes', 'Yes', ' yes', 'yes', 'YES', ' YES']
        no_strings=[' No', 'No', ' no', 'no', 'NO', ' NO']
        self.yes_tokens = [self.tokenizer.encode(item, add_special_tokens=False)[0] for item in yes_strings]
        self.no_tokens = [self.tokenizer.encode(item, add_special_tokens=False)[0] for item in no_strings]
        self.model_type = 'vllm'
        self.task = task

    def generate(self, prompts, enable_thinking=False, use_prob=False, **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]

        if kwargs.get("is_message", False):
            messages = prompts
        else:
            messages = [[
                {"role": "system", "content": kwargs.get("system_message", "You are a helpful assistant.")},
                {"role": "user", "content": prompt} 
            ] for prompt in prompts]

        outputs = self.model.chat(
            messages, 
            self.sampling_params,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )

        if use_prob is False:
            responses = [o.outputs[0].text for o in outputs]
            return responses

        if use_prob:
            tok_logps = [o.outputs[0].logprobs for o in outputs]

            scores = []
            for tok_logp in tok_logps:
                yes_ = math.exp(max( 
                    [-1e2] + [
                        v.logprob for k, v in tok_logp[0].items()
                        if k in self.yes_tokens
                    ] 
                ))
                no_ = math.exp(max( 
                    [-1e2] + [
                        v.logprob for k, v in tok_logp[0].items()
                        if k in self.no_tokens
                    ]
                ))
                scores.append( (yes_) / (no_ + yes_) )
            return scores
