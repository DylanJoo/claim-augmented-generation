import math
import argparse
import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncStream
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer
import uuid
from typing import List
import logging
logging.getLogger("vllm.engine.async_llm_engine").setLevel(logging.WARNING)

class LLM:

    def __init__(
        self,
        model_name_or_path: str = 'meta-llama/Llama-3.2-1B-Instruct',
        temperature=0.0,
        top_p=1.0,
        logprobs=None,
        max_tokens=128,
        dtype='half',
        gpu_memory_utilization=0.9,
        num_gpus=1, 
        max_model_len=20480,
    ):
        args = AsyncEngineArgs(
            model=model_name_or_path,
            dtype=dtype,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=False,
            max_model_len=max_model_len,
        )
        self.model = AsyncLLMEngine.from_engine_args(AsyncEngineArgs.from_cli_args(args))

        self.sampling_params = SamplingParams(
            temperature=temperature, 
            top_p=top_p,
            logprobs=logprobs,
            skip_special_tokens=False,
            min_tokens=1,
            max_tokens=max_tokens,
        )
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            # there is no actively running loop
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        yes_strings=[' Yes', 'Yes', ' yes', 'yes', 'YES', ' YES']
        no_strings=[' No', 'No', ' no', 'no', 'NO', ' NO']
        self.yes_tokens = [self.tokenizer.encode(item, add_special_tokens=False)[0] for item in yes_strings]
        self.no_tokens = [self.tokenizer.encode(item, add_special_tokens=False)[0] for item in no_strings]

    async def _iterate_over_output(self, output_iterator: AsyncStream, use_logprobs=False) -> str:
        output = None
        async for output in output_iterator:
            if use_logprobs:
                tok_logps = output.outputs[0].logprobs[0]
                yes_ = math.exp(max(
                    [-1e2] + [
                        item.logprob for tok, item in tok_logps.items() 
                        if tok in self.yes_tokens
                    ]
                ))
                no_ = math.exp(max(
                    [-1e2] + [
                        item.logprob for tok, item in tok_logps.items() 
                        if tok in self.no_tokens 
                    ]
                ))
                output = score = yes_ / (no_ + yes_)
            else:
                output = last_text = output.outputs[0].text
        return output

    # [TODO] not working on vllm v1 engine
    # async def _iterate_over_output(self, output_iterator, use_logprobs=False) -> str:
    #     output = None
    #     otuput = await output_iterator.get()
    #     return output

    # async def _iterate_over_output(self, collector, use_logprobs=False):
    #     while True:
    #         try:
    #             print(f"Collector: {collector}")
    #             output = await collector.get()
    #             print(f"output: {output}")
    #             yield output
    #         except Exception as e:
    #             # Handle or re-raise depending on your needs
    #             raise e

    async def _agenerate_text(self, prompts, sampling_params):
        request_ids = [str(uuid.uuid4()) for _ in range(len(prompts))]

        # Add requests to the engine
        output_iterators = [
            await self.model.add_request(request_id, prompt, sampling_params)
            for request_id, prompt in zip(request_ids, prompts)
        ]

        # Gather all the outputs
        outputs = await asyncio.gather(*[
            self._iterate_over_output(output_iterator)
            for output_iterator in output_iterators
        ])
        return list(outputs)

    async def _agenerate_prob(self, prompts, sampling_params) -> List[float]:
        request_ids = [str(uuid.uuid4()) for _ in range(len(prompts))]

        # Add requests to the engine
        output_iterators = [
            await self.model.add_request(request_id, prompt, sampling_params)
            for request_id, prompt in zip(request_ids, prompts)
        ]

        # Gather all the outputs
        outputs = await asyncio.gather(*[
            self._iterate_over_output(output_iterator, use_logprobs=True)
            for output_iterator in output_iterators
        ])
        return list(outputs)

    def generate(self, prompts, prob=False, **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]
        
        sampling_params = self.sampling_params

        if prob:
            return self.loop.run_until_complete(self._agenerate_prob(prompts, sampling_params))
        else:
            return self.loop.run_until_complete(self._agenerate_text(prompts, sampling_params))
