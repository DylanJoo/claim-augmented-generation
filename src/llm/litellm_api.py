import os
import math
import asyncio
import openai
from typing import List
from transformers import AutoTokenizer

class LLM:

    def __init__(
        self,
        model="llama3.3-70b-instruct",
        temperature=0.0,
        top_p=1.0,
        logprobs=20,
        max_tokens=10,
        **kwargs
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.logprobs = logprobs
        self.model_type = 'litellm'

        self.client = openai.OpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
            base_url='http://10.162.95.158:4000/v1/',
            max_retries=10
        )
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.3-70B-Instruct')
        yes_strings=[' Yes', 'Yes', ' yes', 'yes', 'YES', ' YES']
        no_strings=[' No', 'No', ' no', 'no', 'NO', ' NO']
        self.yes_tokens = [self.tokenizer.tokenize(item)[0] for item in yes_strings]
        self.no_tokens = [self.tokenizer.tokenize(item)[0] for item in no_strings]

    async def _generate_async_prob(self, prompts: List[str]) -> List[float]:

        # singlge function call of selected token prob
        def _generate_prob(prompt: str) -> float:
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                logprobs=self.logprobs,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )

            # dict of scores: {first token: first token logprob}
            tok_logps = response.choices[0].logprobs.top_logprobs[0] 
            yes_ = math.exp(max(
                [-1e2] + [
                    logp for tok, logp in tok_logps.items() 
                    if tok in self.yes_tokens
                ]
            ))
            no_ = math.exp(max(
                [-1e2] + [
                    logp for tok, logp in tok_logps.items() 
                    if tok in self.no_tokens 
                ]
            ))
            score = yes_ / (no_ + yes_)
            return score

        # Gather all the outputs
        outputs = await asyncio.gather(*[
            asyncio.to_thread(_generate_prob, prompt) for prompt in prompts
        ])
        return list(outputs)

    def generate(self, prompts, use_prob=False, **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if use_prob:
            return self.loop.run_until_complete(self._generate_async_prob(prompts))
        else:
            return self.loop.run_until_complete(self._generate_async_text(prompts))

    async def _generate_async_text(self, prompts: List[str]) -> List[float]:

        def _generate_text(prompt: str) -> float:
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                logprobs=None,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                extra_body={
                    "cache": {"no-store": True}
                }
            )
            return response.choices[0].text

        outputs = await asyncio.gather(*[
            asyncio.to_thread(_generate_text, prompt) for prompt in prompts
        ])
        return list(outputs)
