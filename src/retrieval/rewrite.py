from llm.vllm import LLM
from typing import List
from utils import Result

SYSTEM_MESSAGE = """\
You are an intellegence search assistant in the retrieval team. You will be provided with a user's report request, which includes user background and problem statement. The report request is complex and multifaceted, requiring a deep understanding of the topic. The provided background and the problem statement will help you understand the context of the request for the report.

Specifically, your focus is to break down this request into standalone, atomic sub-questions that can be independently verified, retrieved, explored and answered. Each sub-question MUST be clear with comprehensive contexts, and focused on a specific diverse aspect of the report request."""

USER_MESSAGE = """\
The report request is as follows:
- User Background:
{user_background}
- Problem statement:
{problem_statement}

Generate a list of diverse {n_questions} sub-questions that can be independently explored, retrieved and answered. Each sub-question should be specific, clear and focused on a particular aspect without overlapping with other sub-questions in the list. The sub-questions must have comprehensive context. The sub-questions are designed to guide further research and exploration of the topic. Enclose each sub-question with <q> and </q> tags."""


def run(topics, n_questions=10) -> List[Result]:

    llm = LLM(model_name_or_path="Qwen/Qwen3-8B",
              dtype="half",
              temperature=0.7,
              top_p=0.8,
              top_k=20,
              max_tokens=1024,
              logprobs=None,
    )

    messages = [[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": USER_MESSAGE.format(
                user_background=topic["meta"]["background"],
                problem_statement=topic["query"],
                n_questions=n_questions
            )}
    ] for topic in topics]

    llm_outputs = llm.generate(prompts=messages, is_message=True)

    outputs = []
    for topic, llm_output in zip(topics, llm_outputs):
        generated_text = llm_output.strip()
        sub_questions = [q.strip() for q in generated_text.split('<q>') if q.strip()]
        sub_questions = [q.split('</q>')[0].strip() for q in sub_questions if '</q>' in q]
        outputs.append(Result(topic=topic, subquestions=sub_questions))

    return outputs
