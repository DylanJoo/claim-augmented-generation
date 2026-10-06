"""
Iterative RAG generation.

For each topic, retrieved documents are consumed in fixed-size batches:
  - Round 0: draft a report from the first batch.
  - Round 1+: update the draft by integrating the next batch.

The loop stops when all batches are exhausted or the character budget is hit.
"""

import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_DRAFT = (
    "You are a research analyst. Write a well-structured, focused report that directly "
    "addresses the information need. Use only the provided documents as sources. "
    "Cite each document inline as [N] after the sentence it supports. "
    "Stay narrowly on the topic — do not include tangential information."
)

_SYSTEM_UPDATE = (
    "You are a research analyst. You have a draft report and a set of new documents. "
    "Revise and extend the draft to incorporate relevant new information. "
    "Maintain inline citations [N]. Do not repeat information already covered "
    "unless you are adding meaningful new detail."
)


def _format_docs(docs: list[dict], offset: int = 1) -> str:
    parts = []
    for i, d in enumerate(docs):
        prefix = f"[{i + offset}] "
        title = (d.get("title") or "").strip()
        text = (d.get("text") or "").strip()
        parts.append(prefix + (f"{title}: {text}" if title else text))
    return "\n\n".join(parts)


def _draft_prompt(topic: dict, docs: list[dict], limit: int) -> list[dict]:
    background = (topic.get("background") or "").strip()
    problem = topic["problem_statement"]

    user = ""
    if background:
        user += f"Background: {background}\n\n"
    user += f"Documents:\n{_format_docs(docs)}\n\n"
    user += f"Report request: {problem}\n\n"
    user += f"Write a draft report of approximately {limit} characters:"

    return [{"role": "system", "content": _SYSTEM_DRAFT}, {"role": "user", "content": user}]


def _update_prompt(
    topic: dict, docs: list[dict], previous: str, offset: int, limit: int
) -> list[dict]:
    background = (topic.get("background") or "").strip()
    problem = topic["problem_statement"]

    user = ""
    if background:
        user += f"Background: {background}\n\n"
    user += f"Previous draft:\n{previous}\n\n"
    user += f"New documents:\n{_format_docs(docs, offset=offset)}\n\n"
    user += f"Report request: {problem}\n\n"
    user += f"Update and extend your report to approximately {limit} characters:"

    return [{"role": "system", "content": _SYSTEM_UPDATE}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    topics: list[dict],
    run_results: dict[str, list[tuple[str, float]]],
    corpus: dict[str, dict],
    llm,
    n_rounds: int = 3,
    docs_per_round: int = 5,
    max_chars: int = 2000,
) -> dict[str, str]:
    """
    Args:
        topics:         list of topic dicts with keys request_id, problem_statement,
                        and optionally background.
        run_results:    {qid: [(docid, score), ...]} sorted by score descending.
        corpus:         {docid: {"text": ..., "title": ...}}.
        llm:            callable (messages: list[dict]) -> str.
        n_rounds:       maximum number of generation rounds.
        docs_per_round: documents fed per round.
        max_chars:      target character budget for the final report.

    Returns:
        {qid: response_text}
    """
    outputs = {}

    for topic in tqdm(topics, desc="Iterative RAG"):
        qid = topic["request_id"]
        ranked = run_results.get(qid, [])

        all_docs = []
        for docid, _ in ranked:
            doc = corpus.get(docid)
            if doc:
                all_docs.append(doc)

        if not all_docs:
            logger.warning("topic %s: no retrieved documents found in corpus", qid)
            outputs[qid] = ""
            continue

        response = ""
        doc_offset = 1

        for round_idx in range(n_rounds):
            start = round_idx * docs_per_round
            batch = all_docs[start : start + docs_per_round]
            if not batch:
                break

            if round_idx == 0:
                messages = _draft_prompt(topic, batch, limit=max_chars)
            else:
                if len(response) >= max_chars:
                    logger.info("topic %s: reached char budget after round %d", qid, round_idx)
                    break
                messages = _update_prompt(topic, batch, response, doc_offset, limit=max_chars)

            response = llm(messages)
            doc_offset += len(batch)
            logger.info(
                "topic %s round %d/%d: %d chars", qid, round_idx + 1, n_rounds, len(response)
            )

        outputs[qid] = response

    return outputs
