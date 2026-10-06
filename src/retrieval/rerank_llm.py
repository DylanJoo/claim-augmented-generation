import logging
import sys
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def run(
    topics: List[Dict],
    initial_run: Dict[str, List[Tuple[str, float]]],
    corpus: Dict[str, Dict],
    april_src: str,
    method: str = "rankgpt",
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    backend: str = "request",
    base_url: str = "http://localhost:8000/v1",
    temperature: float = 0.0,
    k: int = 100,
    query_batch_size: int = 32,
    max_doc_length: int = 1024,
    output_subquestions: str = None,
) -> Dict[str, Dict[str, float]]:
    """Rerank a first-stage run with APRIL's ModularReranker ($HOME/APRIL).

    APRIL's own CLI loaders (loader_dev/irds.py, loader_dev/neuclir.py) don't
    know this project's data layout (topics.jsonl + *.processed-claims.jsonl.gz),
    so queries/corpus are assembled here from utils.load_topics/load_corpus
    and handed to APRIL through its documented Python API
    (AutoLLMReranker.from_prebuilt(...).rerank(...) -- see $HOME/APRIL/example/README.md)
    instead.

    initial_run: {qid: [(docid, score), ...]} as returned by utils.load_run.
    corpus: {docid: {"title":..., "text":..., ...}} as returned by utils.load_corpus.
    output_subquestions: only consumed by APRIL's Lancer method, which dumps its
        generated {qid: [subquestion, ...]} to this path as JSON; ignored otherwise.
    """
    if april_src not in sys.path:
        sys.path.insert(0, april_src)
    from autollmrerank.wrapper import AutoLLMReranker

    queries = {t["qid"]: t["query"] for t in topics}

    run_input = {
        qid: {docid: score for docid, score in hits[:k]}
        for qid, hits in initial_run.items()
        if qid in queries
    }
    missing = set(initial_run) - set(run_input)
    if missing:
        logger.warning("Dropping %d quer(ies) with no matching topic: %s", len(missing), sorted(missing))

    # APRIL's formatter reads 'contents' or 'text' plus a separate 'title',
    # so hand it this project's corpus dict fields as-is (no need to
    # concatenate title into the text ourselves).
    reranker_corpus = {
        docid: {"text": doc.get("text", ""), "title": doc.get("title", "")}
        for docid, doc in corpus.items()
    }

    reranker = AutoLLMReranker.from_prebuilt(
        method,
        model,
        llm={"backend": backend, "base_url": base_url, "temperature": temperature},
        top_k=k,
        rank_end=k,
        max_doc_length=max_doc_length,
        data={"output_subquestions": output_subquestions}
    )
    reranked_run = reranker.rerank(
        run=run_input,
        queries=queries,
        corpus=reranker_corpus,
        query_batch_size=query_batch_size,
    )
    return reranked_run
