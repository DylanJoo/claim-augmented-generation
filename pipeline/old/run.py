"""Document-mode RAG pipeline: retrieval → clustering → greedy compilation → reranking.

Usage:
    python pipeline/run.py config/neuclir-document.yaml
"""
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import clustering
import retrieval
from retrieval import rewrite
from generation import greedy
from generation import refine as reranking
from utils import convert_result_to_submission, Result

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

exp_cfg = cfg["exp"]
data_cfg = cfg["data"]
ret_cfg = cfg["retrieval"]
clu_cfg = cfg.get("clustering", {})
gen_cfg = cfg["generation"]
rer_cfg = cfg["reranking"]

exp_name = exp_cfg["exp_name"]
output_path = exp_cfg["output_path"].format(exp_name=exp_name)
submission_path = exp_cfg["submission_path"].format(exp_name=exp_name)

# 0. Load topics
with open(data_cfg["topics"], "r", encoding="utf-8") as f:
    topics = [json.loads(line) for line in f if line.strip()]
results = [Result(topic=topic, subquestions=[]) for topic in topics]
print(f"Loaded {len(results)} topic(s).")

# 1. Retrieval: query decomposition
# if ret_cfg.get("n_subquestions", 0) > 0:
#     results = retrieval.rewrite.run(topics, n_questions=n_subquestions)
#     print(f"Query decomposition done: {len(results[0].subquestions)} subquestions for first topic")

# 2. Retrieval: search
results = retrieval.search.run(
    results, 
    index=ret_cfg["index"], 
    k=ret_cfg.get("k", 1000)
)
print(f"Retrieval done: {len(results[0].evidences)} evidences for first topic")

# 3. Clustering
# results = clustering.run(
#     results,
#     n_centroids=clu_cfg.get("n_centroids", 10),
#     n_neighbors=clu_cfg.get("n_neighbors", 20),
#     sim_threshold=clu_cfg.get("sim_threshold", 0.9),
# )
# print(f"Clustering done: {len(results[0].clusters)} clusters for first topic")

# 4. Greedy compilation
# results = greedy.run(results)
# print(f"Generation done: {len(results[0].responses)} responses for first topic")

# 5. Reranking
# results = reranking.run(
#     results,
#     max_char_limit=rer_cfg.get("max_char_limit", 8000),
#     max_sent_limit=rer_cfg.get("max_sent_limit", 30),
# )
# print("Reranking done")
