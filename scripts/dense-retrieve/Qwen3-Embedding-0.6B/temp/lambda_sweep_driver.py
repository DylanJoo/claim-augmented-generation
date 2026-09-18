"""One-off driver: load claim-rep shards once, then run cc_kmeans._select
across a lambda_mult (and alpha) grid, writing one TREC run file per config.
Avoids reloading the ~62k claim-embedding shards (~21 min) once per config,
which the sbatch grid scripts normally pay for every single run.
"""
import copy
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from retrieval import cc_kmeans
from utils import Result, Hit, load_topics, load_run

TOPICS = "data/ragtime2025.topics.test.jsonl"
RUN_FILE = "runs/run.ragtime1.documents.Qwen3-Embedding-0.6B.txt"
CORPUS = None  # unused downstream by cc_kmeans.run's own logic; we skip it here too
CLAIM_REPS = os.path.expanduser("~/scratch/ragtime1/Qwen3-Embedding-0.6B/claims_emb/claims_emb.*.pkl")
OUT_DIR = "runs/ragtime1"
K = 100
N_CLUSTERS = 50
TOP_M = 20
LABEL_MODE = "binary"
KMEANS_N_INIT = 10

os.chdir(os.path.expanduser("~/claim-augmented-generation"))

topics = load_topics(TOPICS)
print(f"Loaded {len(topics)} topics")

base_run = load_run(RUN_FILE, k=K)
needed_docids = {docid for pool in base_run.values() for docid, _ in pool}
print(f"{len(needed_docids)} unique pooled docids need claim embeddings")

claim_reps_by_id = cc_kmeans._load_claim_reps(CLAIM_REPS, needed_docids)
rows_by_parent = cc_kmeans._rows_by_parent(claim_reps_by_id)
print(f"Loaded {len(claim_reps_by_id)} claim vectors -- shard loading done, starting sweep")


def write_trec(qid_to_evidences, output_path, tag):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for qid, evidences in qid_to_evidences.items():
            for hit in evidences:
                out.write(f"{qid} Q0 {hit.docid} {hit.rank} {hit.score:.6f} {tag}\n")
    print(f"wrote {output_path}")


def run_config(top_m, alpha, lambda_mult, out_suffix, tag_suffix):
    qid_to_evidences = {}
    for topic in topics:
        qid = str(topic["qid"])
        pool = base_run.get(qid, [])
        hits = [
            Hit(docid=docid, score=score, rank=rank, content_dict={})
            for rank, (docid, score) in enumerate(pool, start=1)
        ]
        qid_to_evidences[qid] = cc_kmeans._select(
            hits, claim_reps_by_id, rows_by_parent, K,
            n_clusters=N_CLUSTERS, top_m=top_m, label_mode=LABEL_MODE,
            kmeans_n_init=KMEANS_N_INIT, alpha=alpha, lambda_mult=lambda_mult, qid=qid,
        )
    out_path = f"{OUT_DIR}/run.ragtime1.documents.Qwen3-Embedding-0.6B.{out_suffix}.txt"
    write_trec(qid_to_evidences, out_path, f"doc-{tag_suffix}")


if __name__ == "__main__":
    import contextlib
    import io

    configs = []
    # sanity: lambda_mult=0.0 must reproduce the existing k50 alpha-0.5 run bit-for-bit
    configs.append((None, 0.5, 0.0, "cckmeans.k50-binary.alpha-0.5.lambda-0.0-sanity", "cckmeans-k50-binary-a0.5-l0.0-sanity"))
    # lambda sweep on the plain (top_m=None) k50 alpha-0.5 config
    for lam in [0.3, 0.5, 0.7, 0.9]:
        configs.append((None, 0.5, lam, f"cckmeans.k50-binary.alpha-0.5.lambda-{lam}", f"cckmeans-k50-binary-a0.5-l{lam}"))
    # lambda sweep on the "core" (top_m=20) config at alpha=0.7 (mid of the deleted 0.2/0.3/0.7/0.8 sweep)
    for lam in [0.0, 0.3, 0.5, 0.7, 0.9]:
        configs.append((20, 0.7, lam, f"cckmeans-core.top20-k50-binary.alpha-0.7.lambda-{lam}", f"cckmeans-core-top20-k50-binary-a0.7-l{lam}"))

    for top_m, alpha, lambda_mult, out_suffix, tag_suffix in configs:
        print(f"--- running top_m={top_m} alpha={alpha} lambda_mult={lambda_mult} ---")
        # _select prints big per-topic matrices/assignments to stdout; silence those, keep our own progress prints
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_config(top_m, alpha, lambda_mult, out_suffix, tag_suffix)
        print(f"--- done top_m={top_m} alpha={alpha} lambda_mult={lambda_mult} ---")
