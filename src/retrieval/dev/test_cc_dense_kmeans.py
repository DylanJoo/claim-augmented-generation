"""
Standalone sanity check for cc_dense.py's agg="kmeans" path -- not pytest,
just a script you run directly and eyeball the printed output against the
hand-picked synthetic setup below (matches this repo's no-test-infra
convention, see src/retrieval/dev/*.py).

Builds a tiny synthetic pool for one topic "q1" with a known cluster
structure in a 4-d embedding space (3 near-orthonormal directions, c0/c1/c2,
each with a splash of noise so k-means has to actually do work instead of
landing on a degenerate answer):

    docA: 3 claims near c0                       -> should end up ~[1,0,0] (binary)
    docB: 3 claims near c1                       -> should end up ~[0,1,0]
    docC: 2 claims near c0 + 2 claims near c1     -> should end up ~[1,1,0] (binary)
                                                      or ~[0.5,0.5,0] (scaled)
    docD: 3 claims near c2                       -> should end up ~[0,0,1]
    docE: no claims in the shard at all           -> all-zero vector (exercises
                                                      the "missing" code path)

Base relevance is set so docA > docB > docC > docD > docE, i.e. already
sorted -- so any change in output order after re-ranking is entirely the
k-means diversity signal at work, nothing else.

Run: python src/retrieval/dev/test_cc_dense_kmeans.py
"""
import json
import os
import pickle
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from retrieval import cc_dense
from utils import Result

RNG = np.random.default_rng(0)
DIM = 4


def _unit(v):
    return v / np.linalg.norm(v)

def _noisy(base, noise_scale=0.05):
    v = base + RNG.normal(scale=noise_scale, size=DIM).astype(np.float32)
    return _unit(v).astype(np.float32)


def build_fixtures(tmpdir):
    c0 = _unit(np.array([1, 0, 0, 0], dtype=np.float32))
    c1 = _unit(np.array([0, 1, 0, 0], dtype=np.float32))
    c2 = _unit(np.array([0, 0, 1, 0], dtype=np.float32))

    # docid -> list of claim vectors (docE intentionally has none)
    claims_by_doc = {
        "docA": [_noisy(c0) for _ in range(3)],
        "docB": [_noisy(c1) for _ in range(3)],
        "docC": [_noisy(c0), _noisy(c0), _noisy(c1), _noisy(c1)],
        "docD": [_noisy(c2) for _ in range(3)],
        "docE": [],
    }

    # -- claim-level embedding shard (tevatron pickle format: (reps, lookup)) --
    reps, lookup = [], []
    for docid, vecs in claims_by_doc.items():
        for i, vec in enumerate(vecs):
            reps.append(vec)
            lookup.append(f"{docid}#{i}")
    claim_reps_path = os.path.join(tmpdir, "claims_emb.shard0.pkl")
    with open(claim_reps_path, "wb") as f:
        pickle.dump((np.stack(reps), lookup), f)

    # -- doc-level TREC run file: already sorted A > B > C > D > E --
    run_path = os.path.join(tmpdir, "run.txt")
    base_scores = {"docA": 5.0, "docB": 4.5, "docC": 4.0, "docD": 3.5, "docE": 3.0}
    with open(run_path, "w") as f:
        for rank, (docid, score) in enumerate(base_scores.items(), start=1):
            f.write(f"q1 Q0 {docid} {rank} {score:.4f} base\n")

    # -- corpus jsonl (unused by cc_dense.py's run() -- kept only so the
    # --corpus CLI/plumbing shape matches the real pipeline) --
    corpus_path = os.path.join(tmpdir, "corpus.jsonl")
    with open(corpus_path, "w") as f:
        for docid in base_scores:
            f.write(json.dumps({"id": docid, "title": docid, "text": "",
                                 "statements": [f"{docid} claim {i}" for i in range(len(claims_by_doc[docid]))]}) + "\n")

    return run_path, corpus_path, claim_reps_path


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    assert cond, label


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_path, corpus_path, claim_reps_path = build_fixtures(tmpdir)
        inputs = [Result(topic={"qid": "q1", "query": "dummy"}, subquestions=[])]

        for label_mode in ("binary", "scaled"):
            print(f"\n{'='*70}\nlabel_mode={label_mode}\n{'='*70}")
            outputs = cc_dense.run(
                inputs,
                run_file=run_path,
                corpus=[corpus_path],
                claim_reps=claim_reps_path,
                k=10,
                lambda_mult=0.5,
                mode="subtract",
                agg="kmeans",
                n_clusters=3,
                label_mode=label_mode,
                kmeans_n_init=5,
            )

            evidences = outputs[0].evidences
            print(f"\nFinal selection order: {[(h.docid, round(h.score, 4)) for h in evidences]}")

            check("all 5 pooled docs returned", len(evidences) == 5)
            check("ranks are 1..5 contiguous", [h.rank for h in evidences] == list(range(1, 6)))
            scores = [h.score for h in evidences]
            check("scores non-increasing (subtract mode)", all(scores[i] >= scores[i+1] for i in range(len(scores)-1)))

            picked_docids = {h.docid for h in evidences}
            check("docE (no claims) present and didn't crash the pipeline", "docE" in picked_docids)

        # -- clamping check: ask for more clusters than claims exist (13 claims total) --
        print(f"\n{'='*70}\nn_clusters clamp check (request 100, only 13 claims exist)\n{'='*70}")
        cc_dense.run(
            inputs, run_file=run_path, corpus=[corpus_path], claim_reps=claim_reps_path,
            k=10, lambda_mult=0.5, mode="subtract", agg="kmeans", n_clusters=100, label_mode="binary",
        )
        print("[PASS] n_clusters clamp ran without raising (watch for the '[cc_dense] ... clamping' line above)")

        # -- min_doc_support check: docD is the *only* doc near c2, so its
        # cluster has doc-frequency 1 -- with min_doc_support=2 that cluster
        # must get zeroed, leaving docD's vector all-zero (same as docE's).
        print(f"\n{'='*70}\nmin_doc_support=2 check (docD's cluster has doc-frequency 1)\n{'='*70}")
        doc_vecs, _ = cc_dense._kmeans_doc_vectors(
            list_docids=["docA", "docB", "docC", "docD", "docE"],
            claim_reps_by_id=cc_dense._load_claim_reps(claim_reps_path, {"docA", "docB", "docC", "docD", "docE"}),
            rows_by_parent=cc_dense._rows_by_parent(
                cc_dense._load_claim_reps(claim_reps_path, {"docA", "docB", "docC", "docD", "docE"})
            ),
            n_clusters=3,
            label_mode="binary",
            kmeans_n_init=5,
            min_doc_support=2,
        )
        docD_idx = 3
        check("docD's vector is all-zero once its singleton cluster is dropped",
              not doc_vecs[docD_idx].any())
        check("docA (shares a cluster with docC) is unaffected", doc_vecs[0].any())

        # -- mode="add" sanity: scores should still come back sorted desc after the re-sort fix --
        print(f"\n{'='*70}\nmode=add check\n{'='*70}")
        outputs = cc_dense.run(
            inputs, run_file=run_path, corpus=[corpus_path], claim_reps=claim_reps_path,
            k=10, lambda_mult=0.2, mode="add", agg="kmeans", n_clusters=3, label_mode="binary",
        )
        scores = [h.score for h in outputs[0].evidences]
        check("mode=add scores non-increasing after re-sort", all(scores[i] >= scores[i+1] for i in range(len(scores)-1)))

        # -- single-hit early-return path --
        print(f"\n{'='*70}\nsingle-doc pool (early return) check\n{'='*70}")
        single_run_path = os.path.join(tmpdir, "run_single.txt")
        with open(single_run_path, "w") as f:
            f.write("q1 Q0 docA 1 5.0000 base\n")
        outputs = cc_dense.run(
            inputs, run_file=single_run_path, corpus=[corpus_path], claim_reps=claim_reps_path,
            k=10, agg="kmeans", n_clusters=3,
        )
        check("single-doc pool returned unchanged (len==1)", len(outputs[0].evidences) == 1)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
