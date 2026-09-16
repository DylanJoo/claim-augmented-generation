"""
Standalone sanity check for cc_kmeans_core.py -- not pytest, just a script
you run directly and eyeball the printed output (matches this repo's
no-test-infra convention, see src/retrieval/dev/*.py).

Extends test_cc_dense_kmeans.py's synthetic setup with two extra docs placed
*beyond* top_m, to specifically exercise the fit-on-core/predict-on-tail
split that makes this module different from cc_dense.py's agg="kmeans":

    docA: 3 claims near c0     -- core, should end up ~[1,0,0] (binary)
    docB: 3 claims near c1     -- core, should end up ~[0,1,0]
    docC: 2 claims near c0 + 2 near c1 -- core, should end up ~[1,1,0]
    docD: 3 claims near c2     -- core, should end up ~[0,0,1]
    docE: no claims at all     -- core, all-zero (exercises "missing" path)
    docF: 2 claims near c0     -- TAIL (beyond top_m=5): k-means never sees
                                   these at fit time, only via .predict();
                                   should land in the same cluster as docA.
    docG: 2 claims near a 4th, never-fit direction c3 -- TAIL: predict() has
                                   no "correct" cluster for this, but must
                                   still return *some* valid cluster id
                                   without crashing.

Base relevance is set so docA > ... > docG, already sorted -- so any change
in output order after re-ranking is entirely the diversity signal at work.

Run: python src/retrieval/dev/test_cc_kmeans_core.py
"""
import json
import os
import pickle
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from retrieval import cc_kmeans_core
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
    c3 = _unit(np.array([0.5, -0.5, -0.5, 0.5], dtype=np.float32))  # never in the core fit

    claims_by_doc = {
        "docA": [_noisy(c0) for _ in range(3)],
        "docB": [_noisy(c1) for _ in range(3)],
        "docC": [_noisy(c0), _noisy(c0), _noisy(c1), _noisy(c1)],
        "docD": [_noisy(c2) for _ in range(3)],
        "docE": [],
        "docF": [_noisy(c0) for _ in range(2)],
        "docG": [_noisy(c3) for _ in range(2)],
    }

    reps, lookup = [], []
    for docid, vecs in claims_by_doc.items():
        for i, vec in enumerate(vecs):
            reps.append(vec)
            lookup.append(f"{docid}#{i}")
    claim_reps_path = os.path.join(tmpdir, "claims_emb.shard0.pkl")
    with open(claim_reps_path, "wb") as f:
        pickle.dump((np.stack(reps), lookup), f)

    run_path = os.path.join(tmpdir, "run.txt")
    base_scores = {"docA": 7.0, "docB": 6.5, "docC": 6.0, "docD": 5.5, "docE": 5.0, "docF": 4.5, "docG": 4.0}
    with open(run_path, "w") as f:
        for rank, (docid, score) in enumerate(base_scores.items(), start=1):
            f.write(f"q1 Q0 {docid} {rank} {score:.4f} base\n")

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

        print(f"\n{'='*70}\ntop_m=5 (docF, docG are tail-only, predict()-scored)\n{'='*70}")
        outputs = cc_kmeans_core.run(
            inputs,
            run_file=run_path,
            corpus=[corpus_path],
            claim_reps=claim_reps_path,
            k=10,
            lambda_mult=0.5,
            mode="subtract",
            n_clusters=3,
            top_m=5,
            label_mode="binary",
            kmeans_n_init=5,
        )

        evidences = outputs[0].evidences
        print(f"\nFinal selection order: {[(h.docid, round(h.score, 4)) for h in evidences]}")

        check("all 7 pooled docs returned", len(evidences) == 7)
        check("ranks are 1..7 contiguous", [h.rank for h in evidences] == list(range(1, 8)))
        scores = [h.score for h in evidences]
        check("scores non-increasing (subtract mode)", all(scores[i] >= scores[i+1] for i in range(len(scores)-1)))
        check("docE (no claims) present and didn't crash the pipeline",
              "docE" in {h.docid for h in evidences})

        # -- the actual point of this module: verify docF (tail, never seen
        # by k-means fit) gets predict()-assigned into docA's cluster, not
        # left at all-zero or crashing.
        doc_vecs, labels = cc_kmeans_core._core_doc_cluster_vectors(
            list_docids=["docA", "docB", "docC", "docD", "docE", "docF", "docG"],
            claim_reps_by_id=cc_kmeans_core._load_claim_reps(
                claim_reps_path, {"docA", "docB", "docC", "docD", "docE", "docF", "docG"}),
            rows_by_parent=cc_kmeans_core._rows_by_parent(
                cc_kmeans_core._load_claim_reps(
                    claim_reps_path, {"docA", "docB", "docC", "docD", "docE", "docF", "docG"})),
            n_clusters=3,
            top_m=5,
            label_mode="binary",
            kmeans_n_init=5,
        )
        docA_idx, docF_idx, docG_idx = 0, 5, 6
        check("docF (tail, predict()-scored) lands in docA's (core) cluster",
              bool(np.array_equal(doc_vecs[docA_idx], doc_vecs[docF_idx])))
        check("docG (tail, off-manifold claims) still gets *some* valid cluster, doesn't crash",
              doc_vecs[docG_idx].any())

        # -- degenerate case: top_m=0 means an empty core to fit on, even
        # though the tail (everyone) has claims -- must fall back gracefully.
        print(f"\n{'='*70}\ntop_m=0 (empty core) check\n{'='*70}")
        outputs = cc_kmeans_core.run(
            inputs, run_file=run_path, corpus=[corpus_path], claim_reps=claim_reps_path,
            k=10, lambda_mult=0.5, mode="subtract", n_clusters=3, top_m=0, label_mode="binary",
        )
        check("top_m=0 (empty core) ran without raising and returned every doc",
              len(outputs[0].evidences) == 7)

        # -- mode="add" sanity: scores should still come back sorted desc after the re-sort fix --
        print(f"\n{'='*70}\nmode=add check\n{'='*70}")
        outputs = cc_kmeans_core.run(
            inputs, run_file=run_path, corpus=[corpus_path], claim_reps=claim_reps_path,
            k=10, lambda_mult=0.2, mode="add", n_clusters=3, top_m=5, label_mode="binary",
        )
        scores = [h.score for h in outputs[0].evidences]
        check("mode=add scores non-increasing after re-sort", all(scores[i] >= scores[i+1] for i in range(len(scores)-1)))

        # -- single-hit early-return path --
        print(f"\n{'='*70}\nsingle-doc pool (early return) check\n{'='*70}")
        single_run_path = os.path.join(tmpdir, "run_single.txt")
        with open(single_run_path, "w") as f:
            f.write("q1 Q0 docA 1 5.0000 base\n")
        outputs = cc_kmeans_core.run(
            inputs, run_file=single_run_path, corpus=[corpus_path], claim_reps=claim_reps_path,
            k=10, n_clusters=3, top_m=5,
        )
        check("single-doc pool returned unchanged (len==1)", len(outputs[0].evidences) == 1)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
