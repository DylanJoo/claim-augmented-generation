"""
Standalone sanity check for cc_kmeans.py -- not pytest, just a script you
run directly and eyeball the printed output (matches this repo's
no-test-infra convention, see src/retrieval/dev/*.py).

Covers both fit modes this module supports:
  - top_m=None (whole-pool fit, the original cc_dense.py agg="kmeans"
    behavior)
  - top_m=<int> (core-filtered fit/predict, the original cc_kmeans_core.py
    behavior)

Synthetic pool for one topic "q1" with a known cluster structure in a 4-d
embedding space (3 near-orthonormal directions, c0/c1/c2, each with a splash
of noise so k-means has to actually do work instead of landing on a
degenerate answer), plus two extra docs placed *beyond* top_m=5 to
specifically exercise the fit-on-core/predict-on-tail split:

    docA: 3 claims near c0                       -> core, ~[1,0,0] (binary)
    docB: 3 claims near c1                       -> core, ~[0,1,0]
    docC: 2 claims near c0 + 2 claims near c1     -> core, ~[1,1,0] (binary)
                                                      or ~[0.5,0.5,0] (scaled)
    docD: 3 claims near c2                       -> core, ~[0,0,1]
    docE: no claims in the shard at all           -> core, all-zero (exercises
                                                      the "missing" code path)
    docF: 2 claims near c0                        -> TAIL (beyond top_m=5):
                                                      k-means never sees these
                                                      at fit time, only via
                                                      .predict(); should land
                                                      in the same cluster as docA.
    docG: 2 claims near a 4th, never-fit direction -> TAIL: predict() has no
                                                      "correct" cluster for
                                                      this, but must still
                                                      return *some* valid
                                                      cluster id without
                                                      crashing.

Base relevance is set so docA > docB > ... > docG, already sorted -- so any
change in output order after re-ranking is entirely the k-means diversity
signal at work, nothing else.

Run: python src/retrieval/dev/test_cc_kmeans.py
"""
import json
import os
import pickle
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from retrieval import cc_kmeans
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

        # ============================================================
        # top_m=None: whole-pool fit (the old cc_dense agg="kmeans" path)
        # ============================================================
        for label_mode in ("binary", "scaled"):
            print(f"\n{'='*70}\ntop_m=None, label_mode={label_mode}\n{'='*70}")
            outputs = cc_kmeans.run(
                inputs,
                run_file=run_path,
                corpus=[corpus_path],
                claim_reps=claim_reps_path,
                k=10,
                alpha=0.5,
                n_clusters=3,
                top_m=None,
                label_mode=label_mode,
                kmeans_n_init=5,
            )

            evidences = outputs[0].evidences
            print(f"\nFinal selection order: {[(h.docid, round(h.score, 4)) for h in evidences]}")

            check("all 7 pooled docs returned", len(evidences) == 7)
            check("ranks are 1..7 contiguous", [h.rank for h in evidences] == list(range(1, 8)))
            scores = [h.score for h in evidences]
            check("scores non-increasing (alpha-nDCG coverage gain)", all(scores[i] >= scores[i+1] for i in range(len(scores)-1)))
            check("docE (no claims) present and didn't crash the pipeline",
                  "docE" in {h.docid for h in evidences})

        # ============================================================
        # top_m=5: core-filtered fit/predict (the old cc_kmeans_core path)
        # ============================================================
        print(f"\n{'='*70}\ntop_m=5 (docF, docG are tail-only, predict()-scored)\n{'='*70}")
        outputs = cc_kmeans.run(
            inputs,
            run_file=run_path,
            corpus=[corpus_path],
            claim_reps=claim_reps_path,
            k=10,
            alpha=0.5,
            n_clusters=3,
            top_m=5,
            label_mode="binary",
            kmeans_n_init=5,
        )
        evidences = outputs[0].evidences
        print(f"\nFinal selection order: {[(h.docid, round(h.score, 4)) for h in evidences]}")
        check("all 7 pooled docs returned (top_m=5)", len(evidences) == 7)
        check("docE (no claims) present and didn't crash the pipeline (top_m=5)",
              "docE" in {h.docid for h in evidences})

        doc_vecs, labels = cc_kmeans._kmeans_doc_vectors(
            list_docids=["docA", "docB", "docC", "docD", "docE", "docF", "docG"],
            claim_reps_by_id=cc_kmeans._load_claim_reps(
                claim_reps_path, {"docA", "docB", "docC", "docD", "docE", "docF", "docG"}),
            rows_by_parent=cc_kmeans._rows_by_parent(
                cc_kmeans._load_claim_reps(
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

        # -- top_m >= pool size must degenerate to the top_m=None case exactly --
        print(f"\n{'='*70}\ntop_m=100 (>= pool size) matches top_m=None\n{'='*70}")
        doc_vecs_full, _ = cc_kmeans._kmeans_doc_vectors(
            list_docids=["docA", "docB", "docC", "docD", "docE", "docF", "docG"],
            claim_reps_by_id=cc_kmeans._load_claim_reps(
                claim_reps_path, {"docA", "docB", "docC", "docD", "docE", "docF", "docG"}),
            rows_by_parent=cc_kmeans._rows_by_parent(
                cc_kmeans._load_claim_reps(
                    claim_reps_path, {"docA", "docB", "docC", "docD", "docE", "docF", "docG"})),
            n_clusters=3,
            top_m=100,
            label_mode="binary",
            kmeans_n_init=5,
        )
        doc_vecs_none, _ = cc_kmeans._kmeans_doc_vectors(
            list_docids=["docA", "docB", "docC", "docD", "docE", "docF", "docG"],
            claim_reps_by_id=cc_kmeans._load_claim_reps(
                claim_reps_path, {"docA", "docB", "docC", "docD", "docE", "docF", "docG"}),
            rows_by_parent=cc_kmeans._rows_by_parent(
                cc_kmeans._load_claim_reps(
                    claim_reps_path, {"docA", "docB", "docC", "docD", "docE", "docF", "docG"})),
            n_clusters=3,
            top_m=None,
            label_mode="binary",
            kmeans_n_init=5,
        )
        check("top_m>=pool size gives identical vectors to top_m=None",
              bool(np.array_equal(doc_vecs_full, doc_vecs_none)))

        # -- n_clusters clamp check --
        print(f"\n{'='*70}\nn_clusters clamp check (request 100, only 13 claims exist)\n{'='*70}")
        cc_kmeans.run(
            inputs, run_file=run_path, corpus=[corpus_path], claim_reps=claim_reps_path,
            k=10, alpha=0.5, n_clusters=100, top_m=None, label_mode="binary",
        )
        print("[PASS] n_clusters clamp ran without raising (watch for the '[cc_kmeans] ... clamping' line above)")

        # -- alpha=0.0 sanity: no novelty discount at all, so gain reduces to
        # raw cluster-touch count and ties are broken by pool order --
        print(f"\n{'='*70}\nalpha=0.0 check (no novelty discount)\n{'='*70}")
        outputs = cc_kmeans.run(
            inputs, run_file=run_path, corpus=[corpus_path], claim_reps=claim_reps_path,
            k=10, alpha=0.0, n_clusters=3, top_m=5, label_mode="binary",
        )
        scores = [h.score for h in outputs[0].evidences]
        check("alpha=0.0 scores non-increasing", all(scores[i] >= scores[i+1] for i in range(len(scores)-1)))

        # -- single-hit early-return path --
        print(f"\n{'='*70}\nsingle-doc pool (early return) check\n{'='*70}")
        single_run_path = os.path.join(tmpdir, "run_single.txt")
        with open(single_run_path, "w") as f:
            f.write("q1 Q0 docA 1 5.0000 base\n")
        outputs = cc_kmeans.run(
            inputs, run_file=single_run_path, corpus=[corpus_path], claim_reps=claim_reps_path,
            k=10, n_clusters=3, top_m=None,
        )
        check("single-doc pool returned unchanged (len==1)", len(outputs[0].evidences) == 1)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
