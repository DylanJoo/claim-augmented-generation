"""
Sanity-check tool: for the top-K documents of one or more TREC run files,
compute query-document and query-claim cosine similarity (Qwen3 embeddings
are pre-normalized, so dot product == cosine). Answers "are all the claims
in a top-ranked (possibly diversity-reranked) document actually on-topic for
the query, or does the doc only make the cut because of one narrow claim?"

Usage:
    python pipeline/analyze_topk_relevance.py \
        --queries-emb <queries_emb.pkl> \
        --docs-emb '<docs_emb/docs_emb.*.pkl>' \
        --claims-emb '<claims_emb/claims_emb.*.pkl>' \
        --corpus <collection.jsonl.gz> [<more files/globs>...] \
        --runs base=<path> cc_kmeans=<path> dd_dense=<path> \
        --topk 3 \
        --output <report.md>
"""

import argparse
import glob
import gzip
import json
import logging
import pickle
from collections import defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _pickle_load(path):
    with open(path, "rb") as f:
        reps, lookup = pickle.load(f)
    return np.asarray(reps), lookup


def load_queries(path):
    reps, lookup = _pickle_load(path)
    return {qid: reps[i] for i, qid in enumerate(lookup)}


def load_run_topk(path, k):
    run = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, docid, rank, score = parts[0], parts[2], int(parts[3]), float(parts[4])
            run[qid].append((rank, docid, score))
    return {qid: sorted(hits, key=lambda x: x[0])[:k] for qid, hits in run.items()}


def load_doc_reps(doc_reps_glob, needed_docids):
    files = sorted(glob.glob(doc_reps_glob))
    reps_by_id = {}
    for i, fpath in enumerate(files, 1):
        reps, lookup = _pickle_load(fpath)
        for vec, repid in zip(reps, lookup):
            if repid in needed_docids:
                reps_by_id[repid] = vec.copy()
        del reps, lookup
        logger.info("doc reps: [%d/%d] loaded %s, %d/%d matched so far",
                    i, len(files), fpath, len(reps_by_id), len(needed_docids))
    return reps_by_id


def load_claim_reps_by_parent(claim_reps_glob, needed_docids):
    files = sorted(glob.glob(claim_reps_glob))
    by_parent = defaultdict(list)
    n_matched = 0
    for i, fpath in enumerate(files, 1):
        reps, lookup = _pickle_load(fpath)
        for vec, repid in zip(reps, lookup):
            parent = repid.rsplit("#", 1)[0]
            if parent in needed_docids:
                by_parent[parent].append((repid, vec.copy()))
                n_matched += 1
        del reps, lookup
        logger.info("claim reps: [%d/%d] loaded %s, %d claim(s) matched so far",
                    i, len(files), fpath, n_matched)
    return by_parent


def _open(fpath):
    return gzip.open(fpath, "rt", encoding="utf-8") if fpath.endswith(".gz") else open(fpath, encoding="utf-8")


def load_statements(corpus_patterns, needed_docids):
    files = []
    for pat in corpus_patterns:
        files.extend(sorted(glob.glob(pat)))
    statements = {}
    remaining = set(needed_docids)
    for fpath in files:
        if not remaining:
            break
        with _open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                docid = doc["id"]
                if docid in remaining:
                    statements[docid] = doc.get("statements") or []
                    remaining.discard(docid)
    if remaining:
        logger.warning("no statements found for %d docid(s), e.g. %s", len(remaining), list(remaining)[:3])
    return statements


def main():
    parser = argparse.ArgumentParser(description="Query-doc / query-claim cosine similarity for top-K of one or more runs")
    parser.add_argument("--queries-emb", required=True, help="Path to queries_emb.pkl")
    parser.add_argument("--docs-emb", required=True, help="Glob for doc-level embedding shards")
    parser.add_argument("--claims-emb", required=True, help="Glob for claim-level embedding shards")
    parser.add_argument("--corpus", required=True, nargs="+", help="Corpus jsonl(.gz) file(s)/globs with a 'statements' field")
    parser.add_argument("--runs", required=True, nargs="+", help="label=path pairs, e.g. base=runs/foo.txt")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--output", required=True, help="Markdown report output path")
    args = parser.parse_args()

    run_paths = dict(item.split("=", 1) for item in args.runs)
    runs = {label: load_run_topk(path, args.topk) for label, path in run_paths.items()}

    needed_docids = set()
    for run in runs.values():
        for hits in run.values():
            needed_docids.update(docid for _, docid, _ in hits)
    logger.info("Need embeddings/text for %d unique docid(s) across %d run(s)", len(needed_docids), len(runs))

    queries = load_queries(args.queries_emb)
    doc_reps = load_doc_reps(args.docs_emb, needed_docids)
    claim_reps_by_parent = load_claim_reps_by_parent(args.claims_emb, needed_docids)
    statements = load_statements(args.corpus, needed_docids)

    rows = []
    for label, run in runs.items():
        for qid, hits in run.items():
            qvec = queries.get(str(qid))
            if qvec is None:
                continue
            for rank, docid, score in hits:
                dvec = doc_reps.get(docid)
                qdoc_cos = float(np.dot(qvec, dvec)) if dvec is not None else None
                claims = claim_reps_by_parent.get(docid, [])
                sims = [(cid, float(np.dot(qvec, cvec))) for cid, cvec in claims]
                sims.sort(key=lambda x: x[1])
                stmts = statements.get(docid, [])
                lowest_text = ""
                if sims:
                    lowest_id = sims[0][0]
                    idx = int(lowest_id.rsplit("#", 1)[1])
                    if idx < len(stmts):
                        lowest_text = stmts[idx][:120]
                rows.append({
                    "label": label, "qid": qid, "rank": rank, "docid": docid,
                    "qdoc_cos": qdoc_cos,
                    "n_claims": len(sims),
                    "min_claim_cos": sims[0][1] if sims else None,
                    "mean_claim_cos": float(np.mean([s for _, s in sims])) if sims else None,
                    "max_claim_cos": sims[-1][1] if sims else None,
                    "lowest_claim_text": lowest_text,
                })

    with open(args.output, "w", encoding="utf-8") as out:
        out.write("| label | qid | rank | docid | q-doc cos | n_claims | min q-claim | mean q-claim | max q-claim | lowest-sim claim (truncated) |\n")
        out.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            out.write(
                f"| {r['label']} | {r['qid']} | {r['rank']} | {r['docid'][:8]} | "
                f"{r['qdoc_cos']:.4f} | {r['n_claims']} | {r['min_claim_cos']:.4f} | "
                f"{r['mean_claim_cos']:.4f} | {r['max_claim_cos']:.4f} | {r['lowest_claim_text']} |\n"
            )
        out.write("\n\n### Aggregate summary (mean over all topic x rank rows)\n\n")
        out.write("| label | mean q-doc cos | mean of per-doc mean q-claim cos | mean of per-doc min q-claim cos |\n")
        out.write("|---|---|---|---|\n")
        by_label = defaultdict(list)
        for r in rows:
            by_label[r["label"]].append(r)
        for label, rs in by_label.items():
            mqd = np.mean([r["qdoc_cos"] for r in rs if r["qdoc_cos"] is not None])
            mmean = np.mean([r["mean_claim_cos"] for r in rs if r["mean_claim_cos"] is not None])
            mmin = np.mean([r["min_claim_cos"] for r in rs if r["min_claim_cos"] is not None])
            out.write(f"| {label} | {mqd:.4f} | {mmean:.4f} | {mmin:.4f} |\n")

    logger.info("Done. Report saved to %s", args.output)


if __name__ == "__main__":
    main()
