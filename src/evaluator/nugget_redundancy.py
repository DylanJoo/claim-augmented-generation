"""
Per-rank nugget (subtopic) redundancy diagnostic.

rac_eval.py's alpha_nDCG/StRecall summarize a whole ranked list into one
number per topic. This script instead walks each topic's ranked list rank by
rank and reports, at every position, how many *new* ground-truth nuggets
(subtopics, from the same diversity qrel StRecall/alpha_nDCG read) that
rank's document contributes on top of everything ranked above it:

    covered = union of subtopics(doc) for doc in ranked[:rank-1]
    new(rank) = |subtopics(ranked[rank]) - covered|

A document with new(rank) == 0 despite being qrel-relevant to at least one
subtopic is "fully redundant" -- every nugget it covers was already covered
by a higher-ranked document, so it adds nothing a downstream reader/RAG step
couldn't already get from what came before it. This is the mechanism behind
a StRecall/alpha_nDCG drop that a single aggregate score doesn't show
directly: a reranker can look fine on relevance and still push redundant
picks above documents that would have covered fresh nuggets.

Pure stdlib (no pandas/ir_measures) so it runs without the pytorch module
environment the rest of this repo's eval scripts need.

Usage:
    python3 src/evaluator/nugget_redundancy.py \
        --qrel $HOME/trec2026/data/neuclir/neuclir24-test-request.qrel \
        --depth 20 \
        baseline=runs/run.neuclir1.documents.Qwen3-Embedding-0.6B.txt \
        dc-gap=runs/run.neuclir1.documents.Qwen3-Embedding-0.6B.dc-gap.topn.txt
"""
import argparse
import sys
from collections import defaultdict


def load_run(path, depth):
    """qid -> [docid, ...] ordered by rank, truncated to `depth`."""
    ranked = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank = parts[0], parts[1], parts[2], int(parts[3])
            ranked[qid].append((rank, docid))
    out = {}
    for qid, rows in ranked.items():
        rows.sort(key=lambda r: r[0])
        out[qid] = [docid for _, docid in rows[:depth]]
    return out


def load_subtopic_map(path):
    """qid -> {docid: set(subtopic_id)} for relevance > 0 rows, and
    qid -> set(all subtopic ids seen for that qid) as the nugget universe."""
    doc_subtopics = defaultdict(lambda: defaultdict(set))
    universe = defaultdict(set)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 4:
                continue
            qid, subtopic, docid, rel = parts
            universe[qid].add(subtopic)
            if int(rel) > 0:
                doc_subtopics[qid][docid].add(subtopic)
    return doc_subtopics, universe


def per_rank_stats(ranked_docids, doc_subtopics, universe_size, depth):
    """Walk one topic's ranked list, return a list of per-rank dicts."""
    covered = set()
    rows = []
    for rank in range(1, depth + 1):
        if rank - 1 >= len(ranked_docids):
            break
        docid = ranked_docids[rank - 1]
        subtopics = doc_subtopics.get(docid, set())
        new = subtopics - covered
        rows.append({
            "rank": rank,
            "judged_relevant": bool(subtopics),
            "n_new": len(new),
            "fully_redundant": bool(subtopics) and len(new) == 0,
            "cum_recall": (len(covered | new) / universe_size) if universe_size else 0.0,
        })
        covered |= new
    return rows


def summarize(run_name, run, doc_subtopics, universe, depth):
    per_rank = defaultdict(list)  # rank -> list of n_new across topics
    per_rank_recall = defaultdict(list)  # rank -> list of cum_recall across topics
    redundant_at_or_before = defaultdict(int)  # rank -> count of topics with >=1 fully_redundant doc by that rank
    n_topics = 0
    total_redundant_in_top10 = 0
    total_redundant_in_top20 = 0
    first_redundant_rank = []

    for qid, ranked_docids in run.items():
        subtopic_map = doc_subtopics.get(qid, {})
        uni = len(universe.get(qid, set()))
        if uni == 0:
            continue
        n_topics += 1
        rows = per_rank_stats(ranked_docids, subtopic_map, uni, depth)
        seen_redundant = False
        for r in rows:
            per_rank[r["rank"]].append(r["n_new"])
            per_rank_recall[r["rank"]].append(r["cum_recall"])
            if r["fully_redundant"]:
                if r["rank"] <= 10:
                    total_redundant_in_top10 += 1
                if r["rank"] <= 20:
                    total_redundant_in_top20 += 1
                if not seen_redundant:
                    first_redundant_rank.append(r["rank"])
                    seen_redundant = True
        if rows:
            per_rank["_final_recall"].append(rows[-1]["cum_recall"])

    print(f"\n=== {run_name} ===  ({n_topics} topics with qrel)")
    print(f"{'rank':>5} {'mean_new_nuggets':>18} {'cum_recall(mean)':>18}")
    for rank in range(1, depth + 1):
        vals = per_rank.get(rank, [])
        if not vals:
            continue
        mean_new = sum(vals) / len(vals)
        rvals = per_rank_recall.get(rank, [])
        mean_recall = sum(rvals) / len(rvals) if rvals else float("nan")
        print(f"{rank:>5} {mean_new:>18.3f} {mean_recall:>18.4f}")
    final_recalls = per_rank.get("_final_recall", [])
    mean_final_recall = sum(final_recalls) / len(final_recalls) if final_recalls else 0.0
    mean_first_redundant = sum(first_redundant_rank) / len(first_redundant_rank) if first_redundant_rank else float("nan")
    print(f"mean subtopic recall @depth={depth}: {mean_final_recall:.4f}")
    print(f"fully-redundant docs in top-10 (summed over topics): {total_redundant_in_top10}")
    print(f"fully-redundant docs in top-20 (summed over topics): {total_redundant_in_top20}")
    print(f"mean rank of first fully-redundant doc: {mean_first_redundant:.2f} "
          f"({len(first_redundant_rank)}/{n_topics} topics had one by rank {depth})")
    return {
        "mean_final_recall": mean_final_recall,
        "redundant_top10": total_redundant_in_top10,
        "redundant_top20": total_redundant_in_top20,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qrel", required=True)
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("runs", nargs="+", help="name=path, e.g. baseline=runs/foo.txt")
    args = ap.parse_args()

    doc_subtopics, universe = load_subtopic_map(args.qrel)

    results = {}
    for spec in args.runs:
        name, _, path = spec.partition("=")
        if not path:
            name, path = path or spec, spec
        run = load_run(path, args.depth)
        results[name] = summarize(name, run, doc_subtopics, universe, args.depth)

    if len(results) > 1:
        print("\n=== summary ===")
        for name, r in results.items():
            print(f"{name:>30}  recall@depth={r['mean_final_recall']:.4f}  "
                  f"redundant_top10={r['redundant_top10']}  redundant_top20={r['redundant_top20']}")


if __name__ == "__main__":
    main()
