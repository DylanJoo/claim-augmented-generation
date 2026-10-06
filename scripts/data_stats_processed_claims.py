#!/usr/bin/env python3
"""
Calculate data statistics for *.processed-claims*.jsonl.gz files.
Usage: python data_stats_processed_claims.py ~/scratch/neuclir1/*.processed-claims*.jsonl.gz
       python data_stats_processed_claims.py --threshold 500 --save-outliers outliers.jsonl *.jsonl.gz
"""
import sys
import gzip
import json
import argparse
from pathlib import Path


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0
    idx = int(len(sorted_vals) * p / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def compute_stats(values):
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    return {
        "count": n,
        "sum": sum(s),
        "min": s[0],
        "max": s[-1],
        "mean": sum(s) / n,
        "p25": percentile(s, 25),
        "p50": percentile(s, 50),
        "p75": percentile(s, 75),
        "p90": percentile(s, 90),
        "p99": percentile(s, 99),
    }


def print_stats(label, stats, indent=2):
    pad = " " * indent
    print(
        f"{pad}{label:32s}  min={stats['min']:6}  mean={stats['mean']:7.1f}"
        f"  p90={stats['p90']:6}  p99={stats['p99']:6}  max={stats['max']:6}"
    )


def analyze_file(path, threshold):
    opener = gzip.open if str(path).endswith(".gz") else open
    n_docs = 0
    n_claims_per_doc = []
    doc_text_chars = []
    claim_text_chars = []
    n_empty_statements = 0
    outliers = []  # list of {doc_id, claim, char_len}

    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_docs += 1

            doc_id = rec.get("id", rec.get("doc_id", rec.get("_id", f"doc_{n_docs}")))
            text = rec.get("text", "")
            doc_text_chars.append(len(text))

            statements = rec.get("statements", [])
            n_claims_per_doc.append(len(statements))
            if len(statements) == 0:
                n_empty_statements += 1
            for s in statements:
                char_len = len(s)
                claim_text_chars.append(char_len)
                if threshold is not None and char_len >= threshold:
                    outliers.append({"doc_id": doc_id, "char_len": char_len, "claim": s})

    return {
        "n_docs": n_docs,
        "n_docs_no_claims": n_empty_statements,
        "n_claims_per_doc": n_claims_per_doc,
        "doc_text_chars": doc_text_chars,
        "claim_text_chars": claim_text_chars,
        "outliers": outliers,
    }


def print_outlier_examples(outliers, n=5):
    if not outliers:
        return
    top = sorted(outliers, key=lambda x: x["char_len"], reverse=True)[:n]
    print(f"\n  Top-{len(top)} longest claims:")
    for i, o in enumerate(top, 1):
        preview = o["claim"][:200].replace("\n", " ")
        ellipsis = "..." if len(o["claim"]) > 200 else ""
        print(f"    [{i}] doc_id={o['doc_id']}  chars={o['char_len']}")
        print(f"        {preview}{ellipsis}")


def main():
    parser = argparse.ArgumentParser(description="Stats for processed-claims files")
    parser.add_argument("files", nargs="+", help="Path(s) to .jsonl or .jsonl.gz files")
    parser.add_argument(
        "--threshold", type=int, default=None,
        help="Char length threshold to flag outlier claims (e.g. 500)"
    )
    parser.add_argument(
        "--save-outliers", metavar="PATH", default=None,
        help="Save outlier claim records to this .jsonl file"
    )
    parser.add_argument(
        "--show-examples", type=int, default=5, metavar="N",
        help="Number of longest-claim examples to print per file (default: 5)"
    )
    args = parser.parse_args()

    all_n_claims_per_doc = []
    all_doc_text_chars = []
    all_claim_text_chars = []
    all_outliers = []
    total_docs = 0
    total_no_claims = 0

    for path_str in args.files:
        path = Path(path_str).expanduser()
        if not path.exists():
            print(f"[WARN] File not found: {path}", file=sys.stderr)
            continue

        print(f"\n{'='*60}")
        print(f"File: {path.name}")
        print(f"{'='*60}")

        res = analyze_file(path, args.threshold)
        n = res["n_docs"]
        total_docs += n
        total_no_claims += res["n_docs_no_claims"]
        all_n_claims_per_doc.extend(res["n_claims_per_doc"])
        all_doc_text_chars.extend(res["doc_text_chars"])
        all_claim_text_chars.extend(res["claim_text_chars"])
        all_outliers.extend(res["outliers"])

        total_claims = sum(res["n_claims_per_doc"])
        print(f"  Documents : {n:,}    Total claims : {total_claims:,}    Docs with no claims : {res['n_docs_no_claims']:,}")
        print_stats("Claims per doc", compute_stats(res["n_claims_per_doc"]))
        print_stats("Doc length (chars)", compute_stats(res["doc_text_chars"]))
        print_stats("Claim length (chars)", compute_stats(res["claim_text_chars"]))

        if args.threshold is not None:
            n_out = len(res["outliers"])
            pct = 100 * n_out / total_claims if total_claims else 0
            print(f"\n  Outlier claims (>={args.threshold} chars) : {n_out:,}  ({pct:.2f}%)")
            print_outlier_examples(res["outliers"], n=args.show_examples)

    if len(args.files) > 1:
        print(f"\n{'='*60}")
        print("AGGREGATE (all files)")
        print(f"{'='*60}")
        total_claims = sum(all_n_claims_per_doc)
        print(f"  Documents : {total_docs:,}    Total claims : {total_claims:,}    Docs with no claims : {total_no_claims:,}")
        print_stats("Claims per doc", compute_stats(all_n_claims_per_doc))
        print_stats("Doc length (chars)", compute_stats(all_doc_text_chars))
        print_stats("Claim length (chars)", compute_stats(all_claim_text_chars))

        if args.threshold is not None:
            n_out = len(all_outliers)
            pct = 100 * n_out / total_claims if total_claims else 0
            print(f"\n  Outlier claims (>={args.threshold} chars) : {n_out:,}  ({pct:.2f}%)")
            print_outlier_examples(all_outliers, n=args.show_examples)

    if args.save_outliers and all_outliers:
        out_path = Path(args.save_outliers)
        with open(out_path, "w", encoding="utf-8") as f:
            for o in sorted(all_outliers, key=lambda x: x["char_len"], reverse=True):
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        print(f"\n[INFO] Saved {len(all_outliers):,} outlier records to {out_path}")


if __name__ == "__main__":
    main()
