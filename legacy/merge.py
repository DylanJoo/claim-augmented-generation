"""
Merge corpus documents and claim files into a single unified JSONL collection.

Each output line: {"id": ..., "title": ..., "text": ..., "statements": [...]}

Documents without a matching claim entry get statements=[].
Claims without a matching document are silently skipped.
"""

import argparse
import gzip
import json
import logging
import glob
import os
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def resolve_files(inputs):
    files = []
    for pattern in inputs:
        expanded = glob.glob(pattern, recursive=True)
        files.extend(expanded if expanded else [pattern])
    return sorted(set(files))


def open_file(fpath):
    if fpath.endswith(".gz"):
        return gzip.open(fpath, "rt", encoding="utf-8")
    return open(fpath, "r", encoding="utf-8")


def load_claim_map(claim_inputs):
    """Return {doc_id: [statements]} from one or more claim JSONL files."""
    claim_map = {}
    for fpath in tqdm(resolve_files(claim_inputs), desc="Loading claims"):
        with open_file(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                claim_map[entry["id"]] = entry.get("statements") or []
    logger.info("Loaded claims for %d documents", len(claim_map))
    return claim_map


def main():
    parser = argparse.ArgumentParser(
        description="Merge corpus docs + claim files into a unified JSONL collection."
    )
    parser.add_argument(
        "--input", required=True, nargs="+",
        help="Document JSONL or .jsonl.gz file(s); globs accepted",
    )
    parser.add_argument(
        "--claims", required=True, nargs="+",
        help="Claim JSONL file(s) with 'id' and 'statements' fields; globs accepted",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSONL file path (.jsonl or .jsonl.gz)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    claim_map = load_claim_map(args.claims)

    doc_files = resolve_files(args.input)
    logger.info("Found %d document file(s)", len(doc_files))

    open_out = gzip.open if args.output.endswith(".gz") else open
    n_merged = n_missing = 0

    with open_out(args.output, "wt", encoding="utf-8") as out:
        for fpath in tqdm(doc_files, desc="Merging corpus files"):
            with open_file(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    doc = json.loads(line)
                    doc_id = doc["id"]
                    statements = claim_map.get(doc_id)
                    if statements is None:
                        n_missing += 1
                        statements = []
                    merged = {
                        "id": doc_id,
                        "title": doc.get("title", "") or "",
                        "text": doc.get("text", "") or "",
                        "statements": statements,
                    }
                    out.write(json.dumps(merged, ensure_ascii=False) + "\n")
                    n_merged += 1

    logger.info("Wrote %d entries to %s", n_merged, args.output)
    if n_missing:
        logger.warning("%d documents had no matching claims (statements=[])", n_missing)


if __name__ == "__main__":
    main()
