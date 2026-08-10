"""
Flatten processed-claims JSONL into one entry per claim, for dense encoding
(e.g. with tevatron) rather than BM25 indexing.

Each output line: {"id": "{parent_id}#{i}", "title": ..., "text": claim}
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


def open_out(fpath):
    return gzip.open(fpath, "wt", encoding="utf-8")


def flatten(doc_inputs, output_path):
    files = resolve_files(doc_inputs)
    if not files:
        logger.warning("No corpus files found for inputs: %s", doc_inputs)
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    logger.info("Found %d corpus file(s)", len(files))

    n_claims = 0
    with open_out(output_path) as out:
        for fpath in tqdm(files, desc="Flattening claims"):
            with open_file(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    doc = json.loads(line)
                    parent_id = doc["id"]
                    title = (doc.get("title", "") or "").strip()
                    claims = doc.get("statements") or []

                    # Title or not would be decided during encoding.
                    for i, claim in enumerate(claims):
                        out.write(json.dumps({
                            "id": f"{parent_id}#{i}",
                            "title": title,
                            "text": claim.strip(),
                        }, ensure_ascii=False) + "\n")
                        n_claims += 1

    logger.info("Wrote %d claims to %s", n_claims, output_path)
    return n_claims


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, nargs="+",
                        help="Processed-claims JSONL or .jsonl.gz file(s); globs accepted")
    parser.add_argument("--output", required=True,
                        help="Output path, one entry per claim; always written gzip-compressed as .jsonl.gz")
    parser.add_argument("--include-title", action="store_true", default=False,
                        help="Include the title field on each flattened claim")
    args = parser.parse_args()

    flatten(args.input, args.output)


if __name__ == "__main__":
    main()
