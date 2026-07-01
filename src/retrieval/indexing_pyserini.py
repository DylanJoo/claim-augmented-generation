import argparse
import gzip
import json
import logging
import os
import glob
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


def preprocess_docs_and_claims(doc_inputs, output_dir, include_title=False):
    """Stream-convert processed-claims JSONL to Pyserini JsonCollection format.

    Each claim becomes one entry: {"id": "{parent}#{i}", "contents": title + doc_text + claim}
    Written line-by-line to avoid holding the corpus in RAM.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = resolve_files(doc_inputs)
    if not files:
        logger.warning("No corpus files found for inputs: %s", doc_inputs)
        return

    logger.info("Found %d corpus file(s)", len(files))
    out_path = os.path.join(output_dir, "collection.jsonl")
    count = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for fpath in tqdm(files, desc="Processing files"):
            with open_file(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    doc = json.loads(line)
                    parent_id = doc["id"]
                    title = (doc.get("title", "") or "").strip() if include_title else ""
                    doc_text = (doc.get("text", "") or "").strip()
                    claims = doc.get("statements") or []

                    for i, claim in enumerate(claims):
                        parts = [p for p in [title, doc_text, claim.strip()] if p]
                        entry = {
                            "id": f"{parent_id}#{i}",
                            "contents": " ".join(parts),
                        }
                        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        count += 1

    logger.info("Wrote %d entries to %s", count, out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess processed-claims JSONL into Pyserini JsonCollection format."
    )
    parser.add_argument("--input", required=True, nargs="+",
                        help="Processed-claims JSONL or .jsonl.gz file(s); globs accepted")
    parser.add_argument("--output", required=True,
                        help="Output directory for the Pyserini collection JSONL")
    parser.add_argument("--include-title", action="store_true", default=False,
                        help="Prepend the title field to each indexed unit")
    args = parser.parse_args()

    preprocess_docs_and_claims(args.input, args.output, include_title=args.include_title)
    logger.info("Done. Run pyserini.index.lucene on: %s", args.output)


if __name__ == "__main__":
    main()
