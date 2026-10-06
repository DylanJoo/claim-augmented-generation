import argparse
import gzip
import json
import logging
import os
import glob
import bm25s
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def resolve_files(inputs):
    """Expand a list of paths/globs into a sorted list of file paths."""
    files = []
    for pattern in inputs:
        expanded = glob.glob(pattern, recursive=True)
        files.extend(expanded if expanded else [pattern])
    return sorted(set(files))


def open_file(fpath):
    if fpath.endswith(".gz"):
        return gzip.open(fpath, "rt", encoding="utf-8")
    return open(fpath, "r", encoding="utf-8")


def load_corpus_doc(doc_inputs, include_title=False, concat_claims=False):
    """Doc-level indexing: one entry per source document.

    concat_claims=False → text = title + doc text
    concat_claims=True  → text = title + all statements concatenated
    """
    docs, docids = [], []
    files = resolve_files(doc_inputs)
    if not files:
        logger.warning("No corpus files found for inputs: %s", doc_inputs)
        return docs, docids
    logger.info("Found %d corpus file(s)", len(files))

    for fpath in tqdm(files, desc="Loading corpus files"):
        with open_file(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                parent_id = doc["id"]
                title = (doc.get("title", "") or "").strip() if include_title else ""

                if concat_claims:
                    claims = doc.get("statements") or []
                    content = " ".join(c.strip() for c in claims)
                else:
                    content = (doc.get("text", "") or "").strip()

                parts = [p for p in [title, content] if p]
                docids.append(parent_id)
                docs.append(" ".join(parts))

    return docs, docids


def load_corpus_claims(doc_inputs, include_title=False, doc_augmented=False):
    """Claim-level indexing: one entry per claim, docid = '{parent_id}#{i}'.

    doc_augmented=False → text = title + claim
    doc_augmented=True  → text = title + doc text + claim
    """
    docs, docids = [], []
    files = resolve_files(doc_inputs)
    if not files:
        logger.warning("No corpus files found for inputs: %s", doc_inputs)
        return docs, docids
    logger.info("Found %d corpus file(s)", len(files))

    for fpath in tqdm(files, desc="Loading corpus files"):
        with open_file(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                parent_id = doc["id"]
                title = (doc.get("title", "") or "").strip() if include_title else ""
                doc_text = (doc.get("text", "") or "").strip() if doc_augmented else ""
                claims = doc.get("statements") or []

                for i, claim in enumerate(claims):
                    if doc_augmented:
                        parts = [p for p in [title, doc_text, claim.strip()] if p]
                    else:
                        parts = [p for p in [title, claim.strip()] if p]
                    docids.append(f"{parent_id}#{i}")
                    docs.append(" ".join(parts))

    return docs, docids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+",
                        help="Merged JSONL or .jsonl.gz file(s); globs accepted")
    parser.add_argument("--index", required=True, help="Output directory for the bm25s index")
    parser.add_argument("--stopwords", default="en")
    parser.add_argument("--stemmer", default=None, choices=[None, "snowball"])
    parser.add_argument("--k1", type=float, default=0.5)
    parser.add_argument("--b", type=float, default=0.0)
    parser.add_argument("--claim-level", action="store_true", default=False,
                        help="Claim-level indexing: one entry per claim")
    parser.add_argument("--doc-augmented", action="store_true", default=False,
                        help="With --claim-level: prepend full doc text to each claim")
    parser.add_argument("--concat-claims", action="store_true", default=False,
                        help="Doc-level indexing using concatenated claims instead of doc text")
    parser.add_argument("--include-title", action="store_true", default=False,
                        help="Prepend the title field to each indexed unit")
    args = parser.parse_args()

    if args.doc_augmented and not args.claim_level:
        parser.error("--doc-augmented requires --claim-level")
    if args.concat_claims and args.claim_level:
        parser.error("--concat-claims and --claim-level are mutually exclusive")

    os.makedirs(args.index, exist_ok=True)

    logger.info("Loading corpus from %s", args.input)
    if args.claim_level:
        mode = "doc-augmented" if args.doc_augmented else "claims"
        logger.info("Claim-level indexing (%s)", mode)
        docs, docids = load_corpus_claims(
            args.input,
            include_title=args.include_title,
            doc_augmented=args.doc_augmented,
        )
    else:
        mode = "concat-claims" if args.concat_claims else "documents"
        logger.info("Doc-level indexing (%s)", mode)
        docs, docids = load_corpus_doc(
            args.input,
            include_title=args.include_title,
            concat_claims=args.concat_claims,
        )
    logger.info("Loaded %d documents", len(docs))

    logger.info("Tokenizing (stopwords=%s, stemmer=%s)", args.stopwords, args.stemmer)
    stemmer = None
    if args.stemmer == "snowball":
        import Stemmer
        stemmer = Stemmer.Stemmer("english")

    corpus_tokens = bm25s.tokenize(docs, stopwords=args.stopwords, stemmer=stemmer)

    logger.info("Building BM25 index (k1=%.2f, b=%.2f)", args.k1, args.b)
    retriever = bm25s.BM25(k1=args.k1, b=args.b)
    retriever.index(corpus_tokens)

    logger.info("Saving index to %s", args.index)
    retriever.save(args.index, corpus=docs)

    docids_path = os.path.join(args.index, "docids.json")
    with open(docids_path, "w") as f:
        json.dump(docids, f)
    logger.info("Saved %d docids to %s", len(docids), docids_path)

    logger.info("Done")


if __name__ == "__main__":
    main()
