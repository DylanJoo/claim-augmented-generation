# Indexing

All indexes are built with `src/retrieval/indexing.py` using [bm25s](https://github.com/xhluca/bm25s).

## Indexing modes

Four modes control how source documents are turned into indexed units. Exactly one mode flag may be set at a time; omitting all flags defaults to plain document indexing.

| Mode | Flag | Indexed unit | Docid format | Indexed text | Entries per source doc |
|---|---|---|---|---|---|
| **Documents** | *(none)* | Full document | `{id}` | `[title +] text` | 1 |
| **Claims** | `--claim-level` | Each claim separately | `{id}#{i}` | `[title +] claim` | N (one per claim) |
| **Doc-augmented** | `--doc-augmented` | Each claim + full doc | `{id}#{i}` | `[title +] text + claim` | N (one per claim) |
| **Concat-claims** | `--concat-claims` | All claims as one doc | `{id}` | `[title +] claim₁ … claimₙ` | 1 |

`--include-title` prepends the `title` field and applies to all modes.

### When to use each mode

- **Documents** — standard passage/document retrieval baseline.
- **Claims** — fine-grained claim retrieval; retrieved units are individual atomic statements, with `{id}#{i}` allowing lookup of the parent document.
- **Doc-augmented** — like claim retrieval, but each indexed entry also carries the full document body, so BM25 scoring incorporates both document and claim vocabulary.
- **Concat-claims** — document-level retrieval using only claim vocabulary; retrieves at the same granularity as Documents but scores on the distilled claim text rather than the raw body.

## BM25 parameters

| Parameter | Flag | Documents | Claims | Doc-augmented | Concat-claims |
|---|---|---|---|---|---|
| Term frequency saturation | `--k1` | 1.2 | 1.2 | 1.2 | 1.2 |
| Length normalization | `--b` | 0.75 | 0.5 | 0.75 | 0.75 |
| Stopwords | `--stopwords` | `en` | `en` | `en` | `en` |
| Stemmer | `--stemmer` | *(none)* | *(none)* | *(none)* | *(none)* |

`b=0.5` for claim-level reflects that individual claims are short and less sensitive to length normalization. The doc-augmented and concat-claims modes use `b=0.75` because their indexed units are document-length.

## Slurm scripts

| Script | Mode | Input | Output index |
|---|---|---|---|
| `neuclir1-documents.sh` | Documents | `neuclir1/*.processed.jsonl.gz` | `neuclir1/documents.bm25s` |
| `neuclir1-claims.sh` | Claims | `neuclir1/claims/*.jsonl` | `neuclir1/claims.bm25s` |
| `neuclir1-docs-and-claims.sh` | Doc-augmented | `neuclir1/claims/*.jsonl` | `neuclir1/docs-and-claims.bm25s` |
| `neuclir1-concat-claims.sh` | Concat-claims | `neuclir1/claims/*.jsonl` | `neuclir1/concat-claims.bm25s` |

The claims JSONL files are expected to carry both `text` and `statements` fields so that doc-augmented mode can prepend the document body to each claim.

## Usage

```bash
# Documents
python src/retrieval/indexing.py \
    --input /path/to/*.processed.jsonl.gz \
    --index /path/to/documents.bm25s \
    --include-title --k1 1.2 --b 0.75

# Claims
python src/retrieval/indexing.py \
    --input /path/to/claims/*.jsonl \
    --index /path/to/claims.bm25s \
    --claim-level --k1 1.2 --b 0.5

# Doc-augmented
python src/retrieval/indexing.py \
    --input /path/to/claims/*.jsonl \
    --index /path/to/docs-and-claims.bm25s \
    --doc-augmented --include-title --k1 1.2 --b 0.75

# Concat-claims
python src/retrieval/indexing.py \
    --input /path/to/claims/*.jsonl \
    --index /path/to/concat-claims.bm25s \
    --concat-claims --include-title --k1 1.2 --b 0.75
```
