# Indexing

All indexes are built with `src/retrieval/indexing.py` using [bm25s](https://github.com/xhluca/bm25s).

## Corpus preparation

Before indexing, merge the document corpus and claim files into a single collection:

```bash
python src/retrieval/merge.py \
    --input /path/to/*.processed.jsonl.gz \
    --claims /path/to/claims/*.jsonl \
    --output /path/to/collection.jsonl.gz
```

Each line of the output has `id`, `title`, `text`, and `statements`. All indexing modes read from this file.

## Indexing modes

Four modes control how collection entries are turned into indexed units.

| Mode | Flag | Indexed unit | Docid format | Indexed text | Entries per doc |
|---|---|---|---|---|---|
| **Documents** | *(none)* | Full document | `{id}` | `[title +] text` | 1 |
| **Concat-claims** | `--concat-claims` | All claims as one doc | `{id}` | `[title +] claim₁ … claimₙ` | 1 |
| **Claims** | `--claim-level` | Each claim separately | `{id}#{i}` | `[title +] claim` | N |
| **Doc-augmented** | `--claim-level --doc-augmented` | Each claim + full doc | `{id}#{i}` | `[title +] text + claim` | N |

`--include-title` prepends the `title` field and applies to all modes.  
`--concat-claims` and `--claim-level` are mutually exclusive.

## BM25 parameters

| Parameter | Flag | Documents | Concat-claims | Claims | Doc-augmented |
|---|---|---|---|---|---|
| Term frequency saturation | `--k1` | 1.2 | 1.2 | 1.2 | 1.2 |
| Length normalization | `--b` | 0.75 | 0.75 | 0.5 | 0.75 |
| Stopwords | `--stopwords` | `en` | `en` | `en` | `en` |

`b=0.5` for claims reflects that individual claims are short and less sensitive to length normalization.

## Slurm scripts

| Script | Mode | Output index |
|---|---|---|
| `neuclir1-documents.sh` | Documents | `neuclir1/documents.bm25s` |
| `neuclir1-concat-claims.sh` | Concat-claims | `neuclir1/concat-claims.bm25s` |
| `neuclir1-claims.sh` | Claims | `neuclir1/claims.bm25s` |
| `neuclir1-docs-and-claims.sh` | Doc-augmented | `neuclir1/docs-and-claims.bm25s` |

All scripts read from `neuclir1/collection.jsonl.gz`.

## Usage

```bash
# Documents
python src/retrieval/indexing.py \
    --input /path/to/collection.jsonl.gz \
    --index /path/to/documents.bm25s \
    --include-title --k1 1.2 --b 0.75

# Concat-claims
python src/retrieval/indexing.py \
    --input /path/to/collection.jsonl.gz \
    --index /path/to/concat-claims.bm25s \
    --concat-claims --include-title --k1 1.2 --b 0.75

# Claims
python src/retrieval/indexing.py \
    --input /path/to/collection.jsonl.gz \
    --index /path/to/claims.bm25s \
    --claim-level --k1 1.2 --b 0.5

# Doc-augmented
python src/retrieval/indexing.py \
    --input /path/to/collection.jsonl.gz \
    --index /path/to/docs-and-claims.bm25s \
    --claim-level --doc-augmented --include-title --k1 1.2 --b 0.75
```
