#!/usr/bin/env python3
"""Convert ragtime2026 unprocessed topics into the ragtime/neuclir topic format.

Unprocessed record:
  {"topic_id", "collection_id", "title", "problem_statement", "background", "limit"}

Target record (matches ragtime2025.topics.test.jsonl):
  {"qid", "query", "meta": {"title", "background", "collection_id", "limit", "track"}}
"""
import json

SRC = "ragtime2026.topics.test.jsonl.unprocessed"
DST = "ragtime2026.topics.test.jsonl"

with open(SRC) as fin, open(DST, "w") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out = {
            "qid": str(d["topic_id"]),
            "query": d["problem_statement"],
            "meta": {
                "title": d["title"],
                "background": d["background"],
                "collection_id": d["collection_id"],
                "limit": d["limit"],
                "track": "ragtime",
            },
        }
        fout.write(json.dumps(out) + "\n")
