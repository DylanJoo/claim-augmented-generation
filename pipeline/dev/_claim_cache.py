"""
Shared on-disk cache for cc_dense.py's claim-embedding shard loading.
Scanning every claim_reps shard for a ~100-doc pooled topic takes 15-20
minutes (see run_dominant_doc_check.py's shard-by-shard log) since
_load_claim_reps has to scan every shard file looking for the needed docids.
Repeated dev/analysis scripts over the same topic pool would otherwise pay
that cost on every single run -- this caches the result keyed by
(claim_reps glob pattern, sorted needed_docids), so a different topic/pool
naturally misses and reloads fresh.
"""
import hashlib
import os
import pickle

_DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/cc_dense_dev/claim_reps")


def _cache_path(claim_reps_pattern, needed_docids, cache_dir):
    key = claim_reps_pattern + "|" + ",".join(sorted(needed_docids))
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"claim_reps.{digest}.pkl")


def load_claim_reps_cached(load_fn, claim_reps_pattern, needed_docids, cache_dir=_DEFAULT_CACHE_DIR):
    """load_fn: cc_dense._load_claim_reps, injected so this module doesn't
    need to import cc_dense itself."""
    path = _cache_path(claim_reps_pattern, needed_docids, cache_dir)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    reps = load_fn(claim_reps_pattern, needed_docids)
    with open(path, "wb") as f:
        pickle.dump(reps, f)
    return reps
