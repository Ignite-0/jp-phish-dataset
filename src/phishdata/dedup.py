"""MinHash near-duplicate search, shared by the split and leakage scripts."""

from __future__ import annotations

from datasketch import MinHash, MinHashLSH

from phishdata.preprocess import normalize

NUM_PERM = 128


def minhash(text: str, n: int = 3) -> MinHash:
    """MinHash over character n-grams of the normalised, lowercased text."""
    t = normalize(text).lower()
    m = MinHash(num_perm=NUM_PERM)
    for i in range(max(1, len(t) - n + 1)):
        m.update(t[i : i + n].encode("utf-8"))
    return m


def near_duplicates(
    queries: list[str], corpus: list[str], threshold: float = 0.9
) -> list[tuple[int, int, float]]:
    """Return (query_idx, corpus_idx, estimated_jaccard) for pairs above threshold.

    Pass the same list twice to dedupe within one set; self-pairs are skipped.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    corpus_mh = [minhash(t) for t in corpus]
    for j, m in enumerate(corpus_mh):
        lsh.insert(str(j), m)
    same = queries is corpus
    pairs = []
    for i, text in enumerate(queries):
        q = corpus_mh[i] if same else minhash(text)
        for key in lsh.query(q):
            j = int(key)
            if same and j <= i:
                continue
            score = q.jaccard(corpus_mh[j])
            if score >= threshold:
                pairs.append((i, j, score))
    return pairs
