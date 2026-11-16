"""Split real messages into seed/val/test/ood before any generation, then freeze them.

uv run python scripts/make_splits.py --ood-brands sagawa mizuho docomo [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from phishdata.preprocess import normalize
from phishdata.schema import to_df, validate_df
from phishdata.splits import assign_pools, dedupe

REAL = Path("data/raw/real.csv")
OUT = Path("data/processed")
MANIFEST = OUT / "manifest.json"
POOLS = ["seed", "val", "test", "ood"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ood-brands", nargs="+", required=True, help="one brand key per category"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="overwrite a frozen split")
    a = p.parse_args()
    if MANIFEST.exists() and not a.force:
        raise SystemExit(
            f"{MANIFEST} exists: the split is frozen. Use --force only on purpose."
        )

    raw = pd.read_csv(REAL, dtype=str, keep_default_na=False).replace("", None)
    for col in ("url_list", "seed_ids"):
        raw[col] = raw[col].map(lambda s: json.loads(s) if s else [])
    df = to_df(validate_df(raw))
    n0 = len(df)
    df = assign_pools(dedupe(df), a.ood_brands, a.seed)
    df["text_norm"] = df["text"].map(normalize)
    print(f"{n0} real rows, {n0 - len(df)} near-duplicates dropped")

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": a.seed, "ood_brands": a.ood_brands, "files": {}}
    for pool in POOLS:
        path = OUT / f"real_{pool}.parquet"
        part = df[df["pool"] == pool]
        part.to_parquet(path, index=False)
        manifest["files"][path.name] = {
            "rows": len(part),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    MANIFEST.write_text(json.dumps(manifest, indent=2), "utf-8")
    print(
        pd.crosstab([df["pool"], df["label"]], [df["lang"]], margins=True).to_string()
    )
    print(pd.crosstab(df["category"], df["pool"]).to_string())
    print(f"frozen -> {MANIFEST}")


if __name__ == "__main__":
    main()
