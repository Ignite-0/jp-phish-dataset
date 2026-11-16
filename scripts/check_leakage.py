"""Drop synthetic rows that nearly match any val/test/ood message, and report them.

uv run python scripts/check_leakage.py [--threshold 0.8]

Close matches may mean the generator memorised a public scam; the report keeps them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from phishdata.preprocess import normalize
from phishdata.splits import drop_leaks

SYNTHETIC = Path("data/raw/synthetic.jsonl")
PROCESSED = Path("data/processed")
OUT = PROCESSED / "synthetic.parquet"
REPORT = Path("reports/leakage.json")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threshold", type=float, default=0.8)
    a = p.parse_args()

    syn = pd.read_json(SYNTHETIC, lines=True).drop(columns="k")
    held = pd.concat(
        [
            pd.read_parquet(PROCESSED / f"real_{x}.parquet")
            for x in ("val", "test", "ood")
        ]
    )
    kept, matches = drop_leaks(syn, held.reset_index(drop=True), a.threshold)
    kept = kept.assign(text_norm=kept["text"].map(normalize))
    kept.to_parquet(OUT, index=False)

    REPORT.parent.mkdir(exist_ok=True)
    report = {
        "threshold": a.threshold,
        "synthetic_in": len(syn),
        "dropped": len(syn) - len(kept),
        "kept": len(kept),
        "by_pool": pd.Series([m["held_out_pool"] for m in matches])
        .value_counts()
        .to_dict(),
        "matches": matches,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    print(f"{report['dropped']} of {len(syn)} dropped -> {OUT}; details in {REPORT}")


if __name__ == "__main__":
    main()
