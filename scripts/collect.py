"""Collect real messages.

  uv run python scripts/collect.py antiphishing [--limit N]   # alerts -> review CSV
  uv run python scripts/collect.py uci                        # download UCI zip
  uv run python scripts/collect.py build [--uci-max 300]      # merge all -> real.csv

Only source pages are fetched. URLs found inside messages are stored as text, never opened.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
import urllib.robotparser
import zipfile
from pathlib import Path

import pandas as pd

from phishdata import sources
from phishdata.schema import to_df

RAW = Path("data/raw")
CACHE = RAW / "antiphishing"
DOWNLOADS = RAW / "downloads"
REVIEW = RAW / "review_antiphishing.csv"
MANUAL = RAW / "manual.csv"
REAL = RAW / "real.csv"
UA = "phishdata-research/0.1 (academic phishing-detection project)"
UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DELAY_S = 2.0

_robots: dict[str, urllib.robotparser.RobotFileParser] = {}


def fetch(url: str, cache: Path | None = None) -> bytes:
    """GET a source page politely: cached, robots.txt-checked, rate-limited."""
    if cache and cache.exists():
        return cache.read_bytes()
    root = "/".join(url.split("/")[:3])
    if root not in _robots:
        rp = urllib.robotparser.RobotFileParser(root + "/robots.txt")
        rp.read()
        _robots[root] = rp
    if not _robots[root].can_fetch(UA, url):
        raise PermissionError(f"robots.txt disallows {url}")
    time.sleep(DELAY_S)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read()
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(body)
    return body


def cmd_antiphishing(limit: int | None) -> None:
    brands = sources.load_brands()
    index = fetch(sources.AP_BASE + "/news/alert/", CACHE / "_index.html").decode(
        "utf-8"
    )
    links = sources.ap_alert_links(index)[::-1]  # newest slugs tend to sort last
    rows, failed = [], 0
    for url in links[:limit]:
        try:
            html = fetch(url, CACHE / url.rsplit("/", 1)[-1]).decode(
                "utf-8", errors="replace"
            )
            rows.append(sources.parse_ap_alert(html, url, brands))
        except (
            OSError,
            ValueError,
            AttributeError,
        ) as e:  # one odd page mustn't stop it
            failed += 1
            print(f"skip {url}: {e}")
    new = pd.DataFrame(rows)
    if REVIEW.exists():  # keep text already transcribed
        old = pd.read_csv(REVIEW, dtype=str).fillna("")
        done = old[old["text"].str.strip() != ""]
        new = pd.concat([done, new[~new["campaign_id"].isin(done["campaign_id"])]])
    new.sort_values("collected_on", ascending=False).to_csv(REVIEW, index=False)
    print(f"{len(rows)} alerts parsed, {failed} skipped -> {REVIEW}")
    print(
        "Fill the 'text' column from each screenshot (open the link in your browser)."
    )


def cmd_uci() -> None:
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    data = fetch(UCI_URL, DOWNLOADS / "sms_spam_collection.zip")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        (DOWNLOADS / "SMSSpamCollection").write_bytes(z.read("SMSSpamCollection"))
    print(f"saved {DOWNLOADS / 'SMSSpamCollection'}")


def cmd_build(uci_max: int) -> None:
    brands = sources.load_brands()
    msgs = []
    if REVIEW.exists():
        msgs += sources.ap_review_to_messages(pd.read_csv(REVIEW, dtype=str))
    if (p := DOWNLOADS / "SMSSpamCollection").exists():
        msgs += sources.load_uci(p.read_bytes(), brands, limit=uci_max)
    if (p := DOWNLOADS / "smishtank.csv").exists():
        msgs += sources.load_smishtank(p.read_text("utf-8", errors="replace"), brands)
    if MANUAL.exists():
        msgs += sources.load_manual(MANUAL.read_text("utf-8-sig"), brands)
    else:
        MANUAL.write_text(",".join(sources.MANUAL_HEADER) + "\n", "utf-8")
        print(f"created empty template {MANUAL}")
    df = to_df(msgs).drop_duplicates("id")
    for col in ("url_list", "seed_ids"):
        df[col] = df[col].map(lambda v: json.dumps(v, ensure_ascii=False))
    df.to_csv(REAL, index=False)
    print(f"{len(df)} messages -> {REAL}")
    print(df.groupby(["source", "label", "lang"]).size().to_string())


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("antiphishing").add_argument("--limit", type=int)
    sub.add_parser("uci")
    sub.add_parser("build").add_argument("--uci-max", type=int, default=300)
    a = p.parse_args()
    if a.cmd == "antiphishing":
        cmd_antiphishing(a.limit)
    elif a.cmd == "uci":
        cmd_uci()
    else:
        cmd_build(a.uci_max)


if __name__ == "__main__":
    main()
