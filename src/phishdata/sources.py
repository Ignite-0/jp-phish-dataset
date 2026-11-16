"""Parsers that turn public sources into schema rows. No network access here."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from phishdata.preprocess import detect_lang, extract_urls, mask_pii, normalize
from phishdata.schema import Message

AP_BASE = "https://www.antiphishing.jp"
BRANDS_PATH = Path(__file__).with_name("brands.json")  # ships inside the package

_CATEGORY_WORDS = [
    ("government", ("税", "年金", "保険料", "e-Tax", "国民", "住民", "マイナ")),
    ("banking", ("銀行", "証券", "カード", "信託", "バンク")),
    ("delivery", ("配送", "宅配", "郵便", "荷物", "運輸", "不在通知")),
    ("carrier", ("携帯", "通信", "ISP", "プロバイダ", "電力", "ガス", "Webメール")),
    ("shopping", ("通販", "ショッピング", "EC")),
]


def load_brands(path: Path = BRANDS_PATH) -> dict[str, dict]:
    """Brand key -> {names, category, domains, verified}."""
    return {
        k: v
        for k, v in json.loads(path.read_text("utf-8")).items()
        if not k.startswith("_")
    }


def match_brand(text: str, brands: dict[str, dict]) -> str | None:
    """Brand whose name appears in the text (longest match wins).

    Text is NFKC-normalised first. Japanese names ignore spaces ("三菱 UFJ 銀行").
    Latin names must stand alone ("au" must not match "because"); short or
    all-caps names are case-sensitive so "line" doesn't match "LINE".
    """
    t = normalize(text)
    squeezed = t.replace(" ", "")
    hits = [
        (len(n), k)
        for k, b in brands.items()
        for n in b["names"]
        if _has_name(n, t, squeezed)
    ]
    return max(hits)[1] if hits else None


def _has_name(name: str, text: str, squeezed: str) -> bool:
    if not name.isascii():
        return normalize(name).replace(" ", "") in squeezed
    flags = 0 if len(name) <= 3 or name.isupper() else re.IGNORECASE
    return (
        re.search(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", text, flags)
        is not None
    )


def guess_category(text: str, brand: str | None, brands: dict[str, dict]) -> str:
    """Category from the brand, else from keywords, else 'other'."""
    if brand:
        return brands[brand]["category"]
    for cat, words in _CATEGORY_WORDS:
        if any(w in text for w in words):
            return cat
    return "other"


def row_id(source: str, text: str) -> str:
    """Stable id from source and text."""
    return f"{source}-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"


def campaign_from_urls(urls: list[str], fallback: str) -> str:
    """Group messages that point at the same host; fallback when there is no URL."""
    if not urls:
        return fallback
    host = (
        re.sub(r"^(?:https?://)?", "", urls[0], flags=re.IGNORECASE)
        .split("/")[0]
        .lower()
    )
    return "host-" + ".".join(host.split(".")[-2:])


# --- Council of Anti-Phishing Japan -------------------------------------------------


def ap_alert_links(index_html: str) -> list[str]:
    """Absolute URLs of every alert page linked from the alert index."""
    soup = BeautifulSoup(index_html, "html.parser")
    hrefs = {
        a["href"]
        for a in soup.select('a[href^="/news/alert/"]')
        if a["href"].endswith(".html")
    }
    return sorted(urljoin(AP_BASE, h) for h in hrefs)


def _section(entry: BeautifulSoup, heading: str) -> BeautifulSoup | None:
    for h in entry.select("h4.alert_h4"):
        if heading in h.get_text():
            return h.find_next_sibling()
    return None


def parse_ap_alert(html: str, url: str, brands: dict[str, dict]) -> dict:
    """One review record per alert. The message body is a screenshot, so `text` is empty
    for a human to transcribe; subjects and (partly masked) URLs are kept as text."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.select_one("h3.default_h3").get_text(strip=True)
    entry = soup.select_one("div.entry")
    d = soup.select_one("p.date")
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", d.get_text() if d else "")
    when = date(*map(int, m.groups())).isoformat() if m else ""

    summary = s.get_text(" ", strip=True) if (s := _section(entry, "概要")) else ""
    subj = _section(entry, "件名")
    subjects = [
        ln
        for ln in (subj.get_text("\n", strip=True).splitlines() if subj else [])
        if ln and not ln.startswith("※")
    ]
    site = _section(entry, "URL")
    urls = re.findall(r"https?://\S+", site.get_text("\n") if site else "")
    shot = _section(entry, "文面例")
    img = shot.find("img") if shot else None
    if img is None and shot is not None:
        img = shot.find_next("img")

    is_sms = any(w in summary for w in ("SMS", "ショートメッセージ"))
    brand = match_brand(title, brands)
    return {
        "campaign_id": "ap-" + url.rsplit("/", 1)[-1].removesuffix(".html"),
        "collected_on": when,
        "title": title,
        "brand_claimed": brand or "",
        "category": guess_category(title + summary, brand, brands),
        "channel": "sms" if is_sms else "email",
        "subjects": " | ".join(subjects),
        "urls": " ".join(urls),
        "screenshot": urljoin(AP_BASE, img["src"]) if img else "",
        "source_url": url,
        "text": "",
    }


def ap_review_to_messages(df: pd.DataFrame) -> list[Message]:
    """Turn reviewed alert rows (text filled in by hand) into phishing messages."""
    out = []
    for r in df.fillna("").to_dict("records"):
        if not str(r["text"]).strip():
            continue
        text = mask_pii(str(r["text"]))
        out.append(
            Message(
                id=row_id("antiphishing", text),
                text=text,
                url_list=extract_urls(text) or str(r["urls"]).split(),
                channel=r["channel"],
                lang=detect_lang(text),
                label="phish",
                category=r["category"],
                brand_claimed=r["brand_claimed"] or None,
                source="antiphishing",
                source_url=r["source_url"],
                campaign_id=r["campaign_id"],
                collected_on=r["collected_on"] or None,
            )
        )
    return out


# --- English datasets -------------------------------------------------------------


def load_uci(
    raw: bytes, brands: dict[str, dict], limit: int | None = None
) -> list[Message]:
    """UCI SMS Spam Collection (tab-separated 'label<TAB>text'). Only ham is used:
    spam is not phishing, so it would mislabel the positive class."""
    out = []
    for i, line in enumerate(raw.decode("utf-8", errors="replace").splitlines()):
        tag, _, body = line.partition("\t")
        if tag != "ham" or not body.strip():
            continue
        text = mask_pii(body)
        brand = match_brand(text, brands)
        out.append(
            Message(
                id=row_id("uci", text),
                text=text,
                url_list=extract_urls(text),
                channel="sms",
                lang="en",
                label="legit",
                category=guess_category(text, brand, brands),
                brand_claimed=brand,
                source="uci_sms_spam",
                source_url="https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
                campaign_id=f"uci-{i}",
            )
        )
        if limit and len(out) >= limit:
            break
    return out


def _pick(cols: list[str], *wanted: str) -> str | None:
    low = {c.lower(): c for c in cols}
    for w in wanted:
        for lc, c in low.items():
            if w in lc:
                return c
    return None


def load_smishtank(csv_text: str, brands: dict[str, dict]) -> list[Message]:
    """Smishtank CSV (all phishing). Column names aren't documented, so the text and date
    columns are found by name; raises if no text column is found."""
    df = pd.read_csv(io.StringIO(csv_text), dtype=str).fillna("")
    text_col = _pick(
        list(df.columns), "maintext", "fulltext", "text", "message", "body"
    )
    if text_col is None:
        raise ValueError(f"no text column in Smishtank CSV: {list(df.columns)}")
    date_col = _pick(list(df.columns), "date", "time", "submitted")
    out = []
    for r in df.to_dict("records"):
        if not r[text_col].strip():
            continue
        text = mask_pii(r[text_col])
        urls = extract_urls(text)
        brand = match_brand(text, brands)
        when = pd.to_datetime(r[date_col], errors="coerce") if date_col else pd.NaT
        out.append(
            Message(
                id=row_id("smishtank", text),
                text=text,
                url_list=urls,
                channel="sms",
                lang=detect_lang(text),
                label="phish",
                category=guess_category(text, brand, brands),
                brand_claimed=brand,
                source="smishtank",
                source_url="https://smishtank.com",
                campaign_id=campaign_from_urls(urls, row_id("st", text)),
                collected_on=None if pd.isna(when) else when.date(),
            )
        )
    return out


MANUAL_HEADER = [
    "text",
    "label",
    "channel",
    "category",
    "brand_claimed",
    "source",
    "source_url",
    "campaign_id",
    "collected_on",
]


def load_manual(csv_text: str, brands: dict[str, dict]) -> list[Message]:
    """Hand-entered rows (screenshots, brand pages, own inbox). Text is masked here,
    and blank brand/category/campaign fields are filled in."""
    out = []
    for r in csv.DictReader(io.StringIO(csv_text)):
        if not (r.get("text") or "").strip():
            continue
        text = mask_pii(r["text"])
        urls = extract_urls(text)
        brand = r.get("brand_claimed") or match_brand(text, brands)
        out.append(
            Message(
                id=row_id("manual", text),
                text=text,
                url_list=urls,
                channel=r.get("channel") or "sms",
                lang=detect_lang(text),
                label=r["label"],
                category=r.get("category") or guess_category(text, brand, brands),
                brand_claimed=brand,
                source=r.get("source") or "manual",
                source_url=r.get("source_url") or None,
                campaign_id=r.get("campaign_id")
                or campaign_from_urls(urls, row_id("m", text)),
                collected_on=r.get("collected_on") or None,
            )
        )
    return out
