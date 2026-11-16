"""Text cleanup: PII masking, normalisation and static URL extraction (no network)."""

from __future__ import annotations

import re
import unicodedata

# Invisible characters, incl. variation selectors seen inside real phishing subjects.
ZERO_WIDTH = re.compile(
    r"[\u00ad\u200b-\u200f\u2060-\u2064\ufeff\ufe00-\ufe0f\U000e0100-\U000e01ef]"
)

# Characters allowed inside a URL after NFKC; stops at Japanese text and brackets.
_URL_CHARS = r"[A-Za-z0-9\-._~:/?#@!$&'*+,;=%]"
URL_RE = re.compile(
    rf"(?:https?://|www\.){_URL_CHARS}+"
    rf"|(?<![@A-Za-z0-9.\-])(?:[A-Za-z0-9\-]+\.)+[A-Za-z]{{2,24}}(?:/{_URL_CHARS}*)?",
    re.IGNORECASE,
)
_NOT_TLDS = {
    "pdf",
    "jpg",
    "jpeg",
    "png",
    "gif",
    "txt",
    "zip",
    "csv",
    "doc",
    "docx",
    "xlsx",
}

EMAIL_RE = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")
_SEP = r"[\-－‐ ()（）]?"
PHONE_RE = re.compile(
    rf"(?<!\d)(?:[+＋]\d{{1,3}}{_SEP}\d{{1,4}}{_SEP}\d{{2,4}}{_SEP}\d{{3,4}}"
    rf"|[0０]\d{{1,4}}[\-－‐ ()（）]\d{{1,4}}[\-－‐ ()（）]\d{{3,4}}"
    rf"|[0０]\d{{9,10}})(?!\d)"
)
NUMBER_RE = re.compile(r"(?<!\d)(?:\d[ \-]?){9,15}\d(?!\d)")
NAME_RE = re.compile(
    r"([一-龯々ァ-ヶーA-Za-z][一-龯々ァ-ヶーA-Za-z ]{0,14})(様|さま)(?!々)"
)
# Honorific targets that are not names ("会員様", "ご利用者様", ...).
_NOT_NAMES = (
    "会員",
    "皆",
    "利用者",
    "契約者",
    "顧客",
    "加入者",
    "購入者",
    "受取人",
    "名義人",
    "担当者",
    "各位",
    "客",
    "お客",
    "ユーザー",
    "メンバー",
)


def normalize(text: str) -> str:
    """NFKC-normalise and strip zero-width/invisible characters."""
    return ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", text))


_JA_CHAR = re.compile(r"[ぁ-んァ-ヶ一-龯々]")
_LATIN = re.compile(r"[A-Za-z]")


def detect_lang(text: str) -> str:
    """Rough language tag from script counts: 'ja', 'en' or 'mixed'."""
    t = URL_RE.sub("", normalize(text))
    ja, en = len(_JA_CHAR.findall(t)), len(_LATIN.findall(t))
    if ja + en == 0:
        return "en"
    share = ja / (ja + en)
    return "ja" if share >= 0.6 else "en" if share <= 0.1 else "mixed"


def refang(text: str) -> str:
    """Undo common defanging (hxxp, [.], (.)) so URLs can be extracted."""
    text = re.sub(r"hxxp", "http", text, flags=re.IGNORECASE)
    return re.sub(r"\[\.\]|\(\.\)|\[dot\]", ".", text, flags=re.IGNORECASE)


def extract_urls(text: str) -> list[str]:
    """Return URLs and bare domains found in text, in order, without duplicates.

    Works on the normalised text, so full-width URLs are found too. Static only.
    """
    found: list[str] = []
    for m in URL_RE.finditer(refang(normalize(text))):
        url = m.group(0).rstrip(".,;:!?'")
        host = re.sub(r"^(?:https?://)?", "", url, flags=re.IGNORECASE).split("/")[0]
        if host.rsplit(".", 1)[-1].lower() in _NOT_TLDS or url in found:
            continue
        found.append(url)
    return found


def _mask_segment(text: str) -> str:
    text = PHONE_RE.sub("[PHONE]", text)
    text = NUMBER_RE.sub("[NUMBER]", text)

    def name(m: re.Match[str]) -> str:
        who = m.group(1).strip()
        return m.group(0) if who.endswith(_NOT_NAMES) else f"[NAME]{m.group(2)}"

    return NAME_RE.sub(name, text)


def mask_pii(text: str) -> str:
    """Replace emails, phone numbers, long digit runs and names before 様 with tokens.

    URLs are left untouched so their features survive; raw characters are kept.
    """
    text = EMAIL_RE.sub("[EMAIL]", text)  # first, so "j.smith@" isn't read as a domain
    out, last = [], 0
    for m in URL_RE.finditer(text):
        out.append(_mask_segment(text[last : m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(_mask_segment(text[last:]))
    return "".join(out)
