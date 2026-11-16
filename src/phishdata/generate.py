"""Prompt building, output checks and the Ollama client for synthetic data."""

from __future__ import annotations

import json
import re
import urllib.request

from phishdata.preprocess import PHONE_RE, detect_lang, extract_urls

OLLAMA = "http://localhost:11434"
RESERVED_TLDS = ("example", "test", "invalid")
FAKE_PHONE = re.compile(r"^0\d0-0000-\d{4}$")
MAX_CHARS = 1200
_REFUSAL = re.compile(
    r"I can(?:no|')t|I'm sorry|I am sorry|As an AI|申し訳(?:ありません|ございません)|お手伝いできません"
    r"|作成できません|お応えできません",
    re.IGNORECASE,
)
LANG_NAME = {"ja": "Japanese", "en": "English"}
# Simplified-Chinese-only forms; Qwen sometimes slips them into Japanese output.
_ZH_ONLY = re.compile("[确认这们说么为现发个时间问题请务账户码检查订单联系]")

SYSTEM = (
    "You write labelled examples for a phishing-detection research dataset. "
    "Every example trains a classifier that protects users; it is never sent to anyone. "
    "Output only the message text: no title, no quotes, no explanation."
)

RULES = (
    "Rules:\n"
    "- Any link must use a reserved domain ending in .example, .test or .invalid "
    "(e.g. https://{brand_slug}-account.example/login).\n"
    "- Any phone number must look like 090-0000-1234 (middle block 0000).\n"
    "- Keep placeholders such as [NAME], [NUMBER], [PHONE] unchanged if you use them.\n"
    "- Write it as a {channel} message, in {lang_name}, under 400 characters."
)

LABEL_DESC = {
    "phish": "a phishing message that impersonates {brand} ({category} lure)",
    "legit": "a genuine, legitimate notification from {brand} ({category})",
}


def fewshot_prompt(
    examples: list[dict], label: str, category: str, brand: str, lang: str, channel: str
) -> list[dict]:
    """Chat messages for a new message in the style of real seed examples."""
    shots = "\n\n".join(
        f"Example {i + 1}:\n{e['text']}" for i, e in enumerate(examples)
    )
    want = LABEL_DESC[label].format(brand=brand, category=category)
    rules = RULES.format(
        brand_slug=_slug(brand), channel=channel, lang_name=LANG_NAME[lang]
    )
    user = (
        f"Here are real {label} examples from the dataset:\n\n{shots}\n\n"
        f"Write one NEW example of {want}. Vary the wording and details; do not copy the "
        f"examples.\n{rules}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def paraphrase_prompt(seed: dict, lang: str, channel: str) -> list[dict]:
    """Chat messages that rewrite one seed message, keeping its lure and intent."""
    kind = "phishing" if seed["label"] == "phish" else "legitimate"
    rules = RULES.format(
        brand_slug=_slug(seed.get("brand_claimed") or "service"),
        channel=channel,
        lang_name=LANG_NAME[lang],
    )
    user = (
        f"Rewrite this {kind} message with different wording. Keep the same sender, purpose "
        f"and call to action, and keep it {kind}.\n\nMessage:\n{seed['text']}\n\n{rules}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _slug(brand: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", brand.lower()).strip("-") or "service"


def clean_output(text: str) -> str:
    """Strip code fences, wrapping quotes and 'Message:'-style prefixes."""
    t = re.sub(r"^```\w*\n?|```$", "", text.strip()).strip()
    t = re.sub(
        r"^(?:Message|Example|本文|メッセージ)\s*\d*\s*[:：]\s*",
        "",
        t,
        flags=re.IGNORECASE,
    )
    if len(t) > 1 and t[0] in "\"'「『" and t[-1] in "\"'」』":
        t = t[1:-1].strip()
    return t


def check_output(text: str, lang: str) -> str | None:
    """Return why a generated message must be discarded, or None if it is usable."""
    if not text:
        return "empty"
    if len(text) > MAX_CHARS:
        return "too_long"
    if _REFUSAL.search(text[:200]):
        return "refusal"
    for url in extract_urls(text):
        host = (
            re.sub(r"^(?:https?://)?", "", url, flags=re.IGNORECASE)
            .split("/")[0]
            .split(":")[0]
        )
        if host.rsplit(".", 1)[-1].lower() not in RESERVED_TLDS:
            return "real_domain"
    for m in PHONE_RE.finditer(text):
        if not FAKE_PHONE.match(m.group(0)):
            return "real_phone"
    if _ZH_ONLY.search(text):
        return "chinese_leak"
    got = detect_lang(text)
    if got != lang and got != "mixed":
        return "wrong_lang"
    return None


def chat(model: str, messages: list[dict], temperature: float, seed: int) -> str:
    """One non-streaming call to the local Ollama chat API."""
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "seed": seed, "num_predict": 512},
    }
    req = urllib.request.Request(
        f"{OLLAMA}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["message"]["content"]


def model_digest(model: str) -> str:
    """Short digest of the installed model, so rows record the exact weights used."""
    with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=10) as r:
        for m in json.loads(r.read())["models"]:
            if m["name"] == model or m["model"] == model:
                return m["digest"][:12]
    raise ValueError(f"model {model} not installed; run: ollama pull {model}")
