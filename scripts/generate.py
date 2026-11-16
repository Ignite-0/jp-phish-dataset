"""Generate synthetic training messages with a local Ollama model, from the seed pool only.

  uv run python scripts/generate.py --n 50        # smoke run
  uv run python scripts/generate.py --n 4000      # full run; safe to stop and restart

Accepted rows go to data/raw/synthetic.jsonl, discards (with reason) to
data/raw/synthetic_rejects.jsonl. Job k always uses seed --seed + k, so reruns resume.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

from phishdata import generate as gen
from phishdata.preprocess import detect_lang, extract_urls
from phishdata.schema import Message
from phishdata.sources import load_brands

SEED_FILE = Path("data/processed/real_seed.parquet")
MANIFEST = Path("data/processed/manifest.json")
OUT = Path("data/raw/synthetic.jsonl")
REJECTS = Path("data/raw/synthetic_rejects.jsonl")
SAMPLE = Path("data/raw/synthetic_review_sample.csv")
LABELS, LANGS = ["phish", "legit"], ["ja", "en"]


def load_seed(path: Path) -> pd.DataFrame:
    """Read the seed pool, refusing anything that isn't purely seed rows."""
    if path.name != SEED_FILE.name:
        raise SystemExit(f"generation may only read {SEED_FILE.name}, got {path.name}")
    df = pd.read_parquet(path)
    if set(df["pool"]) != {"seed"}:
        raise SystemExit(f"{path} holds non-seed rows: {set(df['pool'])}")
    return df.astype(object).where(df.notna(), None)  # NaN is truthy; use None


def plan_job(k: int, seed: int, df: pd.DataFrame, brands: dict[str, dict]) -> dict:
    """Pick method, label, language, category, brand and seed examples for job k."""
    rng = random.Random(seed + k)
    label, lang = LABELS[k % 2], LANGS[(k // 2) % 2]
    method = "paraphrase" if rng.random() < 0.5 else "fewshot"
    cats = sorted({b["category"] for b in brands.values()})
    category = rng.choice(cats)
    pool = df[(df["label"] == label) & (df["category"] == category)]
    if len(pool) < 3:  # thin cell: borrow style from the same label
        pool = df[df["label"] == label]
    if pool.empty:
        raise SystemExit(f"seed pool has no {label} rows")
    same_lang = pool[pool["lang"] == lang]
    pick_from = same_lang if len(same_lang) >= 3 else pool
    examples = pick_from.sample(min(3, len(pick_from)), random_state=seed + k)
    names = sorted(k2 for k2, b in brands.items() if b["category"] == category)
    brand_key = rng.choice(names)
    channel = rng.choice(examples["channel"].tolist())
    if method == "paraphrase":  # a rewrite inherits what the seed message is about
        examples = examples.head(1)
        s = examples.iloc[0]
        category, brand_key, channel = s["category"], s["brand_claimed"], s["channel"]
    return {
        "k": k,
        "seed": seed + k,
        "method": method,
        "label": label,
        "lang": lang,
        "category": category,
        "brand_key": brand_key,
        "channel": channel,
        "examples": examples.to_dict("records"),
    }


def prompt_for(job: dict, brands: dict[str, dict]) -> list[dict]:
    b = brands.get(job["brand_key"], {})
    name = next((n for n in b.get("names", []) if detect_lang(n) == job["lang"]), None)
    brand = name or (b["names"][0] if b else job["brand_key"])
    if job["method"] == "paraphrase":
        return gen.paraphrase_prompt(job["examples"][0], job["lang"], job["channel"])
    return gen.fewshot_prompt(
        job["examples"],
        job["label"],
        job["category"],
        brand,
        job["lang"],
        job["channel"],
    )


def done_jobs() -> set[int]:
    ks = set()
    for path in (OUT, REJECTS):
        if path.exists():
            ks |= {
                json.loads(ln)["k"] for ln in path.read_text("utf-8").splitlines() if ln
            }
    return ks


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--seed-file", type=Path, default=SEED_FILE)
    p.add_argument("--n", type=int, default=4000, help="accepted rows to reach")
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    df = load_seed(a.seed_file)
    ood = json.loads(MANIFEST.read_text("utf-8"))["ood_brands"]
    brands = {k: v for k, v in load_brands().items() if k not in ood}  # keep OOD unseen
    generator = f"ollama:{a.model}@{gen.model_digest(a.model)}"
    seen = done_jobs()
    accepted = sum(1 for _ in OUT.open("r", encoding="utf-8")) if OUT.exists() else 0
    k = 0
    with (
        OUT.open("a", encoding="utf-8") as out,
        REJECTS.open("a", encoding="utf-8") as rej,
    ):
        while accepted < a.n and k < a.n * 4:
            if k in seen:
                k += 1
                continue
            job = plan_job(k, a.seed, df, brands)
            text = gen.clean_output(
                gen.chat(a.model, prompt_for(job, brands), a.temperature, job["seed"])
            )
            reason = gen.check_output(text, job["lang"])
            if reason:
                rej.write(
                    json.dumps(
                        {"k": k, "reason": reason, "text": text}, ensure_ascii=False
                    )
                    + "\n"
                )
            else:
                msg = Message(
                    id=f"syn-{a.seed}-{k}",
                    text=text,
                    url_list=extract_urls(text),
                    channel=job["channel"],
                    lang=detect_lang(text),
                    label=job["label"],
                    category=job["category"],
                    brand_claimed=job["brand_key"],
                    source="synthetic",
                    campaign_id=f"syn-{a.seed}-{k}",
                    pool="synthetic",
                    is_synthetic=True,
                    seed_ids=[e["id"] for e in job["examples"]],
                    generator=generator,
                    prompt_id=f"{job['method']}-v1",
                    temperature=a.temperature,
                    seed=job["seed"],
                    collected_on=date.today(),
                )
                out.write(
                    json.dumps(
                        {"k": k, **msg.model_dump(mode="json")}, ensure_ascii=False
                    )
                    + "\n"
                )
                accepted += 1
            out.flush()
            rej.flush()
            if k % 25 == 0:
                print(f"job {k}: {accepted}/{a.n} accepted")
            k += 1

    rows = [json.loads(ln) for ln in OUT.read_text("utf-8").splitlines() if ln]
    rejects = [json.loads(ln) for ln in REJECTS.read_text("utf-8").splitlines() if ln]
    total = len(rows) + len(rejects)
    print(
        f"accepted {len(rows)}, discarded {len(rejects)} ({len(rejects) / max(total, 1):.1%})"
    )
    print(Counter(r["reason"] for r in rejects).most_common())
    pd.DataFrame(rows).sample(min(100, len(rows)), random_state=0)[
        ["id", "label", "lang", "category", "brand_claimed", "prompt_id", "text"]
    ].to_csv(SAMPLE, index=False)
    print(f"read these yourself: {SAMPLE}")


if __name__ == "__main__":
    main()
