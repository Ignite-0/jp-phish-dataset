"""Row schema for the message dataset (one row per message)."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, ValidationError, model_validator

COLUMNS = [
    "id",
    "text",
    "url_list",
    "channel",
    "lang",
    "label",
    "category",
    "brand_claimed",
    "source",
    "source_url",
    "campaign_id",
    "pool",
    "is_synthetic",
    "seed_ids",
    "generator",
    "prompt_id",
    "temperature",
    "seed",
    "collected_on",
]


class Channel(StrEnum):
    sms = "sms"
    email = "email"


class Lang(StrEnum):
    ja = "ja"
    en = "en"
    mixed = "mixed"


class Label(StrEnum):
    phish = "phish"
    legit = "legit"


class Category(StrEnum):
    delivery = "delivery"
    banking = "banking"
    account = "account"
    carrier = "carrier"
    shopping = "shopping"
    government = "government"
    other = "other"


class Pool(StrEnum):
    unsplit = "unsplit"
    seed = "seed"
    val = "val"
    test = "test"
    ood = "ood"
    synthetic = "synthetic"


class Message(BaseModel):
    """One labelled message. Synthetic rows must point back to their seeds."""

    id: str
    text: str
    url_list: list[str] = []
    channel: Channel
    lang: Lang
    label: Label
    category: Category
    brand_claimed: str | None = None
    source: str
    source_url: str | None = None
    campaign_id: str
    pool: Pool = Pool.unsplit
    is_synthetic: bool = False
    seed_ids: list[str] = []
    generator: str | None = None
    prompt_id: str | None = None
    temperature: float | None = None
    seed: int | None = None
    collected_on: date | None = None

    @model_validator(mode="after")
    def _synthetic_consistency(self) -> Message:
        if not self.text.strip():
            raise ValueError("text is empty")
        if self.is_synthetic != (self.pool == Pool.synthetic):
            raise ValueError("is_synthetic must match pool == 'synthetic'")
        if self.is_synthetic and not (
            self.seed_ids and self.generator and self.prompt_id
        ):
            raise ValueError("synthetic rows need seed_ids, generator and prompt_id")
        return self


def _clean(value: object) -> object:
    """Turn pandas missing values into None so Pydantic defaults apply."""
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "tolist"):  # numpy arrays from parquet list columns
        return value.tolist()
    return value


def validate_df(df: pd.DataFrame) -> list[Message]:
    """Validate every row; raise ValueError naming the first bad row's index."""
    out = []
    for idx, row in df.iterrows():
        data = {
            k: v for k, v in ((k, _clean(v)) for k, v in row.items()) if v is not None
        }
        try:
            out.append(Message(**data))
        except ValidationError as e:
            raise ValueError(f"row {idx}: {e}") from e
    return out


def to_df(messages: list[Message]) -> pd.DataFrame:
    """Messages to a DataFrame with the canonical column order."""
    return pd.DataFrame([m.model_dump(mode="json") for m in messages], columns=COLUMNS)
