import pandas as pd
import pytest

from phishdata.schema import Message, Pool, to_df, validate_df

BASE = {
    "id": "r1",
    "text": "お荷物をお届けに上がりましたが不在のため持ち帰りました",
    "channel": "sms",
    "lang": "ja",
    "label": "phish",
    "category": "delivery",
    "source": "manual",
    "campaign_id": "c1",
}


def test_valid_row_roundtrip() -> None:
    msgs = validate_df(pd.DataFrame([BASE]))
    assert msgs[0].pool == Pool.unsplit
    assert validate_df(to_df(msgs))[0] == msgs[0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("channel", "fax"),
        ("lang", "fr"),
        ("label", "spam"),
        ("category", "x"),
        ("pool", "x"),
    ],
)
def test_bad_enum_rejected(field: str, value: str) -> None:
    with pytest.raises(ValueError, match="row 0"):
        validate_df(pd.DataFrame([{**BASE, field: value}]))


def test_empty_text_rejected() -> None:
    with pytest.raises(ValueError):
        Message(**{**BASE, "text": "  "})


def test_synthetic_needs_seed_ids() -> None:
    row = {
        **BASE,
        "pool": "synthetic",
        "is_synthetic": True,
        "generator": "g",
        "prompt_id": "p",
    }
    with pytest.raises(ValueError, match="seed_ids"):
        Message(**row)
    assert Message(**row, seed_ids=["r1"]).is_synthetic


def test_synthetic_flag_matches_pool() -> None:
    with pytest.raises(ValueError, match="is_synthetic"):
        Message(**{**BASE, "is_synthetic": True})
