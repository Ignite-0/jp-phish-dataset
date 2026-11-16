import pandas as pd

from phishdata.splits import assign_pools, dedupe, drop_leaks


def _df(n_campaigns: int = 20, per: int = 3) -> pd.DataFrame:
    rows = []
    for c in range(n_campaigns):
        for k in range(per):
            rows.append(
                {
                    "id": f"r{c}-{k}",
                    "text": f"campaign {c} variant {k} "
                    + "abcdefghij"[c % 10] * (c + 5),
                    "campaign_id": f"c{c}",
                    "brand_claimed": "sagawa" if c == 0 else "yamato",
                    "collected_on": f"2026-01-{c + 1:02d}",
                    "pool": "unsplit",
                }
            )
    return pd.DataFrame(rows)


def test_campaigns_never_span_pools() -> None:
    df = assign_pools(_df(), ["sagawa"], seed=1)
    assert df.groupby("campaign_id")["pool"].nunique().max() == 1
    assert set(df.loc[df["campaign_id"] == "c0", "pool"]) == {"ood"}
    assert set(df["pool"]) == {"ood", "seed", "val", "test"}


def test_split_is_reproducible() -> None:
    a = assign_pools(_df(), ["sagawa"], seed=3)["pool"].tolist()
    assert a == assign_pools(_df(), ["sagawa"], seed=3)["pool"].tolist()


def test_dedupe_keeps_earliest() -> None:
    text = "お荷物をお届けしましたが不在のため持ち帰りました。こちらからご確認ください"
    df = pd.DataFrame(
        {
            "id": ["late", "early"],
            "text": [text + "。", text],
            "collected_on": ["2026-02-01", "2026-01-01"],
        }
    )
    assert dedupe(df)["id"].tolist() == ["early"]


def test_drop_leaks() -> None:
    held = pd.DataFrame(
        {
            "id": ["t1"],
            "pool": ["test"],
            "text": ["【重要】アカウントを一時停止しました。本日中にご確認ください"],
        }
    )
    syn = pd.DataFrame(
        {
            "id": ["s1", "s2"],
            "text": [
                held["text"][0] + "!",
                "全く別の正規のお知らせです。配送予定日のご案内",
            ],
        }
    )
    kept, matches = drop_leaks(syn, held)
    assert kept["id"].tolist() == ["s2"]
    assert matches[0]["synthetic_id"] == "s1" and matches[0]["held_out_pool"] == "test"
