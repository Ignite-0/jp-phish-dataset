"""Dedup, campaign-grouped splitting and leakage removal for the dataset."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from phishdata.dedup import near_duplicates

SEED_FRAC, VAL_FRAC, TEST_FRAC = 0.50, 0.15, 0.35


def dedupe(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Drop near-duplicate messages, keeping the earliest-collected copy."""
    df = df.sort_values("collected_on", na_position="last", kind="stable").reset_index(
        drop=True
    )
    texts = df["text"].tolist()
    drop = {j for _, j, _ in near_duplicates(texts, texts, threshold)}
    return df.drop(index=list(drop)).reset_index(drop=True)


def assign_pools(
    df: pd.DataFrame, ood_brands: list[str], seed: int = 0
) -> pd.DataFrame:
    """Set `pool` to ood/seed/val/test. Whole campaigns go to one pool.

    Any campaign that mentions an OOD brand goes to ood. The rest is split by
    campaign_id with GroupShuffleSplit; fractions are of campaigns, not rows.
    """
    df = df.copy()
    ood_campaigns = set(df.loc[df["brand_claimed"].isin(ood_brands), "campaign_id"])
    is_ood = df["campaign_id"].isin(ood_campaigns)
    df.loc[is_ood, "pool"] = "ood"
    rest = df[~is_ood]
    if rest["campaign_id"].nunique() < 3:
        raise ValueError("need at least 3 non-OOD campaigns to split")

    groups = rest["campaign_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    train_idx, test_idx = next(gss.split(rest, groups=groups))
    df.loc[rest.index[test_idx], "pool"] = "test"

    train = rest.iloc[train_idx]
    gss = GroupShuffleSplit(
        n_splits=1, test_size=VAL_FRAC / (SEED_FRAC + VAL_FRAC), random_state=seed
    )
    seed_idx, val_idx = next(gss.split(train, groups=train["campaign_id"]))
    df.loc[train.index[seed_idx], "pool"] = "seed"
    df.loc[train.index[val_idx], "pool"] = "val"
    return df


def drop_leaks(
    synthetic: pd.DataFrame, held_out: pd.DataFrame, threshold: float = 0.8
) -> tuple[pd.DataFrame, list[dict]]:
    """Remove synthetic rows that nearly match any held-out (val/test/ood) message.

    Returns the kept rows and one record per match for the memorisation report.
    """
    pairs = near_duplicates(
        synthetic["text"].tolist(), held_out["text"].tolist(), threshold
    )
    matches = [
        {
            "synthetic_id": synthetic.iloc[i]["id"],
            "held_out_id": held_out.iloc[j]["id"],
            "held_out_pool": held_out.iloc[j]["pool"],
            "jaccard": round(score, 3),
        }
        for i, j, score in pairs
    ]
    drop = {synthetic.index[i] for i, _, _ in pairs}
    return synthetic.drop(index=list(drop)), matches
