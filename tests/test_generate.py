import json
from unittest import mock

import pytest

from phishdata import generate as gen

SEEDS = [{"id": f"s{i}", "text": f"seed text {i}", "label": "phish"} for i in range(3)]


def test_fewshot_prompt_uses_only_given_examples() -> None:
    msgs = gen.fewshot_prompt(SEEDS, "phish", "delivery", "佐川急便", "ja", "sms")
    user = msgs[-1]["content"]
    assert all(s["text"] in user for s in SEEDS)
    assert "Japanese" in user and ".example" in user and "090-0000" in user


def test_paraphrase_prompt_keeps_label() -> None:
    seed = {"text": "Your parcel is held", "label": "legit", "brand_claimed": "yamato"}
    user = gen.paraphrase_prompt(seed, "en", "sms")[-1]["content"]
    assert "keep it legitimate" in user and "Your parcel is held" in user


@pytest.mark.parametrize(
    "text,lang,reason",
    [
        ("お荷物をお預かりしています https://sagawa-x.example/a", "ja", None),
        ("Call 090-0000-1234 about your account at amazon-help.test", "en", None),
        ("お荷物をお預かりしています https://sagawa-x.top/a", "ja", "real_domain"),
        ("お問い合わせは090-1234-5678まで", "ja", "real_phone"),
        ("申し訳ありませんが、そのようなメッセージは作成できません。", "ja", "refusal"),
        ("I'm sorry, but I can't help with that.", "en", "refusal"),
        ("Your parcel is waiting for you at the depot", "ja", "wrong_lang"),
        ("ご注文の确认用リンクです https://a.example/x", "ja", "chinese_leak"),
        ("ご注文の確認、お問い合わせ番号をご入力ください", "ja", None),
        ("", "ja", "empty"),
        ("あ" * 2000, "ja", "too_long"),
    ],
)
def test_check_output(text: str, lang: str, reason: str | None) -> None:
    assert gen.check_output(text, lang) == reason


def test_clean_output() -> None:
    assert gen.clean_output('```\nMessage: "Hello there"\n```') == "Hello there"
    assert gen.clean_output("「お知らせです」") == "お知らせです"


def test_chat_sends_seed_and_temperature() -> None:
    resp = mock.MagicMock()
    resp.__enter__.return_value.read.return_value = json.dumps(
        {"message": {"content": "ok"}}
    ).encode()
    with mock.patch("urllib.request.urlopen", return_value=resp) as op:
        assert gen.chat("m", [{"role": "user", "content": "x"}], 0.7, 42) == "ok"
    body = json.loads(op.call_args.args[0].data)
    assert body["options"]["seed"] == 42 and body["options"]["temperature"] == 0.7
    assert body["stream"] is False


def _script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "generate.py"
    spec = importlib.util.spec_from_file_location("generate_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generation_reads_seed_pool_only(tmp_path) -> None:
    import pandas as pd

    script = _script()
    bad = tmp_path / "real_seed.parquet"
    pd.DataFrame({"pool": ["seed", "test"], "text": ["a", "b"]}).to_parquet(bad)
    with pytest.raises(SystemExit, match="non-seed"):
        script.load_seed(bad)
    with pytest.raises(SystemExit, match="may only read"):
        script.load_seed(tmp_path / "real_test.parquet")


def test_plan_job_is_deterministic_and_uses_seed_rows() -> None:
    import pandas as pd

    script = _script()
    df = pd.DataFrame(
        [
            {
                "id": f"s{i}",
                "text": f"t{i}",
                "label": lab,
                "lang": "ja",
                "category": "delivery",
                "channel": "sms",
                "brand_claimed": None,
                "pool": "seed",
            }
            for i, lab in enumerate(["phish", "legit"] * 4)
        ]
    )
    brands = {"sagawa": {"names": ["佐川急便"], "category": "delivery", "domains": []}}
    a, b = script.plan_job(5, 0, df, brands), script.plan_job(5, 0, df, brands)
    assert a == b
    assert {e["id"] for e in a["examples"]} <= set(df["id"])
    assert all(e["label"] == a["label"] for e in a["examples"])
