import pytest

from phishdata.dedup import near_duplicates
from phishdata.preprocess import extract_urls, mask_pii, normalize

MASK_CASES = [
    ("お問い合わせは090-1234-5678まで", "お問い合わせは[PHONE]まで"),
    ("電話 03-1234-5678", "電話 [PHONE]"),
    ("０９０－１２３４－５６７８へ", "[PHONE]へ"),
    ("連絡先09012345678", "連絡先[PHONE]"),
    ("Call +81 90 1234 5678 now", "Call [PHONE] now"),
    ("Call +1-202-555-0143 today", "Call [PHONE] today"),
    ("(03)1234-5678", "([PHONE]"),
    ("メール taro@example.com へ", "メール [EMAIL] へ"),
    ("Reply to j.smith+x@mail.example.co.jp", "Reply to [EMAIL]"),
    ("お問い合わせ番号 123456789012", "お問い合わせ番号 [NUMBER]"),
    ("カード 4111 1111 1111 1111 を確認", "カード [NUMBER] を確認"),
    ("口座番号1234567890", "口座番号[NUMBER]"),
    ("山田太郎様 お荷物のお知らせ", "[NAME]様 お荷物のお知らせ"),
    ("ヤマダ タロウ様へ", "[NAME]様へ"),
    ("山田さま、こんにちは", "[NAME]さま、こんにちは"),
    ("お客様のアカウント", "お客様のアカウント"),
    ("カード会員様へ重要なお知らせ", "カード会員様へ重要なお知らせ"),
    ("様々な理由で", "様々な理由で"),
    ("認証コードは123456です", "認証コードは123456です"),
    (
        "確認: https://example.top/a/1234567890123 へ",
        "確認: https://example.top/a/1234567890123 へ",
    ),
]


@pytest.mark.parametrize("raw,expected", MASK_CASES)
def test_mask_pii(raw: str, expected: str) -> None:
    assert mask_pii(raw) == expected


def test_normalize_strips_invisible_and_fullwidth() -> None:
    assert normalize("ログ\u200bイン ｙａｍａｔｏ") == "ログイン yamato"
    assert normalize("差し" + chr(0xFE09) + "押さえ") == "差し押さえ"


@pytest.mark.parametrize(
    "text,urls",
    [
        (
            "こちら https://sagawa-kyubin.top/x?id=1 から確認",
            ["https://sagawa-kyubin.top/x?id=1"],
        ),
        ("ｈｔｔｐｓ：／／ｅｘａｍｐｌｅ．ｔｏｐ／ａ", ["https://example.top/a"]),
        ("確認はamazon-verify.xyz/loginまで", ["amazon-verify.xyz/login"]),
        ("hxxps://bad[.]example[.]icu/p", ["https://bad.example.icu/p"]),
        ("see www.example.com.", ["www.example.com"]),
        ("mail me at a@example.com", []),
        ("添付 report.pdf をご確認", []),
        ("no links here", []),
    ],
)
def test_extract_urls(text: str, urls: list[str]) -> None:
    assert extract_urls(text) == urls


def test_near_duplicates() -> None:
    a = "お荷物をお届けしましたが不在のため持ち帰りました。こちらからご確認ください"
    texts = [a, a + "。", "銀行口座の利用を制限しました"]
    assert [(i, j) for i, j, _ in near_duplicates(texts, texts)] == [(0, 1)]
    assert {j for _, j, _ in near_duplicates(["ｘ" + a], texts, 0.8)} == {0, 1}
