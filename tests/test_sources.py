import pandas as pd

from phishdata import sources
from phishdata.preprocess import detect_lang

BRANDS = {
    "sagawa": {
        "names": ["佐川急便", "佐川"],
        "category": "delivery",
        "domains": ["sagawa-exp.co.jp"],
    },
    "amazon": {
        "names": ["Amazon"],
        "category": "shopping",
        "domains": ["amazon.co.jp"],
    },
}

# Hand-written page in the same shape as a real alert (no real message content).
ALERT_HTML = """<html><body>
<h3 class="default_h3">佐川急便をかたるフィッシング (2026/01/02)</h3>
<p class="date">2026年01月02日</p>
<div class="entry">
 <h4 class="alert_h4">概要</h4><div><p>佐川急便をかたるショートメッセージ (SMS) の報告を受けています。</p></div>
 <h4 class="alert_h4">メールの件名</h4><div><p>お荷物のお届けについて<br>不在通知<br>※上記以外の件名も使われる可能性があります。</p></div>
 <h4 class="alert_h4">サイトのURL</h4><div><p>https://sagawa-●●●●.top/a<br>http://x●●●●.com/b</p></div>
 <h4 class="alert_h4">メール・SMSの文面例</h4><p class="magnify">拡大する</p><div><img src="/news/images/x01.png"></div>
</div></body></html>"""


def test_alert_links() -> None:
    html = '<a href="/news/alert/a_1.html">a</a><a href="/news/alert/">idx</a><a href="/x.html">x</a>'
    assert sources.ap_alert_links(html) == [
        "https://www.antiphishing.jp/news/alert/a_1.html"
    ]


def test_parse_alert() -> None:
    r = sources.parse_ap_alert(
        ALERT_HTML,
        "https://www.antiphishing.jp/news/alert/sagawa_20260102.html",
        BRANDS,
    )
    assert r["campaign_id"] == "ap-sagawa_20260102"
    assert r["collected_on"] == "2026-01-02"
    assert (r["brand_claimed"], r["category"], r["channel"]) == (
        "sagawa",
        "delivery",
        "sms",
    )
    assert r["subjects"] == "お荷物のお届けについて | 不在通知"
    assert r["urls"].split() == ["https://sagawa-●●●●.top/a", "http://x●●●●.com/b"]
    assert r["screenshot"].endswith("/news/images/x01.png")
    assert r["text"] == ""


def test_review_rows_need_text_and_get_masked() -> None:
    r = sources.parse_ap_alert(
        ALERT_HTML, "https://www.antiphishing.jp/news/alert/s.html", BRANDS
    )
    df = pd.DataFrame(
        [r, {**r, "text": "山田太郎様 お荷物を持ち帰りました https://sagawa-x.top/a"}]
    )
    msgs = sources.ap_review_to_messages(df)
    assert len(msgs) == 1
    assert msgs[0].text.startswith("[NAME]様") and msgs[0].label == "phish"
    assert msgs[0].url_list == ["https://sagawa-x.top/a"]


def test_uci_uses_ham_only() -> None:
    raw = b"ham\tSee you at 5, call 090-1234-5678\nspam\tWIN a prize now\nham\t\n"
    msgs = sources.load_uci(raw, BRANDS)
    assert [m.label for m in msgs] == ["legit"]
    assert "[PHONE]" in msgs[0].text


def test_smishtank_finds_text_column() -> None:
    csv = "MainText,Timestamp\nYour Amazon account is locked: amaz0n-help.xyz/login,2023-05-01\n"
    (m,) = sources.load_smishtank(csv, BRANDS)
    assert (m.label, m.brand_claimed, m.campaign_id) == (
        "phish",
        "amazon",
        "host-amaz0n-help.xyz",
    )
    assert str(m.collected_on) == "2023-05-01"


def test_manual_fills_blanks() -> None:
    csv = "text,label,channel\nヤマトからのお知らせ、佐川急便です,legit,email\n"
    (m,) = sources.load_manual(csv, BRANDS)
    assert (m.brand_claimed, m.category, m.lang) == ("sagawa", "delivery", "ja")


def test_detect_lang() -> None:
    assert detect_lang("お荷物のお届けについて") == "ja"
    assert detect_lang("Your parcel is waiting") == "en"
    assert detect_lang("【重要】Please verify your account 本日中に") == "mixed"


def test_match_brand_needs_whole_latin_words() -> None:
    brands = {
        **BRANDS,
        "au": {"names": ["au", "KDDI"], "category": "carrier", "domains": []},
        "etc": {"names": ["ETC"], "category": "other", "domains": []},
    }
    assert sources.match_brand("because it was late, etc.", brands) is None
    assert sources.match_brand("【au】ご利用料金のお知らせ", brands) == "au"
    assert sources.match_brand("your amazon order", brands) == "amazon"
    assert sources.match_brand("ETC利用照会サービス", brands) == "etc"


def test_match_brand_spacing_and_case() -> None:
    brands = {
        "mufg": {"names": ["三菱UFJ銀行"], "category": "banking", "domains": []},
        "line": {"names": ["LINE"], "category": "account", "domains": []},
        "linepay": {"names": ["LINE Pay"], "category": "banking", "domains": []},
    }
    assert sources.match_brand("三菱 UFJ 銀行をかたるフィッシング", brands) == "mufg"
    assert sources.match_brand("ＬＩＮＥをかたるフィッシング", brands) == "line"
    assert sources.match_brand("LINE Pay をかたる", brands) == "linepay"
    assert sources.match_brand("drop me a line later", brands) is None
