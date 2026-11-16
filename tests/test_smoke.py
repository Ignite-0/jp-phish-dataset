from phishdata import __version__
from phishdata.sources import load_brands


def test_import_and_packaged_brands() -> None:
    assert __version__
    assert "yamato" in load_brands()
